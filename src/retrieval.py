from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm import get_llm
from dotenv import load_dotenv
load_dotenv()
from src.ingestion import get_vectorstore
from functools import lru_cache

# ---------- Sparse (BM25) ----------

def get_all_chunks(vectorstore):
    """Pull every stored chunk back out of the FAISS index."""
    if vectorstore is None:
        return []

    return list(vectorstore.docstore._dict.values())


def get_bm25_retriever(vectorstore, k=4):
    chunks = get_all_chunks(vectorstore)

    if not chunks:
        raise ValueError("No documents available for BM25 retrieval.")

    return BM25Retriever.from_documents(chunks, k=k)


# ---------- Hybrid (dense + BM25) ----------

def get_hybrid_retriever(vectorstore,k=4, dense_weight=0.5,bm25_weight=0.5):
    dense = get_dense_retriever(vectorstore, k=k)
    bm25 = get_bm25_retriever(vectorstore, k=k)

    return EnsembleRetriever(retrievers=[dense, bm25], weights=[dense_weight, bm25_weight],)


# ---------- Re-ranking ----------

def get_reranked_retriever(vectorstore, candidates=8, top_n=3):
    hybrid = get_hybrid_retriever(vectorstore, k=candidates)
    cross_encoder = get_cross_encoder()
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid,
    )
@lru_cache(maxsize=1)
def get_cross_encoder():
    return HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

# ---------- Multi-query ----------

def get_multiquery_retriever(vectorstore, k=4):
    base_retriever = get_dense_retriever(vectorstore, k=k)
    llm = get_llm()

    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        include_original=True,
        parser_key="lines",
    )


# ---------- HyDE ----------

HYDE_PROMPT = ChatPromptTemplate.from_template(
    """Write a short hypothetical passage that would answer this question,
as if it came from a document. Do not mention that it's hypothetical.

Question: {question}

Passage:"""
)


def get_hyde_retriever(vectorstore, k=4):
    llm = get_llm()
    hyde_chain = HYDE_PROMPT | llm | StrOutputParser()

    def retrieve(question):
        hypothetical_doc = hyde_chain.invoke({"question": question}).strip()

        if not hypothetical_doc:
            return vectorstore.similarity_search(question, k=k)

        return vectorstore.similarity_search(hypothetical_doc, k=k)

    return retrieve  # a plain function, not a LangChain Retriever object


# ---------- Router (automatic strategy selection) ----------

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """Classify the retrieval strategy that best fits this question.
Reply with exactly one word: "standard", "multiquery", "hyde", or "mmr".

- "standard": Simple, direct questions with clear keywords.
- "multiquery": Broad, vague, or ambiguous questions.
- "hyde": Questions where a hypothetical passage may improve semantic matching.
- "mmr": Questions requiring diverse information from multiple parts of the documents.

Question: {question}

Strategy:"""
)


def get_router():
    llm = get_llm()
    return ROUTER_PROMPT | llm | StrOutputParser()


def route_and_retrieve(question, vectorstore, router, hybrid_retriever, k=4):
    strategy = "standard"

    try:
        selected_strategy = (
            router.invoke({"question": question})
            .strip()
            .lower()
        )

        if "multiquery" in selected_strategy:
            strategy = "multiquery"
            docs = get_multiquery_retriever(
                vectorstore, k=k
            ).invoke(question)

        elif "hyde" in selected_strategy:
            strategy = "hyde"
            docs = get_hyde_retriever(
                vectorstore, k=k
            )(question)

        elif "mmr" in selected_strategy:
            strategy = "mmr"
            docs = get_mmr_retriever(
                vectorstore,
                k=k,
                fetch_k=20,
                lambda_mult=0.5
            ).invoke(question)

        else:
            strategy = "standard"
            docs = get_reranked_retriever(
                vectorstore, candidates=8, top_n=3
            ).invoke(question)

        return docs, strategy

    except Exception as e:
        print(f"Retrieval strategy '{strategy}' failed: {e}")
        print("Falling back to standard hybrid retrieval.")

        docs = hybrid_retriever.invoke(question)
        return docs, "standard_fallback"

def get_dense_retriever(vectorstore, k=4):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
#----------MMR---------------------
def get_mmr_retriever(
    vectorstore,
    k=4,
    fetch_k=20,
    lambda_mult=0.5
):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        }
    )

# ---------- Quick test ----------

if __name__ == "__main__":
    vectorstore = get_vectorstore()

    question = "Explain the different topics covered in these documents."

    retriever = get_mmr_retriever(
        vectorstore,
        k=4,
        fetch_k=20,
        lambda_mult=0.5
    )

    docs = retriever.invoke(question)

    for i, doc in enumerate(docs, 1):
        print(f"\n--- Result {i} ---")
        print("Source:", doc.metadata.get("source"))
        print("Content:", doc.page_content[:500])