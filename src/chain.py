from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.llm import get_llm

from src.ingestion import get_vectorstore
from src.retrieval import get_reranked_retriever

PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the context doesn't contain the answer, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
)


def format_context(docs):
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{source}, page {page + 1}" if page is not None else source
        parts.append(f"[{label}]\n{doc.page_content}")
    return "\n\n".join(parts)


def get_rag_chain():
    llm = get_llm()
    return (
        {"context": lambda x: format_context(x["docs"]), "question": lambda x: x["question"]}
        | PROMPT
        | llm
        | StrOutputParser()
    )

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Given the conversation history and a follow-up question, rewrite the follow-up
into a standalone question that makes sense without the history.
If the question is already standalone, return it unchanged.

Chat history:
{history}

Follow-up question: {question}

Standalone question:"""
)


def get_query_rewriter():
    llm = get_llm()
    return REWRITE_PROMPT | llm | StrOutputParser()


def format_history(messages):
    """messages: list of (role, text) tuples, e.g. ('user', 'hi')"""
    return "\n".join(f"{role}: {text}" for role, text in messages)

from src.retrieval import route_and_retrieve

def answer_question(question, history, rag_chain, rewriter, checker, vectorstore, router, hybrid_retriever):
    if history:
        question = rewriter.invoke({
            "history": format_history(history),
            "question": question,
        })

    docs, strategy = route_and_retrieve(question, vectorstore, router, hybrid_retriever)
    context = format_context(docs)
    answer = rag_chain.invoke({"docs": docs, "question": question})

    check = checker.invoke({"context": context, "answer": answer})

    return {
        "question": question,
        "answer": answer,
        "sources": docs,
        "is_faithful": check.is_faithful,
        "faithfulness_reason": check.reason,
        "strategy": strategy,
    }

def prepare_answer(question, history, rewriter, vectorstore, router, hybrid_retriever):
    """Rewrite the follow-up (if any) and retrieve. Returns what the stream needs."""
    if history:
        question = rewriter.invoke({
            "history": format_history(history),
            "question": question,
        })

    docs, strategy = route_and_retrieve(question, vectorstore, router, hybrid_retriever)
    return question, docs, strategy


def check_answer(checker, docs, answer):
    """Run the faithfulness check on a finished answer."""
    return checker.invoke({"context": format_context(docs), "answer": answer})   
 

from pydantic import BaseModel, Field


FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template(
    """Given the context and the answer below, judge whether the answer is faithful
to the context.

An answer is faithful if every claim it makes is supported by the context.
An answer can be faithful even if it leaves out details from the context — leaving
things out is fine. An answer is NOT faithful only if it adds information, facts,
or claims that the context does not contain or contradict.

Context:
{context}

Answer:
{answer}

"""
)


class FaithfulnessResult(BaseModel):
    is_faithful: bool = Field(description="True if the answer is fully supported by the context")
    reason: str = Field(description="One short sentence explaining the judgment")


def get_faithfulness_checker():
    llm = get_llm()
    structured_llm = llm.with_structured_output(FaithfulnessResult)
    return FAITHFULNESS_PROMPT | structured_llm


if __name__ == "__main__":
    from src.ingestion import get_vectorstore
    from src.retrieval import get_hybrid_retriever, get_router

    vectorstore = get_vectorstore()
    hybrid_retriever = get_hybrid_retriever(vectorstore)
    router = get_router()
    rag_chain = get_rag_chain()
    rewriter = get_query_rewriter()
    checker = get_faithfulness_checker()

    history = []
    q1 = "What do teams collect in distributed approaches to digital ecosystems?"
    result1 = answer_question(q1, history, rag_chain, rewriter, checker, vectorstore, router, hybrid_retriever)
    print(result1)