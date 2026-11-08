from typing import Literal
from pydantic import BaseModel, Field
from src.ingestion import get_vectorstore
from src.retrieval import (
    get_router,
    get_hybrid_retriever,
    route_and_retrieve
)
from langchain_core.prompts import ChatPromptTemplate
from src.llm import get_llm
import logging
from src.crag import rewrite_query
from functools import lru_cache
logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2

STRICT_NOTE = (
    "5. IMPORTANT: Use ONLY facts explicitly stated in the context. "
    "Remove any claim you cannot point to in the context."
)
logging.basicConfig( level=logging.INFO,  # change to logging.DEBUG to see everything
                    format="%(asctime)s | %(levelname)s | %(message)s" )

# 1. Retrieval Decision
class RetrievalDecision(BaseModel):

    reasoning: str = Field(
        description=(
            "Reason for the retrieval decision. "
            "Write this before deciding."
        )
    )
    decision: Literal["retrieve", "no_retrieve"] = Field(
        description="Whether document retrieval is required, based on the reasoning above."
    )

# 2. Document Relevance Evaluation
class DocumentRelevance(BaseModel):

    document_number: int = Field(
        description=(
            "The number of the document being evaluated, exactly as "
            "shown in the prompt (Document 1, Document 2, ...)."
        )
    )
    reasoning: str = Field(
        description=(
            "Reason for the relevance decision. "
            "Write this before deciding relevance."
        )
    )
    relevance: Literal["relevant", "irrelevant"] = Field(
        description="Whether the document is relevant, based on the reasoning above."
    )
class BatchDocumentRelevance(BaseModel):

    evaluations: list[DocumentRelevance] = Field(
        description="Relevance evaluation for every supplied document, in the same order."
    )
class AnswerEvaluation(BaseModel):

    reasoning: str = Field(
        description=(
            "Explain whether the answer is supported by the context and "
            "whether it addresses the question. Write this before deciding."
        )
    )

    support: Literal["supported", "unsupported", "partially_supported"] = Field(
        description="Whether the answer is supported by the context, based on the reasoning above."
    )

    usefulness: Literal["useful", "not_useful"] = Field(
        description="Whether the answer addresses the user's question, based on the reasoning above."
    )


# Step 2: Create Self-RAG evaluation chains
@lru_cache(maxsize=None)
def get_retrieval_decision_chain():

        prompt = ChatPromptTemplate.from_template("""
    You are a retrieval decision evaluator.

    Determine whether the question requires retrieving
    information from the knowledge base.

    The knowledge base contains the user's own documents: project
    write-ups, notes, and PDFs. Questions about the user's projects,
    their details, or anything that could be in those documents need
    retrieval.

    Question:
    {question}

    Return:
    - retrieve: if the question could be answered from the knowledge base,
      or if you are unsure.
    - no_retrieve: only if the question is clearly general knowledge or
      simple conversation (greetings, thanks, math, definitions).
    """)

        return prompt | get_llm().with_structured_output(RetrievalDecision,method="json_schema"
)

@lru_cache(maxsize=None)
def get_batch_document_relevance_chain():

    prompt = ChatPromptTemplate.from_template("""
    You are a document relevance evaluator.

    Evaluate ALL the documents against the user's question.

    Question:
    {question}

    Documents:
    {documents}

    For every document, determine whether it is:
    - relevant: contains information that helps answer the question.
    - irrelevant: does not help answer the question.

    Return exactly one evaluation for each document.
    Preserve the original document order.
    """)

    return prompt | get_llm().with_structured_output(BatchDocumentRelevance,method="json_schema"
)

@lru_cache(maxsize=None)
def get_answer_evaluation_chain():

    prompt = ChatPromptTemplate.from_template("""
    You are a Self-RAG answer evaluator.

    Evaluate the generated answer using the provided context
    and the user's question.

    Question:
    {question}

    Context:
    {context}

    Generated Answer:
    {answer}

    Evaluate TWO things:

    1. SUPPORT:
       - supported: All claims are supported by the context.
       - partially_supported: Some claims are supported, but others are not.
       - unsupported: The answer is not supported by the context.

    2. USEFULNESS:
       - useful: The answer clearly addresses the user's question.
       - not_useful: The answer does not properly address the question.

    Provide reasoning for your evaluations.
    """)

    return prompt | get_llm().with_structured_output(AnswerEvaluation, method="json_schema"
)

def evaluate_documents(question, documents, chain=None):

    if not documents:
        return []

    if chain is None:
        chain = get_batch_document_relevance_chain()

    formatted_documents = "\n\n".join(
        f"Document {i + 1}:\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )

    # Try the batched call, retrying once if the numbers don't line up
    for attempt in range(2):

        result = chain.invoke({
            "question": question,
            "documents": formatted_documents
        })

        by_number = {e.document_number: e for e in result.evaluations}

        if all(n in by_number for n in range(1, len(documents) + 1)):
            return [
                (doc, by_number[n])
                for n, doc in enumerate(documents, 1)
            ]

        logger.warning(
            "Evaluation numbers did not match documents (attempt %d)",
            attempt + 1
        )

    # Fallback: evaluate each document on its own
    logger.warning("Falling back to one-by-one document evaluation")

    evaluated = []

    for doc in documents:

        single = chain.invoke({
            "question": question,
            "documents": f"Document 1:\n{doc.page_content}"
        })

        if single.evaluations:
            evaluated.append((doc, single.evaluations[0]))
        else:
            evaluated.append((
                doc,
                DocumentRelevance(
                    document_number=1,
                    reasoning="Evaluation failed; treated as irrelevant.",
                    relevance="irrelevant"
                )
            ))

    return evaluated

def evaluate_answer(question, context, answer, chain=None):

    if chain is None:
        chain = get_answer_evaluation_chain()

    evaluation = chain.invoke({
        "question": question,
        "context": context,
        "answer": answer
    })

    return evaluation

def decide_retrieval(question, chain=None):

    if chain is None:
        chain = get_retrieval_decision_chain()

    decision = chain.invoke({
        "question": question
    })

    return decision

@lru_cache(maxsize=None)
def get_answer_generation_chain():

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant.

    Answer the user's question using the provided context.

    Question:
    {question}

    Context:
    {context}

    Instructions:
    1. If context is available, prioritize information from it.
    2. Do not invent facts that are not supported by the context.
    3. If the context is insufficient, clearly mention that.
    4. Provide a clear and direct answer.
    {extra_instructions}
    """)

    return prompt | get_llm()


def generate_answer(question, context="", chain=None, strict=False):

    if chain is None:
        chain = get_answer_generation_chain()

    response = chain.invoke({
        "question": question,
        "context": context,
        "extra_instructions": STRICT_NOTE if strict else ""
    })

    # Extract the response content
    content = response.content

    # Handle normal string responses
    if isinstance(content, str):
        return content

    # Handle Claude's multiple content blocks
    if isinstance(content, list):

        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
        ]

        return "\n".join(text_parts)

    return str(content)

@lru_cache(maxsize=None)
def get_retrieval_components(k=4):
    vectorstore = get_vectorstore()
    router = get_router()
    hybrid_retriever = get_hybrid_retriever(
        vectorstore=vectorstore,
        k=k
    )
    return vectorstore, router, hybrid_retriever


def retrieve_documents(question, k=4):

    vectorstore, router, hybrid_retriever = get_retrieval_components(k)

    documents, strategy = route_and_retrieve(
        question=question,
        vectorstore=vectorstore,
        router=router,
        hybrid_retriever=hybrid_retriever,
        k=k
    )

    return documents, strategy


def retrieve_and_evaluate(question, k=4):

    # Step 1: Retrieve documents
    documents, strategy = retrieve_documents(question, k)

    logger.info("Retrieved %d documents using strategy: %s", len(documents), strategy)

    for index, document in enumerate(documents, start=1):
        logger.debug("Document %d: %s", index, document.page_content[:500])

    # Step 2: Evaluate documents
    evaluations = evaluate_documents(question, documents)

    for index, (document, evaluation) in enumerate(evaluations, start=1):
        logger.info("Document %d relevance: %s", index, evaluation.relevance)
        logger.debug("Reasoning: %s", evaluation.reasoning)

    return evaluations, strategy

def filter_relevant_documents(evaluations):

    # Keep only documents classified as relevant
    relevant_documents = [
        document
        for document, evaluation in evaluations
        if evaluation.relevance == "relevant"
    ]

    return relevant_documents

def format_context(documents):

    if not documents:
        return ""

    parts = []

    for document in documents:

        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page")

        label = f"Source: {source}"
        if page is not None:
            label += f" (page {page})"

        parts.append(f"{label}\n{document.page_content}")

    return "\n\n".join(parts)

def self_rag_pipeline(question):

    # Step 1: Decide whether retrieval is necessary
    decision = decide_retrieval(question)

    logger.info("Retrieval decision: %s", decision.decision)
    logger.debug("Reasoning: %s", decision.reasoning)

    # Step 2: Generate directly if retrieval is unnecessary
    if decision.decision == "no_retrieve":

        answer = generate_answer(question)

        return {
            "answer": answer,
            "strategy": None,
            "documents": [],
            "evaluation": None
        }

    query = question
    strict = False
    need_retrieval = True
    relevant_documents, strategy, context = [], None, ""
    result = None

    for attempt in range(1, MAX_ATTEMPTS + 1):

        logger.info("Attempt %d/%d", attempt, MAX_ATTEMPTS)

        if need_retrieval:
            evaluations, strategy = retrieve_and_evaluate(query)
            relevant_documents = filter_relevant_documents(evaluations)
            context = format_context(relevant_documents)

        # Nothing relevant: rewrite the query and retry
        if not relevant_documents:
            if attempt < MAX_ATTEMPTS:
                query = rewrite_query(question)
                logger.info("No relevant documents. Rewritten query: %s", query)
                need_retrieval = True
                continue
            break

        answer = generate_answer(question, context, strict=strict)
        evaluation = evaluate_answer(question, context, answer)

        logger.info(
            "Support: %s | Usefulness: %s",
            evaluation.support, evaluation.usefulness
        )

        result = {
            "answer": answer,
            "strategy": strategy,
            "documents": relevant_documents,
            "evaluation": evaluation
        }

        if evaluation.support == "supported" and evaluation.usefulness == "useful":
            break

        if attempt == MAX_ATTEMPTS:
            break

        if evaluation.usefulness == "not_useful":
            # Retrieval was the problem: rewrite and retrieve again
            query = rewrite_query(question, context)
            logger.info("Answer not useful. Rewritten query: %s", query)
            need_retrieval = True
        else:
            # Context is fine, the answer drifted: regenerate strictly
            strict = True
            need_retrieval = False

    if result is None:
        return {
            "answer": "I couldn't find sufficient relevant information in the knowledge base to answer this question.",
            "strategy": strategy,
            "documents": [],
            "evaluation": None
        }

    return result



if __name__ == "__main__":

    question = "Which technologies are used in the Multi-Document RAG Assistant project?"

    # Run the complete Self-RAG pipeline
    result = self_rag_pipeline(question)

    print("\n========== FINAL SELF-RAG RESULT ==========")

    print("\nSelected Strategy:")
    print(result["strategy"])

    print("\nRelevant Documents:")
    print(len(result["documents"]))

    print("\nGenerated Answer:")
    print(result["answer"])

    print("\nAnswer Evaluation:")

    if result["evaluation"]:
        print("Support:", result["evaluation"].support)
        print("Usefulness:", result["evaluation"].usefulness)
        print("Reasoning:", result["evaluation"].reasoning)
    else:
        print("Answer evaluation skipped.")