"""
services/llm.py — GPT-4.1 powered answer generation and text summarization.

This module contains the three LLM-facing functions used by the application:

- :func:`generate_answer` — The primary RAG answer function.  Takes a user
  question and retrieved context chunks, calls GPT-4.1, and returns a cited
  answer.

- :func:`generate_document_summary` — Generates a comprehensive summary of
  an entire document from all its indexed chunks.
"""

from openai import AzureOpenAI

from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
)


def _get_client() -> AzureOpenAI:
    """Create and return an authenticated AzureOpenAI client.

    Constructs a new ``AzureOpenAI`` client using the endpoint, API key, and
    API version defined in :mod:`config`.  A new client is created on every
    call — this is intentional as the ``openai`` SDK clients are lightweight
    and stateless, and caching them at module level would complicate testing.

    Returns:
        AzureOpenAI: A configured client ready to make chat completion and
            embedding requests against the Azure OpenAI resource.
    """
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def generate_answer(
    query: str, context_chunks: list[dict], language: str = "English"
) -> str:
    """Generate a grounded, cited answer to a user question from document chunks.

    Constructs a structured prompt that presents the retrieved document chunks
    as numbered context blocks (``[Source N - Page P]``) and instructs GPT-4.1
    to answer the question using only those sources.  The model is directed to
    cite sources explicitly so users can verify which pages support each claim.

    Args:
        query (str): The user's natural-language question as entered in the
            chat interface or transcribed from voice input.
        context_chunks (list[dict]): Ranked list of document chunks returned
            by :func:`~services.search.hybrid_search`.  Each dict must contain
            ``content`` (str) and ``page_number`` (int).
        language (str): Target output language display name as defined in
            ``LANGUAGE_CONFIG`` (e.g., ``"English"``, ``"Hindi"``,
            ``"French"``, ``"Telugu"``).  Defaults to ``"English"``,
            in which case no translation is performed.

    Returns:
        str: A comprehensive answer in the target language with inline
            citations referencing ``[Source N - Page P]`` markers.
            If the model cannot find a relevant answer in the context, it
            returns a polite "not found" message.

    Raises:
        openai.AuthenticationError: If the API key is invalid.
        openai.RateLimitError: If the GPT deployment quota is exceeded.
        openai.BadRequestError: If the combined context + question exceeds
            the model's context window.
    """
    client = _get_client()

    context_parts = [
        f"[Source {i} - Page {chunk['page_number']}]\n{chunk['content']}"
        for i, chunk in enumerate(context_chunks, 1)
    ]
    context_text = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant that answers questions based on provided document excerpts. "
        "Always cite your sources using the [Source N - Page P] references provided. "
        "If the answer is not found in the provided context, say so clearly. "
        "Always respond in English."
    )
    user_prompt = (
        f"Context from the document:\n\n{context_text}\n\n---\n\n"
        f"Question: {query}\n\n"
        "Provide a comprehensive answer with citations referencing the source numbers and page numbers above."
    )

    # 5. TODO: Uncomment below code to utilize LLM to generate answer to user queries
    # response = client.chat.completions.create(
    #     model=AZURE_OPENAI_DEPLOYMENT,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ],
    #     temperature=0.3,
    #     max_tokens=1500,
    # )

    english_answer = response.choices[0].message.content
    return english_answer


def generate_document_summary(
    chunks: list[dict], language: str = "English"
) -> str:
    """Generate a comprehensive summary of an entire document from its indexed chunks.

    Concatenates all chunk content in page order (as provided) into a single
    context block and requests a structured, well-organised summary from
    GPT-4.1.  Unlike :func:`generate_answer` (which answers a specific
    question), this function instructs the model to capture the overall
    document: key topics, findings, arguments, and conclusions.

    The system prompt requests:

    * A **well-structured** summary with bullet points where appropriate.
    * **Page-level citations** when referencing specific details.
    * Response directly in the target language.

    The summary uses ``max_tokens=3000`` (compared to 1500 for
    :func:`generate_answer`) because whole-document summaries are inherently
    longer.

    Args:
        chunks (list[dict]): All chunks for a single document, as returned by
            :func:`~services.search.fetch_chunks_by_document`.  Each dict must
            contain ``content`` (str) and ``page_number`` (int).  Chunks should
            be ordered by page then offset.
        language (str): Target output language display name.  Defaults to
            ``"English"``.

    Returns:
        str: A formatted multi-paragraph (or bulleted) summary of the
            document in the target language, with page-level citations.

    Raises:
        openai.AuthenticationError: If the API key is invalid.
        openai.RateLimitError: If the GPT deployment quota is exceeded.
        openai.BadRequestError: If the total document text exceeds the model's
            context window (very large documents may need to be split further).
    """
    client = _get_client()

    context_parts = [
        f"[Page {chunk['page_number']}]\n{chunk['content']}"
        for chunk in chunks
    ]
    context_text = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant that produces comprehensive document summaries. "
        "You will receive the full text of a document split into page-level chunks. "
        "Write a well-structured summary that captures the key topics, findings, "
        "arguments, and conclusions of the document. Use bullet points where appropriate. "
        "Reference page numbers when citing specific details. "
        f"Always respond in {language}."
    )
    user_prompt = (
        f"Document content:\n\n{context_text}\n\n---\n\n"
        "Provide a comprehensive summary of this entire document."
    )

    # 6. TODO: Uncomment below code to generate summary of the document selected by user.
    # response = client.chat.completions.create(
    #     model=AZURE_OPENAI_DEPLOYMENT,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ],
    #     temperature=0.3,
    #     max_tokens=3000,
    # )

    english_summary = response.choices[0].message.content
    return english_summary
