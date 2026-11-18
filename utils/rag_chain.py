"""Retrieval-Augmented Generation pipeline.

Combines ChromaDB retrieval with Gemini generation:

1. Condense follow-up questions into standalone search queries (chat memory).
2. Retrieve the top-k most similar chunks.
3. Build a prompt with numbered sources so answers can cite them.
4. Stream the answer back to the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Mapping, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.config import Settings, get_settings

SYSTEM_PROMPT = """You are GetMeContext, a careful document analyst. Answer questions \
using ONLY the numbered sources provided.

Rules:
- Use only the sources. Never use outside knowledge or make assumptions.
- Cite the sources you rely on with bracketed numbers, e.g. [1] or [2][3].
- If the answer is not in the sources, reply exactly:
  "The answer is not available in the uploaded document."
- Be concise but complete. Use bullet points for lists and steps.
- Use the conversation history only to understand follow-up questions; \
never treat it as a source of facts."""

ANSWER_TEMPLATE = """SOURCES:
{context}

QUESTION: {question}"""

CONDENSE_PROMPT = """Rewrite the follow-up question as a standalone search query for \
document retrieval. Keep all specific names, numbers, and topics from the conversation. \
Return only the rewritten question, with no explanation.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""


@dataclass(frozen=True)
class Source:
    """A citation the answer can reference, numbered to match the prompt."""

    index: int
    document: str
    page: int  # 1-based; 0 means page unknown
    snippet: str


@dataclass
class PreparedAnswer:
    """Everything needed to stream an answer and display its citations."""

    messages: List[BaseMessage]
    sources: List[Source]


def get_llm(settings: Settings | None = None) -> ChatGoogleGenerativeAI:
    """Initialise the Gemini chat model."""
    settings = settings or get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.temperature,
        max_retries=3,
    )


def message_text(message: Any) -> str:
    """Extract plain text from an LLM response or streamed chunk.

    Gemini can return either a string or a list of content blocks depending on
    the SDK version, so normalise both shapes.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        return str(content.get("text", ""))
    if isinstance(content, Sequence):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


def format_history(history: Sequence[Mapping[str, Any]], window: int) -> str:
    """Render the last ``window`` turns as plain text for question condensing."""
    if window <= 0 or not history:
        return ""
    turns = list(history)[-window * 2 :]
    lines = []
    for message in turns:
        role = "User" if message.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {message.get('content', '')}")
    return "\n".join(lines)


def to_chat_messages(
    history: Sequence[Mapping[str, Any]], window: int
) -> List[BaseMessage]:
    """Convert session-state history dicts into LangChain chat messages."""
    if window <= 0 or not history:
        return []
    messages: List[BaseMessage] = []
    for message in list(history)[-window * 2 :]:
        text = str(message.get("content", ""))
        if message.get("role") == "user":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))
    return messages


def format_context(documents: Iterable[Document]) -> Tuple[str, List[Source]]:
    """Build the numbered context block and the matching citation list."""
    blocks: List[str] = []
    sources: List[Source] = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        name = str(metadata.get("source", "document"))
        page = int(metadata.get("page", -1)) + 1
        content = (document.page_content or "").strip()
        snippet = " ".join(content.split())
        if len(snippet) > 240:
            snippet = snippet[:237].rstrip() + "..."

        sources.append(Source(index=index, document=name, page=page, snippet=snippet))
        blocks.append(f"[{index}] {name} — page {page}\n{content}")

    return "\n\n".join(blocks), sources


def build_messages(
    context: str,
    question: str,
    history: Sequence[Mapping[str, Any]] = (),
    window: int = 3,
) -> List[BaseMessage]:
    """Assemble the final chat messages sent to the LLM."""
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        *to_chat_messages(history, window),
        HumanMessage(
            content=ANSWER_TEMPLATE.format(context=context, question=question)
        ),
    ]


class RAGPipeline:
    """Stateful RAG pipeline bound to a retriever and an LLM."""

    def __init__(self, retriever: Any, llm: Any, conversation_window: int = 3):
        self.retriever = retriever
        self.llm = llm
        self.conversation_window = conversation_window

    def standalone_question(
        self, question: str, history: Sequence[Mapping[str, Any]]
    ) -> str:
        """Rewrite a follow-up question so retrieval does not depend on context."""
        history_text = format_history(history, self.conversation_window)
        if not history_text:
            return question

        prompt = CONDENSE_PROMPT.format(history=history_text, question=question)
        try:
            rewritten = message_text(self.llm.invoke(prompt)).strip()
        except Exception:  # noqa: BLE001 - retrieval should still work without it
            return question
        return rewritten or question

    def retrieve(
        self, question: str, history: Sequence[Mapping[str, Any]]
    ) -> List[Document]:
        """Retrieve relevant chunks for a question, resolving follow-ups first."""
        query = self.standalone_question(question, history)
        return list(self.retriever.invoke(query))

    def prepare(
        self, question: str, history: Sequence[Mapping[str, Any]]
    ) -> PreparedAnswer:
        """Retrieve context and build the prompt, before any tokens are streamed."""
        documents = self.retrieve(question, history)
        context, sources = format_context(documents)
        messages = build_messages(
            context=context,
            question=question,
            history=history,
            window=self.conversation_window,
        )
        return PreparedAnswer(messages=messages, sources=sources)

    def stream(self, prepared: PreparedAnswer) -> Iterator[str]:
        """Yield answer tokens as they arrive from Gemini."""
        for chunk in self.llm.stream(prepared.messages):
            text = message_text(chunk)
            if text:
                yield text

    def answer(
        self, question: str, history: Sequence[Mapping[str, Any]] = ()
    ) -> Tuple[str, List[Source]]:
        """Non-streaming convenience method used by tests and scripts."""
        prepared = self.prepare(question, history)
        return "".join(self.stream(prepared)), prepared.sources


def create_rag_pipeline(
    vector_store: Any, settings: Settings | None = None
) -> RAGPipeline:
    """Wire a Chroma vector store into a ready-to-use RAG pipeline."""
    settings = settings or get_settings()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.top_k},
    )
    return RAGPipeline(
        retriever=retriever,
        llm=get_llm(settings),
        conversation_window=settings.conversation_window,
    )
