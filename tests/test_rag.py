from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from utils.rag_chain import (
    RAGPipeline,
    build_messages,
    format_context,
    message_text,
    to_chat_messages,
)


class StubRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.documents


class StubLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.responses.pop(0))

    def stream(self, messages):
        self.calls.append(messages)
        for character in self.responses.pop(0):
            yield AIMessageChunk(content=character)


class FailingCondenseLLM(StubLLM):
    def invoke(self, messages):
        raise RuntimeError("quota exceeded")


def sample_documents():
    return [
        Document(page_content="Submit by 15 May.", metadata={"source": "a.pdf", "page": 2}),
        Document(page_content="Office hours are on Fridays.", metadata={"source": "b.pdf", "page": 0}),
    ]


def test_format_context_numbers_sources_and_pages():
    context, sources = format_context(sample_documents())

    assert "[1] a.pdf — page 3" in context
    assert "[2] b.pdf — page 1" in context
    assert sources[0].index == 1
    assert sources[0].page == 3
    assert sources[1].document == "b.pdf"
    assert "Submit by 15 May." in sources[0].snippet


def test_format_context_truncates_long_snippets():
    document = Document(page_content="word " * 100, metadata={"source": "a.pdf", "page": 0})

    _, sources = format_context([document])

    assert len(sources[0].snippet) <= 240
    assert sources[0].snippet.endswith("...")


def test_build_messages_places_system_history_and_question():
    history = [
        {"role": "user", "content": "Question one"},
        {"role": "assistant", "content": "Answer one"},
        {"role": "user", "content": "Question two"},
        {"role": "assistant", "content": "Answer two"},
    ]

    messages = build_messages("[1] a.pdf — page 1\ntext", "Follow-up?", history, window=1)

    assert isinstance(messages[0], SystemMessage)
    assert len(messages) == 4  # system + 2 history messages + human
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "Question two"
    assert isinstance(messages[3], HumanMessage)
    assert "Follow-up?" in messages[3].content
    assert "[1] a.pdf" in messages[3].content


def test_to_chat_messages_honours_window():
    history = [{"role": "user", "content": f"q{i}"} for i in range(5)]

    messages = to_chat_messages(history, window=2)

    assert [m.content for m in messages] == ["q1", "q2", "q3", "q4"]


def test_message_text_handles_strings_lists_and_dicts():
    assert message_text(AIMessage(content="hello")) == "hello"
    assert message_text(AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])) == "ab"
    assert message_text({"text": "mapping"}) == "mapping"
    assert message_text(SimpleNamespace(content=None)) == ""


def test_pipeline_rewrites_follow_up_and_streams_answer():
    llm = StubLLM(["What is the exam deadline?", "The deadline is 15 May [1]."])
    retriever = StubRetriever(sample_documents())
    pipeline = RAGPipeline(retriever=retriever, llm=llm, conversation_window=2)
    history = [
        {"role": "user", "content": "Tell me about the exam"},
        {"role": "assistant", "content": "The exam covers chapter 3."},
    ]

    answer, sources = pipeline.answer("When is it due?", history)

    assert retriever.queries == ["What is the exam deadline?"]
    assert answer == "The deadline is 15 May [1]."
    assert len(sources) == 2
    assert len(llm.calls) == 2  # one condense call + one streamed answer call


def test_pipeline_skips_condensing_without_history():
    llm = StubLLM(["Direct answer."])
    retriever = StubRetriever(sample_documents())
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    answer, _ = pipeline.answer("What is the deadline?")

    assert retriever.queries == ["What is the deadline?"]
    assert answer == "Direct answer."
    assert len(llm.calls) == 1  # no rewrite call


def test_pipeline_falls_back_when_condensing_fails():
    llm = FailingCondenseLLM(["Fallback answer."])
    retriever = StubRetriever(sample_documents())
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    answer, _ = pipeline.answer("When is it due?", [{"role": "user", "content": "hi"}])

    assert retriever.queries == ["When is it due?"]
    assert answer == "Fallback answer."


def test_streaming_yields_tokens():
    llm = StubLLM(["streamed answer"])
    pipeline = RAGPipeline(retriever=StubRetriever(sample_documents()), llm=llm)

    prepared = pipeline.prepare("question", [])
    tokens = list(pipeline.stream(prepared))

    assert len(tokens) == len("streamed answer")
    assert "".join(tokens) == "streamed answer"
