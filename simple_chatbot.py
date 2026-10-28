"""A small LangChain chatbot that answers only from saved chat history."""

from __future__ import annotations

import argparse
import os
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


load_dotenv()


GREETING = "Hello"
FACT_RESPONSE = "Thanks for sharing this with me"
UNKNOWN_RESPONSE = "Sorry, I don't know the answer to your question"

QUESTION_STARTERS = (
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "whose",
    "whom",
    "am",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
    "have",
    "has",
    "had",
    "tell me",
    "please tell me",
    "explain",
    "please explain",
    "describe",
    "please describe",
    "show me",
    "please show me",
    "give me",
    "please give me",
    "remind me",
    "please remind me",
)

MEMORY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the current question using only information explicitly present "
            "in the saved chat. Do not use general knowledge or make assumptions. "
            "The current question is not evidence and must not be treated as a "
            "source of facts. If the answer cannot be derived from the saved chat, "
            "respond with exactly this sentence and nothing else: "
            "{unknown_response}",
        ),
        (
            "human",
            "Saved chat:\n{saved_chat}\n\nCurrent question:\n{question}",
        ),
    ]
)


def initial_history() -> list[dict[str, str]]:
    """Return a fresh conversation that starts with the bot greeting."""
    return [{"role": "assistant", "content": GREETING}]


class SimpleMemoryChatbot:
    """Use a list of chat messages as short-lived, in-memory knowledge."""

    MODEL = "openai/gpt-oss-120b"

    def __init__(self, model: Any | None = None) -> None:
        self._model = model
        self._chain: Any | None = None

    @property
    def model(self) -> Any:
        """Create the LangChain Groq model wrapper on first use."""
        if self._model is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is missing. Add it to your .env file."
                )

            self._model = ChatGroq(
                model=self.MODEL,
                api_key=api_key,
                temperature=0.0,
                max_tokens=500,
                max_retries=2,
                timeout=30.0,
                reasoning_format="hidden",
            )

        return self._model

    @property
    def chain(self) -> Any:
        """Build the prompt-model-output-parser chain lazily."""
        if self._chain is None:
            self._chain = MEMORY_PROMPT | self.model | StrOutputParser()
        return self._chain

    @staticmethod
    def format_history(history: list[dict[str, str]]) -> str:
        """Format saved messages for insertion into the prompt."""
        return "\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in history
        )

    @staticmethod
    def is_question(message: str) -> bool:
        """Return whether a message is phrased as a question or request."""
        normalized_message = " ".join(message.casefold().split())
        if "?" in normalized_message:
            return True

        return any(
            normalized_message == starter
            or normalized_message.startswith(f"{starter} ")
            for starter in QUESTION_STARTERS
        )

    def answer(
        self,
        question: str,
        history: list[dict[str, str]] | None,
    ) -> tuple[str, list[dict[str, str]]]:
        """Acknowledge a fact or answer a question, then save both messages."""
        question = question.strip()
        saved_history = [dict(message) for message in (history or initial_history())]

        if not question:
            return "", saved_history

        if self.is_question(question):
            answer = self.chain.invoke(
                {
                    "saved_chat": self.format_history(saved_history),
                    "question": question,
                    "unknown_response": UNKNOWN_RESPONSE,
                }
            ).strip()

            # Keep the required fallback wording exact even if the model adds
            # punctuation or introductory text around it.
            if UNKNOWN_RESPONSE.lower() in answer.lower():
                answer = UNKNOWN_RESPONSE
            elif not answer:
                answer = UNKNOWN_RESPONSE
        else:
            answer = FACT_RESPONSE

        saved_history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        return answer, saved_history


def respond(
    message: str,
    memory: list[dict[str, str]] | None,
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    """Handle one Gradio submission and update its session memory."""
    try:
        _, updated_memory = SimpleMemoryChatbot().answer(message, memory)
    except Exception as error:
        updated_memory = [dict(item) for item in (memory or initial_history())]
        if message.strip():
            updated_memory.extend(
                [
                    {"role": "user", "content": message.strip()},
                    {"role": "assistant", "content": f"Error: {error}"},
                ]
            )

    return "", updated_memory, updated_memory


def reset_chat() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Reset both the visible chat and its in-memory session state."""
    history = initial_history()
    return history, history


def build_ui() -> gr.Blocks:
    """Build the stateful Gradio chat interface."""
    with gr.Blocks(title="Simple Memory Chatbot") as demo:
        gr.Markdown(
            "# Simple Memory Chatbot\n"
            "Share facts for the bot to remember, then ask questions about them."
        )
        memory = gr.State(value=initial_history)
        chatbot = gr.Chatbot(
            value=initial_history,
            label="Chat",
            height=450,
            buttons=["copy"],
        )
        message = gr.Textbox(
            label="Your message",
            placeholder="Tell the bot something, then ask about it later...",
        )
        with gr.Row():
            send_button = gr.Button("Send", variant="primary")
            clear_button = gr.Button("Clear")

        send_button.click(
            fn=respond,
            inputs=[message, memory],
            outputs=[message, chatbot, memory],
        )
        message.submit(
            fn=respond,
            inputs=[message, memory],
            outputs=[message, chatbot, memory],
        )
        clear_button.click(
            fn=reset_chat,
            inputs=None,
            outputs=[chatbot, memory],
        )

    return demo


def main() -> None:
    """Launch the local Gradio chatbot."""
    parser = argparse.ArgumentParser(description="Launch the simple memory chatbot.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="create a temporary public Gradio link",
    )
    args = parser.parse_args()

    build_ui().launch(inbrowser=True, share=args.share, show_error=True)


if __name__ == "__main__":
    main()
