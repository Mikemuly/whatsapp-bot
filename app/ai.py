import asyncio
import logging
import os
from typing import Literal, TypedDict

from google import genai
from google.genai import types
from groq import AsyncGroq

from rag import format_context, retrieve_relevant_chunks

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-flash-lite"
GROQ_DEFAULT_MODEL = "qwen/qwen3.8-27b"
REQUEST_TIMEOUT_MS = 9000
GROQ_TIMEOUT_SECONDS = 9.0
SYSTEM_INSTRUCTION = (
    "You provide concise information about the Constitution of Kenya, 2010. "
    "Answer only from the provided excerpts; never use outside knowledge or "
    "guess. Cite every factual claim with its Article and clause when "
    "available. If the excerpts do not answer the question, say that the "
    "Constitution excerpts do not provide the answer. Reply in the user's "
    "language (English or Swahili), keep citations in their original form, "
    "and keep the answer short for WhatsApp. End with: "
    "This is information, not legal advice."
)
FALLBACK_REPLY = (
    "Sorry, I'm having trouble responding right now. Please try again later."
)
INDEX_UNAVAILABLE_REPLY = (
    "I can't access the Constitution reference right now. Please try again "
    "later. This is information, not legal advice."
)


class ConversationMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


async def ask_gemini(
    user_text: str,
    history: list[ConversationMessage] | None = None,
) -> str:
    client: genai.Client | None = None
    try:
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        contents = [
            {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
            for message in history or []
        ]
        contents.append({"role": "user", "parts": [{"text": user_text}]})
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response")
        return response.text
    except Exception:
        logger.exception("Failed to generate a Gemini reply")
        return FALLBACK_REPLY
    finally:
        if client is not None:
            try:
                await client.aio.aclose()
            except Exception:
                logger.exception("Failed to close the Gemini client")


async def ask_groq(
    user_text: str,
    history: list[ConversationMessage] | None = None,
) -> str:
    try:
        model = os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL)
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            *[
                {"role": message["role"], "content": message["content"]}
                for message in history or []
            ],
            {"role": "user", "content": user_text},
        ]
        async with AsyncGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=GROQ_TIMEOUT_SECONDS,
            max_retries=0,
        ) as client:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
            )
        reply = completion.choices[0].message.content
        if not reply:
            raise RuntimeError("Groq returned an empty response")
        return reply
    except Exception:
        logger.exception("Failed to generate a Groq reply")
        return FALLBACK_REPLY


async def get_ai_reply(
    user_text: str,
    history: list[ConversationMessage] | None = None,
) -> str:
    try:
        retrieved_chunks = await asyncio.to_thread(
            retrieve_relevant_chunks,
            user_text,
        )
    except Exception:
        logger.exception("Failed to retrieve Constitution excerpts")
        return INDEX_UNAVAILABLE_REPLY

    prompt = format_context(user_text, retrieved_chunks)
    provider = os.getenv("AI_PROVIDER", "groq").strip().lower()
    if provider == "groq":
        return await ask_groq(prompt, history)
    if provider == "gemini":
        return await ask_gemini(prompt, history)

    logger.error("Unsupported AI_PROVIDER: %s", provider)
    return FALLBACK_REPLY
