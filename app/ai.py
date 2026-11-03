import asyncio
import logging
import os
from typing import Literal, TypedDict

from google import genai
from google.genai import types
from groq import AsyncGroq
from openai import AsyncOpenAI

from app.prompts import (
    LEGAL_SYSTEM_PROMPT,
    NO_RELEVANT_SOURCES_REPLY,
    UNVERIFIED_CITATION_REPLY,
)
from rag import (
    format_context,
    has_supported_citation,
    retrieve_relevant_chunks,
)

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-flash-lite"
GROQ_DEFAULT_MODEL = "qwen/qwen3.8-27b"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT_MS = 9000
GROQ_TIMEOUT_SECONDS = 9.0
OPENAI_TIMEOUT_SECONDS = 9.0
SYSTEM_INSTRUCTION = LEGAL_SYSTEM_PROMPT
Language = Literal["english", "swahili"]
LANGUAGE_PICKER_REPLY = (
    "Hi! I'm a Kenyan legal information assistant, not a lawyer. "
    "I can also chat briefly. Please choose English, Kiswahili, or Other below."
)
FALLBACK_REPLY = (
    "Sorry, I'm having trouble responding right now. Please try again later."
)
INDEX_UNAVAILABLE_REPLY = (
    "I can't access the legal references right now. Please try again later. "
    "This is legal information, not legal advice."
)
NO_RELEVANT_EXCERPTS_REPLY = NO_RELEVANT_SOURCES_REPLY

GREETINGS = {
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "how are you doing",
    "habari",
    "hujambo",
    "mambo",
}
THANKS = {"thanks", "thank you", "asante"}


class ConversationMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


def is_small_talk_message(user_text: str) -> bool:
    message = user_text.strip().casefold().rstrip("!?.")
    return message in GREETINGS | THANKS


def _small_talk_reply(user_text: str, language: Language | None) -> str | None:
    message = user_text.strip().casefold().rstrip("!?.")
    if not is_small_talk_message(user_text):
        return None

    if language is None:
        return LANGUAGE_PICKER_REPLY
    if message in THANKS:
        if language == "swahili":
            return "Karibu! Una swali lingine?"
        return "You're welcome! What else can I help with?"
    return (
        "Habari! Ninaweza kukusaidiaje leo?"
        if language == "swahili"
        else "Hi! I'm doing well, thanks. What can I help you with today?"
    )


def _system_instruction(language: Language | None) -> str:
    if language is None:
        return SYSTEM_INSTRUCTION
    requested_language = "Kiswahili" if language == "swahili" else "English"
    return f"{SYSTEM_INSTRUCTION}\n\nReply in {requested_language}."


async def ask_gemini(
    user_text: str,
    history: list[ConversationMessage] | None = None,
    language: Language | None = None,
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
                system_instruction=_system_instruction(language)
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
    language: Language | None = None,
) -> str:
    try:
        model = os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL)
        messages = [
            {"role": "system", "content": _system_instruction(language)},
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


async def ask_openai(
    user_text: str,
    history: list[ConversationMessage] | None = None,
    language: Language | None = None,
) -> str:
    try:
        model = os.getenv("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)
        messages = [
            {"role": "system", "content": _system_instruction(language)},
            *[
                {"role": message["role"], "content": message["content"]}
                for message in history or []
            ],
            {"role": "user", "content": user_text},
        ]
        async with AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=0,
        ) as client:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
            )
        reply = completion.choices[0].message.content
        if not reply:
            raise RuntimeError("OpenAI returned an empty response")
        return reply
    except Exception:
        logger.exception("Failed to generate an OpenAI reply")
        return FALLBACK_REPLY


async def get_ai_reply(
    user_text: str,
    history: list[ConversationMessage] | None = None,
    case_mode: bool = False,
    language: Language | None = None,
) -> str:
    small_talk_reply = _small_talk_reply(user_text, language)
    if small_talk_reply is not None:
        return small_talk_reply

    retrieval_query = user_text
    if case_mode and history:
        prior_facts = [
            message["content"]
            for message in history
            if message["role"] == "user"
        ]
        retrieval_query = "\n".join([*prior_facts, user_text]).strip()

    try:
        retrieved_chunks = await asyncio.to_thread(
            retrieve_relevant_chunks,
            retrieval_query,
        )
    except Exception:
        logger.exception("Failed to retrieve legal excerpts")
        return INDEX_UNAVAILABLE_REPLY

    if not retrieved_chunks:
        if case_mode:
            return (
                "I don't have relevant law for analysis in my current sources. "
                "I can still help organize your facts without legal analysis. "
                "What happened, and when? "
                "Bado tuko free mode mkuu."
            )
        return NO_RELEVANT_EXCERPTS_REPLY

    prompt = format_context(user_text, retrieved_chunks, case_mode=case_mode)
    provider = os.getenv("AI_PROVIDER", "groq").strip().lower()
    if provider == "groq":
        if language:
            reply = await ask_groq(prompt, history, language)
        else:
            reply = await ask_groq(prompt, history)
    elif provider == "openai":
        if language:
            reply = await ask_openai(prompt, history, language)
        else:
            reply = await ask_openai(prompt, history)
    elif provider == "gemini":
        if language:
            reply = await ask_gemini(prompt, history, language)
        else:
            reply = await ask_gemini(prompt, history)
    else:
        logger.error("Unsupported AI_PROVIDER: %s", provider)
        return FALLBACK_REPLY

    if reply == FALLBACK_REPLY:
        return reply
    if not has_supported_citation(reply, retrieved_chunks):
        logger.warning("Generated reply did not contain verifiable citations")
        return UNVERIFIED_CITATION_REPLY
    return reply
