import logging
import os

from google import genai
from google.genai import types
from groq import AsyncGroq

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-flash-lite"
GROQ_DEFAULT_MODEL = "qwen/qwen3.8-27b"
REQUEST_TIMEOUT_MS = 9000
GROQ_TIMEOUT_SECONDS = 9.0
SYSTEM_INSTRUCTION = (
    "You are a helpful WhatsApp assistant. Keep your replies concise, "
    "friendly, and easy to understand."
)
FALLBACK_REPLY = (
    "Sorry, I'm having trouble responding right now. Please try again later."
)


async def ask_gemini(user_text: str) -> str:
    client: genai.Client | None = None
    try:
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_text,
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


async def ask_groq(user_text: str) -> str:
    try:
        model = os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL)
        async with AsyncGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=GROQ_TIMEOUT_SECONDS,
            max_retries=0,
        ) as client:
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_text},
                ],
            )
        reply = completion.choices[0].message.content
        if not reply:
            raise RuntimeError("Groq returned an empty response")
        return reply
    except Exception:
        logger.exception("Failed to generate a Groq reply")
        return FALLBACK_REPLY


async def get_ai_reply(user_text: str) -> str:
    provider = os.getenv("AI_PROVIDER", "groq").strip().lower()
    if provider == "groq":
        return await ask_groq(user_text)
    if provider == "gemini":
        return await ask_gemini(user_text)

    logger.error("Unsupported AI_PROVIDER: %s", provider)
    return FALLBACK_REPLY
