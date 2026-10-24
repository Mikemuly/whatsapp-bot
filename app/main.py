import asyncio
import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.responses import PlainTextResponse

from app.ai import ConversationMessage, get_ai_reply

logger = logging.getLogger(__name__)

load_dotenv()

WHATSAPP_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
WHATSAPP_SEND_ATTEMPTS = 2
WHATSAPP_RETRY_DELAY_SECONDS = 1
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

app = FastAPI()
processed_message_ids: set[str] = set()
MAX_HISTORY_MESSAGES = 10
conversation_history: dict[str, list[ConversationMessage]] = {}


@app.get("/")
async def read_root() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(
    mode: str | None = Query(default=None, alias="hub.mode"),
    verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    challenge: str | None = Query(default=None, alias="hub.challenge"),
) -> PlainTextResponse:
    if (
        mode == "subscribe"
        and VERIFY_TOKEN
        and verify_token == VERIFY_TOKEN
        and challenge is not None
    ):
        return PlainTextResponse(content=challenge, status_code=200)
    return PlainTextResponse(content="Forbidden", status_code=403)


async def send_whatsapp_message(to: str, message: str) -> None:
    if not WHATSAPP_TOKEN:
        raise RuntimeError("WHATSAPP_TOKEN is not configured")
    if not PHONE_NUMBER_ID:
        raise RuntimeError("PHONE_NUMBER_ID is not configured")

    url = f"https://graph.facebook.com/v23.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    body = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message},
    }
    async with httpx.AsyncClient() as client:
        for attempt in range(WHATSAPP_SEND_ATTEMPTS):
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=WHATSAPP_TIMEOUT,
                )
                response.raise_for_status()
                return
            except httpx.HTTPStatusError as error:
                logger.error(
                    "WhatsApp rejected the message with status %s: %s",
                    error.response.status_code,
                    error.response.text,
                )
                return
            except httpx.RequestError:
                if attempt + 1 == WHATSAPP_SEND_ATTEMPTS:
                    logger.exception(
                        "WhatsApp message delivery failed after %s attempts",
                        WHATSAPP_SEND_ATTEMPTS,
                    )
                    return
                logger.warning(
                    "WhatsApp connection failed; retrying in %s second",
                    WHATSAPP_RETRY_DELAY_SECONDS,
                    exc_info=True,
                )
                await asyncio.sleep(WHATSAPP_RETRY_DELAY_SECONDS)


async def process_incoming_message(
    sender: str,
    text: str,
    history: list[ConversationMessage],
) -> None:
    reply = await get_ai_reply(text, history)
    conversation_history[sender].append({"role": "assistant", "content": reply})
    del conversation_history[sender][:-MAX_HISTORY_MESSAGES]
    await send_whatsapp_message(sender, reply)


@app.post("/webhook")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    payload: Any = await request.json()
    entries = payload.get("entry") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
        return {"status": "ok"}

    changes = entries[0].get("changes")
    if not isinstance(changes, list) or not changes or not isinstance(changes[0], dict):
        return {"status": "ok"}

    value = changes[0].get("value")
    if not isinstance(value, dict):
        return {"status": "ok"}

    messages = value.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        return {"status": "ok"}

    message = messages[0]
    message_id = message.get("id")
    if not isinstance(message_id, str) or message_id in processed_message_ids:
        return {"status": "ok"}

    if message.get("type") != "text":
        return {"status": "ok"}

    sender = message.get("from")
    text = message.get("text")
    if not isinstance(sender, str) or not isinstance(text, dict):
        return {"status": "ok"}
    text_body = text.get("body")
    if not isinstance(text_body, str):
        return {"status": "ok"}

    history = conversation_history.setdefault(sender, [])
    history_snapshot = (
        history[-(MAX_HISTORY_MESSAGES - 1) :]
        if MAX_HISTORY_MESSAGES > 1
        else []
    )
    history.append({"role": "user", "content": text_body})
    del history[:-MAX_HISTORY_MESSAGES]

    processed_message_ids.add(message_id)
    background_tasks.add_task(
        process_incoming_message,
        sender,
        text_body,
        history_snapshot,
    )
    return {"status": "ok"}