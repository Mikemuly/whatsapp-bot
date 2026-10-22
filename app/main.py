import os
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse

load_dotenv()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

app = FastAPI()


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
        "text": {"body": f"You said: {message}"},
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=body)
    print(response.status_code, response.text)


@app.post("/webhook")
async def receive_webhook(request: Request) -> dict[str, str]:
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
    if message.get("type") != "text":
        return {"status": "ok"}

    sender = message.get("from")
    text = message.get("text")
    if not isinstance(sender, str) or not isinstance(text, dict):
        return {"status": "ok"}
    text_body = text.get("body")
    if not isinstance(text_body, str):
        return {"status": "ok"}

    await send_whatsapp_message(sender, text_body)
    return {"status": "ok"}