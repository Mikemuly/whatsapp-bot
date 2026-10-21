import os

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse

load_dotenv()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

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


@app.post("/webhook")
async def receive_webhook(request: Request) -> dict[str, str]:
    payload = await request.json()
    print(payload)
    return {"status": "received"}