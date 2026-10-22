import json
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from app import ai
from fastapi import BackgroundTasks
from google.genai.errors import APIError
from starlette.requests import Request


class GetAIReplyTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.ai.genai.Client")
    async def test_generates_reply_with_system_instruction(self, client_factory):
        client = client_factory.return_value
        client.aio.models.generate_content = AsyncMock(
            return_value=Mock(text="Hello!")
        )
        client.aio.aclose = AsyncMock()

        reply = await ai.ask_gemini("Hi")

        self.assertEqual(reply, "Hello!")
        client.aio.models.generate_content.assert_awaited_once()
        call = client.aio.models.generate_content.call_args
        self.assertEqual(call.kwargs["model"], ai.GEMINI_MODEL)
        self.assertEqual(call.kwargs["contents"], "Hi")
        self.assertEqual(
            call.kwargs["config"].system_instruction,
            ai.SYSTEM_INSTRUCTION,
        )
        http_options = client_factory.call_args.kwargs["http_options"]
        self.assertEqual(http_options.timeout, ai.REQUEST_TIMEOUT_MS)
        self.assertEqual(http_options.retry_options.attempts, 1)
        client.aio.aclose.assert_awaited_once()

    @patch("app.ai.AsyncGroq")
    async def test_ask_groq_returns_chat_completion_text(self, groq_factory):
        client = groq_factory.return_value
        client.__aenter__.return_value = client
        client.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[Mock(message=Mock(content="Groq reply"))]
            )
        )

        reply = await ai.ask_groq("Hi")

        self.assertEqual(reply, "Groq reply")
        groq_factory.assert_called_once_with(
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=ai.GROQ_TIMEOUT_SECONDS,
            max_retries=0,
        )
        call = client.chat.completions.create.call_args
        self.assertEqual(
            call.kwargs["model"],
            ai.GROQ_DEFAULT_MODEL,
        )
        self.assertEqual(
            call.kwargs["messages"],
            [
                {"role": "system", "content": ai.SYSTEM_INSTRUCTION},
                {"role": "user", "content": "Hi"},
            ],
        )
        client.__aexit__.assert_awaited_once()

    async def test_dispatches_to_selected_provider(self):
        with (
            patch.dict(os.environ, {"AI_PROVIDER": "groq"}),
            patch(
                "app.ai.ask_groq",
                new_callable=AsyncMock,
                return_value="Groq reply",
            ) as ask_groq,
            patch(
                "app.ai.ask_gemini",
                new_callable=AsyncMock,
            ) as ask_gemini,
        ):
            reply = await ai.get_ai_reply("Hi")

        self.assertEqual(reply, "Groq reply")
        ask_groq.assert_awaited_once_with("Hi")
        ask_gemini.assert_not_awaited()

    @patch("app.ai.genai.Client")
    async def test_returns_fallback_without_retry_on_503(self, client_factory):
        client = client_factory.return_value
        client.aio.models.generate_content = AsyncMock(
            side_effect=APIError(503, {})
        )
        client.aio.aclose = AsyncMock()

        with self.assertLogs("app.ai", level="ERROR"):
            reply = await ai.ask_gemini("Hi")

        self.assertEqual(reply, ai.FALLBACK_REPLY)
        client.aio.models.generate_content.assert_awaited_once()
        client.aio.aclose.assert_awaited_once()

    @patch("app.ai.genai.Client")
    async def test_returns_fallback_when_generation_fails(self, client_factory):
        client = client_factory.return_value
        client.aio.models.generate_content = AsyncMock(side_effect=(
            RuntimeError("Gemini unavailable")
        ))
        client.aio.aclose = AsyncMock()

        with self.assertLogs("app.ai", level="ERROR"):
            reply = await ai.ask_gemini("Hi")

        self.assertEqual(reply, ai.FALLBACK_REPLY)
        client.aio.aclose.assert_awaited_once()


class WebhookReplyTests(unittest.IsolatedAsyncioTestCase):
    def make_request(self, message_id: str) -> Request:
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "id": message_id,
                                        "from": "15551234567",
                                        "type": "text",
                                        "text": {"body": "Hi"},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }

        async def receive():
            return {
                "type": "http.request",
                "body": json.dumps(payload).encode(),
                "more_body": False,
            }

        return Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/webhook",
                "headers": [],
                "query_string": b"",
            },
            receive,
        )

    async def test_schedules_generated_reply_after_webhook_response(self):
        from app import main

        with (
            patch.object(main, "processed_message_ids", set()),
            patch(
                "app.main.get_ai_reply",
                new=AsyncMock(return_value="AI response"),
            ) as generate_reply,
            patch(
                "app.main.send_whatsapp_message",
                new=AsyncMock(),
            ) as send_message,
        ):
            tasks = BackgroundTasks()
            response = await main.receive_webhook(
                self.make_request("wamid.test-1"), tasks
            )

            self.assertEqual(response, {"status": "ok"})
            generate_reply.assert_not_awaited()
            send_message.assert_not_awaited()

            await tasks()

            generate_reply.assert_awaited_once_with("Hi")
            send_message.assert_awaited_once_with("15551234567", "AI response")

    async def test_duplicate_message_id_is_scheduled_only_once(self):
        from app import main

        with (
            patch.object(main, "processed_message_ids", set()),
            patch(
                "app.main.get_ai_reply",
                new=AsyncMock(return_value="AI response"),
            ) as generate_reply,
            patch(
                "app.main.send_whatsapp_message",
                new=AsyncMock(),
            ) as send_message,
        ):
            first_tasks = BackgroundTasks()
            duplicate_tasks = BackgroundTasks()
            message_id = "wamid.duplicate-test"

            first_response = await main.receive_webhook(
                self.make_request(message_id), first_tasks
            )
            duplicate_response = await main.receive_webhook(
                self.make_request(message_id), duplicate_tasks
            )

            self.assertEqual(first_response, {"status": "ok"})
            self.assertEqual(duplicate_response, {"status": "ok"})
            self.assertEqual(len(first_tasks.tasks), 1)
            self.assertEqual(len(duplicate_tasks.tasks), 0)

            await first_tasks()

            generate_reply.assert_awaited_once_with("Hi")
            send_message.assert_awaited_once_with("15551234567", "AI response")


class WhatsAppDeliveryTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.main.asyncio.sleep", new_callable=AsyncMock)
    @patch("app.main.httpx.AsyncClient")
    async def test_retries_connection_timeout_then_sends(
        self, client_factory, sleep
    ):
        from app import main

        client = client_factory.return_value.__aenter__.return_value
        client.post = AsyncMock(
            side_effect=[
                httpx.ConnectTimeout("connection timed out"),
                Mock(status_code=200, is_success=True),
            ]
        )

        with (
            patch.object(main, "WHATSAPP_TOKEN", "test-token"),
            patch.object(main, "PHONE_NUMBER_ID", "test-phone-id"),
            self.assertLogs("app.main", level="WARNING"),
        ):
            await main.send_whatsapp_message("15551234567", "Reply")

        self.assertEqual(client.post.await_count, 2)
        self.assertEqual(
            client.post.call_args.kwargs["timeout"],
            main.WHATSAPP_TIMEOUT,
        )
        sleep.assert_awaited_once_with(main.WHATSAPP_RETRY_DELAY_SECONDS)

    @patch("app.main.asyncio.sleep", new_callable=AsyncMock)
    @patch("app.main.httpx.AsyncClient")
    async def test_logs_and_stops_after_repeated_connection_timeouts(
        self, client_factory, sleep
    ):
        from app import main

        client = client_factory.return_value.__aenter__.return_value
        client.post = AsyncMock(
            side_effect=httpx.ConnectTimeout("connection timed out")
        )

        with (
            patch.object(main, "WHATSAPP_TOKEN", "test-token"),
            patch.object(main, "PHONE_NUMBER_ID", "test-phone-id"),
            self.assertLogs("app.main", level="ERROR") as captured,
        ):
            await main.send_whatsapp_message("15551234567", "Reply")

        self.assertIn("failed after 2 attempts", captured.output[-1])
        self.assertEqual(client.post.await_count, main.WHATSAPP_SEND_ATTEMPTS)
        sleep.assert_awaited_once_with(main.WHATSAPP_RETRY_DELAY_SECONDS)


if __name__ == "__main__":
    unittest.main()
