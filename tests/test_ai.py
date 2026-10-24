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
        self.assertEqual(
            call.kwargs["contents"],
            [{"role": "user", "parts": [{"text": "Hi"}]}],
        )
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

    @patch("app.ai.AsyncGroq")
    async def test_ask_groq_includes_prior_conversation(self, groq_factory):
        client = groq_factory.return_value
        client.__aenter__.return_value = client
        client.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[Mock(message=Mock(content="Continuing reply"))]
            )
        )

        history = [
            {"role": "user", "content": "My name is Alex."},
            {"role": "assistant", "content": "Nice to meet you, Alex."},
        ]
        reply = await ai.ask_groq("What is my name?", history)

        self.assertEqual(reply, "Continuing reply")
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["messages"],
            [
                {"role": "system", "content": ai.SYSTEM_INSTRUCTION},
                *history,
                {"role": "user", "content": "What is my name?"},
            ],
        )
        client.__aexit__.assert_awaited_once()

    @patch("app.ai.genai.Client")
    async def test_ask_gemini_converts_prior_conversation(self, client_factory):
        client = client_factory.return_value
        client.aio.models.generate_content = AsyncMock(
            return_value=Mock(text="Your name is Alex.")
        )
        client.aio.aclose = AsyncMock()
        history = [
            {"role": "user", "content": "My name is Alex."},
            {"role": "assistant", "content": "Nice to meet you, Alex."},
        ]

        reply = await ai.ask_gemini("What is my name?", history)

        self.assertEqual(reply, "Your name is Alex.")
        self.assertEqual(
            client.aio.models.generate_content.call_args.kwargs["contents"],
            [
                {"role": "user", "parts": [{"text": "My name is Alex."}]},
                {
                    "role": "model",
                    "parts": [{"text": "Nice to meet you, Alex."}],
                },
                {"role": "user", "parts": [{"text": "What is my name?"}]},
            ],
        )

    async def test_dispatches_to_selected_provider(self):
        with (
            patch.dict(os.environ, {"AI_PROVIDER": "groq"}),
            patch(
                "app.ai.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=[
                    {
                        "text": "Article 43 excerpt",
                        "article": "43",
                        "title": "Economic and social rights",
                        "chapter": "Four - The Bill of Rights",
                        "clause": "1",
                    }
                ],
            ) as retrieve,
            patch(
                "app.ai.format_context",
                return_value="Question with Article 43 context",
            ) as format_context,
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
            reply = await ai.get_ai_reply("Hi", [{"role": "user", "content": "Earlier"}])

        self.assertEqual(reply, "Groq reply")
        retrieve.assert_awaited_once_with(ai.retrieve_relevant_chunks, "Hi")
        format_context.assert_called_once()
        ask_groq.assert_awaited_once_with(
            "Question with Article 43 context",
            [{"role": "user", "content": "Earlier"}],
        )
        ask_gemini.assert_not_awaited()

    async def test_returns_clear_reply_when_constitution_index_unavailable(self):
        with (
            patch(
                "app.ai.asyncio.to_thread",
                new_callable=AsyncMock,
                side_effect=RuntimeError("index missing"),
            ),
            patch("app.ai.ask_groq", new_callable=AsyncMock) as ask_groq,
            self.assertLogs("app.ai", level="ERROR"),
        ):
            reply = await ai.get_ai_reply("What is housing?")

        self.assertEqual(reply, ai.INDEX_UNAVAILABLE_REPLY)
        ask_groq.assert_not_awaited()

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
            patch.object(main, "conversation_history", {}),
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

            generate_reply.assert_awaited_once_with("Hi", [])
            send_message.assert_awaited_once_with("15551234567", "AI response")
            self.assertEqual(
                main.conversation_history["15551234567"],
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "AI response"},
                ],
            )

    async def test_duplicate_message_id_is_scheduled_only_once(self):
        from app import main

        with (
            patch.object(main, "processed_message_ids", set()),
            patch.object(main, "conversation_history", {}),
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

            generate_reply.assert_awaited_once_with("Hi", [])
            send_message.assert_awaited_once_with("15551234567", "AI response")

    async def test_conversation_history_is_capped_and_passed_to_ai(self):
        from app import main

        sender = "15551234567"
        initial_history = [
            {"role": "user", "content": f"old message {index}"}
            for index in range(main.MAX_HISTORY_MESSAGES)
        ]
        conversation = {sender: initial_history.copy()}
        with (
            patch.object(main, "processed_message_ids", set()),
            patch.object(main, "conversation_history", conversation),
            patch(
                "app.main.get_ai_reply",
                new=AsyncMock(return_value="AI response"),
            ) as generate_reply,
            patch(
                "app.main.send_whatsapp_message",
                new=AsyncMock(),
            ),
        ):
            tasks = BackgroundTasks()
            response = await main.receive_webhook(
                self.make_request("wamid.history-cap"), tasks
            )

            self.assertEqual(response, {"status": "ok"})
            self.assertEqual(
                generate_reply.await_count,
                0,
            )
            self.assertEqual(len(conversation[sender]), main.MAX_HISTORY_MESSAGES)
            self.assertEqual(
                conversation[sender][-1],
                {"role": "user", "content": "Hi"},
            )

            await tasks()

        generate_reply.assert_awaited_once_with(
            "Hi",
            initial_history[-(main.MAX_HISTORY_MESSAGES - 1) :],
        )
        self.assertEqual(len(conversation[sender]), main.MAX_HISTORY_MESSAGES)
        self.assertEqual(
            conversation[sender][-1],
            {"role": "assistant", "content": "AI response"},
        )


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
