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
    async def test_greeting_asks_for_language_without_legal_retrieval(self):
        with patch(
            "app.ai.asyncio.to_thread",
            new=AsyncMock(),
        ) as retrieve:
            reply = await ai.get_ai_reply("Hello!")

        self.assertEqual(reply, ai.LANGUAGE_PICKER_REPLY)
        retrieve.assert_not_awaited()

    async def test_small_talk_uses_selected_language(self):
        with patch(
            "app.ai.asyncio.to_thread",
            new=AsyncMock(),
        ) as retrieve:
            reply = await ai.get_ai_reply("Habari", language="swahili")

        self.assertEqual(reply, "Habari! Ninaweza kukusaidiaje leo?")
        retrieve.assert_not_awaited()

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

    @patch("app.ai.AsyncOpenAI")
    async def test_ask_openai_returns_chat_completion_text(self, openai_factory):
        client = openai_factory.return_value
        client.__aenter__.return_value = client
        client.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[Mock(message=Mock(content="OpenAI reply"))]
            )
        )

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-openai-key", "OPENAI_MODEL": "test-model"},
        ):
            reply = await ai.ask_openai("Hi")

        self.assertEqual(reply, "OpenAI reply")
        openai_factory.assert_called_once_with(
            api_key="test-openai-key",
            timeout=ai.OPENAI_TIMEOUT_SECONDS,
            max_retries=0,
        )
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs,
            {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": ai.SYSTEM_INSTRUCTION},
                    {"role": "user", "content": "Hi"},
                ],
            },
        )
        client.__aexit__.assert_awaited_once()

    @patch("app.ai.AsyncOpenAI")
    async def test_ask_openai_includes_prior_conversation(self, openai_factory):
        client = openai_factory.return_value
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

        reply = await ai.ask_openai("What is my name?", history)

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
                        "source": "Constitution of Kenya, 2010",
                        "section": "43",
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
                return_value="Article 43(1) applies.",
            ) as ask_groq,
            patch(
                "app.ai.ask_gemini",
                new_callable=AsyncMock,
            ) as ask_gemini,
        ):
            reply = await ai.get_ai_reply(
                "What does Article 43 protect?",
                [{"role": "user", "content": "Earlier"}],
            )

        self.assertEqual(reply, "Article 43(1) applies.")
        retrieve.assert_awaited_once_with(
            ai.retrieve_relevant_chunks,
            "What does Article 43 protect?",
        )
        format_context.assert_called_once()
        ask_groq.assert_awaited_once_with(
            "Question with Article 43 context",
            [{"role": "user", "content": "Earlier"}],
        )
        ask_gemini.assert_not_awaited()

    async def test_dispatches_to_openai_provider(self):
        with (
            patch.dict(os.environ, {"AI_PROVIDER": "openai"}),
            patch(
                "app.ai.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=[
                    {
                        "text": "Article 43 excerpt",
                        "source": "Constitution of Kenya, 2010",
                        "section": "43",
                        "article": "43",
                        "title": "Economic and social rights",
                        "chapter": "Four - The Bill of Rights",
                        "clause": "1",
                    }
                ],
            ),
            patch(
                "app.ai.format_context",
                return_value="Question with Article 43 context",
            ),
            patch(
                "app.ai.ask_openai",
                new_callable=AsyncMock,
                return_value="Article 43(1) applies.",
            ) as ask_openai,
            patch("app.ai.ask_groq", new_callable=AsyncMock) as ask_groq,
            patch("app.ai.ask_gemini", new_callable=AsyncMock) as ask_gemini,
        ):
            reply = await ai.get_ai_reply("What does Article 43 protect?")

        self.assertEqual(reply, "Article 43(1) applies.")
        ask_openai.assert_awaited_once_with(
            "Question with Article 43 context",
            None,
        )
        ask_groq.assert_not_awaited()
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

    async def test_quick_mode_refuses_when_no_excerpts_are_retrieved(self):
        with (
            patch(
                "app.ai.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.ai.ask_groq", new_callable=AsyncMock) as ask_groq,
        ):
            reply = await ai.get_ai_reply("Can my landlord lock me out?")

        self.assertEqual(reply, ai.NO_RELEVANT_EXCERPTS_REPLY)
        ask_groq.assert_not_awaited()

    async def test_case_mode_retrieves_using_prior_user_facts(self):
        history = [
            {"role": "user", "content": "My employer dismissed me."},
            {"role": "assistant", "content": "When did this happen?"},
            {"role": "user", "content": "Yesterday."},
        ]
        with (
            patch(
                "app.ai.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=[
                    {
                        "text": "Section 45 excerpt",
                        "source": "Employment Act",
                        "section": "45",
                        "article": "",
                        "title": "Unfair termination",
                        "chapter": "Part VI - Termination",
                        "clause": "",
                    }
                ],
            ) as retrieve,
            patch(
                "app.ai.format_context",
                return_value="Case context",
            ) as format_context,
            patch(
                "app.ai.ask_groq",
                new_callable=AsyncMock,
                return_value="Employment Act, s. 45 may be relevant.",
            ) as ask_groq,
            patch.dict(os.environ, {"AI_PROVIDER": "groq"}),
        ):
            reply = await ai.get_ai_reply(
                "I received no letter.",
                history,
                case_mode=True,
            )

        self.assertEqual(reply, "Employment Act, s. 45 may be relevant.")
        retrieve.assert_awaited_once_with(
            ai.retrieve_relevant_chunks,
            "My employer dismissed me.\nYesterday.\nI received no letter.",
        )
        format_context.assert_called_once_with(
            "I received no letter.",
            [
                {
                    "text": "Section 45 excerpt",
                    "source": "Employment Act",
                    "section": "45",
                    "article": "",
                    "title": "Unfair termination",
                    "chapter": "Part VI - Termination",
                    "clause": "",
                }
            ],
            case_mode=True,
        )
        ask_groq.assert_awaited_once_with(
            "Case context",
            history,
        )

    async def test_rejects_answer_with_unsupported_citation(self):
        with (
            patch(
                "app.ai.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=[
                    {
                        "text": "Article 43 excerpt",
                        "source": "Constitution of Kenya, 2010",
                        "section": "43",
                        "article": "43",
                        "title": "Housing",
                        "chapter": "Four",
                        "clause": "1",
                    }
                ],
            ),
            patch(
                "app.ai.ask_groq",
                new_callable=AsyncMock,
                return_value="Article 49(1) gives you this right.",
            ),
            patch.dict(os.environ, {"AI_PROVIDER": "groq"}),
            self.assertLogs("app.ai", level="WARNING"),
        ):
            reply = await ai.get_ai_reply("Question")

        self.assertEqual(reply, ai.UNVERIFIED_CITATION_REPLY)

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
    def make_request(
        self,
        message_id: str,
        text_body: str = "Hi",
        message_type: str = "text",
        button_reply_id: str | None = None,
        list_reply_id: str | None = None,
    ) -> Request:
        message = {
            "id": message_id,
            "from": "15551234567",
            "type": message_type,
        }
        if message_type == "text":
            message["text"] = {"body": text_body}
        elif message_type == "interactive":
            if list_reply_id is not None:
                message["interactive"] = {
                    "type": "list_reply",
                    "list_reply": {
                        "id": list_reply_id,
                        "title": list_reply_id,
                    },
                }
            else:
                message["interactive"] = {
                    "type": "button_reply",
                    "button_reply": {
                        "id": button_reply_id,
                        "title": button_reply_id,
                    },
                }
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [message]
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
            patch.object(main, "conversation_last_activity", {}),
            patch.object(main, "case_mode_senders", set()),
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

            generate_reply.assert_awaited_once_with("Hi", [], case_mode=False)
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
            patch.object(main, "conversation_last_activity", {}),
            patch.object(main, "case_mode_senders", set()),
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

            generate_reply.assert_awaited_once_with("Hi", [], case_mode=False)
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
            patch.object(main, "conversation_last_activity", {}),
            patch.object(main, "case_mode_senders", set()),
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
            case_mode=False,
        )
        self.assertEqual(len(conversation[sender]), main.MAX_HISTORY_MESSAGES)
        self.assertEqual(
            conversation[sender][-1],
            {"role": "assistant", "content": "AI response"},
        )

    async def test_case_command_starts_case_mode(self):
        from app import main

        sender = "15551234567"
        with (
            patch.object(main, "processed_message_ids", set()),
            patch.object(main, "conversation_history", {}),
            patch.object(main, "conversation_last_activity", {}),
            patch.object(main, "case_mode_senders", set()),
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
            await main.receive_webhook(
                self.make_request("wamid.case", "/case"), tasks
            )
            await tasks()

            self.assertIn(sender, main.case_mode_senders)
            generate_reply.assert_not_awaited()
            send_message.assert_awaited_once_with(sender, main.CASE_START_REPLY)
            self.assertEqual(
                main.conversation_history[sender],
                [{"role": "assistant", "content": main.CASE_START_REPLY}],
            )

    async def test_case_command_passes_narrative_to_case_analysis(self):
        from app import main

        with (
            patch.object(main, "processed_message_ids", set()),
            patch.object(main, "conversation_history", {}),
            patch.object(main, "conversation_last_activity", {}),
            patch.object(main, "case_mode_senders", set()),
            patch(
                "app.main.get_ai_reply",
                new=AsyncMock(return_value="What outcome do you want?"),
            ) as generate_reply,
            patch("app.main.send_whatsapp_message", new=AsyncMock()),
        ):
            tasks = BackgroundTasks()
            await main.receive_webhook(
                self.make_request("wamid.case-narrative", "/case I was dismissed"),
                tasks,
            )
            await tasks()

        generate_reply.assert_awaited_once_with(
            "I was dismissed",
            [],
            case_mode=True,
        )

    async def test_reset_clears_history_and_case_mode(self):
        from app import main

        sender = "15551234567"
        history = {
            sender: [{"role": "user", "content": "Sensitive case facts"}]
        }
        case_senders = {sender}
        with (
            patch.object(main, "processed_message_ids", set()),
            patch.object(main, "conversation_history", history),
            patch.object(
                main,
                "conversation_last_activity",
                {sender: 1.0},
            ) as activity,
            patch.object(main, "case_mode_senders", case_senders),
            patch.object(
                main,
                "preferred_languages",
                {sender: "swahili"},
            ) as languages,
            patch.object(
                main,
                "awaiting_language_other_senders",
                {sender},
            ) as awaiting_other,
            patch(
                "app.main.send_whatsapp_message",
                new=AsyncMock(),
            ) as send_message,
        ):
            tasks = BackgroundTasks()
            await main.receive_webhook(
                self.make_request("wamid.reset", "/reset"), tasks
            )
            await tasks()

        self.assertEqual(history, {})
        self.assertEqual(case_senders, set())
        self.assertEqual(activity, {})
        self.assertEqual(languages, {})
        self.assertEqual(awaiting_other, set())
        send_message.assert_awaited_once_with(sender, main.RESET_REPLY)

    def test_expired_history_is_removed(self):
        from app import main

        sender = "15551234567"
        with (
            patch.object(
                main,
                "conversation_history",
                {sender: [{"role": "user", "content": "Old case"}]},
            ) as history,
            patch.object(
                main,
                "conversation_last_activity",
                {sender: 10.0},
            ) as activity,
            patch.object(main, "case_mode_senders", {sender}) as case_senders,
            patch.object(
                main,
                "preferred_languages",
                {sender: "english"},
            ) as languages,
            patch.object(
                main,
                "awaiting_language_other_senders",
                {sender},
            ) as awaiting_other,
        ):
            main._prune_expired_conversations(
                now=10.0 + main.HISTORY_TTL_SECONDS
            )

        self.assertEqual(history, {})
        self.assertEqual(activity, {})
        self.assertEqual(case_senders, set())
        self.assertEqual(languages, {})
        self.assertEqual(awaiting_other, set())

    async def test_reply_in_flight_when_reset_is_not_sent_or_restored(self):
        from app import main

        old_history = [{"role": "user", "content": "Old case"}]
        with (
            patch.object(main, "conversation_history", {}),
            patch.object(main, "preferred_languages", {}),
            patch(
                "app.main.get_ai_reply",
                new=AsyncMock(return_value="Old case analysis"),
            ) as generate_reply,
            patch(
                "app.main.send_whatsapp_message",
                new=AsyncMock(),
            ) as send_message,
        ):
            await main.process_incoming_message(
                "15551234567",
                "Old case",
                old_history,
                True,
                old_history,
            )

        generate_reply.assert_awaited_once()
        send_message.assert_not_awaited()
        self.assertEqual(old_history, [{"role": "user", "content": "Old case"}])

    async def test_direct_commands_are_answered_without_ai(self):
        from app import main

        for index, (command, expected_reply) in enumerate(
            [
                ("/start", main.WELCOME_REPLY),
                ("/help", main.HELP_REPLY),
                ("/sources", main.SOURCES_REPLY),
            ]
        ):
            with (
                patch.object(main, "processed_message_ids", set()),
                patch.object(main, "conversation_history", {}),
                patch.object(main, "preferred_languages", {}),
                patch(
                    "app.main.get_ai_reply",
                    new=AsyncMock(),
                ) as generate_reply,
                patch(
                    "app.main.send_whatsapp_message",
                    new=AsyncMock(),
                ) as send_message,
            ):
                tasks = BackgroundTasks()
                await main.receive_webhook(
                    self.make_request(f"wamid.command-{index}", command),
                    tasks,
                )
                await tasks()

            generate_reply.assert_not_awaited()
            if command == "/start":
                send_message.assert_awaited_once_with(
                    "15551234567",
                    expected_reply,
                    language_picker=True,
                )
            elif command == "/help":
                send_message.assert_awaited_once_with(
                    "15551234567",
                    expected_reply,
                    command_picker=True,
                    language="english",
                )
            else:
                send_message.assert_awaited_once_with(
                    "15551234567",
                    expected_reply,
                )

    async def test_language_choice_sends_localized_command_list(self):
        from app import main

        sender = "15551234567"
        with (
            patch.object(main, "processed_message_ids", set()),
            patch.object(main, "preferred_languages", {}),
            patch.object(main, "conversation_last_activity", {}) as activity,
            patch(
                "app.main.get_ai_reply",
                new=AsyncMock(),
            ) as generate_reply,
            patch(
                "app.main.send_whatsapp_message",
                new=AsyncMock(),
            ) as send_message,
        ):
            tasks = BackgroundTasks()
            await main.receive_webhook(
                self.make_request("wamid.language", "Kiswahili"),
                tasks,
            )
            await tasks()

            self.assertEqual(main.preferred_languages[sender], "swahili")
            self.assertIn(sender, activity)
            generate_reply.assert_not_awaited()
            send_message.assert_awaited_once_with(
                sender,
                main.HELP_REPLY_SWAHILI,
                command_picker=True,
                language="swahili",
            )

    async def test_language_command_switches_preferred_language(self):
        from app import main

        sender = "15551234567"
        with (
            patch.object(main, "processed_message_ids", set()),
            patch.object(main, "preferred_languages", {sender: "english"}) as languages,
            patch.object(main, "conversation_last_activity", {}),
            patch(
                "app.main.get_ai_reply",
                new=AsyncMock(),
            ) as generate_reply,
            patch(
                "app.main.send_whatsapp_message",
                new=AsyncMock(),
            ) as send_message,
        ):
            language_tasks = BackgroundTasks()
            await main.receive_webhook(
                self.make_request("wamid.language-command", "/language"),
                language_tasks,
            )
            await language_tasks()

            send_message.assert_awaited_once_with(
                sender,
                main.LANGUAGE_SWITCH_PROMPTS["english"],
                supported_language_picker=True,
            )

            switch_tasks = BackgroundTasks()
            await main.receive_webhook(
                self.make_request(
                    "wamid.language-switch-button",
                    message_type="interactive",
                    button_reply_id=main.LANGUAGE_BUTTON_IDS["swahili"],
                ),
                switch_tasks,
            )
            await switch_tasks()

        self.assertEqual(languages[sender], "swahili")
        generate_reply.assert_not_awaited()
        send_message.assert_awaited_with(
            sender,
            main.HELP_REPLY_SWAHILI,
            command_picker=True,
            language="swahili",
        )

    async def test_list_reply_selects_a_command(self):
        from app import main

        sender = "15551234567"
        with (
            patch.object(main, "processed_message_ids", set()),
            patch.object(main, "conversation_history", {}),
            patch.object(main, "preferred_languages", {sender: "swahili"}),
            patch(
                "app.main.get_ai_reply",
                new=AsyncMock(),
            ) as generate_reply,
            patch(
                "app.main.send_whatsapp_message",
                new=AsyncMock(),
            ) as send_message,
        ):
            tasks = BackgroundTasks()
            await main.receive_webhook(
                self.make_request(
                    "wamid.command-list",
                    message_type="interactive",
                    list_reply_id="/sources",
                ),
                tasks,
            )
            await tasks()

        generate_reply.assert_not_awaited()
        send_message.assert_awaited_once_with(sender, main.SOURCES_REPLY)

    async def test_greeting_sends_language_picker_buttons(self):
        from app import main

        sender = "15551234567"
        history: list[dict[str, str]] = []
        with (
            patch.object(main, "conversation_history", {sender: history}),
            patch.object(main, "preferred_languages", {}),
            patch(
                "app.main.send_whatsapp_message",
                new=AsyncMock(),
            ) as send_message,
        ):
            await main.process_incoming_message(
                sender,
                "Hello",
                [],
                False,
                history,
            )

        send_message.assert_awaited_once_with(
            sender,
            main.WELCOME_REPLY,
            language_picker=True,
        )

    async def test_unrecognized_message_sends_intro_and_command_picker(self):
        from app import main

        sender = "15551234567"
        history: list[dict[str, str]] = []
        with (
            patch.object(main, "conversation_history", {sender: history}),
            patch.object(main, "preferred_languages", {sender: "swahili"}),
            patch(
                "app.main.get_ai_reply",
                new=AsyncMock(),
            ) as generate_reply,
            patch(
                "app.main.send_whatsapp_message",
                new=AsyncMock(),
            ) as send_message,
        ):
            await main.process_incoming_message(
                sender,
                "I need some help",
                [],
                False,
                history,
            )

        generate_reply.assert_not_awaited()
        send_message.assert_awaited_once_with(
            sender,
            main.UNRECOGNIZED_MESSAGE_REPLIES["swahili"],
            command_picker=True,
            language="swahili",
        )

    async def test_interactive_language_button_selects_language(self):
        from app import main

        sender = "15551234567"
        with (
            patch.object(main, "processed_message_ids", set()),
            patch.object(main, "preferred_languages", {}) as languages,
            patch.object(main, "conversation_last_activity", {}),
            patch(
                "app.main.get_ai_reply",
                new=AsyncMock(),
            ) as generate_reply,
            patch(
                "app.main.send_whatsapp_message",
                new=AsyncMock(),
            ) as send_message,
        ):
            tasks = BackgroundTasks()
            await main.receive_webhook(
                self.make_request(
                    "wamid.language-button",
                    message_type="interactive",
                    button_reply_id=main.LANGUAGE_BUTTON_IDS["swahili"],
                ),
                tasks,
            )
            await tasks()

        self.assertEqual(languages[sender], "swahili")
        generate_reply.assert_not_awaited()
        send_message.assert_awaited_once_with(
            sender,
            main.HELP_REPLY_SWAHILI,
            command_picker=True,
            language="swahili",
        )

    async def test_other_button_prompts_for_free_text(self):
        from app import main

        sender = "15551234567"
        with (
            patch.object(main, "processed_message_ids", set()),
            patch.object(main, "conversation_history", {}),
            patch.object(
                main,
                "awaiting_language_other_senders",
                set(),
            ) as awaiting_other,
            patch.object(main, "conversation_last_activity", {}),
            patch(
                "app.main.get_ai_reply",
                new=AsyncMock(return_value="Here is some help."),
            ) as generate_reply,
            patch(
                "app.main.send_whatsapp_message",
                new=AsyncMock(),
            ) as send_message,
        ):
            tasks = BackgroundTasks()
            await main.receive_webhook(
                self.make_request(
                    "wamid.language-other",
                    message_type="interactive",
                    button_reply_id=main.LANGUAGE_BUTTON_IDS["other"],
                ),
                tasks,
            )
            await tasks()
            self.assertIn(sender, awaiting_other)
            send_message.assert_awaited_once_with(
                sender,
                main.OTHER_LANGUAGE_REPLY,
                supported_language_picker=True,
            )

            free_text_tasks = BackgroundTasks()
            await main.receive_webhook(
                self.make_request(
                    "wamid.language-other-text",
                    "Can you help me?",
                ),
                free_text_tasks,
            )
            await free_text_tasks()

            self.assertNotIn(sender, awaiting_other)
            generate_reply.assert_not_awaited()
            self.assertEqual(send_message.await_count, 2)
            send_message.assert_awaited_with(
                sender,
                main.UNRECOGNIZED_MESSAGE_REPLIES["english"],
                command_picker=True,
                language="english",
            )

    @patch("app.main.httpx.AsyncClient")
    async def test_language_picker_sends_whatsapp_interactive_buttons(
        self,
        client_factory,
    ):
        from app import main

        client = client_factory.return_value.__aenter__.return_value
        client.post = AsyncMock(return_value=Mock(status_code=200))

        with (
            patch.object(main, "WHATSAPP_TOKEN", "test-token"),
            patch.object(main, "PHONE_NUMBER_ID", "test-phone-id"),
        ):
            await main.send_whatsapp_message(
                "15551234567",
                main.WELCOME_REPLY,
                language_picker=True,
            )

        sent_body = client.post.call_args.kwargs["json"]
        self.assertEqual(sent_body["type"], "interactive")
        buttons = sent_body["interactive"]["action"]["buttons"]
        self.assertEqual(
            [button["reply"]["title"] for button in buttons],
            ["English", "Kiswahili", "Other"],
        )
        self.assertEqual(
            [button["reply"]["id"] for button in buttons],
            [
                main.LANGUAGE_BUTTON_IDS["english"],
                main.LANGUAGE_BUTTON_IDS["swahili"],
                main.LANGUAGE_BUTTON_IDS["other"],
            ],
        )

    @patch("app.main.httpx.AsyncClient")
    async def test_supported_language_picker_sends_two_buttons(
        self,
        client_factory,
    ):
        from app import main

        client = client_factory.return_value.__aenter__.return_value
        client.post = AsyncMock(return_value=Mock(status_code=200))

        with (
            patch.object(main, "WHATSAPP_TOKEN", "test-token"),
            patch.object(main, "PHONE_NUMBER_ID", "test-phone-id"),
        ):
            await main.send_whatsapp_message(
                "15551234567",
                main.OTHER_LANGUAGE_REPLY,
                supported_language_picker=True,
            )

        sent_body = client.post.call_args.kwargs["json"]
        buttons = sent_body["interactive"]["action"]["buttons"]
        self.assertEqual(
            [button["reply"]["title"] for button in buttons],
            ["English", "Kiswahili"],
        )

    @patch("app.main.httpx.AsyncClient")
    async def test_command_picker_sends_tappable_whatsapp_list(
        self,
        client_factory,
    ):
        from app import main

        client = client_factory.return_value.__aenter__.return_value
        client.post = AsyncMock(return_value=Mock(status_code=200))

        with (
            patch.object(main, "WHATSAPP_TOKEN", "test-token"),
            patch.object(main, "PHONE_NUMBER_ID", "test-phone-id"),
        ):
            await main.send_whatsapp_message(
                "15551234567",
                main.HELP_REPLY_SWAHILI,
                command_picker=True,
                language="swahili",
            )

        sent_body = client.post.call_args.kwargs["json"]
        self.assertEqual(sent_body["type"], "interactive")
        interactive = sent_body["interactive"]
        self.assertEqual(interactive["type"], "list")
        rows = interactive["action"]["sections"][0]["rows"]
        self.assertEqual(
            [row["id"] for row in rows],
            [option["id"] for option in main.COMMAND_OPTIONS["swahili"]],
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(interactive["action"]["button"], "Chagua amri")

    async def test_selected_language_is_passed_to_legal_reply(self):
        from app import main

        sender = "15551234567"
        with (
            patch.object(main, "conversation_history", {}),
            patch.object(main, "preferred_languages", {sender: "swahili"}),
            patch(
                "app.main.get_ai_reply",
                new=AsyncMock(return_value="Jibu"),
            ) as generate_reply,
            patch("app.main.send_whatsapp_message", new=AsyncMock()),
        ):
            history: list[dict[str, str]] = []
            await main.process_incoming_message(
                sender,
                "Swali la kisheria",
                [],
                True,
                history,
            )

        generate_reply.assert_awaited_once_with(
            "Swali la kisheria",
            [],
            case_mode=True,
            language="swahili",
        )

    async def test_case_summary_contains_user_facts_but_no_assistant_claims(self):
        from app import main

        history = [
            {"role": "user", "content": "I was dismissed on 12 June."},
            {"role": "assistant", "content": "You will win."},
            {"role": "user", "content": "I have my payslips."},
        ]

        summary = main._format_case_summary(history)

        self.assertIn("I was dismissed on 12 June.", summary)
        self.assertIn("I have my payslips.", summary)
        self.assertNotIn("You will win.", summary)
        self.assertIn("not legal advice", summary)

    async def test_delete_command_wipes_sender_case_data(self):
        from app import main

        sender = "15551234567"
        with (
            patch.object(main, "processed_message_ids", set()),
            patch.object(
                main,
                "conversation_history",
                {sender: [{"role": "user", "content": "Private facts"}]},
            ) as history,
            patch.object(
                main,
                "conversation_last_activity",
                {sender: 42.0},
            ) as activity,
            patch.object(main, "case_mode_senders", {sender}) as case_senders,
            patch(
                "app.main.send_whatsapp_message",
                new=AsyncMock(),
            ) as send_message,
        ):
            tasks = BackgroundTasks()
            await main.receive_webhook(
                self.make_request("wamid.delete", "/delete"),
                tasks,
            )
            await tasks()

        self.assertEqual(history, {})
        self.assertEqual(activity, {})
        self.assertEqual(case_senders, set())
        send_message.assert_awaited_once_with(sender, main.DELETE_REPLY)

    async def test_non_text_message_gets_polite_reply_and_is_deduplicated(self):
        from app import main

        with (
            patch.object(main, "processed_message_ids", set()),
            patch(
                "app.main.get_ai_reply",
                new=AsyncMock(),
            ) as generate_reply,
            patch(
                "app.main.send_whatsapp_message",
                new=AsyncMock(),
            ) as send_message,
        ):
            tasks = BackgroundTasks()
            await main.receive_webhook(
                self.make_request(
                    "wamid.voice",
                    message_type="audio",
                ),
                tasks,
            )
            duplicate_tasks = BackgroundTasks()
            await main.receive_webhook(
                self.make_request(
                    "wamid.voice",
                    message_type="audio",
                ),
                duplicate_tasks,
            )
            await tasks()

        self.assertEqual(len(duplicate_tasks.tasks), 0)
        generate_reply.assert_not_awaited()
        send_message.assert_awaited_once_with(
            "15551234567",
            main.NON_TEXT_REPLY,
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
