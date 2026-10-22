# whatsapp-bot

Set `VERIFY_TOKEN`, `WHATSAPP_TOKEN`, and `PHONE_NUMBER_ID` in a `.env` file (use
`.env.example` as a template) to configure Meta webhook verification and replies.
The webhook replies to incoming text messages with `You said: <message>`. Run the
app with `uvicorn app.main:app --reload`.