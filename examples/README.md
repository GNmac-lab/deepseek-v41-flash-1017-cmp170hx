# Examples

All talk to the running server (`./start.sh` first) on `127.0.0.1:8000` by default; pass `--port` otherwise.

- `chat.py "question" [--effort none|low|high|xhigh|max]` — one request; prints the reasoning and the answer.
- `stream.py "prompt"` — streams the reply and reports tokens per second after the first token.
- `tool_call.py` — a full tool-calling round trip.
- `smoke.py` — what `./start.sh smoke` runs: one chat, one tool call, one image.
- `tp4.env`, `pp4.env` — `.env` starting points for the two layouts.
