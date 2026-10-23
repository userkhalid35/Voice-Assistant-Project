import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure project root is in sys.path when executed directly as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.models_and_data import (
    LIVE_ORDERS,
    reset_orders,
    VoiceTurnRequest,
    VoiceTurnResponse,
    SummaryRequest,
    CallOutcomeJSON,
)
from app.tools_and_voice import (
    generate_agent_reply,
    synthesize_speech_base64,
    build_structured_summary,
)

GREETINGS = {
    "hinglish": "Namaste! Main Aria hoon Aura Skincare customer support se. Aaj main aapke skincare order ya policy query mein kaise madad kar sakti hoon?",
    "english": "Hello! I am Aria from Aura Skincare support. How can I help you with your order or skincare questions today?",
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm greeting audio into RAM on startup so Start Call has 0ms lag
    for mode, text in GREETINGS.items():
        try:
            await synthesize_speech_base64(text, mode)
        except Exception:
            pass
    yield

app = FastAPI(title="Aura Skincare Voice CX Agent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/api/orders")
async def get_all_orders():
    return {"orders": list(LIVE_ORDERS.values())}

@app.post("/api/reset")
async def reset_demo_orders():
    updated = reset_orders()
    return {"orders": list(updated.values())}

@app.get("/api/greeting")
async def get_greeting(language_mode: str = "hinglish"):
    text = GREETINGS.get(language_mode, GREETINGS["hinglish"])
    b64, mime = await synthesize_speech_base64(text, language_mode)
    return {
        "reply_text": text,
        "audio_base64": b64,
        "audio_mime": mime,
        "orders": list(LIVE_ORDERS.values()),
    }

@app.post("/api/voice-turn", response_model=VoiceTurnResponse)
async def process_voice_turn(req: VoiceTurnRequest):
    reply_text, tool_used, tool_result = await generate_agent_reply(
        user_text=req.user_text,
        history=req.history,
        language_mode=req.language_mode,
    )
    b64, mime = await synthesize_speech_base64(reply_text, req.language_mode)
    return VoiceTurnResponse(
        reply_text=reply_text,
        audio_base64=b64,
        audio_mime=mime,
        tool_used=tool_used,
        tool_result=tool_result,
        orders=list(LIVE_ORDERS.values()),
    )

@app.post("/api/summary", response_model=CallOutcomeJSON)
async def create_call_summary(req: SummaryRequest):
    return await build_structured_summary(req.transcript)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
