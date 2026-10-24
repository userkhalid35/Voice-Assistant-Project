import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure project root is in sys.path when executed directly as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models_and_data import (
    LIVE_ORDERS,
    CallOutcomeJSON,
    SummaryRequest,
    VoiceAudioTurnRequest,
    VoiceTurnRequest,
    VoiceTurnResponse,
    reset_orders,
)
from app.tools_and_voice import (
    build_structured_summary,
    generate_agent_reply,
    synthesize_speech_base64,
    transcribe_wav_audio,
)

GREETINGS = {
    "auto": "Namaste! I am Aria from Aura Skincare support. How can I help you with your order, skincare routine, or policy questions today? Aap Hindi ya English kisi me bhi pooch sakte hain.",
    "hinglish": "Namaste! Main Aria hoon Aura Skincare customer support se. Aaj main aapke skincare order ya policy query mein kaise madad kar sakti hoon?",
    "english": "Hello! I am Aria from Aura Skincare support. How can I help you with your order or skincare questions today?",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm greeting audio into RAM on startup so Start Call has 0ms lag
    for mode, text in GREETINGS.items():
        try:
            await synthesize_speech_base64(text, mode)
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed pre-warming greeting: %s", e)
    yield


app = FastAPI(title="Aura Skincare Voice CX Agent (Aria)", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Aura Skincare Voice CX Desk"}


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/api/orders")
async def get_all_orders():
    return {"orders": list(LIVE_ORDERS.values())}


@app.post("/api/reset")
async def reset_demo_orders():
    updated = reset_orders()
    return {"orders": list(updated.values())}


@app.get("/api/greeting")
async def get_greeting(language_mode: str = "auto"):
    text = GREETINGS.get(language_mode, GREETINGS["auto"])
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
        transcribed_user_text=req.user_text,
        reply_text=reply_text,
        audio_base64=b64,
        audio_mime=mime,
        tool_used=tool_used,
        tool_result=tool_result,
        orders=list(LIVE_ORDERS.values()),
    )


@app.post("/api/voice-audio", response_model=VoiceTurnResponse)
async def process_voice_audio(req: VoiceAudioTurnRequest):
    transcribed_text = await transcribe_wav_audio(
        audio_base64=req.audio_base64,
        browser_hint=req.browser_transcript_hint,
        language_mode=req.language_mode,
    )
    if not transcribed_text or not transcribed_text.strip():
        return VoiceTurnResponse(
            transcribed_user_text="",
            reply_text="",
            audio_base64=None,
            audio_mime="audio/mp3",
            tool_used=None,
            tool_result=None,
            orders=list(LIVE_ORDERS.values()),
        )
    reply_text, tool_used, tool_result = await generate_agent_reply(
        user_text=transcribed_text,
        history=req.history,
        language_mode=req.language_mode,
    )
    b64, mime = await synthesize_speech_base64(reply_text, req.language_mode)
    return VoiceTurnResponse(
        transcribed_user_text=transcribed_text,
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
    port = int(os.environ.get("PORT", "8000"))
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
