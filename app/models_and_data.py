from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field

INITIAL_ORDERS: dict[str, dict[str, Any]] = {
    "ORD-101": {
        "order_id": "ORD-101",
        "customer": "Priya Sharma",
        "product": "Vitamin C Serum (30ml)",
        "value": "₹699",
        "status": "Out for Delivery",
        "days_since_delivery": None,
        "notes": "BlueDart - BD-982103. Expected by 6 PM today",
        "can_cancel": False,
        "can_return": False,
    },
    "ORD-102": {
        "order_id": "ORD-102",
        "customer": "Rahul Verma",
        "product": "Hydrating Sunscreen SPF 50",
        "value": "₹499",
        "status": "Delivered",
        "days_since_delivery": 14,
        "notes": "Delhivery DL-441029. Delivered 14 days ago",
        "can_cancel": False,
        "can_return": False,
    },
    "ORD-103": {
        "order_id": "ORD-103",
        "customer": "Ananya Patel",
        "product": "Green Tea Face Wash + Toner",
        "value": "₹850",
        "status": "Processing",
        "days_since_delivery": None,
        "notes": "Ordered 3 hours ago. Eligible for cancellation",
        "can_cancel": True,
        "can_return": False,
    },
}

LIVE_ORDERS: dict[str, dict[str, Any]] = deepcopy(INITIAL_ORDERS)


def reset_orders() -> dict[str, dict[str, Any]]:
    global LIVE_ORDERS
    LIVE_ORDERS = deepcopy(INITIAL_ORDERS)
    return LIVE_ORDERS


BRAND_POLICIES = """
Aura Skincare Brand & Policy Guidelines:
1. Brand Overview: Premium organic Indian skincare brand focused on simple, effective, and clean formulations.
2. Official Product Catalog & Pricing:
   - Vitamin C Serum (30ml) — ₹699: Enriched with pure Vitamin C; brightens skin tone, fades dark spots and pigmentation, and boosts collagen for radiant skin.
   - Hydrating Sunscreen SPF 50 — ₹499: Lightweight, non-comedogenic broad-spectrum UVA/UVB protection with deep hydration for all skin types.
   - Green Tea Face Wash + Toner — ₹850: Clarifying antioxidant cleanser and alcohol-free toner combo; controls excess oil, clears pores, and calms irritated or acne-prone skin.
3. Shipping Policy: Free delivery on orders above ₹499. Orders below ₹499 have a ₹50 shipping fee. Standard delivery takes 3-5 business days.
4. Return & Refund Policy: Returns are accepted ONLY within 7 days of delivery for unopened, unused products in original packaging. Opened or used products CANNOT be returned. Damaged or defective products must be reported within 48 hours of delivery with photos for replacement.
5. Cancellation Policy: Orders can be cancelled ONLY while their status is 'Processing'. Once an order is 'Shipped' or 'Out for Delivery', it cannot be cancelled (customers may refuse delivery at doorstep).
6. Cash on Delivery (COD): Available for orders up to ₹2,500. Customers can pay by cash or UPI at the doorstep.
"""


def build_system_prompt(language_mode: str = "hinglish") -> str:
    lang_instruction = (
        "Speak in natural urban Indian Hinglish (a warm, conversational mix of Hindi and English in Roman script, e.g., "
        "'Haan ji, main abhi check karti hoon', 'Aapka order ORD-101 aaj shaam 6 PM tak deliver ho jayega')."
        if language_mode == "hinglish"
        else "Speak in warm, natural, concise Indian English."
    )
    return f"""You are Aria, a friendly, professional, and concise customer support specialist from Aura Skincare.
{lang_instruction}

CRITICAL VOICE RULES:
- Keep every response between 1 and 3 short sentences (under 45 words) so it sounds natural when spoken aloud.
- NEVER use markdown symbols (*, #, -, bullet points) because your text goes directly into a Text-to-Speech engine.
- Strictly enforce Aura Skincare policies. Never promise a return for opened products or items delivered more than 7 days ago.
- Never promise cancellation for orders that are 'Out for Delivery' or 'Delivered' (politely suggest doorstep refusal for Out for Delivery items).
- If the customer asks something unrelated to Aura Skincare (like booking flights to Goa, weather, sports, coding), politely explain that you can only help with Aura Skincare orders and products.
- If an order ID does not exist in our database or speech is unclear, politely ask the customer to verify and repeat the Order ID (such as ORD-101, ORD-102, or ORD-103).

{BRAND_POLICIES}
"""


class TranscriptTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    text: str
    timestamp: str | None = None
    tool_used: str | None = None


class VoiceTurnRequest(BaseModel):
    user_text: str
    history: list[TranscriptTurn] = []
    language_mode: str = "hinglish"


class VoiceAudioTurnRequest(BaseModel):
    audio_base64: str = Field(
        ..., description="Base64 encoded 16kHz mono WAV audio from browser mic"
    )
    browser_transcript_hint: str | None = ""
    history: list[TranscriptTurn] = []
    language_mode: str = "hinglish"


class VoiceTurnResponse(BaseModel):
    transcribed_user_text: str | None = None
    reply_text: str
    audio_base64: str | None = None
    audio_mime: str = "audio/mp3"
    tool_used: str | None = None
    tool_result: dict[str, Any] | None = None
    orders: list[dict[str, Any]]


class SummaryRequest(BaseModel):
    transcript: list[TranscriptTurn]


class CallOutcomeJSON(BaseModel):
    customer_intent: str
    order_id: str | None = None
    resolution_status: str
    call_summary: str
