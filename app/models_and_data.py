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


PRODUCTS_CATALOG: dict[str, dict[str, Any]] = {
    "vitamin-c-serum": {
        "name": "Vitamin C Serum (30ml)",
        "price": "₹699",
        "category": "Brightening Serum",
        "skin_type": [
            "dull skin",
            "dark spots",
            "pigmentation",
            "uneven skin tone",
            "all skin types",
        ],
        "description": "Brightening organic serum enriched with pure Vitamin C that fades dark spots and uneven pigmentation while boosting collagen for radiant skin.",
        "usage": "Best used in the morning or evening before moisturizer. In the morning, always pair it under Hydrating Sunscreen SPF 50.",
    },
    "hydrating-sunscreen": {
        "name": "Hydrating Sunscreen SPF 50",
        "price": "₹499",
        "category": "Broad-Spectrum Sunscreen",
        "skin_type": [
            "all skin types",
            "oily skin",
            "sensitive skin",
            "dry skin",
            "combination skin",
        ],
        "description": "Lightweight, no-white-cast broad-spectrum UVA/UVB organic sunscreen providing deep hydration without clogging pores.",
        "usage": "Apply generously every morning as the final step in your routine over Vitamin C Serum. Reapply every 3-4 hours outdoors.",
    },
    "green-tea-combo": {
        "name": "Green Tea Face Wash + Toner",
        "price": "₹850",
        "category": "Clarifying Cleanser & Pore Toner",
        "skin_type": [
            "oily skin",
            "acne-prone skin",
            "combination skin",
            "open pores",
            "blemishes",
        ],
        "description": "Gentle purifying cleanser and pore-tightening toner combo with green tea antioxidants that controls excess sebum and calms irritated or acne-prone skin.",
        "usage": "Use face wash morning and night to purify pores, then gently pat toner to tighten pores and balance skin pH.",
    },
}

BRAND_POLICIES = """
Aura Skincare Brand & Policy Guidelines:
1. Brand Overview: Premium organic Indian skincare brand focused on simple, effective, clean formulations and personalized routine guidance.
2. Official Product Catalog & Pricing:
   - Vitamin C Serum (30ml) — ₹699: Brightening organic serum for dull skin, dark spots, pigmentation, and uneven skin tone. Best used morning/evening before moisturizer. Pairs well under Hydrating Sunscreen in the morning.
   - Hydrating Sunscreen SPF 50 — ₹499: Lightweight, no-white-cast broad-spectrum organic sunscreen for all skin types (including oily and sensitive skin). Pairs well over Vitamin C Serum in the morning.
   - Green Tea Face Wash + Toner — ₹850: Gentle purifying cleanser and pore-tightening toner combo ideal for oily, acne-prone, and combination skin.
3. Skincare Consulting Rules:
   - Match products accurately to customer skin type and concerns (e.g., Green Tea Combo for oily/acne-prone skin, Vitamin C for dullness/pigmentation, Sunscreen for daily protection across all skin types).
   - Recommend simple morning/evening routines (Cleanser -> Toner -> Vitamin C Serum -> Sunscreen in AM).
   - Never promise medical cures or prescription dermatological treatments.
   - Never invent unauthorized discounts or unlisted products.
4. Shipping Policy: Free delivery on orders above ₹499. Orders below ₹499 have a ₹50 shipping fee. Standard delivery takes 3-5 business days.
5. Return & Refund Policy: Returns are accepted ONLY within 7 days of delivery for unopened, unused products in original packaging. Opened or used products CANNOT be returned. Damaged or defective products must be reported within 48 hours of delivery with photos for replacement.
6. Cancellation Policy: Orders can be cancelled ONLY while their status is 'Processing'. Once an order is 'Shipped' or 'Out for Delivery', it cannot be cancelled (customers may refuse delivery at doorstep).
7. Cash on Delivery (COD): Available for orders up to ₹2,500. Customers can pay by cash or UPI at the doorstep.
"""


def build_system_prompt(language_mode: str = "auto") -> str:
    lang_instruction = """
LANGUAGE & TONE RULES (UNIFIED MULTILINGUAL):
- You are seamlessly bilingual in Indian English, Hindi, and conversational Hinglish.
- Automatically mirror the customer's language on every turn without needing any manual setting:
  * If the customer speaks in English, respond in warm, concise Indian English.
  * If the customer speaks in Hindi or Hinglish (or asks 'Hindi mein baat karo'), immediately switch and respond in natural conversational Hinglish written in clean Roman/Latin script (e.g., 'Haan ji, main abhi check karti hoon', 'Aapka order ORD-101 aaj shaam 6 PM tak deliver ho jayega').
  * Keep Roman Hinglish clean and TTS-friendly so Indian English and Hindi voice synthesizers pronounce it naturally. Never spell out punctuation or use Devanagari script in output.
"""
    if language_mode == "english":
        lang_instruction += "\n- Primary Mode Preference: Respond in warm, natural Indian English."
    elif language_mode == "hinglish":
        lang_instruction += "\n- Primary Mode Preference: Respond in natural urban Hinglish (Roman script)."

    return f"""You are Aria, a friendly, empathetic, and concise skincare specialist and customer support expert from Aura Skincare.
{lang_instruction}

CRITICAL VOICE & CX RULES:
- Keep every response between 1 and 3 short sentences (strictly under 45 words) so it sounds natural when spoken aloud.
- NEVER use markdown symbols (*, #, -, bullet points) because your text goes directly into a Text-to-Speech engine.
- Skincare Expertise: You give helpful product recommendations and routine advice based on Aura Skincare's catalog. Never recommend medical cures or non-Aura brands.
- Order Details: When reporting order status, always mention the courier partner (e.g., BlueDart or Delhivery) and expected delivery timeline from the live order notes.
- Strictly enforce Aura Skincare policies. Never promise a return for opened products or items delivered more than 7 days ago.
- Never promise cancellation for orders that are 'Out for Delivery' or 'Delivered' (politely suggest doorstep refusal for Out for Delivery items).
- Out-of-Scope: If the customer asks something unrelated to Aura Skincare orders or products (like booking flights to Goa, weather, sports, general coding/math), politely explain that you can only help with Aura Skincare orders, products, and policies.
- Order ID Not Found: If an order ID does not exist in our database or speech is unclear, politely ask the customer to verify and repeat the Order ID (such as ORD-101, ORD-102, or ORD-103).

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
    language_mode: str = "auto"


class VoiceAudioTurnRequest(BaseModel):
    audio_base64: str = Field(
        ..., description="Base64 encoded 16kHz mono WAV audio from browser mic"
    )
    browser_transcript_hint: str | None = ""
    history: list[TranscriptTurn] = []
    language_mode: str = "auto"


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
