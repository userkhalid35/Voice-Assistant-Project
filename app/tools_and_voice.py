import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import edge_tts
import httpx
import speech_recognition as sr
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.models_and_data import (
    LIVE_ORDERS,
    PRODUCTS_CATALOG,
    CallMetrics,
    CallOutcomeJSON,
    TranscriptTurn,
    build_system_prompt,
)

load_dotenv()

AUDIO_RAM_CACHE: dict[str, tuple[str, str]] = {}
MAX_AUDIO_CACHE_SIZE = 128


def _cache_audio(key: str, val: tuple[str, str]) -> None:
    if len(AUDIO_RAM_CACHE) >= MAX_AUDIO_CACHE_SIZE:
        try:
            first_key = next(iter(AUDIO_RAM_CACHE))
            del AUDIO_RAM_CACHE[first_key]
        except (StopIteration, KeyError):
            pass
    AUDIO_RAM_CACHE[key] = val


HINDI_HINGLISH_INDICATORS = {
    "mera", "meri", "mere", "kahan", "kab", "karo", "karein", "karna", "karta", "karti",
    "hai", "hain", "ho", "hoon", "tha", "thi", "the", "kya", "kaise", "kyun", "kaunsa",
    "kaunsi", "kaunse", "acha", "accha", "theek", "shukriya", "dhanyawad", "namaste",
    "batao", "bataiye", "bata", "chahiye", "radd", "wapas", "khola", "madad", "chehra",
    "daag", "dhabbe", "dhoop", "kitne", "paisa", "rupaye", "rupayee", "din", "pehle",
    "aaj", "shaam", "subah", "raat", "kharab", "lekin", "aur", "ya", "nahi", "nahin",
    "abhi", "turant", "mujhe", "aapka", "aapki", "aapke", "hoga", "hogi", "dena", "liya",
    "baat", "mein", "par", "kisi", "kuch", "suno", "sunao", "chalega"
}


def detect_language(text: str) -> str:
    """Detects whether text is Hindi/Hinglish or English."""
    if not text:
        return "english"
    if re.search(r"[\u0900-\u097f]", text):
        return "hinglish"
    lower = text.lower()
    # Explicit user instruction to speak English
    if re.search(r"\b(?:only english|in english|speak english|no hindi|not hindi)\b", lower):
        return "english"
    if "hindi" in lower or "hinglish" in lower:
        return "hinglish"
    words = set(re.findall(r"\b[a-zA-Z]+\b", lower))
    if len(words.intersection(HINDI_HINGLISH_INDICATORS)) >= 1:
        return "hinglish"
    return "english"

WORD_TO_DIGIT = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "shunya": "0",
    "one": "1",
    "won": "1",
    "ek": "1",
    "two": "2",
    "to": "2",
    "too": "2",
    "do": "2",
    "three": "3",
    "teen": "3",
    "four": "4",
    "char": "4",
    "five": "5",
    "panch": "5",
    "six": "6",
    "che": "6",
    "seven": "7",
    "saat": "7",
    "eight": "8",
    "aath": "8",
    "nine": "9",
    "nau": "9",
}


def extract_and_normalize_order_id(text: str) -> str | None:
    if not text:
        return None
    cleaned = text.lower()
    cleaned = re.sub(r"\bo\s*r\s*d\b", "ord", cleaned)
    cleaned = re.sub(r"\b(dash|hyphen|minus)\b", "-", cleaned)
    cleaned = re.sub(r"\bone\s+hundred\s+(?:and\s+)?one\b", "101", cleaned)
    cleaned = re.sub(r"\bone\s+hundred\s+(?:and\s+)?two\b", "102", cleaned)
    cleaned = re.sub(r"\bone\s+hundred\s+(?:and\s+)?three\b", "103", cleaned)
    cleaned = re.sub(r"\bek\s+sau\s+(?:aur\s+)?ek\b", "101", cleaned)
    cleaned = re.sub(r"\bek\s+sau\s+(?:aur\s+)?do\b", "102", cleaned)
    cleaned = re.sub(r"\bek\s+sau\s+(?:aur\s+)?teen\b", "103", cleaned)

    for word, digit in WORD_TO_DIGIT.items():
        cleaned = re.sub(rf"\b{word}\b", digit, cleaned)

    cleaned = re.sub(r"(?<=\d)[\s\-_]+(?=\d)", "", cleaned)

    explicit = re.search(r"\bord[\s\-_]*(\d{3})\b", cleaned)
    if explicit:
        return f"ORD-{explicit.group(1)}"

    contextual = re.search(
        r"\b(?:order|id|number|status|cancel|return|track|mera|parcel|package|shipment)\D*(\d{3})\b", cleaned
    )
    if contextual:
        return f"ORD-{contextual.group(1)}"

    standalone = re.search(r"\b(10[1-9]|999)\b", cleaned)
    if standalone:
        return f"ORD-{standalone.group(1)}"

    return None


def get_order_details(order_id: str) -> dict[str, Any]:
    raw_id = re.sub(r"[^\w\-]", "", (order_id or "").upper().strip())
    norm_id = raw_id
    if not norm_id.startswith("ORD-"):
        digits = re.sub(r"\D", "", norm_id)
        norm_id = f"ORD-{digits}" if digits else norm_id

    order = LIVE_ORDERS.get(norm_id)
    if not order:
        return {
            "found": False,
            "requested_id": norm_id or order_id,
            "error": f"Order {norm_id or order_id} was not found in Aura Skincare's database. Valid sample IDs are ORD-101, ORD-102, ORD-103.",
        }
    return {"found": True, "order": order}


def cancel_order(order_id: str) -> dict[str, Any]:
    lookup = get_order_details(order_id)
    if not lookup.get("found"):
        return lookup

    order = lookup["order"]
    status = order["status"]
    if status == "Processing":
        order["status"] = "Cancelled"
        order["can_cancel"] = False
        order["notes"] = "Cancelled by customer via Aria Voice CX Agent."
        return {
            "found": True,
            "action": "CANCEL_ORDER",
            "success": True,
            "order": order,
            "message": f"Order {order['order_id']} ({order['product']}) has been successfully cancelled.",
        }
    elif status == "Cancelled":
        return {
            "found": True,
            "action": "CANCEL_ORDER",
            "success": False,
            "order": order,
            "message": f"Order {order['order_id']} is already cancelled.",
        }
    else:
        return {
            "found": True,
            "action": "CANCEL_ORDER",
            "success": False,
            "order": order,
            "message": (
                f"Cannot cancel {order['order_id']} because its status is '{status}'. "
                "Per Aura Skincare policy, cancellation is only allowed while 'Processing'. "
                "If it is Out for Delivery, the customer can refuse delivery at the doorstep."
            ),
        }


def get_product_recommendations(query_or_skin_type: str) -> dict[str, Any]:
    query = (query_or_skin_type or "").lower().strip()
    matched = []

    # 1. Vitamin C Serum
    if any(
        k in query
        for k in [
            "vitamin c",
            "serum",
            "brighten",
            "glow",
            "dull",
            "dark spot",
            "pigment",
            "uneven",
            "सीरम",
        ]
    ):
        matched.append(PRODUCTS_CATALOG["vitamin-c-serum"])

    # 2. Hydrating Sunscreen SPF 50
    if re.search(r"\b(?:sunscreen|spf|sun|dhoop|tan|protect|hydration|moistur\w*|uv)\b", query) or "सनस्क्रीन" in query:
        matched.append(PRODUCTS_CATALOG["hydrating-sunscreen"])

    # 3. Green Tea Face Wash + Toner
    if re.search(r"\b(?:green tea|face wash|wash|toner|acne|pimple|pore|pores|cleanse|cleanser|muhase)\b", query) or any(k in query for k in ["मुंहासे", "पिंपल"]):
        matched.append(PRODUCTS_CATALOG["green-tea-combo"])

    # Skin type heuristics
    if re.search(r"\b(?:oily|greasy|oil|tel|chipchip\w*)\b", query):
        if PRODUCTS_CATALOG["green-tea-combo"] not in matched:
            matched.append(PRODUCTS_CATALOG["green-tea-combo"])
        if PRODUCTS_CATALOG["hydrating-sunscreen"] not in matched:
            matched.append(PRODUCTS_CATALOG["hydrating-sunscreen"])

    if re.search(r"\b(?:dry|sukhi|dehydrat\w*)\b", query):
        if PRODUCTS_CATALOG["hydrating-sunscreen"] not in matched:
            matched.append(PRODUCTS_CATALOG["hydrating-sunscreen"])
        if PRODUCTS_CATALOG["vitamin-c-serum"] not in matched:
            matched.append(PRODUCTS_CATALOG["vitamin-c-serum"])

    if re.search(r"\b(?:combination|sensitive)\b", query):
        if PRODUCTS_CATALOG["green-tea-combo"] not in matched:
            matched.append(PRODUCTS_CATALOG["green-tea-combo"])
        if PRODUCTS_CATALOG["hydrating-sunscreen"] not in matched:
            matched.append(PRODUCTS_CATALOG["hydrating-sunscreen"])

    # General / Catalog / Fallback
    if not matched or any(
        k in query
        for k in [
            "catalog",
            "all products",
            "entire catalog",
            "whole range",
            "sab products",
            "sari range",
            "all range",
        ]
    ):
        matched = list(PRODUCTS_CATALOG.values())

    return {
        "query": query_or_skin_type,
        "matched_count": len(matched),
        "products": matched,
        "routine_tip": "Recommended routine: Purify with Green Tea Face Wash and Toner, brighten with Vitamin C Serum, and finish with Hydrating Sunscreen SPF 50 every morning.",
    }


def execute_fast_tool_router(
    user_text: str,
) -> tuple[str | None, dict[str, Any] | None]:
    lower = user_text.lower()
    order_id = extract_and_normalize_order_id(user_text)

    if order_id:
        if any(
            k in lower
            for k in [
                "cancel",
                "radd",
                "hatana",
                "dont want",
                "don't want",
                "hata do",
                "nahi chahiye",
            ]
        ):
            return f"cancel_order('{order_id}')", cancel_order(order_id)
        return f"get_order_details('{order_id}')", get_order_details(order_id)

    # Check if query is actually a policy/return/cancellation inquiry before checking product keywords
    is_policy_or_return_query = any(
        k in lower
        for k in [
            "return",
            "refund",
            "replace",
            "exchange",
            "wapas",
            "cancel",
            "radd",
            "policy",
            "shipping charge",
            "delivery fee",
            "cod",
            "cash on delivery",
        ]
    )

    # Check for Skincare / Product inquiries
    skincare_keywords = [
        "vitamin",
        "serum",
        "sunscreen",
        "spf",
        "face wash",
        "toner",
        "green tea",
        "skin",
        "oily",
        "dry",
        "acne",
        "pimple",
        "tan",
        "dull",
        "dark spot",
        "pigment",
        "pore",
        "pores",
        "glow",
        "price",
        "kitne ka",
        "cost",
        "routine",
        "recommend",
        "suggestion",
        "best",
        "kaunsa",
        "product",
        "products",
        "catalog",
        "chehra",
        "cream",
        "cleanser",
        "daag",
        "dhabbe",
        "सीरम",
        "सनस्क्रीन",
    ]
    if not is_policy_or_return_query and any(k in lower for k in skincare_keywords):
        clean_query = user_text.strip()
        if len(clean_query) > 50:
            clean_query = clean_query[:50] + "..."
        return (
            f"get_product_recommendations('{clean_query}')",
            get_product_recommendations(user_text),
        )

    return None, None


def _recognize_wav_with_sr(wav_bytes: bytes, language_mode: str = "hinglish") -> str:
    """Uses Python SpeechRecognition library in RAM (no pyaudio required) to transcribe WAV bytes."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio_data = recognizer.record(source)
    # Try Indian English (captures both English & Romanized Hinglish + digits cleanly)
    try:
        recognize_google_fn = getattr(recognizer, "recognize_google", None)
        if callable(recognize_google_fn):
            text = recognize_google_fn(audio_data, language="en-IN")
            if isinstance(text, str) and text.strip():
                return text.strip()
            if language_mode == "hinglish":
                text_hi = recognize_google_fn(audio_data, language="hi-IN")
                if isinstance(text_hi, str) and text_hi.strip():
                    return text_hi.strip()

    except Exception as e:  # noqa: BLE001
        logger.debug("SpeechRecognition decoding failed: %s", e)
    return ""


async def transcribe_wav_audio(
    audio_base64: str,
    browser_hint: str | None = "",
    language_mode: str = "hinglish",
) -> str:
    """
    4-Layer Fail-Safe Audio-to-Text Pipeline:
    1. Python `speech_recognition` (Google Web Speech STT on WAV bytes in RAM - 100% Free, Zero API Key)
    2. OpenAI Whisper (`whisper-large-v3-turbo` via Groq REST API if GROQ_API_KEY is set)
    3. Gemini 2.5 Flash Multimodal Audio STT (if GEMINI_API_KEY is set)
    4. Browser Web Speech API live transcript hint
    """
    hint_clean = (browser_hint or "").strip()

    try:
        raw_b64 = audio_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        wav_bytes = base64.b64decode(raw_b64)
    except Exception as e:  # noqa: BLE001
        logger.debug("Base64 audio decode failed: %s", e)
        return hint_clean

    if len(wav_bytes) < 1000:
        return hint_clean

    # Layer 1: Python SpeechRecognition (Free, Zero API Keys, High Accuracy for en-IN)
    try:
        sr_text = await asyncio.to_thread(
            _recognize_wav_with_sr, wav_bytes, language_mode
        )
        if sr_text:
            return sr_text
    except Exception as e:  # noqa: BLE001
        logger.debug("Layer 1 SpeechRecognition error: %s", e)

    # Layer 2: OpenAI Whisper via Groq API
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and groq_key != "your_optional_groq_api_key_here":
        try:
            files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
            data = {
                "model": "whisper-large-v3-turbo",
                "prompt": "Aura Skincare support call in Indian English and Hinglish. Order IDs: ORD-101, ORD-102, ORD-103.",
                "temperature": "0.0",
            }
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {groq_key}"},
                    data=data,
                    files=files,
                )
                if resp.status_code == 200:
                    raw_text = resp.json().get("text")
                    if isinstance(raw_text, str) and raw_text.strip():
                        return raw_text.strip()
        except Exception as e:  # noqa: BLE001
            logger.debug("Layer 2 Groq Whisper error: %s", e)

    # Layer 3: Gemini 2.5 Flash Native Audio STT
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": (
                                    "Transcribe the spoken words from this customer audio in Roman script (English or Hinglish). "
                                    "Return ONLY the raw spoken text. If silence, return EMPTY."
                                )
                            },
                            {
                                "inline_data": {
                                    "mime_type": "audio/wav",
                                    "data": audio_base64,
                                }
                            },
                        ]
                    }
                ],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 100},
            }
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    candidates = resp.json().get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and isinstance(parts[0].get("text"), str):
                            t_text = parts[0]["text"].strip()
                            if t_text and t_text.upper() != "EMPTY":
                                return t_text
        except Exception as e:  # noqa: BLE001
            logger.debug("Layer 3 Gemini Audio error: %s", e)

    # Layer 4: Browser Web Speech Hint
    return hint_clean


def deterministic_fallback_brain(
    user_text: str,
    language_mode: str,
    tool_result: dict[str, Any] | None,
) -> str:
    lower = user_text.lower()
    effective_lang = detect_language(user_text) if language_mode == "auto" else language_mode
    is_hi = effective_lang == "hinglish"

    out_of_scope_terms = [
        "flight",
        "goa",
        "hotel",
        "train",
        "weather",
        "cricket",
        "movie",
        "pizza",
        "biryani",
        "stock",
        "bitcoin",
        "crypto",
        "coding",
        "python",
        "math",
    ]
    # Check if the query is actually skincare or order related before considering out-of-scope
    has_skincare_or_order_intent = bool(
        extract_and_normalize_order_id(user_text)
        or any(
            re.search(rf"\b{w}\b", lower)
            for w in [
                "sunscreen", "serum", "face wash", "toner", "spf", "cream", "skin",
                "oily", "dry", "acne", "pimple", "order", "delivery", "track", "cancel", "return", "product"
            ]
        )
    )
    if not has_skincare_or_order_intent and any(re.search(rf"\b{term}\b", lower) for term in out_of_scope_terms):
        return (
            "Maaf kijiyega, main sirf Aura Skincare ke products, orders aur policies mein aapki madad kar sakti hoon. Kya main aapke kisi order ya skincare query mein help karoon?"
            if is_hi
            else "I'm sorry, I can only assist you with Aura Skincare orders, products, and policies. Would you like me to check an order or recommend a product?"
        )

    # 1. Order details / cancellation tool result
    if tool_result is not None and "found" in tool_result:
        if not tool_result.get("found"):
            bad_id = tool_result.get("requested_id", "that ID")
            return (
                f"Mujhe {bad_id} naam ka koi order nahi mila. Kya aap please apna Order ID dobara check karke bata sakte hain, jaise ORD-101, ORD-102 ya ORD-103?"
                if is_hi
                else f"I couldn't locate an order with ID {bad_id}. Could you please verify and repeat your order number, such as ORD-101, ORD-102, or ORD-103?"
            )

        order = tool_result["order"]
        oid = order["order_id"]
        prod = order["product"]
        status = order["status"]

        if tool_result.get("action") == "CANCEL_ORDER":
            if tool_result.get("success"):
                return (
                    f"Haan ji, aapka order {oid} ({prod}) abhi Processing mein tha, isliye maine ise successfully cancel kar diya hai."
                    if is_hi
                    else f"Your order {oid} for {prod} was still Processing, so I have successfully cancelled it for you right away."
                )
            else:
                return (
                    f"Maaf kijiyega, order {oid} abhi '{status}' status mein hai aur hum sirf Processing orders hi cancel kar sakte hain. Aap delivery ke waqt doorstep par parcel refuse kar sakte hain."
                    if is_hi
                    else f"I'm sorry, order {oid} is currently '{status}', and cancellations are only possible while an order is Processing. You may refuse the package at your doorstep."
                )

        if any(
            w in lower for w in ["return", "refund", "wapas", "replace", "exchange"]
        ):
            if oid == "ORD-101":
                return (
                    f"Aapka order {oid} ({prod}) abhi Out for Delivery hai. Returns sirf delivery ke baad unopened products par hi request kiye ja sakte hain, ya aap delivery ke waqt doorstep par parcel refuse kar sakte hain."
                    if is_hi
                    else f"Order {oid} for {prod} is currently Out for Delivery. Returns can only be initiated after delivery for unopened items, or you may refuse delivery at the doorstep."
                )
            if oid == "ORD-102" or (order.get("days_since_delivery") and order["days_since_delivery"] > 7):
                return (
                    f"Aapka order {oid} ({prod}) 14 din pehle deliver hua tha. Hamari policy ke mutabik returns sirf delivery ke 7 din ke andar unopened products par hi valid hain, isliye yeh return nahi ho payega."
                    if is_hi
                    else f"Order {oid} for {prod} was delivered 14 days ago. Since our return policy only covers unopened items within 7 days of delivery, this order is not eligible for a return."
                )
            if oid == "ORD-103":
                return (
                    f"Aapka order {oid} ({prod}) abhi Processing status mein hai. Dispatch se pehle aap ise return karne ki jagah turant cancel kar sakte hain."
                    if is_hi
                    else f"Your order {oid} for {prod} is still Processing. Before dispatch, you may cancel it right away instead of waiting for return."
                )

        if oid == "ORD-101":
            return (
                f"Aapka order {oid}, {prod}, abhi BlueDart ke saath Out for Delivery hai aur aaj shaam 6 PM tak deliver ho jayega."
                if is_hi
                else f"Your order {oid} for {prod} is currently Out for Delivery via BlueDart and is expected to arrive by 6 PM today."
            )
        if oid == "ORD-102":
            return (
                f"Aapka order {oid}, {prod}, 14 din pehle Delhivery ke through deliver ho chuka hai."
                if is_hi
                else f"Your order {oid} for {prod} was delivered 14 days ago via Delhivery."
            )
        if oid == "ORD-103":
            return (
                f"Aapka order {oid}, {prod} (₹850), abhi {status} status mein hai. Kyunki yeh 3 ghante pehle place hua tha, aap chahein toh ise abhi cancel bhi karwa sakte hain."
                if is_hi
                else f"Your order {oid} for {prod} is currently {status}. Since it was ordered 3 hours ago, it is still eligible for cancellation if needed."
            )

    # 2. Policy & Return questions (checked before generic product recommendations)
    if any(
        w in lower
        for w in [
            "return",
            "refund",
            "opened",
            "khola",
            "20 days",
            "14 days",
            "damage",
            "defective",
        ]
    ):
        if any(w in lower for w in ["open", "used", "khola", "20", "14", "10"]):
            return (
                "Maaf kijiyega, Aura Skincare ki policy ke mutabik hum sirf delivery ke 7 din ke andar unopened aur unused products hi return le sakte hain. Opened ya 7 din se purane products return nahi ho sakte."
                if is_hi
                else "I'm sorry, Aura Skincare only accepts returns within 7 days of delivery for unopened, unused products in their original packaging. Opened items or orders past 7 days cannot be returned."
            )
        if any(w in lower for w in ["damage", "broken", "leak", "defective", "kharab"]):
            return (
                "Agar aapko damaged ya defective product mila hai, toh please delivery ke 48 hours ke andar photos ke saath report karein taaki hum turant replacement bhej sakein."
                if is_hi
                else "For damaged or defective items, please report the issue with photos within 48 hours of delivery and we will arrange a replacement."
            )
        return (
            "Hamari return policy ke tahat aap delivery ke 7 din ke andar unopened aur unused products original packaging mein return kar sakte hain. Damaged items 48 hours mein photos ke saath report karne par replace kiye jaate hain."
            if is_hi
            else "We accept returns within 7 days of delivery for unopened and unused products in original packaging. Damaged products must be reported within 48 hours with photos for a replacement."
        )

    if any(w in lower for w in ["ship", "delivery charge", "kitne din", "fee"]):
        return (
            "₹499 se upar ke sabhi orders par shipping bilkul free hai, aur ₹499 se kam par ₹50 shipping charge lagta hai. Standard delivery mein 3 se 5 business days lagte hain."
            if is_hi
            else "We offer free shipping on orders above ₹499, and a ₹50 shipping fee on orders below ₹499. Standard delivery takes 3 to 5 business days."
        )

    if any(w in lower for w in ["cod", "cash on delivery", "upi", "pay"]):
        return (
            "Haan ji, ₹2,500 tak ke orders par Cash on Delivery available hai, aur aap doorstep par cash ya UPI dono se payment kar sakte hain."
            if is_hi
            else "Yes, Cash on Delivery is available for orders up to ₹2,500, and you can pay by cash or UPI right at your doorstep."
        )

    if any(w in lower for w in ["cancel", "radd"]):
        return (
            "Orders sirf tabhi cancel ho sakte hain jab unka status Processing ho. Shipped ya Out for Delivery hone ke baad cancel nahi hota, par aap doorstep par delivery refuse kar sakte hain. Apna Order ID batayein toh main check karoon?"
            if is_hi
            else "Orders can only be cancelled while in Processing status. Once Shipped or Out for Delivery, they cannot be cancelled, though you may refuse doorstep delivery. Please share your Order ID so I can check."
        )

    # 3. Product recommendations tool result
    if tool_result is not None and "products" in tool_result:
        if any(w in lower for w in ["oil", "acne", "pimple", "pore", "pores", "face wash", "toner"]):
            return (
                "Oily aur acne-prone skin ke liye hamara Green Tea Face Wash + Toner combo (₹850) sabse best hai, jo excess oil aur pores ko control karta hai. Iske baad Hydrating Sunscreen SPF 50 (₹499) lagayein."
                if is_hi
                else "For oily and acne-prone skin, our Green Tea Face Wash + Toner combo (₹850) is ideal for clearing pores, paired with our lightweight Hydrating Sunscreen SPF 50 (₹499)."
            )
        if any(w in lower for w in ["vitamin c", "serum", "brighten", "dull", "dark spot", "pigment"]):
            return (
                "Dull skin aur dark spots ke liye hamara Vitamin C Serum (30ml) ₹699 ka hai. Yeh skin ko brighten karta hai aur collagen boost karta hai. Morning mein ise Sunscreen ke neeche lagayein."
                if is_hi
                else "For dull skin and dark spots, our Vitamin C Serum (30ml) is priced at ₹699. It brightens skin and fades pigmentation. Apply it before sunscreen in the morning."
            )
        if any(w in lower for w in ["sunscreen", "spf", "dhoop", "sun"]):
            return (
                "Hamara Hydrating Sunscreen SPF 50 ₹499 ka hai. Yeh lightweight, zero-white-cast broad-spectrum UVA aur UVB protection deta hai aur sabhi skin types ke liye suitable hai."
                if is_hi
                else "Our Hydrating Sunscreen SPF 50 is ₹499. It provides lightweight, no-white-cast broad-spectrum UVA and UVB protection with deep hydration for all skin types."
            )
        if any(w in lower for w in ["dry", "sensitive"]):
            return (
                "Dry aur sensitive skin ke liye Hydrating Sunscreen SPF 50 (₹499) deep moisture deta hai, aur Vitamin C Serum (₹699) glowing nourishment provide karta hai."
                if is_hi
                else "For dry and sensitive skin, our Hydrating Sunscreen SPF 50 (₹499) delivers deep hydration, while Vitamin C Serum (₹699) provides gentle brightening nourishment."
            )
        # General catalog or routine
        return (
            "Aura Skincare mein Vitamin C Serum (₹699), Hydrating Sunscreen SPF 50 (₹499) aur Green Tea Face Wash + Toner (₹850) available hain. Aap apni skin type batayein toh main perfect routine suggest karoon?"
            if is_hi
            else "Aura Skincare offers Vitamin C Serum (₹699), Hydrating Sunscreen SPF 50 (₹499), and Green Tea Face Wash + Toner (₹850). Tell me your skin type and I can recommend the right routine."
        )

    if any(w in lower for w in ["cancel", "radd"]):
        return (
            "Orders sirf tabhi cancel ho sakte hain jab unka status Processing ho. Shipped ya Out for Delivery hone ke baad cancel nahi hota, par aap doorstep par delivery refuse kar sakte hain. Apna Order ID batayein toh main check karoon?"
            if is_hi
            else "Orders can only be cancelled while in Processing status. Once Shipped or Out for Delivery, they cannot be cancelled, though you may refuse doorstep delivery. Please share your Order ID so I can check."
        )

    # 4. Product pricing & general info
    if any(
        w in lower
        for w in [
            "vitamin c",
            "serum",
            "sunscreen",
            "spf",
            "green tea",
            "face wash",
            "toner",
            "product",
            "catalog",
            "price",
            "kitne ka",
            "cost",
            "कीमत",
            "सीरम",
            "सनस्क्रीन",
        ]
    ):
        if any(w in lower for w in ["vitamin c", "serum", "सीरम"]):
            return (
                "Hamara Vitamin C Serum (30ml) ₹699 ka hai. Yeh skin tone ko brighten karta hai, dark spots aur pigmentation ko reduce karta hai aur collagen boost karta hai."
                if is_hi
                else "Our Vitamin C Serum (30ml) is priced at ₹699. It brightens skin tone, fades dark spots and pigmentation, and boosts collagen for radiant skin."
            )
        if any(w in lower for w in ["sunscreen", "spf", "धूप", "सनस्क्रीन"]):
            return (
                "Hamara Hydrating Sunscreen SPF 50 ₹499 ka hai. Yeh lightweight aur non-greasy hai jo high UVA/UVB protection ke saath deep hydration deta hai."
                if is_hi
                else "Our Hydrating Sunscreen SPF 50 is ₹499. It offers lightweight, non-greasy broad-spectrum UVA/UVB protection with deep hydration for all skin types."
            )
        if any(
            w in lower
            for w in ["green tea", "face wash", "toner", "मुंहासे", "पिंपल", "acne", "oil"]
        ):
            return (
                "Hamara Green Tea Face Wash + Toner combo ₹850 ka hai. Yeh excess oil control karta hai, pores ko deeply clean karta hai aur acne-prone skin ko calm karta hai."
                if is_hi
                else "Our Green Tea Face Wash + Toner combo is ₹850. It gently controls excess oil, clears pores, and calms irritated or acne-prone skin."
            )
        return (
            "Aura Skincare ke catalog mein Vitamin C Serum (₹699), Hydrating Sunscreen SPF 50 (₹499) aur Green Tea Face Wash + Toner (₹850) available hain. Aap kiske baare mein jaan-na chahenge?"
            if is_hi
            else "Aura Skincare offers Vitamin C Serum (₹699), Hydrating Sunscreen SPF 50 (₹499), and Green Tea Face Wash + Toner (₹850). Which product would you like to know more about?"
        )

    return (
        "Main aapki baat sun rahi hoon! Aap apna Order ID jaise ORD-101 bata sakte hain, ya apni skin type ke anusaar product recommendation aur routine pooch sakte hain."
        if is_hi
        else "I'm listening! You can share your Order ID such as ORD-101, ORD-102, or ORD-103, or ask about our skincare products and routine recommendations."
    )


async def generate_agent_reply(
    user_text: str,
    history: list[TranscriptTurn],
    language_mode: str = "auto",
) -> tuple[str, str | None, dict[str, Any] | None]:
    tool_used, tool_result = execute_fast_tool_router(user_text)

    # Determine effective language mode for prompt and mirroring
    effective_lang = (
        detect_language(user_text) if language_mode == "auto" else language_mode
    )
    system_prompt = build_system_prompt(effective_lang)

    tool_context = ""
    if tool_result is not None:
        tool_context = (
            f"\n\n[LIVE DATABASE TOOL RESULT for {tool_used}]:\n"
            f"{json.dumps(tool_result, ensure_ascii=False)}\n"
            "INSTRUCTION: Use this exact live tool data to answer the customer accurately and concisely. "
            "If reporting order tracking, always mention the courier partner (e.g., BlueDart or Delhivery) and expected delivery timeline from the live notes. "
            "If answering product/skincare advice, include the exact price and skin type benefits. Keep response under 3 short sentences."
        )

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            contents = []
            for turn in history[-6:]:
                role = "user" if turn.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": turn.text}]})
            contents.append(
                {"role": "user", "parts": [{"text": user_text + tool_context}]}
            )

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
            payload = {
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": contents,
                "generationConfig": {"temperature": 0.25, "maxOutputTokens": 140},
            }
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates and isinstance(candidates[0], dict):
                        content = candidates[0].get("content")
                        if isinstance(content, dict):
                            parts = content.get("parts", [])
                            if parts and isinstance(parts[0], dict) and isinstance(parts[0].get("text"), str):
                                text = parts[0]["text"].strip()
                                clean = re.sub(r"[\*\#\`]", "", text)
                                return clean, tool_used, tool_result
        except Exception as e:  # noqa: BLE001
            logger.debug("Gemini LLM reply error: %s", e)

    if groq_key and groq_key != "your_optional_groq_api_key_here":
        try:
            messages = [{"role": "system", "content": system_prompt}]
            for turn in history[-6:]:
                messages.append({"role": turn.role, "content": turn.text})
            messages.append({"role": "user", "content": user_text + tool_context})

            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": messages,
                        "temperature": 0.25,
                        "max_tokens": 130,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        msg = choices[0].get("message")
                        if isinstance(msg, dict):
                            msg_content = msg.get("content")
                            if isinstance(msg_content, str):
                                clean = re.sub(r"[\*\#\`]", "", msg_content.strip())
                                return clean, tool_used, tool_result
        except Exception as e:  # noqa: BLE001
            logger.debug("Groq LLM reply error: %s", e)

    fallback_reply = deterministic_fallback_brain(user_text, effective_lang, tool_result)
    return fallback_reply, tool_used, tool_result


async def synthesize_speech_base64(
    text: str, language_mode: str = "auto"
) -> tuple[str | None, str]:
    if not text or not text.strip():
        return None, "audio/mp3"

    # Determine effective language for TTS voice selection
    effective_lang = language_mode
    if effective_lang == "auto":
        effective_lang = detect_language(text)

    cache_key = f"{effective_lang}:{text.strip()}"
    if cache_key in AUDIO_RAM_CACHE:
        return AUDIO_RAM_CACHE[cache_key]

    sarvam_key = os.getenv("SARVAM_API_KEY", "").strip()
    if sarvam_key and sarvam_key != "your_optional_sarvam_api_key_here":
        try:
            async with httpx.AsyncClient(timeout=4.5) as client:
                resp = await client.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers={
                        "api-subscription-key": sarvam_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text[:2400],
                        "target_language_code": "hi-IN"
                        if effective_lang == "hinglish"
                        else "en-IN",
                        "speaker": "priya",
                        "model": "bulbul:v3",
                        "pace": 1.05,
                        "speech_sample_rate": 24000,
                    },
                )
                if resp.status_code == 200:
                    audios = resp.json().get("audios", [])
                    if audios:
                        result = (audios[0], "audio/wav")
                        _cache_audio(cache_key, result)
                        return result
        except Exception as e:  # noqa: BLE001
            logger.debug("Sarvam TTS error: %s", e)

    voice = (
        "hi-IN-SwaraNeural"
        if effective_lang == "hinglish"
        else "en-IN-NeerjaNeural"
    )
    try:
        communicate = edge_tts.Communicate(text, voice, rate="+6%")
        buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunk_data = chunk.get("data")
                if chunk_data:
                    buffer.write(chunk_data)
        audio_bytes = buffer.getvalue()
        if audio_bytes:
            b64 = base64.b64encode(audio_bytes).decode("utf-8")
            result = (b64, "audio/mp3")
            _cache_audio(cache_key, result)
            return result
    except Exception as e:  # noqa: BLE001
        logger.debug("Edge TTS error: %s", e)

    return None, "audio/mp3"


async def build_structured_summary(transcript: list[TranscriptTurn]) -> CallOutcomeJSON:
    full_text = "\n".join([f"{t.role.upper()}: {t.text}" for t in transcript])
    all_text_lower = full_text.lower()
    detected_order = extract_and_normalize_order_id(full_text)

    user_turns = [t for t in transcript if t.role == "user"]
    user_texts = [t.text for t in user_turns]
    lower_user = " ".join([t.lower() for t in user_texts])

    # Language detection across user turns
    has_hinglish = any(detect_language(t) == "hinglish" for t in user_texts)
    has_english = any(detect_language(t) == "english" for t in user_texts)
    if has_hinglish and has_english:
        lang_detected = "English / Hinglish"
    elif has_hinglish:
        lang_detected = "Hinglish"
    else:
        lang_detected = "English"

    # Products discussed extraction
    products_discussed: list[str] = []
    if any(k in all_text_lower for k in ["vitamin c", "serum", "brighten", "pigmentation"]) or detected_order == "ORD-101":
        products_discussed.append("Vitamin C Serum (30ml)")
    if any(k in all_text_lower for k in ["sunscreen", "spf", "dhoop", "sun protection"]) or detected_order == "ORD-102":
        products_discussed.append("Hydrating Sunscreen SPF 50")
    if any(k in all_text_lower for k in ["green tea", "face wash", "toner", "acne", "pimple", "oily", "pores", "combo"]) or detected_order == "ORD-103":
        products_discussed.append("Green Tea Face Wash + Toner")

    # Intent classification
    intent = "GENERAL_SUPPORT"
    status = "RESOLVED"
    policy_applied = "None"
    recommended_next_step = "Contact Aura Skincare support if further assistance is needed."
    call_summary = "Customer connected with Aria from Aura Skincare support."

    is_cancellation = any(w in lower_user for w in ["cancel", "radd", "cancellation"])
    is_return = any(w in lower_user for w in ["return", "refund", "wapas", "opened", "exchange"])
    is_tracking = detected_order is not None or any(w in lower_user for w in ["where", "track", "status", "kahan", "kab", "pahuncha", "delivery date"])
    is_product = any(
        w in lower_user
        for w in [
            "vitamin", "serum", "sunscreen", "spf", "face wash", "toner", "green tea",
            "skin", "oily", "dry", "acne", "pimple", "tan", "pore", "pores", "glow",
            "price", "cost", "kitne", "routine", "recommend", "suggestion", "product", "catalog"
        ]
    )
    is_policy = any(w in lower_user for w in ["shipping", "ship", "delivery fee", "delivery charge", "cod", "cash on delivery", "policy"])
    is_out_of_scope = any(w in lower_user for w in ["flight", "goa", "hotel", "weather", "cricket", "movie", "pizza", "bitcoin"])

    if is_out_of_scope:
        intent = "GENERAL_SUPPORT"
        status = "POLICY_DECLINED"
        policy_applied = "Support Scope Policy (Aura Skincare customer support assists exclusively with brand products, orders, and policies)"
        call_summary = "Customer asked questions outside the Aura Skincare support domain. Aria politely redirected the customer to skincare and order assistance."
        recommended_next_step = "Ask an Aura Skincare related product, routine, or order inquiry."

    elif is_cancellation:
        intent = "CANCELLATION_REQUEST"
        if detected_order == "ORD-103":
            status = "RESOLVED"
            policy_applied = "Cancellation Policy (Allowed while order status is 'Processing')"
            call_summary = f"Customer requested cancellation for order ORD-103 ({LIVE_ORDERS['ORD-103']['product']}). As the order was in Processing status, Aria successfully cancelled it."
            recommended_next_step = "No further action needed; order cancellation is confirmed."
        elif detected_order == "ORD-101":
            status = "POLICY_DECLINED"
            policy_applied = "Cancellation Policy (Orders that are 'Out for Delivery' cannot be cancelled)"
            call_summary = "Customer requested cancellation for order ORD-101. Aria declined per policy because the order is Out for Delivery with BlueDart."
            recommended_next_step = "Refuse delivery at doorstep when BlueDart arrives."
        elif detected_order == "ORD-102":
            status = "POLICY_DECLINED"
            policy_applied = "Cancellation Policy (Delivered orders cannot be cancelled)"
            call_summary = "Customer requested cancellation for order ORD-102. Aria declined per policy as the order was already delivered."
            recommended_next_step = "Check eligibility under the 7-day unopened return policy if applicable."
        else:
            status = "RESOLVED"
            policy_applied = "Cancellation Policy (Orders can only be cancelled while in Processing status)"
            call_summary = "Customer inquired about order cancellation policy. Aria explained that cancellations are permitted only while orders are Processing."
            recommended_next_step = "Share your Order ID (such as ORD-103) while it is still Processing to proceed with cancellation."

    elif is_return:
        intent = "RETURN_REQUEST"
        if detected_order == "ORD-102" or any(w in lower_user for w in ["open", "used", "14 days", "20 days"]):
            status = "POLICY_DECLINED"
            policy_applied = "Return Policy (Returns accepted only within 7 days of delivery for unopened, unused products in original packaging)"
            call_summary = "Customer requested a return for an opened product or an order delivered over 7 days ago. Aria enforced the 7-day unopened return policy and declined the request."
            recommended_next_step = "Retain product; opened items or orders delivered over 7 days ago are not eligible for return."
        else:
            status = "RESOLVED"
            policy_applied = "Return Policy (7 days unopened & unused in original packaging; 48 hours for damaged items with photos)"
            call_summary = "Customer inquired about return procedure. Aria explained Aura Skincare's 7-day unopened return policy and 48-hour damaged item reporting window."
            recommended_next_step = "Ensure item is unopened in original packaging and submit return request within 7 days of delivery."

    elif is_tracking:
        intent = "ORDER_TRACKING"
        if detected_order == "ORD-101":
            status = "RESOLVED"
            policy_applied = "Shipping & Live Courier Tracking Policy (BlueDart BD-982103)"
            call_summary = "Customer inquired about tracking order ORD-101 (Vitamin C Serum). Aria confirmed it is Out for Delivery via BlueDart and expected by 6 PM today."
            recommended_next_step = "Wait for BlueDart delivery by 6 PM today."
        elif detected_order == "ORD-102":
            status = "RESOLVED"
            policy_applied = "Courier Delivery Confirmation Policy (Delhivery DL-441029)"
            call_summary = "Customer inquired about order ORD-102 (Hydrating Sunscreen). Aria verified it was delivered 14 days ago via Delhivery."
            recommended_next_step = "Verify package delivery with household members or building reception."
        elif detected_order == "ORD-103":
            status = "RESOLVED"
            policy_applied = "Order Processing Status Policy"
            call_summary = "Customer inquired about order ORD-103 (Green Tea Combo). Aria confirmed the order is currently Processing and eligible for cancellation if needed."
            recommended_next_step = "Wait for dispatch tracking update or request cancellation while in Processing."
        else:
            status = "UNRESOLVED"
            policy_applied = "None"
            call_summary = "Customer requested tracking for an order, but no matching order ID was identified in the database."
            recommended_next_step = "Verify your order number on the order confirmation email (e.g. ORD-101, ORD-102, or ORD-103) and try again."

    elif is_product:
        intent = "PRODUCT_INQUIRY"
        status = "RESOLVED"
        policy_applied = "None"
        if any(w in lower_user for w in ["oil", "acne", "pimple", "pore", "pores"]):
            call_summary = "Customer asked for recommendations for oily and acne-prone skin. Aria recommended the Green Tea Face Wash + Toner combo (₹850) paired with lightweight Hydrating Sunscreen SPF 50 (₹499)."
            recommended_next_step = "Use Green Tea Face Wash and Toner morning and night, followed by Hydrating Sunscreen SPF 50 in the daytime."
        elif any(w in lower_user for w in ["vitamin c", "serum", "brighten", "dark spot", "pigment"]):
            call_summary = "Customer inquired about Vitamin C Serum. Aria provided pricing (₹699) and benefits for fading pigmentation and boosting collagen."
            recommended_next_step = "Apply Vitamin C Serum in the morning on clean skin before moisturizer and sunscreen."
        elif any(w in lower_user for w in ["sunscreen", "spf", "dhoop", "sun"]):
            call_summary = "Customer asked about Hydrating Sunscreen SPF 50. Aria provided pricing (₹499) and broad-spectrum UVA/UVB no-white-cast protection details."
            recommended_next_step = "Apply Hydrating Sunscreen SPF 50 generously each morning as the final step in skincare routine."
        else:
            call_summary = "Customer inquired about Aura Skincare's organic product catalog and pricing. Aria explained product options and recommended routines."
            recommended_next_step = "Browse Aura Skincare formulations and choose products matching your skin type."

    elif is_policy:
        intent = "POLICY_INQUIRY"
        status = "RESOLVED"
        policy_applied = "Shipping Policy (Free above ₹499; ₹50 below) & COD Policy (Up to ₹2,500)"
        call_summary = "Customer inquired about Aura Skincare store policies (shipping charges, delivery timelines, or Cash on Delivery). Aria provided accurate policy details."
        recommended_next_step = "Proceed with order placement; standard shipping takes 3 to 5 business days."

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key and gemini_key != "your_gemini_api_key_here" and len(transcript) > 1:
        try:
            prompt = (
                "You are an AI auditor for Aura Skincare customer support calls.\n"
                "Analyze the following support transcript and generate a structured post-call summary JSON.\n"
                "Return ONLY a valid JSON object matching this schema:\n"
                "{\n"
                '  "customer_intent": "ORDER_TRACKING | CANCELLATION_REQUEST | RETURN_REQUEST | PRODUCT_INQUIRY | POLICY_INQUIRY | GENERAL_SUPPORT",\n'
                '  "order_id": "ORD-101 | ORD-102 | ORD-103 | null",\n'
                '  "resolution_status": "RESOLVED | POLICY_DECLINED | UNRESOLVED",\n'
                '  "call_summary": "Clear, concise 1-2 sentence summary of customer inquiry and resolution",\n'
                '  "policy_applied": "Name and rule of Aura Skincare policy applied, or \'None\'",\n'
                '  "products_discussed": ["list of product names discussed, e.g. Vitamin C Serum (30ml)"],\n'
                '  "recommended_next_step": "Clear actionable next step for customer",\n'
                '  "call_metrics": {\n'
                f'    "total_turns": {len(transcript)},\n'
                f'    "language_detected": "{lang_detected}"\n'
                "  }\n"
                "}\n\n"
                f"Transcript:\n{full_text}"
            )
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    url,
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "temperature": 0.1,
                        },
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates and isinstance(candidates[0], dict):
                        content = candidates[0].get("content")
                        if isinstance(content, dict):
                            parts = content.get("parts", [])
                            if parts and isinstance(parts[0], dict) and isinstance(parts[0].get("text"), str):
                                raw_text = parts[0]["text"].strip()
                                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
                                if match:
                                    raw_text = match.group(1).strip()
                                parsed = json.loads(raw_text)
                                if isinstance(parsed, dict):
                                    parsed_status = parsed.get("resolution_status")
                                    if parsed_status not in {"RESOLVED", "POLICY_DECLINED", "UNRESOLVED"}:
                                        parsed_status = status

                                    parsed_intent = parsed.get("customer_intent")
                                    if parsed_intent not in {
                                        "ORDER_TRACKING",
                                        "CANCELLATION_REQUEST",
                                        "RETURN_REQUEST",
                                        "PRODUCT_INQUIRY",
                                        "POLICY_INQUIRY",
                                        "GENERAL_SUPPORT",
                                    }:
                                        parsed_intent = intent

                                    parsed_prods = parsed.get("products_discussed")
                                    if isinstance(parsed_prods, list):
                                        valid_prods = [str(p) for p in parsed_prods if p]
                                    else:
                                        valid_prods = products_discussed
                                    valid_prods = list(dict.fromkeys(valid_prods))

                                    cm = parsed.get("call_metrics")
                                    cm_dict = cm if isinstance(cm, dict) else {}
                                    lang_val = cm_dict.get("language_detected")
                                    final_lang = str(lang_val) if lang_val else lang_detected

                                    return CallOutcomeJSON(
                                        customer_intent=parsed_intent,
                                        order_id=parsed.get("order_id", detected_order),
                                        resolution_status=parsed_status,
                                        call_summary=parsed.get("call_summary", call_summary),
                                        policy_applied=parsed.get("policy_applied", policy_applied),
                                        products_discussed=valid_prods,
                                        recommended_next_step=parsed.get("recommended_next_step", recommended_next_step),
                                        call_metrics=CallMetrics(
                                            total_turns=len(transcript),
                                            language_detected=final_lang,
                                        ),
                                    )
        except Exception as e:  # noqa: BLE001
            logger.debug("Gemini structured summary error: %s", e)

    products_discussed = list(dict.fromkeys(products_discussed))
    return CallOutcomeJSON(
        customer_intent=intent,
        order_id=detected_order,
        resolution_status=status,
        call_summary=call_summary,
        policy_applied=policy_applied,
        products_discussed=products_discussed,
        recommended_next_step=recommended_next_step,
        call_metrics=CallMetrics(
            total_turns=len(transcript),
            language_detected=lang_detected,
        ),
    )
