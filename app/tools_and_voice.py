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
    CallOutcomeJSON,
    TranscriptTurn,
    build_system_prompt,
)

load_dotenv()

AUDIO_RAM_CACHE: dict[str, tuple[str, str]] = {}

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
    cleaned = re.sub(r"\bone\s+hundred\s+(?:and\s+)?one\b", "101", cleaned)
    cleaned = re.sub(r"\bone\s+hundred\s+(?:and\s+)?two\b", "102", cleaned)
    cleaned = re.sub(r"\bone\s+hundred\s+(?:and\s+)?three\b", "103", cleaned)

    for word, digit in WORD_TO_DIGIT.items():
        cleaned = re.sub(rf"\b{word}\b", digit, cleaned)

    cleaned = re.sub(r"(?<=\d)[\s\-_]+(?=\d)", "", cleaned)

    explicit = re.search(r"\bord[\s\-_]*(\d{3})\b", cleaned)
    if explicit:
        return f"ORD-{explicit.group(1)}"

    contextual = re.search(
        r"\b(?:order|id|number|status|cancel|return|track|mera)\D*(\d{3})\b", cleaned
    )
    if contextual:
        return f"ORD-{contextual.group(1)}"

    standalone = re.search(r"\b(10[1-9]|999)\b", cleaned)
    if standalone:
        return f"ORD-{standalone.group(1)}"

    return None


def get_order_details(order_id: str) -> dict[str, Any]:
    norm_id = order_id.upper().strip()
    if not norm_id.startswith("ORD-"):
        digits = re.sub(r"\D", "", norm_id)
        norm_id = f"ORD-{digits}" if digits else norm_id

    order = LIVE_ORDERS.get(norm_id)
    if not order:
        return {
            "found": False,
            "requested_id": norm_id,
            "error": f"Order {norm_id} was not found in Aura Skincare's database. Valid sample IDs are ORD-101, ORD-102, ORD-103.",
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
    if any(
        k in query
        for k in [
            "sunscreen",
            "spf",
            "sun",
            "dhoop",
            "tan",
            "protect",
            "hydration",
            "moistur",
            "सनस्क्रीन",
        ]
    ):
        matched.append(PRODUCTS_CATALOG["hydrating-sunscreen"])

    # 3. Green Tea Face Wash + Toner
    if any(
        k in query
        for k in [
            "green tea",
            "face wash",
            "wash",
            "toner",
            "acne",
            "pimple",
            "pore",
            "pores",
            "cleanse",
            "cleanser",
            "muhase",
            "मुंहासे",
            "पिंपल",
        ]
    ):
        matched.append(PRODUCTS_CATALOG["green-tea-combo"])

    # Skin type heuristics
    if "oily" in query or "greasy" in query or "oil" in query or "tel" in query:
        if PRODUCTS_CATALOG["green-tea-combo"] not in matched:
            matched.append(PRODUCTS_CATALOG["green-tea-combo"])
        if PRODUCTS_CATALOG["hydrating-sunscreen"] not in matched:
            matched.append(PRODUCTS_CATALOG["hydrating-sunscreen"])

    if "dry" in query or "sukhi" in query or "dehydrat" in query:
        if PRODUCTS_CATALOG["hydrating-sunscreen"] not in matched:
            matched.append(PRODUCTS_CATALOG["hydrating-sunscreen"])
        if PRODUCTS_CATALOG["vitamin-c-serum"] not in matched:
            matched.append(PRODUCTS_CATALOG["vitamin-c-serum"])

    if "combination" in query or "sensitive" in query:
        if PRODUCTS_CATALOG["green-tea-combo"] not in matched:
            matched.append(PRODUCTS_CATALOG["green-tea-combo"])
        if PRODUCTS_CATALOG["hydrating-sunscreen"] not in matched:
            matched.append(PRODUCTS_CATALOG["hydrating-sunscreen"])

    # General / Catalog / Fallback
    if not matched or any(
        k in query
        for k in [
            "catalog",
            "all",
            "product",
            "routine",
            "best",
            "recommend",
            "sab",
            "kaunsa",
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
    if any(k in lower for k in skincare_keywords):
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
            if text and text.strip():
                return text.strip()

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
        wav_bytes = base64.b64decode(audio_base64)
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
                    text = resp.json().get("text", "").strip()
                    if text:
                        return text
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
                        if parts:
                            t_text = parts[0].get("text", "").strip()
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
        "code",
        "python",
        "math",
    ]
    if any(term in lower for term in out_of_scope_terms):
        return (
            "Maaf kijiyega, main sirf Aura Skincare ke products, orders aur policies mein aapki madad kar sakti hoon. Kya main aapke kisi order ya skincare query mein help karoon?"
            if is_hi
            else "I'm sorry, I can only assist you with Aura Skincare orders, products, and policies. Would you like me to check an order or recommend a product?"
        )

    # 1. Product recommendations tool result
    if tool_result is not None and "products" in tool_result:
        matched_prods = tool_result.get("products", [])
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

    # 2. Order details / cancellation tool result
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
        ) and (
            oid == "ORD-102"
            or (order.get("days_since_delivery") and order["days_since_delivery"] > 7)
        ):
            return (
                f"Aapka order {oid} ({prod}) 14 din pehle deliver hua tha. Hamari policy ke mutabik returns sirf delivery ke 7 din ke andar unopened products par hi valid hain, isliye yeh return nahi ho payega."
                if is_hi
                else f"Order {oid} for {prod} was delivered 14 days ago. Since our return policy only covers unopened items within 7 days of delivery, this order is not eligible for a return."
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

    # 3. Policy questions
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
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
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
                    text = data["choices"][0]["message"]["content"].strip()
                    clean = re.sub(r"[\*\#\`]", "", text)
                    return clean, tool_used, tool_result
        except Exception as e:  # noqa: BLE001
            logger.debug("Groq LLM reply error: %s", e)

    fallback_reply = deterministic_fallback_brain(user_text, effective_lang, tool_result)
    return fallback_reply, tool_used, tool_result


async def synthesize_speech_base64(
    text: str, language_mode: str = "auto"
) -> tuple[str | None, str]:
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
                        AUDIO_RAM_CACHE[cache_key] = result
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
                buffer.write(chunk["data"])
        audio_bytes = buffer.getvalue()
        if audio_bytes:
            b64 = base64.b64encode(audio_bytes).decode("utf-8")
            result = (b64, "audio/mp3")
            AUDIO_RAM_CACHE[cache_key] = result
            return result
    except Exception as e:  # noqa: BLE001
        logger.debug("Edge TTS error: %s", e)

    return None, "audio/mp3"


async def build_structured_summary(transcript: list[TranscriptTurn]) -> CallOutcomeJSON:
    full_text = "\n".join([f"{t.role.upper()}: {t.text}" for t in transcript])
    detected_order = extract_and_normalize_order_id(full_text)

    lower_user = " ".join([t.text.lower() for t in transcript if t.role == "user"])
    intent = "GENERAL_INQUIRY"
    status = "RESOLVED"

    if any(w in lower_user for w in ["flight", "goa", "hotel", "weather", "cricket"]):
        intent = "OUT_OF_SCOPE"
        status = "REJECTED_BY_POLICY"
    elif any(w in lower_user for w in ["cancel", "radd"]):
        intent = "CANCELLATION_REQUEST"
        if detected_order == "ORD-101" or detected_order == "ORD-102":
            status = "REJECTED_BY_POLICY"
    elif any(w in lower_user for w in ["return", "refund", "wapas", "opened"]):
        intent = "RETURN_REQUEST"
        if (
            any(w in lower_user for w in ["open", "20", "14"])
            or detected_order == "ORD-102"
        ):
            status = "REJECTED_BY_POLICY"
    elif detected_order or any(
        w in lower_user for w in ["where", "track", "status", "kahan", "kab"]
    ):
        intent = "ORDER_TRACKING"
    elif any(
        w in lower_user
        for w in [
            "oily",
            "acne",
            "pimple",
            "dry",
            "sensitive",
            "routine",
            "tan",
            "pore",
            "pores",
            "kaunsa",
        ]
    ):
        intent = "SKINCARE_ADVICE"
    elif any(
        w in lower_user
        for w in [
            "vitamin c",
            "serum",
            "sunscreen",
            "spf",
            "face wash",
            "toner",
            "green tea",
            "price",
            "catalog",
            "cost",
            "kitne ka",
            "products",
        ]
    ):
        intent = "PRODUCT_INQUIRY"
    elif any(w in lower_user for w in ["shipping", "cod", "policy", "delivery"]):
        intent = "POLICY_INQUIRY"

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key and gemini_key != "your_gemini_api_key_here" and len(transcript) > 1:
        try:
            prompt = (
                "Analyze this Aura Skincare customer support transcript and return ONLY a raw JSON object with keys: "
                "customer_intent (one of ORDER_TRACKING, CANCELLATION_REQUEST, RETURN_REQUEST, POLICY_INQUIRY, PRODUCT_INQUIRY, SKINCARE_ADVICE, OUT_OF_SCOPE), "
                "order_id (string like 'ORD-101' or null), "
                "resolution_status (one of RESOLVED, REJECTED_BY_POLICY, UNRESOLVED), "
                "call_summary (concise 1-2 sentence English summary of the call outcome).\n\n"
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
                    raw_json = resp.json()["candidates"][0]["content"]["parts"][0][
                        "text"
                    ]
                    parsed = json.loads(raw_json)
                    return CallOutcomeJSON(
                        customer_intent=parsed.get("customer_intent", intent),
                        order_id=parsed.get("order_id", detected_order),
                        resolution_status=parsed.get("resolution_status", status),
                        call_summary=parsed.get("call_summary", ""),
                    )
        except Exception as e:  # noqa: BLE001
            logger.debug("Gemini structured summary error: %s", e)

    summary_text = "Customer connected with Aria from Aura Skincare support."
    if intent == "ORDER_TRACKING" and detected_order in LIVE_ORDERS:
        o = LIVE_ORDERS[detected_order]
        summary_text = f"Customer inquired about {detected_order} ({o['product']}). Informed that status is '{o['status']}' ({o['notes']})."
    elif intent == "CANCELLATION_REQUEST" and detected_order in LIVE_ORDERS:
        o = LIVE_ORDERS[detected_order]
        if o["status"] == "Cancelled":
            summary_text = f"Customer requested cancellation for {detected_order} ({o['product']}). Order was Processing and successfully cancelled."
        else:
            summary_text = f"Customer requested cancellation for {detected_order}. Politely declined as order status is '{o['status']}' and advised doorstep refusal."
    elif intent == "RETURN_REQUEST":
        summary_text = "Customer requested a product return. Agent enforced Aura Skincare's 7-day unopened return policy."
    elif intent == "PRODUCT_INQUIRY":
        summary_text = "Customer inquired about Aura Skincare product catalog and pricing. Agent provided accurate product details."
    elif intent == "SKINCARE_ADVICE":
        summary_text = "Customer asked for personalized skincare advice and routine recommendations. Agent provided tailored product guidance based on skin type."
    elif intent == "OUT_OF_SCOPE":
        summary_text = "Customer made an out-of-scope request unrelated to Aura Skincare. Agent politely redirected to Aura Skincare support topics."
    elif intent == "POLICY_INQUIRY":
        summary_text = "Customer asked about Aura Skincare's shipping, COD, and return policies. Agent provided accurate policy details."

    return CallOutcomeJSON(
        customer_intent=intent,
        order_id=detected_order,
        resolution_status=status,
        call_summary=summary_text,
    )
