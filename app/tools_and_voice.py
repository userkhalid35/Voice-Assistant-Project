import os
import re
import io
import json
import base64
import httpx
import edge_tts
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv

# Ensure project root is in sys.path when executed directly
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.models_and_data import (
    LIVE_ORDERS,
    build_system_prompt,
    TranscriptTurn,
    CallOutcomeJSON,
)

load_dotenv()

AUDIO_RAM_CACHE: Dict[str, Tuple[str, str]] = {}

DEV_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")

WORD_TO_DIGIT = {
    "zero": "0", "oh": "0", "o": "0", "shunya": "0", "शून्य": "0",
    "one": "1", "won": "1", "ek": "1", "एक": "1",
    "two": "2", "to": "2", "too": "2", "do": "2", "दो": "2",
    "three": "3", "teen": "3", "तीन": "3",
    "four": "4", "chaar": "4", "char": "4", "चार": "4",
    "five": "5", "paanch": "5", "panch": "5", "पांच": "5", "पाँच": "5",
    "six": "6", "chhe": "6", "che": "6", "छह": "6", "छः": "6",
    "seven": "7", "saat": "7", "सात": "7",
    "eight": "8", "aath": "8", "आठ": "8",
    "nine": "9", "nau": "9", "नौ": "9",
}

def extract_and_normalize_order_id(text: str) -> Optional[str]:
    # 1. Convert Devanagari numerals (e.g. १०१ -> 101)
    cleaned = text.translate(DEV_DIGIT_MAP).lower()

    # 2. Convert Devanagari keywords for Order / ORD
    cleaned = re.sub(r"\b(ओआरडी|ओ\s*आर\s*डी)\b", "ord", cleaned)
    cleaned = re.sub(r"\b(आर्डर|ऑर्डर)\b", "order", cleaned)

    # 3. Convert Hindi compound numbers like 'ek sau ek' -> '101'
    cleaned = re.sub(r"\b(?:एक\s*सौ\s*एक|ek\s*sau\s*ek)\b", "101", cleaned)
    cleaned = re.sub(r"\b(?:एक\s*सौ\s*दो|ek\s*sau\s*do)\b", "102", cleaned)
    cleaned = re.sub(r"\b(?:एक\s*सौ\s*तीन|ek\s*sau\s*teen)\b", "103", cleaned)

    for word, digit in WORD_TO_DIGIT.items():
        cleaned = re.sub(rf"\b{word}\b", digit, cleaned)

    # Collapse spaces or hyphens between digits (e.g. '1 0 1' -> '101')
    cleaned = re.sub(r"(?<=\d)[\s\-]+(?=\d)", "", cleaned)

    explicit = re.search(r"\bord[\s\-_]*(\d{3})\b", cleaned)
    if explicit:
        return f"ORD-{explicit.group(1)}"

    contextual = re.search(r"\b(?:order|id|number|status|cancel|return|track|ord)\D*(\d{3})\b", cleaned)
    if contextual:
        return f"ORD-{contextual.group(1)}"

    standalone = re.search(r"\b(10[1-9]|999)\b", cleaned)
    if standalone:
        return f"ORD-{standalone.group(1)}"

    return None

def get_order_details(order_id: str) -> Dict[str, Any]:
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

def cancel_order(order_id: str) -> Dict[str, Any]:
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

def execute_fast_tool_router(user_text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    lower = user_text.lower()
    order_id = extract_and_normalize_order_id(user_text)

    if not order_id:
        return None, None

    if any(k in lower for k in ["cancel", "radd", "hatana", "dont want", "don't want", "कैंसिल", "रद्द", "हटाना", "नहीं चाहिए"]):
        return f"cancel_order('{order_id}')", cancel_order(order_id)

    return f"get_order_details('{order_id}')", get_order_details(order_id)

def deterministic_fallback_brain(
    user_text: str,
    language_mode: str,
    tool_result: Optional[Dict[str, Any]],
) -> str:
    lower = user_text.lower()
    is_hi = language_mode == "hinglish"

    # Out-of-scope detection
    out_of_scope_terms = ["flight", "goa", "hotel", "train", "weather", "cricket", "movie", "pizza", "biryani", "stock", "bitcoin"]
    if any(term in lower for term in out_of_scope_terms):
        return (
            "Maaf kijiyega, main sirf Aura Skincare ke products, orders aur policies mein aapki madad kar sakti hoon. Kya main aapke kisi skincare order ko check karoon?"
            if is_hi
            else "I'm sorry, I can only assist you with Aura Skincare orders, products, and policies. Would you like me to check an order for you?"
        )

    # Tool-driven responses
    if tool_result is not None:
        if not tool_result.get("found"):
            bad_id = tool_result.get("requested_id", "that ID")
            return (
                f"Mujhe {bad_id} naam ka koi order nahi mila. Kya aap please apna Order ID dobara check karke bata sakte hain, jaise ORD-101, ORD-102 ya ORD-103?"
                if is_hi
                else f"I couldn't locate an order with ID {bad_id}. Could you please verify and repeat your order number?"
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
                    else f"Your order {oid} for {prod} was still Processing, so I have cancelled it for you right away."
                )
            else:
                return (
                    f"Maaf kijiyega, order {oid} abhi '{status}' status mein hai aur hum sirf Processing orders hi cancel kar sakte hain. Aap delivery ke waqt doorstep par parcel refuse kar sakte hain."
                    if is_hi
                    else f"I'm sorry, order {oid} is currently '{status}', and cancellations are only possible while an order is Processing. You may refuse the package at your doorstep."
                )

        if any(w in lower for w in ["return", "refund", "wapas", "replace", "exchange", "वापस", "रिटर्न"]):
            if oid == "ORD-102" or (order.get("days_since_delivery") and order["days_since_delivery"] > 7):
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

    # Policy questions without a specific order ID
    if any(w in lower for w in ["return", "refund", "opened", "khola", "20 days", "14 days", "damage", "defective", "वापस", "रिटर्न", "खोला"]):
        if any(w in lower for w in ["open", "used", "khola", "20", "14", "10", "खोला", "इस्तेमाल"]):
            return (
                "Maaf kijiyega, Aura Skincare ki policy ke mutabik hum sirf delivery ke 7 din ke andar unopened aur unused products hi return le sakte hain. Opened ya 7 din se purane products return nahi ho sakte."
                if is_hi
                else "I'm sorry, Aura Skincare only accepts returns within 7 days of delivery for unopened, unused products in their original packaging. Opened items or orders past 7 days cannot be returned."
            )
        if any(w in lower for w in ["damage", "broken", "leak", "defective", "kharab", "खराब", "टूटा"]):
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

    if any(w in lower for w in ["ship", "delivery charge", "kitne din", "fee", "डिलीवरी", "शिपिंग"]):
        return (
            "₹499 se upar ke sabhi orders par shipping bilkul free hai, aur ₹499 se kam par ₹50 shipping charge lagta hai. Standard delivery mein 3 se 5 business days lagte hain."
            if is_hi
            else "We offer free shipping on orders above ₹499, and a ₹50 shipping fee on orders below ₹499. Standard delivery takes 3 to 5 business days."
        )

    if any(w in lower for w in ["cod", "cash on delivery", "upi", "pay", "कैश ऑन डिलीवरी", "पेमेंट"]):
        return (
            "Haan ji, ₹2,500 tak ke orders par Cash on Delivery available hai, aur aap doorstep par cash ya UPI dono se payment kar sakte hain."
            if is_hi
            else "Yes, Cash on Delivery is available for orders up to ₹2,500, and you can pay by cash or UPI right at your doorstep."
        )

    if any(w in lower for w in ["cancel", "radd", "कैंसिल", "रद्द"]):
        return (
            "Orders sirf tabhi cancel ho sakte hain jab unka status Processing ho. Shipped ya Out for Delivery hone ke baad cancel nahi hota, par aap doorstep par delivery refuse kar sakte hain. Apna Order ID batayein toh main check karoon?"
            if is_hi
            else "Orders can only be cancelled while in Processing status. Once Shipped or Out for Delivery, they cannot be cancelled, though you may refuse doorstep delivery. Please share your Order ID so I can check."
        )

    # Product & catalog inquiries without order ID
    if any(w in lower for w in ["vitamin c", "serum", "sunscreen", "spf", "green tea", "face wash", "toner", "product", "catalog", "price", "kitne ka", "cost", "फायदे", "कीमत", "सीरम", "सनस्क्रीन"]):
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
        if any(w in lower for w in ["green tea", "face wash", "toner", "मुंहासे", "पिंपल", "acne", "oil"]):
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
        "Namaste! Main Aria hoon Aura Skincare se. Main aapke order status, cancellations, ya return aur shipping policies mein madad kar sakti hoon. Aap apna Order ID jaise ORD-101 bata sakte hain."
        if is_hi
        else "Hello! I'm Aria from Aura Skincare. I can help you track an order, cancel a processing order, or answer questions about shipping and returns. Please share your Order ID or question."
    )

async def generate_agent_reply(
    user_text: str,
    history: List[TranscriptTurn],
    language_mode: str = "hinglish",
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    tool_used, tool_result = execute_fast_tool_router(user_text)
    system_prompt = build_system_prompt(language_mode)

    tool_context = ""
    if tool_result is not None:
        tool_context = (
            f"\n\n[LIVE DATABASE TOOL RESULT for {tool_used}]:\n"
            f"{json.dumps(tool_result, ensure_ascii=False)}\n"
            "Use this exact live tool data to answer the customer accurately and concisely."
        )

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    # 1. Try Gemini 2.5 Flash via pure REST
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            contents = []
            for turn in history[-6:]:
                role = "user" if turn.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": turn.text}]})
            contents.append({"role": "user", "parts": [{"text": user_text + tool_context}]})

            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
            payload = {
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.25,
                    "maxOutputTokens": 140,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            }
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(
                    url,
                    headers={"Content-Type": "application/json", "x-goog-api-key": gemini_key},
                    json=payload,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    clean = re.sub(r"[\*\#\`]", "", text)
                    return clean, tool_used, tool_result
        except Exception:
            pass

    # 2. Try Groq Llama 3.3 70B via pure REST
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
                    json={"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.25, "max_tokens": 130},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    clean = re.sub(r"[\*\#\`]", "", text)
                    return clean, tool_used, tool_result
        except Exception:
            pass

    # 3. Zero-Failure Deterministic Fallback Brain
    fallback_reply = deterministic_fallback_brain(user_text, language_mode, tool_result)
    return fallback_reply, tool_used, tool_result

async def synthesize_speech_base64(text: str, language_mode: str = "hinglish") -> Tuple[Optional[str], str]:
    cache_key = f"{language_mode}:{text.strip()}"
    if cache_key in AUDIO_RAM_CACHE:
        return AUDIO_RAM_CACHE[cache_key]

    sarvam_key = os.getenv("SARVAM_API_KEY", "").strip()
    if sarvam_key and sarvam_key != "your_optional_sarvam_api_key_here":
        try:
            async with httpx.AsyncClient(timeout=4.5) as client:
                resp = await client.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers={"api-subscription-key": sarvam_key, "Content-Type": "application/json"},
                    json={
                        "text": text[:2400],
                        "target_language_code": "hi-IN" if language_mode == "hinglish" else "en-IN",
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
        except Exception:
            pass

    # Free Zero-Key Edge Neural TTS in RAM (hi-IN-SwaraNeural handles Hinglish smoothly)
    voice = "hi-IN-SwaraNeural" if language_mode == "hinglish" else "en-IN-NeerjaNeural"
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
    except Exception:
        pass

    return None, "audio/mp3"

async def build_structured_summary(transcript: List[TranscriptTurn]) -> CallOutcomeJSON:
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
        if any(w in lower_user for w in ["open", "20", "14"]) or detected_order == "ORD-102":
            status = "REJECTED_BY_POLICY"
    elif detected_order or any(w in lower_user for w in ["where", "track", "status", "kahan", "kab"]):
        intent = "ORDER_TRACKING"
    elif any(w in lower_user for w in ["shipping", "cod", "policy", "delivery"]):
        intent = "POLICY_INQUIRY"

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key and gemini_key != "your_gemini_api_key_here" and len(transcript) > 1:
        try:
            prompt = (
                "Analyze this Aura Skincare customer support transcript and return ONLY a raw JSON object with keys: "
                "customer_intent (one of ORDER_TRACKING, CANCELLATION_REQUEST, RETURN_REQUEST, POLICY_INQUIRY, OUT_OF_SCOPE), "
                "order_id (string like 'ORD-101' or null), "
                "resolution_status (one of RESOLVED, REJECTED_BY_POLICY, UNRESOLVED), "
                "call_summary (concise 1-2 sentence English summary of the call outcome).\n\n"
                f"Transcript:\n{full_text}"
            )
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
            async with httpx.AsyncClient(timeout=7.0) as client:
                resp = await client.post(
                    url,
                    headers={"Content-Type": "application/json", "x-goog-api-key": gemini_key},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "temperature": 0.1,
                            "thinkingConfig": {"thinkingBudget": 0},
                        },
                    },
                )
                if resp.status_code == 200:
                    raw_json = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = json.loads(raw_json)
                    return CallOutcomeJSON(
                        customer_intent=parsed.get("customer_intent", intent),
                        order_id=parsed.get("order_id", detected_order),
                        resolution_status=parsed.get("resolution_status", status),
                        call_summary=parsed.get("call_summary", ""),
                    )
        except Exception:
            pass

    # Deterministic Summary Fallback
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
