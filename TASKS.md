# TASKMASTER & RALPH LOOP EXECUTION CHECKLIST

- [x] **Task 1: Verify `uv` Environment & Create Directory Skeleton**
  - Ensure `pyproject.toml` has `fastapi`, `uvicorn`, `httpx`, `pydantic`, `python-dotenv`, and `edge-tts`.
  - Create folders `app/` and `static/`, plus `.env.example` and `app/__init__.py` from `SPEC.md`.
  - **Verification Gate:** Run `uv run python -c "import fastapi, httpx, pydantic, edge_tts; print('DEPS_OK')"`

- [x] **Task 2: Build Data Models, Mock Order Database & Brand Policies (`app/models_and_data.py`)**
  - Write `app/models_and_data.py` exactly as specified in `SPEC.md` with `ORD-101`, `ORD-102`, `ORD-103`, Pydantic schemas, and the Hinglish/English system prompt builder.
  - **Verification Gate:** Run `uv run python -c "from app.models_and_data import LIVE_ORDERS, build_system_prompt; assert len(LIVE_ORDERS) == 3; print('MODELS_OK')"`

- [x] **Task 3: Build 0.2ms Tool Router, LLM Brain & Hinglish TTS Engine (`app/tools_and_voice.py`)**
  - Write `app/tools_and_voice.py` from `SPEC.md` implementing `get_order_details`, `cancel_order`, `generate_agent_reply`, `synthesize_speech_base64`, and `build_structured_summary`.
  - **Verification Gate:** Run `uv run python -c "from app.tools_and_voice import get_order_details, cancel_order, extract_and_normalize_order_id; assert extract_and_normalize_order_id('order one zero one') == 'ORD-101'; assert get_order_details('ORD-101')['found'] == True; print('TOOLS_OK')"`

- [x] **Task 4: Build FastAPI Backend Server (`app/main.py`) & Editorial Skincare UI (`static/index.html`)**
  - Write `app/main.py` and `static/index.html` exactly as defined in `SPEC.md`.
  - **Verification Gate:** Run `uv run python -m py_compile app/main.py app/models_and_data.py app/tools_and_voice.py`

- [x] **Task 5: End-to-End Automated API & Policy Test**
  - Run an inline `uv run python` test using `fastapi.testclient` or async functions to verify:
    1. `ORD-101` lookup returns `Out for Delivery` and expected by 6 PM today.
    2. `ORD-103` cancellation succeeds and changes status to `Cancelled`.
    3. `ORD-102` return inquiry enforces the 7-day unopened policy.
    4. `/api/summary` returns valid JSON with `customer_intent`, `order_id`, `resolution_status`, and `call_summary`.

- [x] **Task 6: Create `Dockerfile` and `README.md` (with Section 9 Engineering Answers)**
  - Create `Dockerfile` (using Astral `uv`) and `README.md` answering all 4 questions from Section 9.