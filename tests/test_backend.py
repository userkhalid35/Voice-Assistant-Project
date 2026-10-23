import unittest
from fastapi.testclient import TestClient

from app.main import app
from app.models_and_data import LIVE_ORDERS, reset_orders, VoiceTurnRequest, VoiceTurnResponse

class TestAriaBackend(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        reset_orders()

    def test_01_health_check(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "ok")

    def test_02_get_orders_and_reset(self):
        res = self.client.get("/api/orders")
        self.assertEqual(res.status_code, 200)
        orders = res.json().get("orders", [])
        self.assertGreaterEqual(len(orders), 3)

        res_reset = self.client.post("/api/reset")
        self.assertEqual(res_reset.status_code, 200)
        reset_orders_list = res_reset.json().get("orders", [])
        self.assertGreaterEqual(len(reset_orders_list), 3)

    def test_03_greeting_endpoint(self):
        res_hi = self.client.get("/api/greeting?language_mode=hinglish")
        self.assertEqual(res_hi.status_code, 200)
        data_hi = res_hi.json()
        self.assertIn("Aria", data_hi.get("reply_text", ""))
        self.assertTrue(len(data_hi.get("audio_base64", "")) > 100)

        res_en = self.client.get("/api/greeting?language_mode=english")
        self.assertEqual(res_en.status_code, 200)
        data_en = res_en.json()
        self.assertIn("Aria", data_en.get("reply_text", ""))
        self.assertTrue(len(data_en.get("audio_base64", "")) > 100)

    def test_04_order_tracking_lookup(self):
        payload = {
            "user_text": "Mera order ORD-101 kahan pahuncha hai?",
            "history": [],
            "language_mode": "hinglish"
        }
        res = self.client.post("/api/voice-turn", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("get_order_details", data.get("tool_used", ""))
        self.assertIn("ORD-101", data.get("reply_text", ""))
        self.assertIn("BlueDart", data.get("reply_text", ""))
        self.assertTrue(len(data.get("audio_base64", "")) > 100)

    def test_05_order_cancellation_allowed(self):
        payload = {
            "user_text": "Please cancel my order ORD-103 immediately",
            "history": [],
            "language_mode": "english"
        }
        res = self.client.post("/api/voice-turn", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("cancel_order", data.get("tool_used", ""))
        self.assertIn("cancel", data.get("reply_text", "").lower())
        # Check order status changed in returned orders table
        ord_103 = next((o for o in data.get("orders", []) if o["order_id"] == "ORD-103"), None)
        self.assertIsNotNone(ord_103)
        self.assertEqual(ord_103["status"], "Cancelled")

    def test_06_order_cancellation_policy_block(self):
        payload = {
            "user_text": "Cancel my order ORD-101",
            "history": [],
            "language_mode": "english"
        }
        res = self.client.post("/api/voice-turn", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        # Out for delivery cannot be cancelled
        reply_lower = data.get("reply_text", "").lower()
        self.assertTrue("out for delivery" in reply_lower or "processing" in reply_lower or "cannot" in reply_lower)
        # Verify status remained Out for Delivery
        ord_101 = next((o for o in data.get("orders", []) if o["order_id"] == "ORD-101"), None)
        self.assertEqual(ord_101["status"], "Out for Delivery")

    def test_07_product_pricing_knowledge(self):
        payload = {
            "user_text": "What is the price of your Vitamin C serum?",
            "history": [],
            "language_mode": "english"
        }
        res = self.client.post("/api/voice-turn", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("699", data.get("reply_text", ""))

    def test_08_policy_opened_return_block(self):
        payload = {
            "user_text": "I opened ORD-102 14 days ago and used it. Can I return it for a refund?",
            "history": [],
            "language_mode": "english"
        }
        res = self.client.post("/api/voice-turn", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        reply_lower = data.get("reply_text", "").lower()
        self.assertTrue("unopened" in reply_lower or "7 days" in reply_lower or "cannot" in reply_lower or "policy" in reply_lower)

    def test_09_three_consecutive_questions_session(self):
        history = []
        turns = [
            ("Hi, do you have sunscreen?", "english"),
            ("How much does it cost?", "english"),
            ("Where is ORD-101?", "english"),
        ]
        for user_q, lang in turns:
            payload = {
                "user_text": user_q,
                "history": history,
                "language_mode": lang
            }
            res = self.client.post("/api/voice-turn", json=payload)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            reply = data.get("reply_text", "")
            self.assertTrue(len(reply) > 0)
            history.append({"role": "user", "text": user_q})
            history.append({"role": "assistant", "text": reply})

        self.assertEqual(len(history), 6)

        # Generate summary
        res_sum = self.client.post("/api/summary", json={"transcript": history})
        self.assertEqual(res_sum.status_code, 200)
        outcome = res_sum.json()
        self.assertIn("customer_intent", outcome)
        self.assertIn("resolution_status", outcome)

if __name__ == "__main__":
    unittest.main()
