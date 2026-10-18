import unittest
import json
from app import app

class TestResearchIntegration(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_01_system_status_and_stats(self):
        """Verify /api/status returns active model, scanner, and database stats."""
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "online")
        self.assertIn("model", data)
        self.assertIn("scanner", data)
        self.assertIn("research_stats", data)
        
        # Verify model metadata
        model = data["model"]
        self.assertIn("EXP-06C", model["name"])
        self.assertEqual(model["folds_loaded"], 5)
        self.assertEqual(model["canonical_classes"], ["A+", "AB+", "B+", "O+"])
        
        # Verify research stats
        stats = data["research_stats"]
        self.assertIn("total_sessions", stats)
        self.assertIn("valid_captures", stats)

    def test_02_registration_validation(self):
        """Verify participant registration enforces consent and pseudonymous identity."""
        # Case A: Missing consent
        res_no_consent = self.client.post("/api/research/session", json={
            "participant_id": "P-TEST-999",
            "consent_given": False
        })
        self.assertEqual(res_no_consent.status_code, 400)
        data = res_no_consent.get_json()
        self.assertIn("consent is mandatory", data.get("error", "").lower())

        # Case B: Valid registration with self-reported label
        res_valid = self.client.post("/api/research/session", json={
            "consent_given": True,
            "self_reported_group": "O+",
            "label_source": "self_reported",
            "notes": "Unit test session"
        })
        self.assertEqual(res_valid.status_code, 201)
        valid_data = res_valid.get_json()
        self.assertTrue(valid_data["success"])
        session = valid_data["session"]
        self.assertTrue(session["participant_id"].startswith("P-"))
        self.assertEqual(session["self_reported_group"], "O+")
        self.assertEqual(session["verification_status"], "unverified")
        self.assertEqual(session["status"], "CREATED")

    def test_03_three_capture_workflow_and_inference(self):
        """Test full left-thumb 3-capture protocol, quality analysis, and multi-scan inference."""
        # 1. Create a fresh session
        reg_res = self.client.post("/api/research/session", json={
            "consent_given": True,
            "notes": "3-scan integration test"
        })
        self.assertEqual(reg_res.status_code, 201)
        session_id = reg_res.get_json()["session"]["session_id"]
        participant_id = reg_res.get_json()["session"]["participant_id"]

        # 2. Capture impression 1
        cap1_res = self.client.post("/api/research/capture", json={
            "session_id": session_id,
            "finger": "L_THUMB",
            "impression_number": 1
        })
        self.assertEqual(cap1_res.status_code, 200)
        c1_data = cap1_res.get_json()
        self.assertTrue(c1_data["success"])
        cap1 = c1_data["capture"]
        self.assertEqual(cap1["impression_number"], 1)
        self.assertEqual(cap1["finger"], "L_THUMB")
        self.assertIn("quality_analysis", cap1)
        self.assertIn("fg_coverage", cap1["quality_analysis"])
        self.assertIn("ridge_coherence", cap1["quality_analysis"])
        self.assertIn("ridge_contrast", cap1["quality_analysis"])

        # 3. Attempt inference prematurely (only 1 of 3 impressions) -> MUST fail
        early_inf = self.client.post("/api/research/infer", json={
            "session_id": session_id
        })
        self.assertEqual(early_inf.status_code, 400)
        self.assertIn("requires exactly 3", early_inf.get_json().get("error", ""))

        # 4. Capture impression 2
        cap2_res = self.client.post("/api/research/capture", json={
            "session_id": session_id,
            "finger": "L_THUMB",
            "impression_number": 2
        })
        self.assertEqual(cap2_res.status_code, 200)

        # 5. Capture impression 3
        cap3_res = self.client.post("/api/research/capture", json={
            "session_id": session_id,
            "finger": "L_THUMB",
            "impression_number": 3
        })
        self.assertEqual(cap3_res.status_code, 200)

        # 6. Attempt 4th capture -> MUST be rejected
        cap4_res = self.client.post("/api/research/capture", json={
            "session_id": session_id,
            "finger": "L_THUMB",
            "impression_number": 4
        })
        self.assertEqual(cap4_res.status_code, 400)
        self.assertIn("Invalid 'impression_number'", cap4_res.get_json().get("error", ""))

        # 7. Execute 3-scan feature fusion inference
        inf_res = self.client.post("/api/research/infer", json={
            "session_id": session_id
        })
        self.assertEqual(inf_res.status_code, 200)
        inf_data = inf_res.get_json()
        self.assertTrue(inf_data["success"])
        
        # Verify canonical classes: A+, AB+, B+, O+
        pred_group = inf_data["predicted_group"]
        self.assertIn(pred_group, ["A+", "AB+", "B+", "O+"])
        
        probs = inf_data["probabilities"]
        self.assertEqual(set(probs.keys()), {"A+", "AB+", "B+", "O+"})
        total_prob = sum(probs.values())
        self.assertAlmostEqual(total_prob, 1.0, places=3)

        # Verify mandatory clinical safety disclaimer
        self.assertIn("RESEARCH PROTOTYPE", inf_data.get("disclaimer", ""))
        self.assertIn("NOT FOR MEDICAL USE", inf_data.get("disclaimer", ""))

        # 8. Verify session retrieval
        get_sess = self.client.get(f"/api/research/session/{session_id}")
        self.assertEqual(get_sess.status_code, 200)
        s_data = get_sess.get_json()["session"]
        self.assertEqual(s_data["status"], "COMPLETED")
        self.assertEqual(s_data["predicted_group"], pred_group)
        self.assertEqual(len(s_data["captures"]), 3)

    def test_04_session_history_list(self):
        """Verify session history retrieval endpoint."""
        res = self.client.get("/api/research/sessions")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIsInstance(data["sessions"], list)

if __name__ == "__main__":
    unittest.main()
