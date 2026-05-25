import unittest

from fastapi.testclient import TestClient

from backend.app.main import app


class ApiSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

    def test_updated_seasons_and_predictions(self):
        seasons = self.client.get("/seasons").json()
        self.assertIn(2026, seasons)

        predictions = self.client.get("/seasons/2026/predictions").json()
        self.assertGreater(len(predictions), 0)
        self.assertIn("prediction", predictions[0])

    def test_qa_passes(self):
        report = self.client.get("/qa").json()
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["missing_team_rows"], 0)
        self.assertEqual(report["duplicate_player_years"], 0)

    def test_explanation(self):
        response = self.client.get("/players/Nikola%20Joki%C4%87/seasons/2026/explanation")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreater(len(payload["positive_factors"]), 0)


if __name__ == "__main__":
    unittest.main()

