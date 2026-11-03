import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch("dotenv.load_dotenv"), patch.dict(
    "os.environ", {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}
):
    from . import gtm_agent


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})

    def send_email(self, prospect):
        return gtm_agent.send_prospect_email.func(
            prospect=prospect,
            subject="Demo invitation",
            body="Would you like a demo?",
            runtime=self.runtime,
        )

    def test_disqualified_prospect_is_blocked(self):
        prospect = gtm_agent.get_prospect.invoke({"prospect_id": "LEAD-50001"})["prospect"]
        for disqualified in (True, False, None):
            with self.subTest(disqualified=disqualified):
                supplied = {**prospect, "disqualified": disqualified}
                if disqualified is None:
                    supplied.pop("disqualified")
                self.assertEqual(
                    self.send_email(supplied),
                    {
                        "status": "blocked",
                        "error": "Prospect is disqualified; email not sent.",
                    },
                )

    def test_qualified_prospect_still_receives_email(self):
        prospect = gtm_agent.get_prospect.invoke({"prospect_id": "LEAD-70001"})["prospect"]
        result = self.send_email(prospect)
        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual(result["to"], prospect["email"])
        self.assertEqual(result["to_name"], prospect["name"])
        self.assertEqual(result["from"], "ava.mills@northpoint.com")
        self.assertEqual(result["from_name"], "Ava Mills")
        self.assertEqual(result["subject"], "Demo invitation")
        self.assertEqual(result["body"], "Would you like a demo?")

    def test_missing_email_still_fails(self):
        self.assertEqual(
            self.send_email({"prospect_id": "LEAD-70001"}),
            {"status": "failed", "error": "Prospect record has no email address."},
        )


if __name__ == "__main__":
    unittest.main()
