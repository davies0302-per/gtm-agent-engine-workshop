import importlib
import json
import unittest
from unittest.mock import Mock, patch

with patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}), patch(
    "dotenv.load_dotenv"
), patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")

from gtm_agent import data_service


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "TEST-LEAD"
        self.record = {
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "annual_revenue": 1000000,
            "tech_stack": ["Python"],
            "account_details": {"industry": "Software"},
            "engagement_history": [],
            "billing_qualification": {"tax_id": "private-test-value"},
        }
        self.offering = {
            "description": "Test offering",
            "required_tech_stack": ["Python"],
            "min_annual_revenue": 100000,
        }
        self.enterContext(patch.dict(data_service.PROSPECTS, {self.prospect_id: self.record}, clear=True))
        self.enterContext(patch.dict(data_service._PROFILES, {}, clear=True))
        self.enterContext(patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}))
        self.scoring = self.enterContext(patch.object(agent, "_scoring_llm"))
        self.scoring.invoke.return_value = Mock(model_dump=lambda: {"score": 80})

    def assert_no_billing(self, value):
        if isinstance(value, dict):
            self.assertNotIn("billing_qualification", value)
            for nested in value.values():
                self.assert_no_billing(nested)
        elif isinstance(value, list):
            for nested in value:
                self.assert_no_billing(nested)

    def test_get_prospect_retains_contact_fields(self):
        result = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(result, {"found": True, "prospect": {
            "prospect_id": self.prospect_id,
            "name": self.record["name"],
            "email": self.record["email"],
            "annual_revenue": self.record["annual_revenue"],
        }})
        self.assert_no_billing(result)

    def test_build_profile_filters_before_saving(self):
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        expected = {k: v for k, v in self.record.items() if k != "billing_qualification"}
        expected["prospect_id"] = self.prospect_id
        self.assertEqual(result, {"found": True, "prospect_profile": expected})
        self.assertEqual(data_service._PROFILES[self.prospect_id], expected)
        self.assert_no_billing(result)

    def test_score_allows_only_scoring_fields(self):
        for has_id in (True, False):
            with self.subTest(has_id=has_id):
                profile = {**self.record, "unexpected_field": "private-test-value"}
                if has_id:
                    profile.update(prospect_id=self.prospect_id, tech_stack=["stale"])
                original = dict(profile)
                result = agent.score_prospect.invoke({
                    "prospect_profile": profile, "offering": self.offering,
                })
                messages = self.scoring.invoke.call_args.args[0]
                payload = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
                expected = {k: self.record[k] for k in (
                    "name", "annual_revenue", "tech_stack", "account_details", "engagement_history"
                )}
                if has_id:
                    expected["prospect_id"] = self.prospect_id
                self.assertEqual(payload, expected)
                self.assertNotIn("private-test-value", json.dumps(messages))
                self.assertEqual(profile, original)
                self.assert_no_billing(result)

    def test_cached_profile_is_filtered_before_returning(self):
        cached = {"prospect_id": self.prospect_id, **self.record}
        data_service._PROFILES[self.prospect_id] = cached
        expected = {k: v for k, v in cached.items() if k != "billing_qualification"}
        result = data_service.get_profile_from_db(self.prospect_id)
        self.assertEqual(result, {"prospect_profile": expected})
        self.assert_no_billing(result)
        self.assertEqual(agent.build_prospect_profile.invoke({
            "prospect_id": self.prospect_id,
        }), {"found": True, "prospect_profile": expected})
        self.assertIn("billing_qualification", cached)

    def test_lookup_profile_and_scoring_flow_has_no_billing(self):
        outputs = [agent.get_prospect.invoke({"prospect_id": self.prospect_id})]
        for _ in range(2):
            profile_result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
            outputs.append(profile_result)
            scoring_arguments = {"prospect_profile": profile_result["prospect_profile"], "offering": self.offering}
            self.assert_no_billing(scoring_arguments)
            outputs.append(agent.score_prospect.invoke(scoring_arguments))
        self.assert_no_billing(outputs)

    def test_missing_prospect_and_cache(self):
        self.assertEqual(data_service.get_profile_from_db("missing"), {"prospect_profile": None})
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}), {
            "prospect": None, "found": False,
        })
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {
            "prospect_profile": None, "found": False,
        })
