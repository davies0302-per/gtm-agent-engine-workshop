import copy
import importlib
import json
import unittest
from unittest.mock import patch

from langsmith import tracing_context

from gtm_agent import data_service


with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}), \
        patch("dotenv.load_dotenv"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectTechnologyTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(tracing_context(enabled=False))
        self.enterContext(patch.dict(data_service.PROSPECTS, copy.deepcopy(data_service.PROSPECTS)))
        self.enterContext(patch.dict(data_service._PROFILES, {}, clear=True))

    def test_update_persists_stack_without_duplicates(self):
        result = data_service.update_prospect_info("LEAD-39002", "Terraform")

        self.assertTrue(result["found"])
        self.assertTrue(result["updated"])
        self.assertIn("Terraform", data_service.fetch_tech_stack("LEAD-39002"))
        self.assertEqual(result["tech_stack"], data_service.fetch_tech_stack("LEAD-39002"))
        data_service.update_prospect_info("LEAD-39002", "Terraform")
        self.assertEqual(data_service.fetch_tech_stack("LEAD-39002").count("Terraform"), 1)

    def test_update_rebuilds_cached_profile_and_preserves_other_profiles(self):
        other = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-71001"})
        original = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})
        self.assertNotIn("Terraform", original["prospect_profile"]["tech_stack"])

        data_service.update_prospect_info("LEAD-39002", "Terraform")
        rebuilt = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})

        self.assertIn("Terraform", rebuilt["prospect_profile"]["tech_stack"])
        self.assertEqual(data_service.get_profile_from_db("LEAD-71001"),
                         {"prospect_profile": other["prospect_profile"]})

    def test_unknown_prospect_does_not_change_records_or_profiles(self):
        records = copy.deepcopy(data_service.PROSPECTS)
        agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})
        profiles = copy.deepcopy(data_service._PROFILES)

        result = data_service.update_prospect_info("UNKNOWN", "Terraform")

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertEqual(data_service.PROSPECTS, records)
        self.assertEqual(data_service._PROFILES, profiles)

    def test_score_uses_updated_saved_stack_even_with_stale_profile(self):
        stale = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})["prospect_profile"]
        offering = data_service.get_offering("OFFER-10004")
        update = agent.update_prospect_info.invoke({"prospect_id": "LEAD-39002", "technology": "Terraform"})
        self.assertTrue(update["updated"])
        self.assertIn("Terraform", data_service.fetch_tech_stack("LEAD-39002"))
        rebuilt = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})["prospect_profile"]
        self.assertIn("Terraform", rebuilt["tech_stack"])

        def score_messages(messages):
            profile = json.loads(messages[1]["content"].split("\n\nProspect profile:\n", 1)[1])
            missing = set(offering["required_tech_stack"]) - set(profile["tech_stack"])
            self.assertIn("Terraform", profile["tech_stack"])
            self.assertFalse(missing)
            return agent.ProspectScore(
                score=100,
                justification="All required technologies present: " + ", ".join(offering["required_tech_stack"]),
                rubric_breakdown=agent.RubricBreakdown(revenue_fit=100, tech_stack_match=100, segment_fit=100),
            )

        with patch.object(agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score_messages
            for profile in (stale, rebuilt):
                result = agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})
                self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 100)
                self.assertIn("Terraform", result["justification"])


if __name__ == "__main__":
    unittest.main()
