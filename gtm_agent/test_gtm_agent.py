import importlib
import json
import os
import unittest
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}), \
        patch("dotenv.load_dotenv"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.calls = []
        self.responses = []

        def generate(model, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

        self.model = patch.object(ChatOpenAI, "_generate", generate)
        self.model.start()
        self.addCleanup(self.model.stop)

    def tool_call(self, name, args):
        return AIMessage(content="", tool_calls=[{
            "name": name, "args": args, "id": str(uuid.uuid4()), "type": "tool_call",
        }])

    def test_model_receives_only_gtm_prompt_and_seven_tools(self):
        self.responses = [AIMessage(content="Ready.")]
        result = agent.run_agent("Hello", user_id="rep_amills")
        self.assertEqual(result["reply"], "Ready.")
        messages, kwargs = self.calls[0]
        self.assertEqual([m.content for m in messages if isinstance(m, SystemMessage)],
                         [agent.SYSTEM_PROMPT])
        schemas = {t["function"]["name"]: t["function"]["parameters"]
                   for t in kwargs["tools"]}
        self.assertEqual(set(schemas), {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info",
            "get_current_rep",
        })
        for name in ("get_current_rep", "send_prospect_email"):
            self.assertNotIn("runtime", schemas[name]["properties"])
        self.assertEqual(set(agent.gtm_agent.get_graph().nodes),
                         {"__start__", "model", "tools", "__end__"})

    def test_run_agent_preserves_invocation_config(self):
        run_id = uuid.uuid4()
        with patch.object(agent.uuid, "uuid4", return_value=run_id), \
                patch.object(agent.gtm_agent, "invoke", return_value={
                    "messages": [AIMessage(content="Done.")],
                }) as invoke:
            result = agent.run_agent("Email a prospect", user_id="rep_dweiss",
                                     environment="staging", thread_id="thread-123")
        invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Email a prospect"}]},
            config={
                "run_name": "GTM Assistant", "run_id": run_id,
                "metadata": {
                    "thread_id": "thread-123", "user_id": "rep_dweiss",
                    "environment": "staging", "request_intent": "send_email",
                },
            },
        )
        self.assertEqual(result["run_id"], str(run_id))
        self.assertEqual(result["reply"], "Done.")

    def test_gtm_workflow_injects_rep_and_email_runtime(self):
        prospect_id = "LEAD-12853"
        prospect = agent.data_service.get_prospect_record(prospect_id)
        offering = agent.data_service.get_offering("OFFER-10001")
        self.responses = [
            self.tool_call("get_current_rep", {}),
            self.tool_call("lookup_offering", {"offering_id": "OFFER-10001"}),
            self.tool_call("build_prospect_profile", {"prospect_id": prospect_id}),
            self.tool_call("score_prospect", {
                "prospect_profile": {"prospect_id": prospect_id, **prospect},
                "offering": offering,
            }),
            self.tool_call("update_prospect_info", {
                "prospect_id": prospect_id, "technology": "ExampleDB",
            }),
            self.tool_call("get_prospect", {"prospect_id": prospect_id}),
            self.tool_call("send_prospect_email", {
                "prospect": {"name": prospect["name"], "email": prospect["email"]},
                "subject": "Your platform fit", "body": "Let's discuss the offering.",
            }),
            AIMessage(content="Profile scored and email sent."),
        ]
        score = agent.ProspectScore(
            score=85, justification="Good fit", rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=90, tech_stack_match=80, segment_fit=85,
            ),
        )
        with patch.object(agent, "_scoring_llm", **{"invoke.return_value": score}), \
                patch.dict(agent.data_service._PROFILES, {}, clear=True):
            result = agent.run_agent(
                "Score LEAD-12853 for OFFER-10001, update its technology and email the prospect.",
                user_id="rep_dweiss",
            )
        outputs = {m.name: json.loads(m.content) for m in result["messages"]
                   if isinstance(m, ToolMessage)}
        self.assertEqual(len(outputs), 7)
        self.assertEqual(outputs["get_current_rep"], {
            "found": True, "rep": agent.data_service.get_rep("rep_dweiss"),
        })
        self.assertTrue(outputs["lookup_offering"]["found"])
        self.assertTrue(outputs["build_prospect_profile"]["found"])
        self.assertEqual(outputs["score_prospect"]["score"], 85)
        self.assertTrue(outputs["update_prospect_info"]["updated"])
        self.assertTrue(outputs["get_prospect"]["found"])
        email = outputs["send_prospect_email"]
        self.assertEqual(email["status"], "sent")
        self.assertEqual(email["from"], agent.data_service.get_rep("rep_dweiss")["email"])
        self.assertEqual(email["to"], prospect["email"])
        self.assertEqual(email["subject"], "Your platform fit")
        self.assertEqual(email["body"], "Let's discuss the offering.")
        self.assertEqual(result["reply"], "Profile scored and email sent.")


if __name__ == "__main__":
    unittest.main()
