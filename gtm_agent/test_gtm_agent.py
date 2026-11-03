import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class RepContextTests(unittest.TestCase):
    def test_context_is_available_for_every_intent_and_scoped_to_each_request(self):
        requests = [
            "Email LEAD-12853 to schedule a demo.",
            "Update LEAD-71001 with Kafka.",
            "Score LEAD-71001 against OFFER-10005.",
        ]
        with patch.object(agent.gtm_agent, "invoke", return_value={
            "messages": [AIMessage(content="Done")],
        }) as invoke:
            for rep, request in zip(agent.REP_IDS, requests):
                with self.subTest(request=request):
                    agent.run_agent(request, user_id=rep["rep_id"])
                    messages = invoke.call_args.args[0]["messages"]
                    self.assertEqual(len(messages), 2)
                    self.assertEqual(messages[0]["role"], "system")
                    context = messages[0]["content"]
                    for value in rep.values():
                        self.assertIn(value, context)
                    self.assertEqual(messages[1], {"role": "user", "content": request})
                    self.assertEqual(invoke.call_args.kwargs["config"]["metadata"]["user_id"], rep["rep_id"])

    def test_generated_user_id_has_matching_context(self):
        rep = agent.REP_IDS[1]
        with patch.object(agent.random, "choice", return_value=rep), patch.object(
            agent.gtm_agent, "invoke", return_value={"messages": [AIMessage(content="Done")]},
        ) as invoke:
            agent.run_agent("Look up OFFER-10001.")
        self.assertIn(json.dumps(rep), invoke.call_args.args[0]["messages"][0]["content"])
        self.assertEqual(invoke.call_args.kwargs["config"]["metadata"]["user_id"], rep["rep_id"])

    def test_unknown_rep_does_not_fabricate_identity(self):
        with patch.object(agent.gtm_agent, "invoke", return_value={
            "messages": [AIMessage(content="Done")],
        }) as invoke:
            agent.run_agent("Look up OFFER-10001.", user_id="unknown")
        self.assertEqual(invoke.call_args.args[0]["messages"], [
            {"role": "user", "content": "Look up OFFER-10001."},
        ])

    def test_rep_lookup_remains_available_without_context(self):
        rep = agent.REP_IDS[0]
        runtime = SimpleNamespace(config={"metadata": {"user_id": rep["rep_id"]}})
        self.assertEqual(agent.get_current_rep.func(runtime), {"rep": rep, "found": True})
        runtime.config = {}
        self.assertEqual(agent.get_current_rep.func(runtime), {"rep": None, "found": False})

    def test_email_graph_can_send_with_context_and_default_sender_without_rep_tool(self):
        rep = agent.REP_IDS[0]
        prospect = agent.get_prospect.invoke({"prospect_id": "LEAD-12853"})["prospect"]
        body = "Could we schedule a demo next week?\n\n" + rep["name"] + "\n" + rep["email"]
        model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "get_prospect", "args": {"prospect_id": "LEAD-12853"}, "id": "contact",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email",
                "args": {"prospect": prospect, "subject": "Demo", "body": body},
                "id": "email",
            }]),
            AIMessage(content="Sent"),
        ])
        graph = agent.create_deep_agent(
            model=model,
            tools=[agent.get_prospect, agent.send_prospect_email, agent.get_current_rep],
            system_prompt=agent.SYSTEM_PROMPT,
        )
        with patch.object(agent, "gtm_agent", graph), patch.dict(
            os.environ, {"LANGSMITH_TRACING": "false"},
        ):
            result = agent.run_agent("Email LEAD-12853 to schedule a demo.", user_id=rep["rep_id"])
        calls = [call["name"] for message in result["messages"]
                 if isinstance(message, AIMessage) for call in message.tool_calls]
        self.assertEqual(calls, ["get_prospect", "send_prospect_email"])
        sent = next(json.loads(message.content) for message in result["messages"]
                    if isinstance(message, ToolMessage) and message.name == "send_prospect_email")
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["from"], rep["email"])
        self.assertEqual(sent["from_name"], rep["name"])
        self.assertEqual(sent["body"], body)
        self.assertIn(json.dumps(rep), result["messages"][0].content)


if __name__ == "__main__":
    unittest.main()
