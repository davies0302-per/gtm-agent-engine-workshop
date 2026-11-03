import importlib
import json
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent


with patch('dotenv.load_dotenv'), patch.dict(
    'os.environ', {'LANGSMITH_TRACING': 'false'}
), patch('langchain_openai.ChatOpenAI', side_effect=lambda **kwargs: ChatOpenAI(
    api_key='test-key', **kwargs
)):
    agent = importlib.import_module('gtm_agent.gtm_agent')


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = next(iter(agent.data_service.PROSPECTS))
        self.offering_id = next(iter(agent.data_service.OFFERINGS))
        self.ids = {
            'prospect_id': self.prospect_id,
            'offering_id': self.offering_id,
        }
        self.cache = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        scorer_patch = patch.object(agent, '_scoring_llm')
        self.scorer = scorer_patch.start()
        self.addCleanup(scorer_patch.stop)
        self.result = agent.ProspectScore(
            score=80, justification='Good fit', rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=90, tech_stack_match=70, segment_fit=80
            )
        )
        self.scorer.invoke.return_value = self.result

    def scoring_records(self):
        messages = self.scorer.invoke.call_args.args[0]
        offering, profile = messages[1]['content'].split('\n\nProspect profile:\n')
        return json.loads(offering.removeprefix('Offering:\n')), json.loads(profile)

    def test_ids_load_records_and_return_score(self):
        self.assertEqual(agent.score_prospect.invoke(self.ids), self.result.model_dump())
        offering, profile = self.scoring_records()
        self.assertEqual(offering, agent.data_service.get_offering(self.offering_id))
        self.assertEqual(profile, {
            'prospect_id': self.prospect_id,
            **agent.data_service.get_prospect_record(self.prospect_id),
        })

    def test_missing_prospect_does_not_score(self):
        result = agent.score_prospect.invoke({**self.ids, 'prospect_id': 'missing'})
        self.assertEqual(result, {'score': None, 'error': 'Prospect not found.'})
        self.scorer.invoke.assert_not_called()

    def test_missing_offering_does_not_score(self):
        result = agent.score_prospect.invoke({**self.ids, 'offering_id': 'missing'})
        self.assertEqual(result, {
            'score': None, 'error': 'Cannot score without a valid offering.'
        })
        self.scorer.invoke.assert_not_called()

    def test_offering_requires_each_scoring_field(self):
        for field in ('required_tech_stack', 'min_annual_revenue', 'description'):
            with self.subTest(field=field):
                offering = dict(agent.data_service.get_offering(self.offering_id))
                del offering[field]
                with patch.object(agent.data_service, 'get_offering', return_value=offering):
                    result = agent.score_prospect.invoke(self.ids)
                self.assertEqual(result, {
                    'score': None, 'error': 'Cannot score without a valid offering.'
                })
                self.scorer.invoke.assert_not_called()

    def test_scoring_refreshes_cached_tech_stack(self):
        cached = agent.build_prospect_profile.invoke({'prospect_id': self.prospect_id})
        original_stack = list(cached['prospect_profile']['tech_stack'])
        with patch.object(agent.data_service, 'fetch_tech_stack', return_value=['New CRM']):
            agent.score_prospect.invoke(self.ids)
        _, profile = self.scoring_records()
        self.assertEqual(profile['tech_stack'], ['New CRM'])
        self.assertEqual(cached['prospect_profile']['tech_stack'], original_stack)

    def test_display_tools_preserve_records_and_found_flags(self):
        profile = agent.build_prospect_profile.invoke({'prospect_id': self.prospect_id})
        self.assertTrue(profile['found'])
        for field in ('engagement_history', 'account_details', 'tech_stack'):
            self.assertEqual(profile['prospect_profile'][field],
                             agent.data_service.get_prospect_record(self.prospect_id)[field])
        self.assertEqual(agent.lookup_offering.invoke({'offering_id': self.offering_id}), {
            'offering': agent.data_service.get_offering(self.offering_id), 'found': True
        })
        self.assertEqual(agent.build_prospect_profile.invoke({'prospect_id': 'missing'}), {
            'prospect_profile': None, 'found': False
        })
        self.assertEqual(agent.lookup_offering.invoke({'offering_id': 'missing'}), {
            'offering': None, 'found': False
        })

    def run_request(self, request, calls, reply):
        model = ToolCallingModel(responses=[
            AIMessage(content='', tool_calls=calls), AIMessage(content=reply)
        ])
        graph = create_deep_agent(
            model=model,
            tools=[agent.score_prospect, agent.build_prospect_profile, agent.lookup_offering],
            system_prompt=agent.SYSTEM_PROMPT,
        )
        with patch.object(agent, 'gtm_agent', graph):
            return agent.run_agent(request, user_id='rep_amills')

    def test_score_only_request_through_agent(self):
        result = self.run_request(
            f'Score {self.prospect_id} for {self.offering_id}',
            [{'name': 'score_prospect', 'args': self.ids, 'id': 'score'}],
            'Score: 80/100. Good fit.',
        )
        outputs = [message for message in result['messages'] if message.type == 'tool']
        self.assertEqual([message.name for message in outputs], ['score_prospect'])
        self.assertEqual(json.loads(outputs[0].content), self.result.model_dump())
        self.assertEqual(result['reply'], 'Score: 80/100. Good fit.')

    def test_display_request_through_agent(self):
        result = self.run_request(
            f'Show the profile for {self.prospect_id} and offering {self.offering_id}',
            [
                {'name': 'build_prospect_profile', 'args': {'prospect_id': self.prospect_id},
                 'id': 'profile'},
                {'name': 'lookup_offering', 'args': {'offering_id': self.offering_id},
                 'id': 'offering'},
            ],
            'Here are the prospect profile and offering details.',
        )
        outputs = [message for message in result['messages'] if message.type == 'tool']
        self.assertEqual({message.name for message in outputs},
                         {'build_prospect_profile', 'lookup_offering'})
        self.assertTrue(all(json.loads(message.content)['found'] for message in outputs))
        self.scorer.invoke.assert_not_called()
        self.assertEqual(result['reply'], 'Here are the prospect profile and offering details.')


if __name__ == '__main__':
    unittest.main()
