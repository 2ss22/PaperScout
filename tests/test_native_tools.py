"""Native wire protocol, argument validation and usage accounting."""
from types import SimpleNamespace
from uuid import uuid4

from openai.types.chat import ChatCompletion
import pytest

from research_agent import llm
from research_agent.models import Usage
from research_agent.tool_calling import parse_arguments, tool_reply, tool_schemas


def response(calls=None, *, finish='tool_calls', content=None):
    return ChatCompletion.model_validate({
        'id': 'chat-test', 'object': 'chat.completion', 'created': 1, 'model': 'fixture',
        'choices': [{'index': 0, 'finish_reason': finish, 'message': {'role': 'assistant',
            'content': content, 'reasoning_content': 'provider reasoning preserved', 'tool_calls': calls}}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}})


def tool(call_id='call-1'):
    return {'id': call_id, 'type': 'function', 'function': {'name': 'search_papers',
            'arguments': '{"query":"latent reasoning"}'}}


def install(monkeypatch, responses):
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        return responses.pop(0)
    monkeypatch.setattr(llm, 'get_client', lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    return requests


def test_native_protocol_preserves_reasoning_and_matching_result(monkeypatch):
    requests = install(monkeypatch, [response([tool()]), response(finish='stop', content='完成')])
    messages = [{'role': 'user', 'content': '查找论文'}]
    schemas = tool_schemas(['search_papers'])
    usage = Usage()
    with llm.track_usage(usage):
        first = llm.chat_tools(messages, schemas)
        followup = messages + [first, tool_reply('call-1', {'ok': True, 'papers': []})]
        final = llm.chat_tools(followup, schemas)
    assert requests[0]['tools'] == schemas and requests[0]['tool_choice'] == 'auto'
    assert 'response_format' not in requests[0]
    assert requests[1]['messages'][1]['reasoning_content'] == 'provider reasoning preserved'
    assert requests[1]['messages'][2]['tool_call_id'] == first['tool_calls'][0]['id']
    assert final['reasoning_content'] == first['reasoning_content']
    assert (usage.calls, usage.calls_with_usage, usage.prompt_tokens, usage.completion_tokens) == (2, 2, 20, 6)


@pytest.mark.parametrize('finish,calls,content', [
    ('length', [tool()], None), ('content_filter', [], None),
    ('tool_calls', [tool(), tool()], None), ('tool_calls', [tool('')], None),
    ('tool_calls', [tool(str(i)) for i in range(9)], None), ('stop', [], None),
])
def test_invalid_response_never_reaches_executor(monkeypatch, finish, calls, content):
    install(monkeypatch, [response(calls, finish=finish, content=content)])
    usage = Usage()
    with llm.track_usage(usage), pytest.raises(llm.LLMError):
        llm.chat_tools([], tool_schemas(['search_papers']))
    assert usage.calls == usage.calls_with_usage == 1


@pytest.mark.parametrize('name,raw', [
    ('shell', '{}'), ('search_papers', '{bad'), ('search_papers', '{"query":"  "}'),
    ('search_papers', '{"query":"reasoning","url":"http://localhost/private"}'),
    ('read_visual_page', '{"paper_id":"p","page":"2"}'),
    ('read_visual_page', '{"paper_id":"p","page":81}'),
    ('retrieve_passages', '{"paper_id":"p","query":"x"}'),
])
def test_tool_argument_contract_rejects_unexpected_inputs(name, raw):
    with pytest.raises(ValueError):
        parse_arguments(name, raw)
