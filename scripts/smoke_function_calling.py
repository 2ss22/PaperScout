"""Two real model calls around one real arXiv search (not a full research run)."""
from __future__ import annotations

import json
import time

from research_agent import llm
from research_agent.config import configuration_scope, current_configuration
from research_agent.models import Usage
from research_agent.tool_calling import parse_arguments, tool_reply, tool_schemas
from research_agent.tools import ArxivClient


def smoke() -> dict:
    usage = Usage()
    started = time.monotonic()
    with configuration_scope(), llm.track_usage(usage):
        config = current_configuration()
        schemas = tool_schemas(['search_papers'])
        messages = [{'role': 'user', 'content': '请调用 search_papers 一次，查询 latent reasoning。'
                     '取得工具结果后用一句简短中文说明是否找到论文，不再调用工具。'}]
        decision = llm.chat_tools(messages, schemas)
        calls = decision.get('tool_calls', [])
        if len(calls) != 1 or calls[0]['function']['name'] != 'search_papers':
            raise llm.LLMError('未返回预期的一次原生 search_papers 调用，未执行检索。')
        call = calls[0]
        args = parse_arguments(call['function']['name'], call['function']['arguments'])
        client = ArxivClient()
        try:
            papers = client.search(args.query, max_results=1)
        finally:
            client.close()
        messages.extend([decision, tool_reply(call['id'], {'ok': True, 'papers': [
            {'id': p.id, 'title': p.title} for p in papers]})])
        answer = llm.chat_tools(messages, schemas)
        if answer.get('tool_calls') or not answer.get('content'):
            raise llm.LLMError('工具结果返回后未按要求结束，未执行额外调用。')
        return {'provider': config.text.provider, 'model': config.text.model,
                'tool': call['function']['name'], 'paper_ids': [p.id for p in papers],
                'tool_call_id_matched': messages[-1]['tool_call_id'] == call['id'],
                'reasoning_preserved': 'reasoning_content' in decision,
                'answer': answer['content'], 'usage': usage.model_dump(),
                'elapsed_seconds': round(time.monotonic() - started, 3)}


if __name__ == '__main__':
    print(json.dumps(smoke(), ensure_ascii=False, indent=2))
