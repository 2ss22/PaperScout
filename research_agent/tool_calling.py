"""Allow-listed native function schemas and strict executor argument contracts."""
from __future__ import annotations

import json
from pydantic import Field
from .models import Record


class SearchArgs(Record):
    query: str = Field(min_length=2, max_length=500)


class RetrieveArgs(Record):
    paper_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=2, max_length=500)


class VisualArgs(Record):
    paper_id: str = Field(min_length=1, max_length=100)
    page: int = Field(ge=1, le=80, strict=True)


ARGUMENTS = {'search_papers': SearchArgs, 'retrieve_passages': RetrieveArgs, 'read_visual_page': VisualArgs}
DESCRIPTIONS = {
    'search_papers': '按英文关键词搜索 arXiv，返回最多五篇候选。仅用于当前搜索阶段。',
    'retrieve_passages': '检索当前已选论文的相关正文或已读取图表，返回带文本块编号的片段。',
    'read_visual_page': '读取当前论文的一个指定图表页并保存原图；只能选择提供的候选页，受页数预算限制。',
}


def tool_schemas(names: list[str]) -> list[dict]:
    return [{'type': 'function', 'function': {'name': name, 'description': DESCRIPTIONS[name],
            'parameters': ARGUMENTS[name].model_json_schema()}} for name in names]


def parse_arguments(name: str, raw: str):
    if name not in ARGUMENTS:
        raise ValueError('未知工具；只允许当前阶段的注册工具。')
    if not isinstance(raw, str) or len(raw) > 4096:
        raise ValueError('工具参数不是 JSON 字符串或超过长度限制。')
    args = ARGUMENTS[name].model_validate(json.loads(raw))
    if hasattr(args, 'query') and not args.query.strip():
        raise ValueError('检索词不能为空。')
    return args


def tool_reply(call_id: str, result: dict) -> dict:
    return {'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps(result, ensure_ascii=False)}
