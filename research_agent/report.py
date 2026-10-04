"""Answer-first reports built only from reviewed claims and tool-owned sources."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import html
import re

from .models import RunResult
from .reader import quote_matches

LABELS = {'supported': '有支持', 'partial': '部分支持', 'insufficient': '证据不足', 'unreviewed': '未经核验'}
KINDS = {'problem': '研究问题', 'method': '方法', 'dataset': '数据集', 'metric': '指标', 'conclusion': '结论'}
ROLES = {'support': '支持', 'limitation': '限制', 'contradiction': '反驳', 'background': '背景'}


def literal(text: str) -> str:
    """Escape markup without removing any source words or URLs."""
    text = html.escape(text, quote=False).replace('\n', ' ')
    return re.sub(r'([\\`*_{}\[\]()#+.!|>-])', r'\\\1', text)


def safe(text: str) -> str:
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    return literal(re.sub(r'https?://\S+', '（链接略）', text))


def anchor(kind: str, identifier: str) -> str:
    slug = identifier.lower()
    if not re.fullmatch(r'[a-z0-9_-]+', slug):
        slug = hashlib.sha256(identifier.encode()).hexdigest()[:20]
    return f'{kind}-{slug}'


def evidence_refs(run, ids) -> str:
    known = {e.id for e in run.evidence}
    return ' '.join(f'[{safe(eid)}](#{anchor("evidence", eid)})' for eid in dict.fromkeys(ids) if eid in known)


def answer_claims(run):
    """Never promote an unknown claim or structurally broken saved citation."""
    evidence = {e.id: e for e in run.evidence}
    chunks = {c.id: c for c in run.chunks}
    papers = {p.id for p in run.papers}
    result = []
    for claim in run.claims:
        if claim.status not in {'supported', 'partial'} or not claim.evidence_ids:
            continue
        valid = True
        for eid in claim.evidence_ids:
            e = evidence.get(eid)
            c = chunks.get(e.chunk_id) if e else None
            if (not e or not c or e.paper_id not in papers or c.paper_id != e.paper_id or
                    c.page != e.page or c.source != e.source or c.asset_id != e.asset_id or
                    not quote_matches(e.quote, c.text) or
                    (e.source == 'abstract' and claim.kind not in {'problem', 'method'})):
                valid = False
                break
        if valid:
            result.append(claim)
    return result


def answer_evidence_ids(run) -> set[str]:
    return {eid for c in answer_claims(run) for eid in c.evidence_ids}


def evidence_connections(run, evidence_id) -> list[dict]:
    accepted = {c.id for c in answer_claims(run)}
    connections = []
    for c in run.claims:
        if evidence_id not in c.evidence_ids:
            continue
        relation = [a for a in c.evidence_assessments if a.evidence_id == evidence_id]
        assessment = relation[0] if len(relation) == 1 else None
        status = c.status if c.id in accepted or c.status in {'insufficient', 'unreviewed'} else 'insufficient'
        connections.append({'claim_id': c.id, 'text': c.text, 'status': status, 'scope': c.scope,
                            'role': ROLES[assessment.role] if assessment else '未单独标注',
                            'explanation': assessment.explanation if assessment else '',
                            'reason': c.reason if status == c.status else '引用结构未通过检查，不能用于回答。',
                            'used_in_answer': c.id in accepted})
    return connections


def evidence_markdown(run, e, *, backlinks=True) -> str:
    paper = next((p for p in run.papers if p.id == e.paper_id), None)
    location = f'PDF 第 {e.page} 页' if e.page else '摘要（非全文证据）'
    if e.source in {'vision', 'table'}:
        location += ' · 图表转录（非正文逐字引文，需结合原图）'
    title = safe(paper.title) if paper else safe(e.paper_id)
    link = (f'{paper.pdf_url}#page={e.page}' if e.page else paper.url) if paper else ''
    lines = [f'**[{title}]({link}) · {location}**' if link else f'**{title} · {location}**', '',
             '**引用片段：**', '', '> ' + literal(e.quote), '']
    connections = evidence_connections(run, e.id)
    for conn in connections:
        cid = conn['claim_id']
        ref = f'[{safe(cid)}](#{anchor("claim", cid)})' if backlinks else safe(cid)
        lines += [f'- 关联中文判断 {ref}【{LABELS[conn["status"]]}】：{safe(conn["text"])}',
                  f'  - 最终核验说明：{safe(conn["reason"] or "未记录支持说明。")}']
        if conn['explanation']:
            lines.append(f'  - 文本审阅关系（{conn["role"]}）：{safe(conn["explanation"])}')
        if conn['scope']:
            lines.append(f'  - 条件说明：{safe(conn["scope"])}')
    if not connections:
        lines.append('尚未被任何结论引用；保留供核对，不作为回答依据。')
    if e.id not in answer_evidence_ids(run) and connections:
        lines.append('此片段仅关联待核验或未获支持的说法，未用于肯定回答。')
    gap_ids = [f'G{i}' for i, gap in enumerate(run.gaps, 1) if e.id in gap.evidence_ids]
    if gap_ids:
        lines.append('关联实验建议：' + '、'.join(gap_ids) + '（建议尚未独立核验）。')
    return '\n'.join(lines)


def render_scope(run) -> str:
    lines = ['## 检索范围与论文', '']
    lines += [f'- 查询：{safe(q)}' for q in run.queries]
    for p in run.papers:
        level = {'full_text': '全文文本', 'abstract_only': '仅摘要', 'visual_only': '视觉读取', 'not_read': '尚未阅读'}[p.read_status]
        lines += [f'- [{safe(p.title)}]({p.url}) · {safe(p.id)} · {level}',
                  f'  - 入选理由：{safe(p.reason)}']
        if p.read_error:
            lines.append(f'  - 阅读限制：{safe(p.read_error)}')
        if p.assets:
            lines.append(f'  - 图表候选页 {p.visual_pages_available} 页，保存原始截图 {len(p.assets)} 页。')
        lines += [f'  - 图表限制：{safe(v)}' for v in p.visual_limitations]
    if not run.papers:
        lines.append('未获得可用于回答的相关论文，不能据此判断相关研究不存在。')
    return '\n'.join(lines)


def render_run_log(run) -> str:
    lines = ['## 运行记录', '', f'- 停止原因：{safe(run.stop_reason)}',
             f'- 运行 ID：{safe(run.run_id)} · 编排：{safe(run.engine)}',
             f'- 工具调用记录：{len(run.tool_calls)} 次（含拒绝和缓存命中）',
             f'- 补搜次数：{run.supplemental_rounds} / 1',
             f'- 模型调用：{run.usage.calls} 次（应用层请求数）',
             f'- 其中视觉调用：{run.usage.vision_calls} 次', f'- 耗时：{run.elapsed_seconds:.1f} 秒']
    if run.usage.calls_with_usage:
        lines.append(f'- 已返回的 token 用量：输入 {run.usage.prompt_tokens}，输出 {run.usage.completion_tokens}；'
                     f'覆盖 {run.usage.calls_with_usage}/{run.usage.calls} 次调用')
    if run.errors:
        lines += ['', '### 运行限制', ''] + [f'- {safe(error)}' for error in run.errors]
    return '\n'.join(lines)


def render_report(run: RunResult, *, include_appendices: bool = True) -> str:
    accepted = answer_claims(run)
    accepted_ids = {c.id for c in accepted}
    evidence = {e.id: e for e in run.evidence}
    ranked = sorted(accepted, key=lambda c: (c.priority != 'primary', c.kind != 'conclusion'))
    primary = [c for c in accepted if c.priority == 'primary']
    core = (primary or ranked)[:4]
    lines = ['# PaperScout 论文调研报告', '', f'**研究问题：** {safe(run.question)}', '', '## 核心回答', '',
             '以下回答仅使用已完成核验的结论；“部分支持”必须结合限定条件阅读。支持标签来自模型审阅，仍需核对原文。', '']

    def claim_lines(c):
        block = [f'<a id="{anchor("claim", c.id)}"></a>', '',
                 f'**{safe(c.id)} · {LABELS[c.status]}**　{safe(c.text)} {evidence_refs(run, c.evidence_ids)}', '',
                 f'- 适用条件：{safe(c.scope or "未单独提取；适用范围以引文和核验说明为准。")}']
        if c.status == 'partial':
            block.append(f'- 支持边界：{safe(c.reason or "未记录详细核验说明。")}')
        return block + ['']

    for c in core:
        lines += claim_lines(c)
    if not core:
        lines += ['目前没有足够的已核验证据直接回答问题。未完成核验或证据不足的说法不作为结论。', '']
    core_ids = {c.id for c in core}
    groups = defaultdict(list)
    for c in accepted:
        if c.id not in core_ids:
            groups[c.focus or f'{KINDS[c.kind]}方面有哪些依据？'].append(c)
    if groups:
        lines += ['## 分问题分析', '']
        for question, claims in groups.items():
            lines += [f'### {safe(question)}', '']
            for c in claims:
                lines += claim_lines(c)

    lines += ['## 关键论文对比', '', '不同数据集、划分或预算的结果不直接排名；跨论文综合见上文结论。', '',
              '| 论文 | 与问题相关的发现 | 适用条件与边界 |', '|---|---|---|']
    for p in run.papers:
        own = [c for c in accepted if {evidence[e].paper_id for e in c.evidence_ids} == {p.id}]
        findings = '<br>'.join(f'【{LABELS[c.status]}】{safe(c.text)} [{safe(c.id)}](#{anchor("claim", c.id)})' for c in own[:2])
        boundaries = '<br>'.join(safe(c.scope or c.reason) for c in own[:2])
        if len(own) > 2:
            findings += '<br>其他关联结论：' + ' '.join(f'[{safe(c.id)}](#{anchor("claim", c.id)})' for c in own[2:])
        lines.append(f'| [{safe(p.title)}]({p.url}) | {findings or "未获得可用于肯定回答的证据"} | {boundaries or "未获得可比条件"} |')

    lines += ['', '## 分歧、限制与未知', '']
    for c in accepted:
        if c.status == 'partial':
            lines.append(f'- [{safe(c.id)}](#{anchor("claim", c.id)}) 仅获部分支持：{safe(c.reason)}')
    unknown = [c for c in run.claims if c.id not in accepted_ids]
    for c in unknown:
        lines += [f'<a id="{anchor("claim", c.id)}"></a>', '',
                  f'- **{safe(c.id)} · 待核验说法（不作为结论）**：{safe(c.text)} {evidence_refs(run, c.evidence_ids)}',
                  f'  - 原因：{safe(c.reason) if c.status in {"insufficient", "unreviewed"} else "引用结构未通过检查，不能用于回答。"}']
    if not unknown and not any(c.status == 'partial' for c in accepted):
        lines.append('本次审阅未记录明确分歧；这不表示已经覆盖全部研究。')
    lines += ['', '## 本次检索范围内的待验证方向', '', '以下为实验建议，尚未独立核验，不构成新颖性证明。', '']
    for i, gap in enumerate(run.gaps, 1):
        lines += [f'### G{i} · {safe(gap.question)}', '',
                  f'- 关联材料：{evidence_refs(run, gap.evidence_ids) or "未提供有效证据关联，暂不作为有依据的研究缺口。"}',
                  f'- 所需实验：{safe(gap.experiment)}',
                  f'- 支持／否定假设的判据：{safe(gap.criterion or "尚需设计，不能据此判断实验可行性。")}', '']
    if not run.gaps:
        lines.append('现有证据不足以提出具体实验方向。')
    if not include_appendices:
        return '\n'.join(lines) + '\n'

    used = answer_evidence_ids(run)
    lines += ['', '## 附录 A：原文证据', '']
    for title, items in [('回答引用的证据', [e for e in run.evidence if e.id in used]),
                         ('未用于肯定回答的资料', [e for e in run.evidence if e.id not in used])]:
        lines += [f'### {title}', '']
        if not items:
            lines += ['无。', '']
        for e in items:
            lines += [f'<a id="{anchor("evidence", e.id)}"></a>', '', f'#### {safe(e.id)}', '', evidence_markdown(run, e), '']
    if any(p.assets for p in run.papers):
        lines += ['## 附录 B：图表证据索引', '', '原始截图及结构化单元格随图表证据包（ZIP）提供；图像描述为视觉模型解读。', '']
        for p in run.papers:
            for a in p.assets:
                eids = [e.id for e in run.evidence if e.asset_id == a.id]
                claims = [c for c in run.claims if any(eid in c.evidence_ids for eid in eids)]
                label = '回答引用' if used.intersection(eids) else '未用于肯定回答'
                lines += [f'- [{safe(p.title)} · 第 {a.page} 页]({p.pdf_url}#page={a.page}) · {label}',
                          '  - 关联证据：' + (evidence_refs(run, eids) or '尚未抽取可用证据。'),
                          '  - 关联判断：' + (' '.join(f'[{safe(c.id)}](#{anchor("claim", c.id)})' for c in claims) or '无。')]
                if a.reading:
                    for element in a.reading.elements:
                        lines.append(f'  - {safe(element.label)}：{safe(element.description)}')
                        if element.limitations:
                            lines.append(f'    - 识别限制：{safe(element.limitations)}')
                    if a.reading.limitations:
                        lines.append(f'  - 页面限制：{safe(a.reading.limitations)}')
                if a.error:
                    lines.append(f'  - 读取限制：{safe(a.error)}')
    lines += ['', '## 附录 C：检索与运行详情', '', render_scope(run), '', render_run_log(run), '']
    return '\n'.join(lines)
