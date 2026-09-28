"""Deterministic report rendering: model text never controls source links."""
from __future__ import annotations

import html
import re

from .models import RunResult

LABELS = {"supported": "有支持", "partial": "部分支持", "insufficient": "证据不足", "unreviewed": "未经核验"}
KINDS = {"problem": "研究问题", "method": "方法", "dataset": "数据集", "metric": "指标", "conclusion": "结论"}


def literal(text: str) -> str:
    """Escape markup without removing any source words or URLs."""
    text = html.escape(text, quote=False).replace("\n", " ")
    return re.sub(r"([\\`*_{}\[\]()#+.!|>-])", r"\\\1", text)


def safe(text: str) -> str:
    # All citation links are generated below. Strip model-authored links.
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "（链接略）", text)
    return literal(text)


def render_report(run: RunResult) -> str:
    papers = {p.id: p for p in run.papers}
    evidence = {e.id: e for e in run.evidence}
    def refs(claim):
        return " ".join(f"[{eid}](#evidence-{eid.lower()})" for eid in claim.evidence_ids if eid in evidence)

    lines = ["# PaperScout 论文调研报告", "", f"**研究问题：** {safe(run.question)}", "",
             "> 本报告覆盖本次检索到的文本与已读取的图表页。图表识别和支持程度由模型审阅，仍需人工核对。",
             "", "## 检索范围与论文", ""]
    lines += [f"- 查询：{safe(q)}" for q in run.queries]
    for paper in run.papers:
        level = '正文与图表' if paper.read_status == 'full_text' and paper.assets else ('全文文本' if paper.read_status == 'full_text' else ('视觉读取' if paper.read_status == 'visual_only' else '仅摘要'))
        lines.extend([f"- [{safe(paper.title)}]({paper.url}) · {safe(paper.id)} · {level}",
                      f"  - 入选理由：{safe(paper.reason)}"])
        if paper.read_error:
            lines.append(f"  - 阅读限制：{safe(paper.read_error)}")
        if paper.assets:
            lines.append(f'  - 图表候选页 {paper.visual_pages_available} 页，本次保留原始截图 {len(paper.assets)} 页。')
        for limitation in paper.visual_limitations:
            lines.append(f'  - 图表限制：{safe(limitation)}')
    if not run.papers:
        lines.append("未获得可用于回答的相关论文，不能据此判断相关研究不存在。")

    lines += ["", "## 论文对比", "", "不同评测设置的分数不可直接排名；空缺不代表论文没有相关内容。", "",
              "| 论文 | 研究问题 | 方法 | 数据集 | 指标 | 结论 |",
              "|---|---|---|---|---|---|"]
    for paper in run.papers:
        cells = [f"[{safe(paper.title)}]({paper.url})"]
        for kind in KINDS:
            matches = [c for c in run.claims if c.kind == kind and
                       any(eid in evidence and evidence[eid].paper_id == paper.id for eid in c.evidence_ids)]
            if paper.read_status not in {'full_text', 'visual_only'} and kind in {"dataset", "metric", "conclusion"}:
                matches = []
            rendered = []
            for c in matches:
                if c.status in {"supported", "partial"}:
                    # Cross-paper claims remain in the conclusions, not a single-paper cell.
                    owners = {evidence[e].paper_id for e in c.evidence_ids if e in evidence}
                    if len(owners) == 1:
                        rendered.append(f"【{LABELS[c.status]}】{safe(c.text)} {refs(c)}")
            cells.append("<br>".join(rendered) or "未获得证据")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## 关键结论与支持边界", ""]
    for claim in run.claims:
        # Unsupported claims are audit candidates, never affirmative answers.
        prefix = "待核验说法（不作为结论）" if claim.status in {"insufficient", "unreviewed"} else "结论"
        lines += [f"- **{claim.id} · {LABELS[claim.status]}** · {prefix}：{safe(claim.text)} {refs(claim)}",
                  f"  - 核验说明：{safe(claim.reason or '尚未完成审阅。')}"]
    if not run.claims:
        lines.append("未获得足够证据形成可核验结论。")
    lines += ["", "## 本次检索范围内的待验证方向", "",
              "以下为候选问题与实验建议，不构成新颖性证明，也不表示其他研究尚未开展。", ""]
    for gap in run.gaps:
        lines += [f"- 待验证问题：{safe(gap.question)}", f"  - 所需实验：{safe(gap.experiment)}"]
    if not run.gaps:
        lines.append("现有证据不足以提出具体实验方向。")

    lines += ["", "## 原文证据", ""]
    for e in run.evidence:
        paper = papers[e.paper_id]
        location = f"PDF 第 {e.page} 页" if e.page else "摘要（非全文证据）"
        if e.source in {'vision', 'table'}:
            location += ' · 图表转录（非正文逐字引文，需结合原图）'
        link = f"{paper.pdf_url}#page={e.page}" if e.page else paper.url
        lines += [f'<a id="evidence-{e.id.lower()}"></a>',
                  f"### {e.id} · [{safe(paper.title)}]({link}) · {location}", "",
                  "> " + literal(e.quote), ""]
    if any(p.assets for p in run.papers):
        lines += ['## 图表证据索引', '', '原始截图及结构化单元格随图表证据包（ZIP）提供。', '']
        for p_index, paper in enumerate(run.papers, 1):
            for asset in paper.assets:
                lines.append(f'- [{safe(paper.title)} · 第 {asset.page} 页]({paper.pdf_url}#page={asset.page}) · 图片 SHA256：`{asset.image_sha256}`')
                if asset.reading:
                    for element in asset.reading.elements:
                        lines.append(f'  - {safe(element.label)}：{safe(element.description)}')
    lines += ["## 运行记录", "", f"- 停止原因：{safe(run.stop_reason)}",
              f"- 补搜次数：{run.supplemental_rounds} / 1",
              f"- 模型调用：{run.usage.calls} 次（应用层请求数）",
              f'- 其中视觉调用：{run.usage.vision_calls} 次',
              f"- 耗时：{run.elapsed_seconds:.1f} 秒"]
    if run.usage.calls_with_usage:
        lines.append(f"- 已返回的 token 用量：输入 {run.usage.prompt_tokens}，输出 {run.usage.completion_tokens}；"
                     f"覆盖 {run.usage.calls_with_usage}/{run.usage.calls} 次调用")
    else:
        lines.append("- token 用量：服务未提供或尚无调用")
    if run.errors:
        lines += ["", "### 运行限制", ""] + [f"- {safe(error)}" for error in run.errors]
    return "\n".join(lines) + "\n"
