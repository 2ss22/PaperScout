"""Run three demos; export paired drafts; score human judgements without inventing labels.

python -m eval.evaluator run --case all
python -m eval.evaluator prepare eval/runs/*.json --output eval/runs/annotations.csv
python -m eval.evaluator score eval/runs/annotations.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from research_agent.models import RunResult
from research_agent.orchestrator import run_research

CASES = Path(__file__).with_name("cases.json")
FIELDS = ["run_id", "question", "model", "claim_id", "claim", "machine_status", "evidence",
          "human_support", "human_relevant", "reviewer", "notes"]


def prepare(paths: list[str], output: Path, limit: int = 20) -> int:
    # Stratify by run then machine status, but never fill in human judgements.
    groups = []
    for path in paths:
        run = RunResult.model_validate_json(Path(path).read_text())
        evidence = {e.id: e for e in run.evidence}
        papers = {p.id: p for p in run.papers}
        status = {c.id: c.status for c in run.claims}
        rows_by_status = {}
        for c in run.drafts[-1] if run.drafts else []:
            quotes = []
            for eid in c.evidence_ids:
                if eid in evidence:
                    e = evidence[eid]
                    p = papers[e.paper_id]
                    quotes.append(f"{eid} {p.pdf_url}#page={e.page}" if e.page else f"{eid} {p.url} 摘要")
                    if e.source in {'table', 'vision'}:
                        quotes.append(f'图表转录（非正文逐字引文）；原图 ID：{e.asset_id}；请核对原始页面。')
                    quotes.append(e.quote)
                else:
                    quotes.append(f"{eid}: 不存在的证据")
            row = dict(zip(FIELDS, [run.run_id, run.question, run.model, c.id, c.text,
                                    status.get(c.id, "insufficient"), "\n".join(quotes), "", "", "", ""]))
            rows_by_status.setdefault(row["machine_status"], []).append(row)
        groups.extend(rows_by_status.values())
    rows = []
    while groups and len(rows) < limit:
        for group in groups:
            if group and len(rows) < limit:
                rows.append(group.pop(0))
        groups = [g for g in groups if g]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def score(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    allowed = {"supported", "partial", "insufficient"}
    labelled = [r for r in rows if r["human_support"] in allowed and r["human_relevant"] in {"yes", "no"}
                and r["reviewer"].strip()]
    if len({(r["run_id"], r["claim_id"]) for r in rows}) != len(rows):
        raise ValueError("同一运行的结论重复，不能重复计分。")
    result = {"total_sampled": len(rows), "human_labelled": len(labelled),
              "pending": len(rows) - len(labelled), "scope": "小规模配对样本，不代表通用性能"}
    for name, selected in [
        ("without_verification", labelled),
        # Only fully supported model verdicts count as definitive assertions.
        ("verified_assertions", [r for r in labelled if r["machine_status"] == "supported"]),
    ]:
        grounded = sum(r["human_support"] == "supported" for r in selected)
        result[name] = {"assertions": len(selected), "grounded": grounded,
                        "grounded_rate": grounded / len(selected) if selected else None,
                        "unsupported_assertions": sum(r["human_support"] == "insufficient" for r in selected),
                        "partially_supported_assertions": sum(r["human_support"] == "partial" for r in selected),
                        "useful_answers": sum(r["human_support"] == "supported" and r["human_relevant"] == "yes" for r in selected)}
    result["qualified_or_withheld"] = sum(r["machine_status"] != "supported" for r in labelled)
    return result


def main():
    parser = argparse.ArgumentParser(description="PaperScout 小规模评测工具")
    subs = parser.add_subparsers(dest="command", required=True)
    run_parser = subs.add_parser("run")
    run_parser.add_argument("--case", default="all")
    run_parser.add_argument("--output", type=Path, default=Path("eval/runs"))
    prep = subs.add_parser("prepare")
    prep.add_argument("paths", nargs="+")
    prep.add_argument("--output", type=Path, default=Path("eval/runs/annotations.csv"))
    prep.add_argument("--limit", type=int, default=20)
    scoring = subs.add_parser("score")
    scoring.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        cases = json.loads(CASES.read_text())
        chosen = [c for c in cases if args.case in {"all", c["id"]}]
        if not chosen:
            parser.error("未知案例；可选：all、" + "、".join(c["id"] for c in cases))
        args.output.mkdir(parents=True, exist_ok=True)
        for case in chosen:
            for event in run_research(case["question"]):
                print(f"[{case['id']}] {event.message}", flush=True)
                if event.stage == "done":
                    run = event.data
                    prefix = args.output / f"{case['id']}_{run.run_id[:8]}"
                    prefix.with_suffix(".json").write_text(run.model_dump_json(indent=2))
                    prefix.with_suffix(".md").write_text(run.report)
    elif args.command == "prepare":
        count = prepare(args.paths, args.output, args.limit)
        print(f"已准备 {count} 条待人工标注记录：{args.output}；未自动填充人工标签。")
    else:
        print(json.dumps(score(args.path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
