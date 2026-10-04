"""PaperScout: Chinese Streamlit interface with persistent, downloadable runs."""
from __future__ import annotations

from uuid import uuid4
import hashlib
import streamlit as st


def _read_secrets() -> dict:
    keys = ("LLM_PROVIDER", "GROQ_API_KEY", "GROQ_MODEL", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL",
            "DEEPSEEK_BASE_URL", "OLLAMA_MODEL", "OLLAMA_BASE_URL", "DISABLE_SSL_VERIFY")
    keys += ('VISION_MODEL', 'VISION_BASE_URL', 'VISION_API_KEY', 'VISION_ENABLED', 'VISION_MAX_PAGES')
    try:
        return {key: str(st.secrets[key]) for key in keys if key in st.secrets}
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return {}

from research_agent.checkpoints import list_runs
from research_agent.deletion import preview_deletion, delete_run
from research_agent.config import configuration_scope, settings, vision_settings  # noqa: E402
from research_agent.artifacts import evidence_bundle, table_html  # noqa: E402
from research_agent.visuals import image_bytes  # noqa: E402
from research_agent.report import (render_report, render_scope, render_run_log, evidence_markdown,
                                   answer_evidence_ids, evidence_connections, anchor, LABELS)
from research_agent.models import RunResult  # noqa: E402
from research_agent.orchestrator import run_research  # noqa: E402

st.set_page_config(page_title="PaperScout · 论文调研助手", page_icon="🔎", layout="wide")

EXAMPLES = [
    "连续潜在推理有哪些方法？现有证据能否支持它改善组合泛化？",
    "多模态大模型如何使用视觉信息进行多步推理？有哪些评测与限制？",
    "推理时增加计算量与训练连续思维各有什么收益？哪些结果可以直接比较？",
]


def sidebar() -> None:
    with st.sidebar:
        st.header("关于 PaperScout")
        st.markdown("从研究问题出发，自动检索论文、阅读文本并核验结论。\n\n"
                    "1. 规划英文检索词\n2. 搜索并筛选 arXiv 论文\n"
                    "3. 阅读正文、图像与复杂表格\n4. 组织中文证据报告\n5. 审阅，必要时补搜一次")
        st.caption(f"模型服务：{settings.provider} · {settings.model}")
        st.caption('编排：LangGraph · 原生 Function Calling · SQLite 检查点')
        st.checkbox('阅读前确认论文选择', value=False, key='approve_papers')
        st.caption("初始最多 3 篇 · 补搜最多 2 篇 · 最多补搜 1 次")
        st.info('支持论文图像、流程图、曲线图与复杂表格。图表结论会对照原始截图复核；模糊数字和未覆盖页面仍会保留限制。')
        st.caption(f'视觉模型：{vision_settings.provider} · {vision_settings.model or "尚未配置"}')
        st.caption(f'视觉接入：{vision_settings.base_url} · 密钥来源：{vision_settings.key_source}')
        st.checkbox('读取图像与复杂表格', value=True, key='read_visuals')
        st.slider('每篇最多读取的图表页数', 1, 6, vision_settings.max_pages, key='visual_pages')
        if not vision_settings.enabled:
            st.warning('视觉模型未启用；可以提取截图与表格，但不能确认图像内容。请设置 VISION_MODEL。')
        st.caption("报告中的页码为 PDF 文件页序，从第 1 页开始。")
        notice = st.session_state.pop('paperscout_delete_notice', None)
        if notice:
            st.success(notice)
        saved = list_runs()
        if saved:
            st.divider()
            st.subheader('已保存的调研')
            by_id = {r.run_id: r for r in saved}
            if st.session_state.get('saved_run') not in by_id:
                st.session_state.pop('saved_run', None)
            chosen = st.selectbox('选择历史任务', list(by_id), key='saved_run',
                                  format_func=lambda k: f'{by_id[k].question[:24]} · {by_id[k].status} · {k[:8]}')
            if st.button('查看记录', key='load_run'):
                st.session_state['paperscout_result'] = by_id[chosen]
            if by_id[chosen].resumable and by_id[chosen].status != 'paused':
                if st.button('从检查点继续', key='resume_saved'):
                    st.session_state['paperscout_action'] = {'run_id': chosen, 'resume': True}
            with st.expander('删除所选调研'):
                st.write('将删除所选调研的报告记录和恢复检查点；已导出的文件保留。')
                clean_cache = st.checkbox('同时清理关联论文、PDF 和图片缓存', value=True, key=f'clean_cache_{chosen}')
                plan = None
                try:
                    plan = preview_deletion(chosen, clean_cache=clean_cache)
                    st.caption(f'预计清理 {len(plan.files)} 个缓存文件，{plan.bytes / 1024**2:.2f} MB；'
                               f'保留 {plan.shared_files} 个共享缓存文件。')
                    if clean_cache:
                        st.caption(f'PDF：{plan.counts[".pdf"]} · 图片：{plan.counts[".png"]} · '
                                   f'论文文本、检索与图表解读：{plan.counts[".json"]}')
                    for warning in plan.warnings:
                        st.warning(warning)
                except (ValueError, OSError) as exc:
                    st.info(str(exc))
                confirmed = st.checkbox('我确认删除所选调研，删除后无法恢复', key=f'confirm_delete_{chosen}')
                if st.button('删除记录与关联缓存' if clean_cache else '仅删除记录', key='delete_saved',
                             disabled=not confirmed or plan is None):
                    try:
                        removed = delete_run(chosen, clean_cache=clean_cache)
                    except (ValueError, OSError) as exc:
                        st.error(str(exc))
                    else:
                        current = st.session_state.get('paperscout_result')
                        if current and current.run_id == chosen:
                            st.session_state.pop('paperscout_result', None)
                        action = st.session_state.get('paperscout_action', {})
                        if action.get('run_id') == chosen:
                            st.session_state.pop('paperscout_action', None)
                        st.session_state['paperscout_delete_notice'] = (
                            f'已删除调研及检查点，清理 {len(removed.files)} 个缓存文件；'
                            f'保留 {removed.shared_files} 个共享文件。' + ' '.join(removed.warnings))
                        st.rerun()



def render_evidence_card(result, evidence, *, backlinks=True):
    st.markdown(evidence_markdown(result, evidence, backlinks=backlinks), unsafe_allow_html=False)
    if evidence.asset_id:
        asset = next((a for p in result.papers for a in p.assets if a.id == evidence.asset_id), None)
        if asset:
            try:
                st.image(image_bytes(asset), width=360, caption=f'证据对应的原始 PDF 第 {asset.page} 页')
            except (OSError, RuntimeError) as exc:
                st.warning(str(exc))
            if asset.error:
                st.warning(asset.error)
            if asset.reading and asset.reading.limitations:
                st.caption('识别限制：' + asset.reading.limitations)


def render_result(result: RunResult) -> None:
    # Render older records with the current layout without changing checkpoints
    # or invoking another model. All three downloads use this same snapshot.
    result = result.model_copy(deep=True)
    result.report = render_report(result)
    used = answer_evidence_ids(result)
    st.caption(f'运行 ID：{result.run_id} · 编排：{result.engine} · 模型：{result.provider} / {result.model}')
    if result.status == 'paused':
        st.info('候选论文已保存，确认后从当前步骤继续，无需重新检索。')
        pending = result.pending_selection
        candidates = {p['id']: p for p in pending.get('candidates', [])}
        selection_key = hashlib.sha256('|'.join(candidates).encode()).hexdigest()[:12]
        selected = st.multiselect('确认要阅读的论文', list(candidates), default=pending.get('suggested', []),
                                  format_func=lambda k: candidates[k]['title'],
                                  max_selections=pending.get('limit', 3), key=f'selection_{result.run_id}_{selection_key}')
        if st.button('确认论文并继续', key=f'approve_{result.run_id}'):
            st.session_state['paperscout_action'] = {'run_id': result.run_id, 'resume': True, 'selection': selected}
            st.rerun()
    if result.status == "failed":
        st.error("本次运行未完成，已保留可获得的资料和失败记录。")
        if result.resumable and st.button('恢复本次调研', key=f'retry_{result.run_id}'):
            st.session_state['paperscout_action'] = {'run_id': result.run_id, 'resume': True}
            st.rerun()
    elif result.status == "partial":
        st.warning("报告仍有证据缺口或阅读限制，请查看各条结论的支持状态。")
    cols = st.columns(4)
    cols[0].metric("阅读论文", len(result.papers))
    cols[1].metric("原文证据", len(result.evidence))
    cols[2].metric("模型调用", result.usage.calls)
    cols[3].metric("耗时", f"{result.elapsed_seconds:.1f} 秒")
    report_tab, evidence_tab, visual_tab, trace_tab = st.tabs(["研究报告", "原文证据", '图像与表格', "调研过程"])
    with report_tab:
        st.markdown(render_report(result, include_appendices=False), unsafe_allow_html=True)
        st.subheader('回答依据')
        st.caption('点击正文的证据编号定位到下方卡片，展开核对引用片段与关联判断。中文判断不是原文翻译。')
        for evidence in result.evidence:
            if evidence.id in used:
                st.markdown(f'<a id="{anchor("evidence", evidence.id)}"></a>', unsafe_allow_html=True)
                with st.expander(f'{evidence.id} · 展开原文、支持边界与关联结论'):
                    render_evidence_card(result, evidence)
        unused = [e for e in result.evidence if e.id not in used]
        if unused:
            st.subheader('附录：未用于肯定回答的资料')
            st.caption('包含未引用片段，以及仅关联证据不足或未完成核验说法的片段。')
            for evidence in unused:
                st.markdown(f'<a id="{anchor("evidence", evidence.id)}"></a>', unsafe_allow_html=True)
                with st.expander(f'{evidence.id} · 未用于肯定回答'):
                    render_evidence_card(result, evidence)
        with st.expander('附录：检索范围与阅读限制'):
            st.markdown(render_scope(result), unsafe_allow_html=False)
    with evidence_tab:
        selection = st.radio('证据范围', ['回答引用', '全部资料', '未用于肯定回答'], horizontal=True, key='evidence_scope')
        displayed = [e for e in result.evidence if selection == '全部资料' or
                     (e.id in used if selection == '回答引用' else e.id not in used)]
        for evidence in displayed:
            connections = evidence_connections(result, evidence.id)
            summary = '、'.join(c['claim_id'] for c in connections) or '尚未被结论引用'
            with st.expander(f'{evidence.id} · 关联判断：{summary}'):
                render_evidence_card(result, evidence, backlinks=False)
        if not displayed:
            st.info('当前范围没有证据；可切换“全部资料”查看本次保存的片段。')
    with visual_tab:
        filter_text = st.text_input('检索图表内容', placeholder='图号、表号、方法名、指标或单元格内容', key='visual_filter')
        only_used = st.checkbox('只看回答引用的图表', value=False, key='used_visuals')
        count = 0
        for paper in result.papers:
            for asset in paper.assets:
                eids = [e.id for e in result.evidence if e.asset_id == asset.id]
                cited = bool(used.intersection(eids))
                if only_used and not cited:
                    continue
                linked = [c for c in result.claims if any(e in c.evidence_ids for e in eids)]
                reading_text = asset.reading.model_dump_json() if asset.reading else ''
                native_tables = [c for c in result.chunks if c.asset_id == asset.id and c.source == 'table']
                searchable = paper.title + asset.caption + reading_text + ' '.join(c.text for c in native_tables)
                if filter_text and filter_text.casefold() not in searchable.casefold():
                    continue
                count += 1
                with st.expander(f'{paper.title} · 第 {asset.page} 页 · {"回答引用" if cited else "未用于肯定回答"}', expanded=False):
                    st.caption('关联证据：' + ('、'.join(eids) or '未抽取可用证据'))
                    for claim in linked:
                        st.write(f'{claim.id} · {LABELS[claim.status]}：{claim.text}')
                        st.caption('最终核验说明：' + (claim.reason or '尚未记录'))
                    try:
                        st.image(image_bytes(asset), caption=f'原始 PDF 第 {asset.page} 页')
                    except (OSError, RuntimeError) as exc:
                        st.warning(str(exc))
                    if asset.error:
                        st.warning(asset.error)
                    for table in native_tables:
                        st.caption('几何提取的表格单元格：合并表头及行列关系需结合原图核对。')
                        st.dataframe([c.model_dump() for c in table.cells])
                    if asset.reading:
                        for element in asset.reading.elements:
                            st.write(f'**{element.label}**')
                            st.write(element.description)
                            st.text(element.visible_text)
                            if element.cells:
                                try:
                                    st.markdown(table_html(element.cells), unsafe_allow_html=True)
                                except ValueError as exc:
                                    st.warning(str(exc))
                                    st.dataframe([c.model_dump() for c in element.cells])
                            if element.limitations:
                                st.caption(element.limitations)
                        if asset.reading.limitations:
                            st.caption(asset.reading.limitations)
            for limitation in paper.visual_limitations:
                st.caption(f'{paper.title}：{limitation}')
        if not count:
            st.info('没有匹配的图表页面。')
    with trace_tab:
        st.markdown(render_run_log(result), unsafe_allow_html=False)
        with st.expander('图片校验信息'):
            st.json([{'asset_id': a.id, 'page': a.page, 'image_sha256': a.image_sha256,
                      'pdf_sha256': a.pdf_sha256} for p in result.papers for a in p.assets])
        for item in result.trace:
            with st.expander(f"{item.elapsed_seconds:.1f}s · {item.message}"):
                st.json(item.details)
        st.caption("调用数为应用层请求数；服务未返回 token 时不估算费用。")
        st.json(result.usage.model_dump())
        st.caption(f'工具调用记录：{len(result.tool_calls)} 次（含拒绝和缓存命中）')
        if result.tool_calls:
            with st.expander('查看原生工具调用与返回结果'):
                st.json(result.tool_calls)
    left, right, bundle = st.columns(3)
    left.download_button("下载中文报告（Markdown）", result.report,
                         file_name=f"paperscout_{result.run_id[:8]}.md", mime="text/markdown")
    right.download_button("下载完整运行记录（JSON）", result.model_dump_json(indent=2),
                          file_name=f"paperscout_{result.run_id[:8]}.json", mime="application/json")
    bundle.download_button('下载图表证据包（ZIP）', evidence_bundle(result),
                           file_name=f'paperscout_{result.run_id[:8]}_evidence.zip', mime='application/zip')


def safe_events(action):
    try:
        yield from run_research(**action)
    except (ValueError, OSError) as exc:
        st.error(str(exc))


def main() -> None:
    st.title("🔎 PaperScout · 论文调研助手")
    st.caption("自动寻找论文，回到原文核验；证据不足时补搜，仍不足时保留未知。")
    sidebar()
    st.write("**从一个研究问题开始：**")
    picked = None
    for col, example in zip(st.columns(3), EXAMPLES):
        if col.button(example, use_container_width=True):
            picked = example
    typed = st.chat_input("输入研究问题，例如：连续潜在推理能否改善多模态组合泛化？")
    question = picked or typed
    action = st.session_state.pop('paperscout_action', None)
    if question:
        st.session_state.pop("paperscout_result", None)
        action = {'question': question, 'run_id': uuid4().hex,
                  'approve_papers': st.session_state['approve_papers'],
                  'visual_pages': st.session_state['visual_pages'] if st.session_state['read_visuals'] else 0}
    if action:
        with st.status("正在调研…", expanded=True) as status:
            for event in safe_events(action):
                if event.stage == "done" and isinstance(event.data, RunResult):
                    st.session_state["paperscout_result"] = event.data
                    label = '等待确认论文' if event.data.status == 'paused' else ("调研中断，记录已保留" if event.data.status == "failed" else "调研完成")
                    status.update(label=label, state="error" if event.data.status == "failed" else "complete")
                else:
                    st.write(event.message)
                    if event.stage == "plan" and event.data:
                        for query in event.data.get('queries', []):
                            st.caption(f'检索关键词：{query}')
    if "paperscout_result" in st.session_state:
        render_result(st.session_state["paperscout_result"])


if __name__ == "__main__":
    with configuration_scope(_read_secrets()):
        main()
