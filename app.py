"""PaperScout: Chinese Streamlit interface with persistent, downloadable runs."""
from __future__ import annotations

import streamlit as st


def _read_secrets() -> dict:
    keys = ("LLM_PROVIDER", "GROQ_API_KEY", "GROQ_MODEL", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL",
            "DEEPSEEK_BASE_URL", "OLLAMA_MODEL", "OLLAMA_BASE_URL", "DISABLE_SSL_VERIFY")
    keys += ('VISION_MODEL', 'VISION_BASE_URL', 'VISION_API_KEY', 'VISION_ENABLED', 'VISION_MAX_PAGES')
    try:
        return {key: str(st.secrets[key]) for key in keys if key in st.secrets}
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return {}

from research_agent.config import configuration_scope, settings, vision_settings  # noqa: E402
from research_agent.artifacts import evidence_bundle, table_html  # noqa: E402
from research_agent.visuals import image_bytes  # noqa: E402
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
        st.caption("初始最多 3 篇 · 补搜最多 2 篇 · 最多补搜 1 次")
        st.info('支持论文图像、流程图、曲线图与复杂表格。图表结论会对照原始截图复核；模糊数字和未覆盖页面仍会保留限制。')
        st.caption(f'视觉模型：{vision_settings.provider} · {vision_settings.model or "尚未配置"}')
        st.caption(f'视觉接入：{vision_settings.base_url} · 密钥来源：{vision_settings.key_source}')
        st.checkbox('读取图像与复杂表格', value=True, key='read_visuals')
        st.slider('每篇最多读取的图表页数', 1, 6, vision_settings.max_pages, key='visual_pages')
        if not vision_settings.enabled:
            st.warning('视觉模型未启用；可以提取截图与表格，但不能确认图像内容。请设置 VISION_MODEL。')
        st.caption("报告中的页码为 PDF 文件页序，从第 1 页开始。")


def render_result(result: RunResult) -> None:
    if result.status == "failed":
        st.error("本次运行未完成，已保留可获得的资料和失败记录。")
    elif result.status == "partial":
        st.warning("报告仍有证据缺口或阅读限制，请查看各条结论的支持状态。")
    cols = st.columns(4)
    cols[0].metric("阅读论文", len(result.papers))
    cols[1].metric("原文证据", len(result.evidence))
    cols[2].metric("模型调用", result.usage.calls)
    cols[3].metric("耗时", f"{result.elapsed_seconds:.1f} 秒")
    report_tab, evidence_tab, visual_tab, trace_tab = st.tabs(["研究报告", "原文证据", '图像与表格', "调研过程"])
    with report_tab:
        # report.py escapes all external prose; only program-owned citation anchors use HTML.
        st.markdown(result.report, unsafe_allow_html=True)
    with evidence_tab:
        papers = {p.id: p for p in result.papers}
        for e in result.evidence:
            paper = papers[e.paper_id]
            location = f"PDF 第 {e.page} 页" if e.page else "仅摘要"
            with st.expander(f"{e.id} · {paper.title} · {location}"):
                st.text(e.quote)
                if e.source in {'vision', 'table'}:
                    st.caption('此内容为图表转录或结构化提取，不是 PDF 正文逐字引文。请在“图像与表格”核对原图。')
                st.link_button("查看论文原文", f"{paper.pdf_url}#page={e.page}" if e.page else paper.url)
        if not result.evidence:
            st.info("没有获得可匹配的原文证据。")
    with visual_tab:
        filter_text = st.text_input('检索图表内容', placeholder='图号、表号、方法名、指标或单元格内容', key='visual_filter')
        count = 0
        for paper in result.papers:
            for asset in paper.assets:
                reading_text = asset.reading.model_dump_json() if asset.reading else ''
                native_tables = [c for c in result.chunks if c.asset_id == asset.id and c.source == 'table']
                searchable = paper.title + asset.caption + reading_text + ' '.join(c.text for c in native_tables)
                if filter_text and filter_text.casefold() not in searchable.casefold():
                    continue
                count += 1
                with st.expander(f'{paper.title} · 第 {asset.page} 页', expanded=False):
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
        for item in result.trace:
            with st.expander(f"{item.elapsed_seconds:.1f}s · {item.message}"):
                st.json(item.details)
        st.caption("调用数为应用层请求数；服务未返回 token 时不估算费用。")
        st.json(result.usage.model_dump())
    left, right, bundle = st.columns(3)
    left.download_button("下载中文报告（Markdown）", result.report,
                         file_name=f"paperscout_{result.run_id[:8]}.md", mime="text/markdown")
    right.download_button("下载完整运行记录（JSON）", result.model_dump_json(indent=2),
                          file_name=f"paperscout_{result.run_id[:8]}.json", mime="application/json")
    bundle.download_button('下载图表证据包（ZIP）', evidence_bundle(result),
                           file_name=f'paperscout_{result.run_id[:8]}_evidence.zip', mime='application/zip')


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
    if question:
        st.session_state.pop("paperscout_result", None)
        with st.status("正在调研…", expanded=True) as status:
            for event in run_research(question, visual_pages=st.session_state['visual_pages'] if st.session_state['read_visuals'] else 0):
                if event.stage == "done" and isinstance(event.data, RunResult):
                    st.session_state["paperscout_result"] = event.data
                    label = "调研中断，记录已保留" if event.data.status == "failed" else "调研完成"
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
