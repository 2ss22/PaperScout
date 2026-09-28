# PaperScout

**面向 AI 研究的论文调研 Agent：自主检索、图表阅读、结论级证据溯源。**

输入一个研究问题，PaperScout 自动检索 arXiv、筛选论文、阅读正文与图表，生成带原文证据和 PDF 页码的中文报告。审阅发现重要缺口时，系统最多补搜一次；仍找不到依据的判断会保留为未知。

项目采用 Python、Streamlit 和手写编排，将检索、阅读、写作、核验及停止决策直接呈现在代码和运行记录中，适合研究辅助、Agent 工程学习与演示。

[快速开始](#快速开始) · [工作流程](#工作流程) · [模型配置](#模型配置) · [测试与评测](#测试与评测) · [已知边界](#已知边界)

## 核心能力

| 能力 | 实现方式 |
| --- | --- |
| 自主论文检索 | 将中文问题转换为英文查询，搜索 arXiv，按论文 ID 去重并记录筛选理由 |
| 正文与图表阅读 | 按页提取 PDF 文本，使用 BM25 检索相关内容；提取图表页截图、结构化单元格及视觉解读 |
| 结论级引用 | 模型选择程序提供的原文片段编号，程序提取引文并生成证据链接 |
| 双层证据核验 | 检查引用与原文的对应关系，再审阅证据是否支持结论；图表结论额外对照原图复核 |
| 有限自主补搜 | Critic 根据证据缺口选择结束或补搜，执行器限制轮数和论文数量 |
| 可追溯输出 | 展示检索词、来源、阅读限制、核验状态、调用次数、token 用量及停止原因 |
| 可携带证据包 | 下载 Markdown 报告、JSON 运行记录，以及包含原图的 ZIP 证据包 |

### 三个关键设计

**1. 引文由程序提取，模型负责选择。**

程序为连续原文片段分配 `span_id`。模型返回片段编号和判断，程序从对应字符区间取回原文，避免模型在复制引文时改写数字、遗漏限定条件或拼接不连续句子。不存在的编号会被拒绝。

**2. “原文存在”和“原文支持结论”分开核验。**

一段引文可以真实存在，却不足以支持某个判断。Critic 读取结论与证据，给出 `supported`、`partial` 或 `insufficient`；图表相关结论还会对照原始截图核验行列、单位、图例和趋势。模型核验结果仍可能有误。

**3. 补搜有明确预算，停止也有理由。**

系统允许一次审阅驱动的补搜。重复查询、无效查询或用尽预算时停止，并保留剩余缺口。检索请求失败与“没有找到相关论文”分别记录。

## 工作流程

```mermaid
flowchart TD
    Q[中文研究问题] --> P[Planner：生成英文查询]
    P --> S[Searcher：检索并筛选 arXiv 论文]
    S --> R[Reader：PDF 正文、图像与表格]
    R --> E[选择原文片段编号并生成证据]
    E --> W[Writer：组织中文报告]
    W --> C[Critic：文本支持核验与原图复核]
    C --> D{是否需要补搜且尚有预算？}
    D -->|是：最多一次| S
    D -->|否| O[报告、证据与运行记录]
```

搜索优先使用 arXiv API；请求失败时尝试 arXiv 官方网页搜索。两个入口都不可用时，系统保留失败记录，不将网络故障解释为研究不存在。

| 执行预算 | 当前限制 |
| --- | --- |
| 初始查询 | 最多 2 条，每条最多 5 篇候选 |
| 初始阅读 | 最多 3 篇论文 |
| 补充检索 | 最多 1 次，新增最多 2 篇论文 |
| 图表阅读 | 每篇默认 3 个候选页，可调整为 1–6 页或关闭 |
| 抽取上下文 | 每篇最多 4 个正文块和 4 个图表块 |
| PDF 处理 | 单文件最多 25 MB、80 页 |

模型通过 Pydantic 校验的 JSON 做出决策，Python 执行器控制动作和预算。当前实现属于**结构化工具调度**，没有使用原生 function calling 接口。

## 快速开始

### 1. 安装依赖

克隆本仓库并进入项目根目录。项目已在 Python 3.13 的 Conda 环境中验证：

```bash
# 已有 deepresearch_py313 环境时，跳过创建步骤
conda create -n deepresearch_py313 python=3.13 -y
conda activate deepresearch_py313
python -m pip install -r requirements.txt
```

### 2. 配置模型

首次使用时参考 [.env.example](.env.example) 创建 `.env`。已有配置无需覆盖。

以下为 DeepSeek 配置示例，将密钥替换为自己的值：

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com

VISION_ENABLED=true
VISION_MODEL=deepseek-flash
VISION_MAX_PAGES=3
```

文本模型需要支持 JSON 输出，视觉模型需要支持图片输入。模型名称和访问权限以自己的服务账户为准。`.env` 已加入 Git 忽略规则，不要将实际密钥写入源码或提交到仓库。

### 3. 启动应用

```bash
python -m streamlit run app.py
```

打开终端提示的本地地址，输入研究问题即可开始。首次检索与 PDF 下载需要联网；远程模型调用可能产生费用，耗时取决于论文规模、视觉页数和服务响应。

## 使用示例

页面提供三个预置研究问题，也可以输入自己的问题：

> 连续潜在推理有哪些方法？现有证据能否支持它改善组合泛化？

> 多模态大模型如何使用视觉信息进行多步推理？有哪些评测与限制？

> 推理时增加计算量与训练连续思维各有什么收益？哪些结果可以直接比较？

一次调研完成后，可以通过四个标签页检查结果：

| 标签页 | 内容 |
| --- | --- |
| 研究报告 | 入选论文、问题／方法／数据集／指标／结论对比、支持边界与待验证方向 |
| 原文证据 | 结论对应的引用片段、论文链接及 PDF 页码 |
| 图像与表格 | 原始页面截图、视觉解读、结构化表格；支持按图号、指标等关键词筛选 |
| 调研过程 | 查询、阅读、审阅、补搜和停止事件，以及模型调用与用量记录 |

适合演示的顺序是：**提出问题 → 查看入选论文 → 打开结论的原文 → 核对图表 → 检查支持状态与停止理由 → 下载证据包。** 是否补搜由实际审阅结果决定，并非每次都会触发。

### 下载内容

- **Markdown**：中文报告、证据锚点及论文页码链接。
- **JSON**：论文、文本块、证据、结论、审阅前草稿和完整运行记录。
- **ZIP**：报告、运行记录、原始 PNG、图表结构化 JSON 和图片校验清单；包内使用相对图片路径。

单独下载 JSON 不包含图片字节；需要离线保存图表证据时，请下载 ZIP。

## 模型配置

支持 DeepSeek、Groq 和 Ollama。文本与视觉可以使用同一服务，也可以分别配置。

<details>
<summary>Groq 配置示例</summary>

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3.8-27b

VISION_ENABLED=true
VISION_MODEL=qwen/qwen3.8-27b
VISION_MAX_PAGES=3
```

未单独设置视觉接口时，视觉请求复用 Groq 的服务地址和 `GROQ_API_KEY`。服务商配额仍适用于文本与视觉请求。

</details>

<details>
<summary>Ollama 配置示例</summary>

先在本地安装 Ollama，并下载所选模型，例如 `ollama pull llama3.1`。

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434/v1
VISION_ENABLED=false
```

此示例使用文本模型。需要图片理解时，应另行配置支持图片输入的模型。Ollama 仅让推理运行在本地，arXiv 检索和首次 PDF 下载仍需联网。

</details>

### 单独配置视觉服务

```dotenv
VISION_ENABLED=true
VISION_MODEL=your_vision_model
VISION_BASE_URL=https://your-compatible-service/v1
VISION_API_KEY=your_vision_api_key
VISION_MAX_PAGES=3
```

视觉接口需兼容 Chat Completions 的图片输入。未指定 `VISION_BASE_URL` 时复用文本服务地址；同一地址下可复用文本密钥。使用不同地址时，需要明确提供对应的 `VISION_API_KEY`。

配置优先级为：**项目 `.env` → 进程环境变量 → Streamlit Secrets**。页面重新运行时加载配置，已开始的调研保持原配置；修改后刷新页面，让下一次调研使用新配置。侧栏显示实际服务、模型及视觉密钥来源，不显示密钥内容。

## 技术实现

| 层次 | 技术与职责 |
| --- | --- |
| 界面 | Streamlit，中文交互、事件展示与下载 |
| 编排 | Python 状态机，有界搜索与审阅循环 |
| 模型接口 | OpenAI Python SDK，连接兼容 Chat Completions 的服务 |
| 结构化输出 | Pydantic，校验模型决策及共享数据记录 |
| 论文检索 | HTTPX、arXiv API、Beautiful Soup 网页结果解析 |
| 正文检索 | pypdf 按页提取，BM25 检索相关文本块 |
| 图表阅读 | PyMuPDF 页面渲染与表格提取，多模态模型解读与复核 |
| 缓存与导出 | 本地文件缓存、Markdown、JSON、ZIP |
| 测试 | pytest、模拟外部服务及 Streamlit AppTest |

正文证据的追溯关系为：

```text
结论 → evidence_id → 原文片段 → 文本块 → 论文版本与 PDF 页码
```

图表证据还关联原始截图、图片及 PDF 的 SHA-256。图表转录和中文解释与正文逐字引文分开标注；没有可用原图或视觉服务时，不能仅根据转录确认图表结论。

缓存默认位于 `.cache/paperscout/`，可通过 `PAPERSCOUT_CACHE_DIR` 修改。搜索与视觉解读缓存有效期为 24 小时；版本化 PDF 文本可长期复用。arXiv 请求带超时、请求间隔和有限重试。

## 项目结构

```text
.
├── app.py                       # Streamlit 入口
├── research_agent/
│   ├── agents.py                # 规划、筛选、片段选择、写作和审阅
│   ├── orchestrator.py          # 调研状态机、预算与证据校验
│   ├── tools.py                 # arXiv 请求、下载和文件缓存
│   ├── search.py                # 查询规范化与网页搜索回退
│   ├── reader.py                # PDF 文本、BM25 与原文片段
│   ├── visuals.py               # 图表页读取与原图核验
│   ├── models.py                # 数据结构与模型输出契约
│   ├── llm.py                   # 模型客户端与用量记录
│   ├── config.py                # 配置读取与运行期间的固定配置
│   ├── report.py                # 报告和引用生成
│   └── artifacts.py             # 合并表格展示与 ZIP 证据包
├── eval/                        # 案例、组件验证与人工评测工具
├── scripts/                     # 检索连通性诊断
├── tests/                       # 离线回归测试
├── .env.example
└── requirements.txt
```

## 测试与评测

### 离线测试

```bash
python -m pytest -q
```

现有验证记录包含 **71 项通过的测试**，覆盖检索回退、去重与缓存、正文／图表读取、原文片段选择、无效引用拒绝、原图核验降级、局部抽取失败恢复、补搜上限、配置刷新和界面下载。

这些测试验证工程行为，不等同于真实论文结论的正确率。

### 联网案例与人工评测

```bash
# 运行一个预置案例；--case all 可运行全部三个案例
python -m eval.evaluator run --case multimodal_reasoning --output eval/runs/demo

# 从本批运行中导出最多 20 条待人工标注记录
python -m eval.evaluator prepare eval/runs/demo/*.json \
  --output eval/runs/annotations.csv

# 人工填写标签后计分
python -m eval.evaluator score eval/runs/annotations.csv
```

评测固定同一份 Writer 草稿与证据，对比直接输出和经过核验约束后的肯定断言。记录有依据结论占比、无依据断言数和有效回答数量，避免只靠删除内容提高正确率。请选用同一批、同一模型的运行记录，并独立核对原文与图表。

工具不会自动填充人工标签；没有人工标注时，不报告准确率提升。当前评测属于小规模案例分析，不作为通用性能证明。详细流程见 [评测说明](eval/README.md)。

运行产物保存到 `eval/runs/`，论文缓存保存到 `.cache/`，均不随源码提交。联网评测会调用已配置的模型。

## 常见问题

| 现象 | 说明与处理 |
| --- | --- |
| arXiv 返回 HTTP 406 | 系统会尝试官方网页搜索；若两个入口均失败，检查网络或稍后再试 |
| 页面显示旧模型 | 刷新页面，检查项目 `.env` 的优先级；旧报告仍保留当时的模型信息 |
| 模型返回 404 | 检查服务商、完整模型名称和账户访问权限 |
| 模型返回 429 | 属于服务商配额限制；等待恢复或调整调研规模，当前未实现 Groq 专用配额调度 |
| 某条证据被拒绝 | 查看片段编号、论文、页码和文本块；无效引用不会进入肯定结论 |
| 报告显示“部分完成” | 仍有证据缺口、读取限制或局部错误，查看各条结论的支持状态 |
| PDF 只能读取摘要 | 下载失败、文件超限或没有可用文本；摘要不会被用于补造实验细节 |

## 已知边界

- **覆盖范围有限**：当前仅检索 arXiv；阅读全文文本的解析不代表模型逐字审阅全部内容，只读取预算内的相关块和图表页。
- **复杂图表可能识别错误**：支持图像、曲线、多级表头及合并单元格，但密集小字、模糊扫描和复杂布局仍需人工核对；不自动拼接跨页表格。
- **证据存在不保证推断正确**：视觉转录不等于作者原文，模型给出的支持标签也不等于人工确认。
- **实验分数不可随意横向排名**：不同数据集、划分和训练预算下的结果必须保留条件。
- **研究方向不等于新颖性证明**：报告中的 Gap 仅是本次检索范围内的待验证方向。
- **用量统计有边界**：调用数按应用层请求统计，不单独计算 SDK 内部重试；服务未返回 token 时不估算。
