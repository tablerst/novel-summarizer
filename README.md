## novel-summarizer

> **说书人风格小说解说稿生成器** — 把长篇小说逐章"重新讲一遍"，而不是压缩成干巴巴的摘要。

通过 **SQLite 世界观状态** + **LanceDB 语义记忆** + **LangGraph 逐章工作流**，生成可按策略控制篇幅的深度解说稿（默认 `medium`：约原著 40%–50%），保留核心情节、人物博弈和关键对白。

### 快速开始

#### 1. 安装依赖

```bash
uv sync
```

#### 2. 配置环境变量

在项目根目录创建 `.env`，填入：

```
DEEPSEEK_API_KEY=sk-...

# 可选：覆盖 default.yaml 中某个 provider 的 base_url
NOVEL_SUMMARIZER_LLM_PROVIDER_DEEPSEEK_BASE_URL=https://...

# 可选：本地 Ollama embedding（注意用 host 根地址）
NOVEL_SUMMARIZER_LLM_PROVIDER_LOCAL_EMBEDDING_BASE_URL=http://localhost:11434
```

#### 3. 查看生效配置

```bash
novel-summarizer config
novel-summarizer --profile fast config    # 查看 fast profile
```

#### 4. 导入小说

```bash
novel-summarizer ingest --input path/to/novel.txt --title "书名" --author "作者"
```

自动识别章节结构，分块并写入 SQLite。可通过 `--chapter-regex` 自定义章节正则。

#### 5. 生成说书稿（核心）

```bash
# 全书逐章生成
novel-summarizer storytell --book-id 1

# 只处理第 50–100 章（断点续跑 / 调试）
novel-summarizer storytell --book-id 1 --from-chapter 50 --to-chapter 100
```

每处理一章，系统会：
1. 抽取本章出场实体（NER）
2. 查询 SQLite 获取当前世界观状态
3. 从 LanceDB 检索与本章相关的前情记忆
4. 注入三层上下文，生成说书稿
5. 更新世界观状态（人物/道具/事件）
6. 将说书稿向量归档，供后续章节检索

#### 6. 导出 Markdown

```bash
novel-summarizer export --book-id 1
# 如需导出历史 v1 summaries：
novel-summarizer export --book-id 1 --mode legacy
```

> 默认 `--mode storyteller`，不再隐式回退到 legacy。

#### 7. 一键运行（导入 + 说书 + 导出）

```bash
novel-summarizer run --input path/to/novel.txt --title "书名"
```

### 输出产物

```
output/<book_hash>/
  chapters/
    001_第一章_标题.md          # 逐章说书稿
    002_第二章_标题.md
    ...
  full_story.md                # 全部章节说书稿合并
  characters.md                # 主要人物表（姓名/别名/关系/状态轨迹）
  timeline.md                  # 关键事件时间线
  world_state.json             # 最终世界观状态快照
```

### 配置说明

| 配置文件 | 用途 |
|---------|------|
| `configs/default.yaml` | 框架默认值 |
| `configs/profiles/fast.yaml` | 快速模式（更小模型、更低精度） |
| `configs/profiles/quality.yaml` | 高质量模式（更强模型、更高精度） |

关键配置段：

- `llm.providers.*` — 提供方连接配置（base_url、api_key_env）
- `llm.chat_endpoints.*` — Chat 端点（模型、温度、超时、重试、并发）
- `llm.embedding_endpoints.*` — Embedding 端点（模型、超时、重试、并发）
- `llm.routes.*` — 业务路由到端点的映射（storyteller/embedding；legacy summarize 可选）
- `storyteller.*` — 风格、篇幅比例、记忆检索条目数、生成温度
- `storage.*` — SQLite / LanceDB 路径
- `ingest.*` — 章节正则、清洗规则、分块参数

`storyteller` 篇幅策略建议使用：

- `narration_preset: short|medium|long`
  - `short` ≈ `0.20~0.30`
  - `medium` ≈ `0.40~0.50`（默认）
  - `long` ≈ `0.65~0.80`
- 若同时配置 `narration_ratio` 与 `narration_preset`，**`narration_ratio` 优先**（用于精细控制）。

> 注：当 LLM 不可用并进入 fallback 时，系统会按 `narration_ratio` 上限进行截断生成，因此切换 preset/ratio 会直接影响 fallback 输出长度。

环境变量用于密钥和可选 provider 级 base_url 覆盖（例如 `DEEPSEEK_API_KEY`、`NOVEL_SUMMARIZER_LLM_PROVIDER_DEEPSEEK_BASE_URL`）。

### 技术栈

- **Python 3.12+**、包管理 **uv**
- **LangGraph** — 逐章工作流编排
- **LangChain + langchain-openai** — LLM 调用（OpenAI 兼容 API）
- **SQLite + SQLAlchemy Async** — 世界观状态 + 产物归档
- **LanceDB** — 向量语义记忆
- **Pydantic** — 配置校验
- **Rich** — CLI 进度展示

### 架构概览

```
CLI → LangGraph 逐章循环 → 每章 6 节点有向图
                              ├── 1. 实体抽取（NER）
                              ├── 2. 世界观查询（SQLite）
                              ├── 3. 记忆唤醒（LanceDB）
                              ├── 4. 说书稿生成（LLM）
                              ├── 5. 世界观更新（SQLite）
                              └── 6. 记忆归档（LanceDB）
```

详见 [PLAN.md](PLAN.md) 了解完整架构设计。

### 开发

```bash
uv sync                        # 安装依赖
uv run pytest                  # 运行测试
uv run python -m black .       # 格式化（line length 120）
uv run python -m ruff check .  # Lint 检查
```

### 说明

- 当前仅支持 OpenAI 兼容 API 作为 LLM 提供方。
- 所有 LLM 调用结果按 `(prompt_version, model, input_hash)` 缓存，幂等可恢复。
- `summarize` 命令为 legacy（v1 Map-Reduce），默认推荐使用 `storytell` / `run`。
