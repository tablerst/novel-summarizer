## novel-summarizer 实现规划（v2 — 说书人重写架构）

> **架构转型记录（2026-02-23）**：项目从 v1 "分层压缩（Map-Reduce）"架构，全面转向 v2 "逐章重写（Storyteller Rewrite）"架构。v1 的代码与文档已备份至 `PLAN.v1.md`。核心范式从"压缩"变为"重写"——通过 SQLite 锚定世界观硬逻辑、LanceDB 保留故事血肉记忆、LangGraph 编排逐章循环，让大模型在不丢失设定的安全网内，把故事重新"讲"一遍。

---

### 目标（What / Why）

做一个"说书人风格小说解说稿生成器"CLI 工具：把一本长篇小说（txt/markdown 为主）逐章解析，通过 **SQLite 维护世界观状态** + **LanceDB 语义记忆** + **LangGraph 编排流程**，生成一篇**篇幅约为原著 40%–50%** 的深度解说稿。

**核心理念**：不是"压缩"，而是"重写"。滤除冗长对话和景物描写，保留所有核心情节、动作、伏笔和人物心理博弈，以说书人/剧情解说 Up 主的风格重新演绎整本小说。

**成功标准**（可验收）

- 输入：一份小说文本（txt/md），可选提供书名/作者/章节分隔规则。
- 输出（逐章 + 全书）：
  - `output/<book_hash>/chapters/001_第1章.md` ... `NNN_第N章.md`：每章说书稿（≈原文 40%–50% 篇幅）
  - `output/<book_hash>/full_story.md`：全部章节说书稿拼合的完整解说稿
  - `output/<book_hash>/characters.md`：主要人物表（姓名/别名/关系/动机/状态变化轨迹）
  - `output/<book_hash>/timeline.md`：按章节的关键事件时间线
  - `output/<book_hash>/world_state.json`：最终世界观状态快照
- 世界观一致性：处理第 N 章时，系统知道第 1~N-1 章的所有人物状态、关键道具归属、已发生事件。
- 长线依赖：几十章前的伏笔在当前章被触发时，系统能通过语义检索唤醒相关记忆。
- 可恢复：中途终止后再次运行不重复处理已完成的章节（基于 SQLite + 内容 hash 幂等）。
- 可配置：不改代码即可切换模型、篇幅比例、风格、输出语言等。

### 非目标

- 不做通用 EPUB/PDF 高质量排版解析（后续增强）。
- 不做 GUI（先 CLI；后续可加 Web/桌面）。
- 不做传统 Map-Reduce 式摘要（v1 已验证其局限性，v2 全面转向逐章重写）。

---

## 核心架构：三引擎职责分离

### 设计哲学

传统总结是一层层"压缩"（Map-Reduce），越往后细节越少，最后只剩骨架。本架构的核心是**"重写（Rewrite）"**而非"压缩"——通过三个引擎的协作，在不丢失设定的安全网内，把故事重新"讲"一遍。

### 三引擎角色

| 引擎 | 技术栈 | 职责 | 存什么 |
|------|--------|------|--------|
| **世界观与状态引擎** | SQLite + SQLAlchemy Async | 解决"幻觉"和"关系混乱"，只存**硬逻辑** | Characters（人物状态/位置/存活）、Items（道具归属）、PlotEvents（关键事件时间线）、WorldFacts（设定/规则） |
| **潜意识与伏笔记忆** | LanceDB（向量库） | 解决"长线剧情依赖"，存储向量化的文本切片 | 原文 chunks 向量 + **生成的说书稿向量**。当出现几十章前的人物或隐晦设定时，语义检索唤醒 |
| **说书人工作流** | LangGraph StateGraph | 作为大脑，编排"阅读→查阅资料→更新世界观→撰写解说→归档记忆"的循环 | Graph State（每章处理的中间状态流转） |

### 分层职责

```
┌─────────────────────────────────────────────────┐
│  入口层（CLI）                                    │
│  解析参数、加载配置、展示进度（rich）、错误提示     │
│  命令：ingest / storytell / export / run          │
├─────────────────────────────────────────────────┤
│  编排层（LangGraph StateGraph）                   │
│  逐章循环：每章经过 6 个节点的有向图处理            │
│  每个节点幂等，输入/输出有明确 schema               │
├─────────────────────────────────────────────────┤
│  LLM 层（langchain-openai / 可插拔）              │
│  NER 实体抽取 / 说书稿生成 / 状态变更抽取          │
│  统一封装：重试、超时、并发限制、缓存               │
├────────────────────┬────────────────────────────┤
│  SQLite             │  LanceDB                   │
│  世界观状态（可变）   │  语义记忆（append + 检索）  │
│  产物归档（不可变）   │  原文向量 + 说书稿向量     │
├────────────────────┴────────────────────────────┤
│  基础设施                                         │
│  config / hashing / caching / logging             │
└─────────────────────────────────────────────────┘
```

---

## 目录结构

```
novel-summarizer/
  main.py                          # 入口 thin wrapper
  novel_summarizer/
    __init__.py
    cli.py                         # rich CLI（ingest / storytell / export / run）
    config/
      loader.py                    # YAML + ENV 合并与校验
      schema.py                    # Pydantic 配置模型（含 storyteller 配置段）
    domain/
      hashing.py                   # 内容 hash / 幂等键
    ingest/
      parser.py                    # 章节识别、清洗、规范化
      splitter.py                  # 分块（token/字符）
      service.py                   # ingest 编排入口
    storyteller/                   # ★ 核心：说书人工作流
      __init__.py
      graph.py                     # LangGraph StateGraph 定义与编译
      state.py                     # StorytellerState TypedDict
      nodes/
        __init__.py
        entity_extract.py          # 节点1：NER 实体抽取
        state_lookup.py            # 节点2：SQLite 世界观查询
        memory_retrieve.py         # 节点3：LanceDB 记忆唤醒
        storyteller_generate.py    # 节点4：说书稿生成
        state_update.py            # 节点5：世界观状态更新
        memory_commit.py           # 节点6：记忆归档
      prompts/
        __init__.py
        entity.py                  # NER 抽取 prompt
        narration.py               # 说书稿生成 prompt
        state_mutation.py          # 状态变更抽取 prompt
      service.py                   # storytell 编排入口
    llm/
      factory.py                   # 构建 chat 客户端
      embeddings.py                # 构建 embedding 客户端
      cache.py                     # SQLite 缓存
    embeddings/
      service.py                   # 向量索引构建 / 混合检索
    storage/
      db.py                        # SQLAlchemy async engine + session
      repo.py                      # 聚合式仓储
      models.py                    # 共享模型基类
      types.py                     # 类型定义
      books/                       # books 表
      chapters/                    # chapters 表
      chunks/                      # chunks 表
      narrations/                  # ★ 新增：章节说书稿表
      world_state/                 # ★ 新增：世界观状态表
        characters.py              #   人物可变状态
        items.py                   #   关键道具
        plot_events.py             #   关键事件时间线
        world_facts.py             #   世界设定/硬事实
    export/
      markdown.py                  # 导出 md（含逐章说书稿拼合）
    utils/
      logging.py                   # loguru 配置
  configs/
    default.yaml
    profiles/
      fast.yaml
      quality.yaml
  tests/
  PLAN.md
  PLAN.v1.md                      # v1 架构规划（归档参考）
  README.md
  AGENTS.md
```

---

## LangGraph 逐章循环管道（核心设计）

### 总体流程

外层循环逐章迭代，每章经过 LangGraph StateGraph 的 6 个节点：

```
┌──────────────────────────────────────────────────────────────┐
│                    For each chapter (1..N):                   │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │ 1. Entity   │───▶│ 2. State     │───▶│ 3. Memory      │  │
│  │   Extract   │    │   Lookup     │    │   Retrieve     │  │
│  └─────────────┘    └──────────────┘    └────────────────┘  │
│         │                                       │            │
│         ▼                                       ▼            │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │ 6. Memory   │◀───│ 5. State     │◀───│ 4. Storyteller │  │
│  │   Commit    │    │   Update     │    │   Generate     │  │
│  └─────────────┘    └──────────────┘    └────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Graph State 定义

```python
class StorytellerState(TypedDict):
    # 输入（每章初始化）
    book_id: int
    chapter_id: int
    chapter_idx: int
    chapter_title: str
    chapter_text: str                    # 本章原文全文（或 chunks 拼合）

    # 节点1 输出：NER
    entities_mentioned: list[str]        # 本章出场人名
    locations_mentioned: list[str]       # 本章出场地点
    items_mentioned: list[str]           # 本章出场道具/特殊名词

    # 节点2 输出：世界观状态快照
    character_states: list[dict]         # SQLite 查出的人物当前状态
    item_states: list[dict]              # SQLite 查出的道具归属
    recent_events: list[dict]            # 最近 N 章关键事件

    # 节点3 输出：唤醒的记忆
    awakened_memories: list[dict]        # LanceDB 语义检索结果

    # 节点4 输出：说书稿
    narration: str                       # 本章说书稿全文
    key_events: list[dict]               # 本章关键事件（结构化）
    character_updates: list[dict]        # 人物状态变更
    new_items: list[dict]                # 新道具/道具归属变更

    # 节点5 输出：状态更新确认
    mutations_applied: dict              # 已写入 SQLite 的变更摘要

    # 节点6 输出：归档确认
    memory_committed: bool               # 向量归档完成标志
```

### 各节点详细设计

#### 节点 1：实体抽取（Entity Extract）

**目的**：读入本章原文，轻量级 NER 抽取，为后续节点提供查询 key。

- 输入：`chapter_text`
- LLM 调用：低温（0.1），结构化 JSON 输出
- 输出 JSON schema：
  ```json
  {
    "characters": ["角色A", "角色B"],
    "locations": ["地点X"],
    "items": ["道具Y"],
    "key_phrases": ["某个术语", "某个组织名"]
  }
  ```
- 幂等：以 `chapter_hash + entity_prompt_version` 为缓存 key

#### 节点 2：状态查询（State Lookup）

**目的**：用 NER 结果去 SQLite 查世界观状态，为说书稿生成提供"绝对正确"的背景约束。

- 输入：`entities_mentioned`, `locations_mentioned`, `items_mentioned`
- 查询：
  - `SELECT * FROM characters WHERE book_id=? AND canonical_name IN (?)`（支持别名模糊匹配）
  - `SELECT * FROM items WHERE book_id=? AND name IN (?)`
  - `SELECT * FROM plot_events WHERE book_id=? AND chapter_idx BETWEEN ? AND ? ORDER BY chapter_idx DESC LIMIT 20`
- 纯数据库操作，不调用 LLM
- 输出：`character_states`, `item_states`, `recent_events`

#### 节点 3：记忆唤醒（Memory Retrieve）

**目的**：将本章核心冲突/意图转化为向量查询，从 LanceDB 检索前 N 章的相关内容。

- 输入：`chapter_text`（或其摘要）+ `entities_mentioned`
- 检索策略（混合检索）：
  - **语义检索**：LanceDB 向量相似度（查原文 chunks 表 + 说书稿 narrations 表）
  - **结构过滤**：只检索当前章之前的内容（`chapter_idx < current`）
  - **章节邻近加分**：离当前章越近，分数越高
- 输出：`awakened_memories[]`（每条包含 text、chapter_idx、chapter_title、source_type）

#### 节点 4：说书稿生成（Storyteller Generate）

**目的**：这是核心生成节点。注入三层上下文，生成本章说书稿。

- 输入 Prompt 组装：
  1. **全局状态**（来自 SQLite）：本章涉及人物的最新情况、道具归属
  2. **过往记忆**（来自 LanceDB）：前情提要、唤醒的伏笔片段
  3. **本章原文全文**

- 系统 Prompt 核心指令：
  > "你是一位资深的评书艺人/剧情解说 Up 主。请将本章的详细情节用极具沉浸感、连贯的叙事手法重新演绎。不要做一句话的概括，保留核心的战斗动作、人物心理博弈和关键对白。剔除无意义的环境描写和水文情节。输出篇幅约为原文的 40%–50%。"

- 输出 JSON schema：
  ```json
  {
    "narration": "说书稿正文...",
    "key_events": [
      {"who": "...", "what": "...", "where": "...", "outcome": "...", "impact": "..."}
    ],
    "character_updates": [
      {"name": "...", "change_type": "status|location|ability|relationship", "before": "...", "after": "...", "evidence": "..."}
    ],
    "new_items": [
      {"name": "...", "owner": "...", "description": "..."}
    ]
  }
  ```

- 温度：0.4–0.6（需要一定创造性但不能偏离事实）
- 幂等：以 `chapter_hash + narration_prompt_version + model + world_state_hash` 为缓存 key

#### 节点 5：世界观更新（State Update）

**目的**：根据本章内容更新 SQLite 世界观状态，保证系统在处理下一章时世界观是最新的。

- 输入：节点 4 输出的 `character_updates`, `new_items`, `key_events`
- 操作：
  - `characters` 表：INSERT OR UPDATE（新人物插入，已有人物更新 status/location/abilities/relationships）
  - `items` 表：INSERT OR UPDATE（新道具插入，归属变更更新 owner）
  - `plot_events` 表：INSERT（本章关键事件追加）
- 纯数据库操作 + 少量规则逻辑（别名归一化、冲突检测）
- 输出：`mutations_applied`（变更摘要，用于日志与审计）

#### 节点 6：记忆归档（Memory Commit）

**目的**：将本章说书稿 Embed 存入 LanceDB，供后续章节检索。

- 输入：`narration`（本章说书稿全文）
- 操作：
  - 将说书稿切分为合适长度的片段
  - Embed 并存入 `narrations_vectors_{book_id}` 表
  - 同时在 SQLite `narrations` 表存储说书稿原文（方便导出）
- 输出：`memory_committed = true`

---

## 数据模型与存储设计

### SQLite 表设计

#### 基础表（沿用 v1，小幅调整）

```sql
-- 书籍元数据
books(
  id INTEGER PRIMARY KEY,
  title TEXT,
  author TEXT,
  book_hash TEXT UNIQUE NOT NULL,    -- sha256(normalized_full_text)
  source_path TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- 章节
chapters(
  id INTEGER PRIMARY KEY,
  book_id INTEGER REFERENCES books(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,               -- 章节序号（1-based）
  title TEXT NOT NULL,
  chapter_hash TEXT UNIQUE NOT NULL,
  start_pos INTEGER,
  end_pos INTEGER,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(book_id, idx)
)

-- 文本切片（原文分块，用于向量索引）
chunks(
  id INTEGER PRIMARY KEY,
  chapter_id INTEGER REFERENCES chapters(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  chunk_hash TEXT UNIQUE NOT NULL,
  text TEXT NOT NULL,
  token_count INTEGER,
  start_pos INTEGER,
  end_pos INTEGER,
  meta_json TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(chapter_id, idx)
)
```

#### ★ 新增：章节说书稿表

```sql
-- 逐章说书稿（每章处理完成后写入）
narrations(
  id INTEGER PRIMARY KEY,
  book_id INTEGER REFERENCES books(id) ON DELETE CASCADE,
  chapter_id INTEGER REFERENCES chapters(id) ON DELETE CASCADE,
  chapter_idx INTEGER NOT NULL,
  narration_text TEXT NOT NULL,         -- 说书稿全文
  key_events_json TEXT,                 -- 结构化关键事件 JSON
  prompt_version TEXT NOT NULL,         -- 缓存失效用
  model TEXT NOT NULL,
  input_hash TEXT NOT NULL,             -- 幂等 key
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(chapter_id, prompt_version, model, input_hash)
)
```

#### ★ 新增：世界观状态表（可变）

```sql
-- 人物可变状态
characters(
  id INTEGER PRIMARY KEY,
  book_id INTEGER REFERENCES books(id) ON DELETE CASCADE,
  canonical_name TEXT NOT NULL,         -- 归一化姓名
  aliases_json TEXT DEFAULT '[]',       -- 别名数组 JSON
  first_chapter_idx INTEGER,            -- 首次出场章节
  last_chapter_idx INTEGER,             -- 最近出场章节
  status TEXT DEFAULT 'active',         -- active / dead / missing / unknown
  location TEXT,                        -- 当前所在地点
  abilities_json TEXT,                  -- 已知能力/修为 JSON
  relationships_json TEXT,              -- 与其他人物的关系 JSON
  motivation TEXT,                      -- 当前动机/目标
  notes TEXT,                           -- 补充说明
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(book_id, canonical_name)
)

-- 关键道具
items(
  id INTEGER PRIMARY KEY,
  book_id INTEGER REFERENCES books(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  owner_name TEXT,                      -- 当前持有者（canonical_name）
  first_chapter_idx INTEGER,
  last_chapter_idx INTEGER,
  description TEXT,
  status TEXT DEFAULT 'active',         -- active / destroyed / lost / transferred
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(book_id, name)
)

-- 关键事件时间线（按章节有序追加）
plot_events(
  id INTEGER PRIMARY KEY,
  book_id INTEGER REFERENCES books(id) ON DELETE CASCADE,
  chapter_idx INTEGER NOT NULL,
  event_summary TEXT NOT NULL,
  involved_characters_json TEXT,        -- 涉及人物名数组 JSON
  event_type TEXT,                      -- battle / revelation / death / travel / power_up / alliance / betrayal / etc.
  impact TEXT,                          -- 事件影响/后果
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- 世界设定/硬事实（M3 阶段引入）
world_facts(
  id INTEGER PRIMARY KEY,
  book_id INTEGER REFERENCES books(id) ON DELETE CASCADE,
  category TEXT NOT NULL,               -- geography / organization / magic_system / rule / etc.
  key TEXT NOT NULL,                    -- 设定名称
  value TEXT NOT NULL,                  -- 设定内容
  source_chapter_idx INTEGER,           -- 来源章节
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(book_id, category, key)
)
```

### LanceDB 向量表

| 向量表名 | 内容 | 用途 |
|---------|------|------|
| `chunks_vectors_{book_id}` | 原文切片向量 | 精确定位原文证据 |
| `narrations_vectors_{book_id}` | **说书稿切片向量**（★ 新增） | 唤醒前文说书稿中的伏笔和关联叙述 |

检索时两张表都查，合并结果，按相关性 + 章节邻近度排序。

### 幂等与版本化

核心思想不变：所有"输入 → 产物"的步骤都用**内容 hash**做幂等键。

- `book_hash = sha256(normalized_full_text)`
- `chapter_hash = sha256(book_hash + chapter_title + chapter_text)`
- `chunk_hash = sha256(chapter_hash + split_params + chunk_text)`
- `narration_input_hash = sha256(chapter_hash + world_state_snapshot_hash + prompt_version)`
- `prompt_version`：每次修改 prompt 都 bump 版本，避免缓存污染

关键变化：说书稿的幂等 key 包含 `world_state_snapshot_hash`——如果之前章节的世界观更新了（比如因为 prompt 升级导致重新处理），后续章节的说书稿也会正确地重新生成。

---

## 配置体系

### 加载优先级

1. `configs/default.yaml`（框架默认值）
2. `configs/profiles/<profile>.yaml`（可选覆盖）
3. CLI 参数（最高优先级）
4. ENV：仅用于密钥/敏感信息（`OPENAI_API_KEY`, `OPENAI_BASE_URL`）

### YAML 配置示例

```yaml
app:
  data_dir: "./data"
  output_dir: "./output"
  log_level: "INFO"

ingest:
  encoding: "auto"             # auto-detect common encodings (utf-8/gb18030/big5/...)
  chapter_regex: "^第[0-9一二三四五六七八九十百千]+章.*$"
  cleanup:
    strip_blank_lines: true
    normalize_fullwidth: true

split:
  chunk_size_tokens: 1200
  chunk_overlap_tokens: 120
  min_chunk_tokens: 200

llm:
  provider: "openai"
  chat_model: "deepseek-v3.2"
  embedding_model: "text-embedding-3-large"
  temperature: 0.3               # 默认低温（NER/状态抽取）
  timeout_s: 120
  max_concurrency: 4
  retries: 3

storyteller:                      # ★ 新增配置段
  language: "zh"
  style: "说书人/评书艺人风格，沉浸感强，保留关键对白和心理博弈"
  narration_ratio: [0.4, 0.5]    # 说书稿篇幅占原文比例
  narration_temperature: 0.45    # 说书稿生成温度（略高，需要创造性）
  entity_temperature: 0.1        # NER 抽取温度（低温精确提取）
  state_temperature: 0.1         # 状态变更抽取温度
  memory_top_k: 8                # 记忆唤醒检索条目数
  recent_events_window: 5        # 最近 N 章事件窗口
  include_key_dialogue: true     # 保留关键对白
  include_inner_thoughts: true   # 保留人物心理活动

storage:
  sqlite_path: "./data/novel.db"
  lancedb_dir: "./data/lancedb"

cache:
  enabled: true
  backend: "sqlite"
  ttl_seconds: 2592000           # 30 天
```

### 配置校验

用 `pydantic` 建立 config schema，启动时校验：
- chunk size/overlap 合法
- narration_ratio 在 (0, 1) 范围内
- storage path 可写
- provider 所需 env 已配置（缺失则给出清晰错误）

---

## CLI 设计

### 命令列表

| 命令 | 参数 | 用途 |
|------|------|------|
| `config` | — | 打印当前生效配置 |
| `ingest` | `--input`, `--title`, `--author`, `--chapter-regex` | 解析文本，写入 books/chapters/chunks |
| `storytell` | `--book-id`, `--from-chapter`, `--to-chapter`, `--no-export` | ★ 核心：逐章说书稿生成 |
| `export` | `--book-id`, `--format md` | 从 DB 导出 Markdown |
| `run` | `--input`, `--title`, ... | 一键 ingest + storytell + export |

### 使用示例

```bash
# 1. 导入小说
novel-summarizer ingest --input ./novels/斗破苍穹.txt --title "斗破苍穹" --author "天蚕土豆"

# 2. 生成说书稿（全书）
novel-summarizer storytell --book-id 1

# 3. 只处理特定章节范围（断点续跑 / 调试）
novel-summarizer storytell --book-id 1 --from-chapter 50 --to-chapter 100

# 4. 一键流程
novel-summarizer run --input ./novels/斗破苍穹.txt --title "斗破苍穹"

# 5. 导出
novel-summarizer export --book-id 1
```

### 输出目录结构

```
output/<book_hash>/
  chapters/
    001_第一章_废材的逆袭.md
    002_第二章_药老现身.md
    ...
  full_story.md                  # 全部章节说书稿合并
  characters.md                  # 人物表（从 SQLite characters 表导出）
  timeline.md                    # 事件时间线（从 SQLite plot_events 表导出）
  world_state.json               # 最终世界观状态快照
  run_report.json                # 运行统计
```

---

## Prompt 设计

### 设计原则

- 所有 prompt 强制 JSON 输出，便于解析和后续处理
- 每个 prompt 有独立 `version` 字符串，用于缓存失效
- NER/状态抽取用低温（0.1），说书稿生成用中温（0.4–0.6）
- prompt 按功能分文件，放在 `storyteller/prompts/` 下

### Prompt 1：实体抽取（entity.py）

```
版本：ENTITY_PROMPT_VERSION = "v1"
温度：0.1
系统：你是一个严谨的命名实体识别器。只输出 JSON。
用户：从以下小说章节中提取所有出场的人物名、地点名、关键道具名和特殊术语。
输出：{"characters": [], "locations": [], "items": [], "key_phrases": []}
```

### Prompt 2：说书稿生成（narration.py）

```
版本：NARRATION_PROMPT_VERSION = "v1"
温度：0.45（可配置）
系统：你是一位资深的评书艺人/剧情解说Up主。你的任务是将小说章节用极具沉浸感、
     连贯的叙事手法重新演绎。保留核心的战斗动作、人物心理博弈和关键对白。
     剔除无意义的环境描写和水文情节。
用户：
  ## 当前人物状态（来自世界观数据库，绝对可信）
  {character_states}

  ## 最近剧情事件
  {recent_events}

  ## 前情回忆（与本章相关的历史片段）
  {awakened_memories}

  ## 本章原文
  {chapter_text}

  请输出 JSON：
  {"narration": "...", "key_events": [...], "character_updates": [...], "new_items": [...]}
```

### Prompt 3：状态变更抽取（state_mutation.py，可选）

```
版本：STATE_MUTATION_PROMPT_VERSION = "v1"
温度：0.1
系统：你是一个严谨的信息抽取器。基于章节内容，识别世界观状态的变化。
用户：基于以下章节内容和已知世界观，提取状态变更。
输出：{"character_mutations": [...], "item_mutations": [...], "new_facts": [...]}
```

> 注意：Prompt 3 是可选的。MVP 阶段可直接复用 Prompt 2 输出中的 `character_updates` 和 `new_items`，无需额外 LLM 调用。当需要更严格的状态管理时再独立出来。

---

## 混合检索策略

### 设计

单纯向量检索在小说场景有明显不足（专名召回差、别名问题、精确匹配弱）。实现混合检索：

$$
score = \alpha \cdot \operatorname{norm}(score_{vector}) + (1-\alpha) \cdot \operatorname{norm}(score_{keyword}) + \beta \cdot proximity
$$

- **向量检索**（LanceDB）：语义相似度，适合"这个人物的性格变化"类查询
- **关键词检索**（SQLite FTS5）：对 chunks.text 建全文索引，适合专名/短语精确定位
- **结构过滤**：按 `chapter_idx < current` 限制，只检索已处理的前文
- **章节邻近度**（proximity）：离当前章越近分数越高

### 检索目标表

| 表 | 向量检索 | 关键词检索 | 用途 |
|----|---------|-----------|------|
| `chunks_vectors_{book_id}` | ✅ | ✅（FTS5） | 查原文细节 |
| `narrations_vectors_{book_id}` | ✅ | — | 查前文说书稿中的伏笔叙述 |

---

## 抗幻觉策略

### 核心原则

世界观状态（SQLite）中的信息是"绝对可信"的约束。说书稿生成时，世界观状态作为 prompt 上下文注入，锚定以下事实：
- 人物当前存活状态（不会让已死人物出现闲聊）
- 人物当前位置（不会出现时空穿越矛盾）
- 道具归属（不会凭空出现已易手的道具）
- 已发生的关键事件（不会遗忘重要剧情节点）

### 分级温度

| 任务 | 温度 | 原因 |
|------|------|------|
| NER 实体抽取 | 0.1 | 纯提取，不需要创造性 |
| 状态变更抽取 | 0.1 | 硬逻辑，必须精确 |
| 说书稿生成 | 0.4–0.6 | 叙事演绎需要一定创造性，但受世界观约束 |

### 后续增强（M3+）

- 一致性校验节点：对比 `WorldState(before)` 和 `WorldState(after)`，检测冲突
- 证据验证：对关键事实 claim 进行原文回溯验证
- 未知标注：证据不足时输出"文本未明确交代"而非编造

---

## 成本与性能控制

- **缓存**：所有 LLM 调用结果按 `(prompt_version, model, input_hash, temperature)` 缓存到 SQLite
- **幂等**：已处理章节不会重复调用 LLM（除非 prompt 版本或世界观变化）
- **并发限制**：`max_concurrency` 控制（注意：逐章流水线天然串行，并发主要用于 chunk embedding）
- **章节范围**：`--from-chapter` / `--to-chapter` 支持只处理部分章节
- **profile**：`fast` profile 用更小的模型和更低的 memory_top_k；`quality` profile 用更强模型和更高精度

---

## 推进步骤（里程碑）

### M0：基础设施迁移

**执行状态（2026-02-23）**：🟡 **进行中（核心已落地）**

**完成情况**

- [x] 新增 `storyteller/` 模块目录结构
- [x] 新增 SQLite 世界观状态表（`characters` / `items` / `plot_events`）
- [x] 新增 SQLite `narrations` 表
- [x] 新增 `StorytellerConfig` Pydantic 配置段
- [x] 更新 `configs/default.yaml` 加入 storyteller 配置
- [x] `storage/repo.py` 新增世界观与说书稿 CRUD 方法
- [ ] 移除 v1 的 `summaries` 表和 `summarize/` 模块（或保留为 legacy）

**备注与风险**

- 当前采取 **v1/v2 并存** 策略，降低迁移风险，但短期会增加维护成本。
- 新增表通过 `create_all` 自动建表，后续若需严格版本治理，建议补 Alembic 迁移脚本。

验收：表结构建立，配置可加载，repo 层可读写世界观状态。

### M1：LangGraph 逐章管道 MVP

**执行状态（2026-02-23）**：🟡 **持续推进（M1.1 已落地）**

**完成情况**

- [x] 定义 `StorytellerState` TypedDict
- [x] 实现 `graph.py`（StateGraph 定义与编译）
- [x] 实现节点 1：entity_extract（MVP 规则抽取）
- [x] 实现节点 2：state_lookup（SQLite 查询）
- [x] 实现节点 3：memory_retrieve（chunks 向量检索已接入，narrations 向量待增强）
- [x] 实现节点 4：storyteller_generate（LLM 生成 + 失败降级草稿）
- [x] 实现节点 5：state_update（写入 plot_events）
- [x] 实现节点 6：memory_commit（占位实现）
- [x] 实现 `storyteller/service.py` 外层循环（逐章调用 graph）
- [x] CLI 新增 `storytell` 命令

**M1.1 增量完成（2026-02-23）**

- [x] `entity_extract` 接入真实 LLM JSON 抽取（失败时自动降级规则抽取）
- [x] `memory_retrieve` 接入向量检索并按 `chapter_idx < current` 过滤
- [x] `storyteller_generate` 接入真实 LLM JSON 生成（失败时自动降级草稿）
- [x] `storytell` 服务接入统一 LLM 缓存与模型标识幂等键
- [x] 新增 `tests/test_storyteller_nodes.py`（节点与 JSON 解析单测）

**备注与风险**

- M1 当前为 **可运行 + 可降级** 状态：即使缺失 API Key 或向量检索失败，也可回退到本地草稿流程，保证流水线不中断。
- 仍存在缺口：`memory_commit` 仍为占位；`narrations_vectors_{book_id}` 尚未落地，前文检索暂以原文 chunks 向量为主。
- 生成质量风险仍在：需要在 M2/M3 引入一致性校验与证据回溯，降低世界观漂移概率。

验收：能对一本小说逐章生成说书稿，世界观状态随章节推进更新。

### M2：导出与完整流程

**执行状态（2026-02-23）**：✅ **已完成**

**备注与风险**

- 已落地：chunks/narrations 双向量索引、FTS5 关键词检索、章节邻近打分融合（hybrid retrieval）。
- 已落地：`embed` 命令升级为“检索资产构建”（chunk/narration 向量 + FTS 索引）。
- 残余风险：在不支持 FTS5 的 SQLite 构建环境中，系统会自动降级为向量检索，召回覆盖会下降。

- [x] export 模块适配新输出格式（逐章 md + 全书拼合 + 人物表 + 时间线）
- [x] CLI 新增 `run` 命令（ingest + storytell + export 一键）
- [x] 说书稿向量索引（narrations_vectors 表）
- [x] 混合检索实现（向量 + FTS5 + 章节邻近）
- [x] world_state.json 导出

验收：一键运行得到完整输出目录，混合检索生效。

### M3：质量增强

**执行状态（2026-02-23）**：✅ **已完成**

**备注与风险**

- 已新增一致性校验节点（去重冲突事件、过滤无效角色更新、别名归一化）。
- 已新增 `world_facts` 持久化层，并在 `state_update` 中回写角色/物品/事件硬事实。
- 已新增运行报告核心指标（LLM 调用估算、缓存命中/未命中、输入输出 token 估算、一致性告警/动作、耗时）。
- 残余风险：运行报告中的 token 为估算值，若需精确统计需接入 provider usage 字段。

- [x] 一致性校验节点（检测世界观冲突）
- [x] world_facts 表引入（世界设定/规则）
- [x] 更精细的别名归一化（LLM 辅助 canonical name 合并）
- [x] 运行报告（LLM 调用次数 / token 估算 / 缓存命中率 / 耗时统计）
- [x] 单元测试与回归样本

### M4：高级特性（可选）

**执行状态（2026-02-23）**：🟡 **进行中（三项已落地）**

**备注与风险**

- 高级特性依赖 M2/M3 稳定，否则会放大调试与维护复杂度。
- 已落地：storyteller 节点级多模型路由（entity 与 narration 可独立配置 endpoint，未配置时自动回退 storyteller 默认路由）。
- 已落地：证据验证节点（基于 chapter + awakened memories 进行 claim 支持度筛选，输出 supported/unsupported 计数）。
- 已落地：Refine pass（二次润色节点），在证据筛选后对叙事初稿做风格统一与连贯性润色。
- 剩余项（UI、MCP）仍待实现。

- [ ] MCP Server 暴露说书能力
- [x] 多模型支持（不同节点用不同模型，如 NER 用小模型、说书稿用大模型）
- [x] 证据验证节点（claim + evidence → supported/unsupported）
- [x] Refine pass（说书稿二次润色，提升叙事连贯性）
- [ ] Web UI / 进度面板

---

## 风险与对策

| 风险 | 对策 |
|------|------|
| 章节识别失败 | 可配置 regex + fallback 按长度切章 |
| 单章原文超上下文窗口 | 超长章节先分 chunk，拼合摘要后再生成说书稿 |
| 世界观累积错误 | 低温抽取 + 一致性校验节点（M3） |
| 成本不可控 | 强缓存 + 幂等 + profile 分级 + 章节范围限制 |
| 说书稿风格不一致 | prompt 固定 + 温度控制 + 可选 refine pass |
| LangGraph 调试困难 | 每个节点独立可测 + 详细日志 + 中间状态可导出 |
| 别名/人名归一化困难 | characters 表维护别名列表 + 模糊匹配 + LLM 辅助 |

---

## 与 v1 架构的关系

v1（Map-Reduce 分层压缩）已完成 M0–M4，作为 MVP 验证了基础设施的可行性。以下基础设施在 v2 中**完整沿用**：

- ✅ CLI 框架（cli.py）
- ✅ 配置系统（config/loader.py, schema.py）
- ✅ Ingest 管道（ingest/parser.py, splitter.py, service.py）
- ✅ SQLite + SQLAlchemy Async（storage/db.py）
- ✅ LLM 客户端（llm/factory.py, cache.py）
- ✅ Embedding 服务（embeddings/service.py）
- ✅ 内容哈希（domain/hashing.py）
- ✅ 日志系统（utils/logging.py）

以下模块将被**替换或重构**：

- 🔄 `summarize/` → `storyteller/`（核心逻辑全部重写）
- 🔄 `llm/prompts.py` → `storyteller/prompts/`（prompt 按功能拆分）
- 🔄 `export/markdown.py`（适配新的输出格式）
- 🔄 `storage/summaries/` → `storage/narrations/` + `storage/world_state/`

v1 代码可保留为 legacy（不删除但不再维护），也可在确认 v2 稳定后移除。

---

## 近期任务清单

- [x] 创建 `storyteller/` 模块目录结构与 `__init__.py`
- [x] 定义世界观状态 ORM 模型（characters / items / plot_events / narrations）
- [x] 扩展 `config/schema.py` 加入 `StorytellerConfig`
- [x] 更新 `configs/default.yaml` 和 profiles
- [x] 实现 LangGraph StateGraph + 6 个节点（MVP）
- [x] 实现 `storyteller/service.py` 逐章循环（MVP）
- [x] CLI 新增 `storytell` 命令
- [x] 导出模块适配（v2 主路径 + legacy 回退）
- [x] 编写测试（已覆盖 storyteller 关键节点）
- [x] run 一键流程改造（ingest/storytell/export）
- [x] narrations 向量索引（`narrations_vectors_{book_id}`）
- [x] 混合检索接入（vector + FTS5 + chapter proximity）
- [x] 一致性校验节点接入（`consistency_check`）
- [x] world_facts 模型/CRUD/repo 接入
- [x] state_update 别名归一 + world_facts 回写
- [x] storytell 运行报告指标输出（CLI）
- [x] M3 测试覆盖（consistency + update 回归）
- [x] storyteller 节点级多模型路由（entity/narration）
- [x] 证据验证节点接入（claim/evidence 支持度过滤）
- [x] Refine pass 接入（evidence 之后二次润色）
