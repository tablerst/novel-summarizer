# Storyteller Step（多章批处理）执行文档

- 创建时间：2026-02-28
- 负责人：Copilot
- 状态：进行中（Phase 0/1/2 已落地，Phase 3/4 待推进）

## 背景与问题

当前 `storyteller` v2 的默认执行粒度是“逐章”，并且每章可能触发多次 LLM/检索调用：

- `entity_extract`：可选（LLM 或 fallback）
- `memory_retrieve`：当 `memory_top_k > 0` 时，每章会触发 **一次远程 embedding**（`OpenAIEmbeddingClient.embed_query`）并结合 LanceDB/FTS 检索
- `storyteller_generate`：每章一次 chat
- `refine_narration`：可选（每章一次 chat）

在“单章 token 量 ~5k、但每章要跑多轮 LLM”的场景下，总耗时会被放大；更隐蔽的是：**每章一次 embedding 查询**经常成为整体瓶颈。

本需求希望新增一个可配置的 Step 功能：一次对多章（例如 5 章）进行处理，并且保证一致性约束可控、可回溯、可切换 step 参数后对比覆盖。

## 目标

1. 新增配置项 `storyteller.step_size`（默认 1），支持一次处理连续 N 章。
2. 在一次运行中保证一致性：
   - Step 内所有 LLM 操作读取 **同一个基准 world_state**（来自 `step_start-1`，即上一个 step 结束时的状态）。
   - Step 结束后再批量/顺序落库更新 world_state（事件/人物/物品/事实）。
3. 支持断点续跑与切换 `step_size`：
   - 允许生成同章多版本 narration（依赖最新版本机制导出/索引不会重复）。
   - 当 step 起点缺失 world_state 数据时，能够 **fallback 到重新生成**，确保基准态正确。
4. 性能收益：
   - 将多章 narration 合并为一次（或少数几次）chat。
   - 将 step 内多章 `memory_retrieve` 的 embedding 查询 **合批**（`embed_documents` 一次处理多 query）。

## 非目标（第一版明确不做）

- 不追求 step 内“逐章推进 world_state 后再喂给下一章 prompt”的严格逐章一致性（那会显著降低 batching 的收益）。
- 不落地“每章 world_state 快照”。第一版只记录 **step 结束章**的 world_state checkpoint。

## 现状（已核实的代码事实）

- 章节循环：`novel_summarizer/storyteller/service.py::storytell_book()`
- LangGraph 节点链：
  `entity_extract` → `state_lookup` → `memory_retrieve` → `storyteller_generate` →
  `consistency_check` → `evidence_verify` → `refine_narration` → `state_update` → `memory_commit`
- world_state 存储方式：
  - `characters/items/world_facts` 为 upsert 表
  - `plot_events` 为 append 表（无去重约束）
- narration 的“是否跳过”判定基于 per-chapter `input_hash`，当前 **不包含** state_lookup 得到的 world_state 内容；step 模式必须补齐，否则切 step 参数会导致错误跳过。

## 方案概览（语义定义）

### Step 分组

- `step_size = N` 时，将章节按连续分组：`[1..N]、[N+1..2N]、...`。
- 对任意 step `[S..E]`：该 step 处理期间读取的基准 world_state 是 chapter `S-1` 的状态（step 边界态）。

### Step 内 world_state 读取与写入

- Step 内生成 narration 时：所有章节统一使用 `S-1` 的 world_state（不随 step 内章节推进）。
- Step 完成后：按章节顺序将 `state_update` 的变更写入 SQLite（得到 `E` 的新边界态）。
- Step 完成后：保存 checkpoint（快照）在 `chapter_idx = E`。

### from/to 对齐规则（避免语义不确定）

当 `step_size > 1` 且用户指定 `--from-chapter`：

- 默认行为：**自动向下对齐到该 step 的起点**。
  - 例：`step_size=5`，用户 `from=7`，则实际从 `6` 开始。
- 原因：step 语义依赖 `S-1` 的边界态；若从 step 中间开始，无法保证正确基准态（除非提前重跑本 step 前置章节）。

### “step 起点没有数据”如何处理（你问的关键点）

基准 world_state 需要来自 `S-1`，但可能不存在 checkpoint。此时必须 fallback：

1. 若存在 `chapter_idx <= S-1` 的最近 checkpoint：
   - restore 到该 checkpoint（恢复 world_state 表到一致状态）；
   - 然后从 `checkpoint.chapter_idx + 1` 继续按 step 语义顺推，直到生成出 `S-1` 的边界态。
2. 若不存在任何 checkpoint：
   - 从“空 world_state”（或 chapter 0 初始态）开始，按 step 语义从第 1 章重跑，直到生成出 `S-1` 的边界态。

该 fallback 可能会重新生成并覆盖已有 narration（通过新的 `input_hash` 产生新版本，不需要删除旧行）。

## 详细实施方案（可拆分任务）

### Phase 0：配置与 CLI（无行为变更）

1. 在 `novel_summarizer/config/schema.py::StorytellerConfig` 增加：
   - `step_size: int = 1`（>=1）
   - `step_align: Literal["auto", "off"] = "auto"`（step_size>1 时 from/to 自动对齐）
   - `step_checkpoint_enabled: bool = True`
   - `step_resume_mode: Literal["continue", "restore"] = "restore"`
   - `step_memory_mode: Literal["per_chapter", "per_step_shared", "off"] = "per_chapter"`
   - （可选）`step_refine_mode: Literal["inline", "off"] = "inline"`（inline = 批量生成时直接产出润色后的稿）
   - （新增）`step_retrieval_concurrency: int = 6`（step 内检索并发度；0/1 表示顺序执行）

2. 在 `novel_summarizer/cli.py`：
   - `storytell` 与 `run` 增加 `--step-size`（通过 overrides 写入 config，不强制改 YAML）。

### Phase 1：Checkpoint（确保可恢复与可切 step）

新增一张表保存 step 边界态：

- 表名建议：`world_state_checkpoints`
- 字段建议：
  - `book_id`
  - `chapter_idx`（step 结束章）
  - `step_size`
  - `snapshot_json`（包含 characters/items/plot_events/world_facts 的快照）
  - `snapshot_hash`（用于幂等/校验）
  - `created_at`

需要在 `storage/repo.py` 增加：

- `get_latest_checkpoint_at_or_before(book_id, chapter_idx)`
- `upsert_checkpoint(book_id, chapter_idx, step_size, snapshot_json, snapshot_hash)`
- `restore_checkpoint(book_id, checkpoint)`：
  - `DELETE` 清空 `characters/items/plot_events/world_facts`（按 book_id）
  - 从 checkpoint 快照重建（避免 plot_events 重复追加污染）

> 备注：`plot_events` 当前无唯一约束，恢复时必须清空再重建，否则 rerun 会无限追加。

### Phase 2：Step 执行主流程（service 层）

在 `novel_summarizer/storyteller/service.py::storytell_book()`：

- `step_size == 1`：保持现有逐章 LangGraph 流程（完全不变）。
- `step_size > 1`：进入 step 模式：

1. 对齐 from/to（若 `step_align==auto`）。
2. 计算 step 起点 `S`，需要基准态 `S-1`。
3. 若 `step_resume_mode==restore`：
   - 获取最近 checkpoint `<= S-1`。
   - 若 checkpoint.chapter_idx == S-1：直接 restore。
   - 若 checkpoint.chapter_idx < S-1 或不存在 checkpoint：按“step 起点没有数据 fallback”策略补跑直到 S-1。
4. 对每个 step `[S..E]`：
   - 只执行一次 `state_lookup` 取得基准 world_state（来自 DB，此时 DB 已处于 `S-1`）。
   - 批量生成 step 内多个章节的 narration（见 Phase 3）。
   - Step 内逐章做 `consistency_check` 与 `evidence_verify`。
   - 将 narration 逐章 `upsert_narration`。
   - step 完成后，再逐章调用 `state_update` 落库。
   - 写 checkpoint（chapter_idx=E）。

### Phase 3：批量 narration 生成（减少 chat 次数）

新增一个批量生成节点/函数（不强制接入 LangGraph，第一版可直接由 service 调用）：

- 位置建议：`novel_summarizer/storyteller/nodes/storyteller_generate_step.py`
- Prompt：新增 `novel_summarizer/storyteller/prompts/step_narration.py`，定义 `STEP_NARRATION_PROMPT_VERSION`
- 输出格式：严格 JSON 数组，长度必须等于 step 内章节数，每项包含：
  - `chapter_idx`
  - `narration`
  - `key_events[]`
  - `character_updates[]`
  - `new_items[]`

鲁棒性策略：

- 若解析失败/数组长度不匹配/某章 narration 为空：
  - 自动将 step 拆分为两半重试；
  - 最小粒度退化到单章时，fallback 到现有 `storyteller_generate`（逐章）路径。

### Phase 4：memory_retrieve embedding 合批（减少 embedding 往返）

在 step 模式下，`memory_retrieve` 的瓶颈主要来自每章一次 `embed_query`。

建议新增 helper：

- `novel_summarizer/embeddings/service.py` 增加 `retrieve_hybrid_memories_batched(...)`：
  - 输入：多个 `query_text`（每章一个）
  - 一次性 `embed_documents(query_texts)` 得到多向量
  - 对每个向量分别做 LanceDB 相似检索 + FTS

并利用现有“预填充短路”机制：step 模式下把每章 `awakened_memories` 直接填入 state，跳过 `memory_retrieve` 节点。

### Phase 5：幂等与“切换 step 覆盖”

当前 `storytell_book()` 的 narration skip 使用 per-chapter `input_hash`，但不包含 world_state。

Step 模式必须把以下信息纳入每章 `input_hash`：

- `step_size`
- `step_start_idx`
- `base_world_state_hash`（对 `state_lookup` 返回的 `character_states/item_states/recent_events/world_facts` 进行稳定序列化后 sha256）

这样可以保证：

- 切换 step_size 会产生新 hash → 新版本 narration 插入 → latest 导出自动“覆盖”。
- checkpoint restore 后若基准态变化，也会触发重新生成，避免错误跳过。

## 任务清单与进度（待更新）

| ID | 任务 | 状态 | 说明 |
|---|---|---|---|
| 1 | 配置模型增加 step_* 字段 | ✅ | `config/schema.py::StorytellerConfig` |
| 2 | CLI 增加 --step-size | ✅ | `cli.py` overrides → `storyteller.step_size` |
| 3 | 新增 world_state_checkpoints 表 + CRUD | ✅ | `storage/world_state/checkpoints.py` + `import_all_models()` |
| 4 | Repo 支持 checkpoint restore + 清理 | ✅ | `SQLAlchemyRepo.restore_world_state_checkpoint()` + `clear_world_state_for_book()` |
| 5 | service 层 step loop + 对齐逻辑 | ✅ | `storyteller/service.py`（step_size>1 分支） |
| 6 | step 批量 narration prompt + node | ✅ | `prompts/step_narration.py` + `nodes/storyteller_generate_step.py` |
| 7 | embedding 合批检索 helper | ✅ | `embeddings/service.py::retrieve_hybrid_memories_batched`，step 模式已接入 |
| 8 | input_hash 纳入 base_world_state_hash | ✅ | step 模式 input_hash 已包含 `step_size/step_start/base_world_state_hash` |
| 9 | 单测/集成测试补齐 | 🟨 | 已新增 step_utils / checkpoint / CLI / config 测试；待补 E2E step 批量生成测试 |

## 实施结果（持续更新）

- 已实现（截至 2026-02-28）：
   - 配置：新增 `storyteller.step_*` 字段，默认 `step_size=1` 不改变旧行为。
   - CLI：`storytell/run` 支持 `--step-size` 覆盖。
   - Checkpoint：新增 `world_state_checkpoints` 表，支持 snapshot 与 restore；restore 会清空并重建 `plot_events/characters/items/world_facts`。
   - Step 主循环（Phase 2 版本）：
      - step 内 narration 生成使用同一基准 world_state（以 step_start 的 `state_lookup` 结果为基准预填充）。
      - 生成阶段使用 draft graph（不执行 `state_update`）。
      - step 结束后按章节顺序执行 `state_update` 并写 checkpoint（chapter_idx=step_end）。
   - Step narration 批量生成（Phase 3 版本）：
      - step 内会先逐章完成 `entity_extract` / `memory_retrieve`，随后对该 step 的章节**一次性批量生成** narration（解析失败则自动拆分重试，最小粒度退化为单章兜底）。
      - 为避免额外 LLM 往返，step 模式会对每章注入 `refine_enabled=False`（润色在批量 prompt 内完成/或直接关闭）。

   - Step memory 检索 embedding 合批（Phase 4 版本）：
      - step 内会收集每章的 retrieval query_text，并通过 `embed_documents` 一次性生成多向量。
      - 之后对每个向量分别做 LanceDB 相似检索 + FTS，再按原有融合打分返回每章的 `awakened_memories`。
      - 检索执行支持并发：step 内多个章节并发执行；且每章内 chunk/narration vector search 与 FTS 并发执行。

   - 断点续跑（增强版）：
      - 新增 `narration_outputs` 表持久化每章用于 `state_update` 的结构化输出（events/updates/items/entities）。
      - 在 restore 后补跑到 `S-1` 时，会优先 replay `narration_outputs` 直接重建 world_state，避免重复调用 LLM；缺失时才退回到“全图重跑”。
   - 幂等与覆盖：step 模式 narration 的 DB `input_hash` 已纳入 `base_world_state_hash`，切换 step_size 会产生新版本。

- 当前限制（下一步推进）：
   - （已解决）narration 已支持 step 内批量生成（Phase 3）。
   - 仍有可优化项：FTS/LanceDB 检索本身仍逐章执行（已消除 embedding 往返，但检索可进一步并发/批量化）。

- 性能数据：
  - step_size=1：
  - step_size=N：
- 一致性与正确性：
- 失败回退统计：

## 验收标准

1. `step_size=1` 时行为与现有版本一致（回归测试全过）。
2. `step_size>1` 时：
   - step 内所有章节使用同一基准态（来自 `S-1`）。
   - step 末写入 checkpoint，且可 restore 后继续生成。
3. 切换 `step_size` 重新运行同一范围时：
   - narration 会产生新版本（latest 导出自动取最新），不出现重复导出；
   - 若缺失 `S-1` checkpoint，会 fallback 补跑生成到 `S-1`。
4. plot_events 不发生“重复无限追加污染”（restore/清理策略生效）。

## 备注与风险

- Step 模式牺牲 step 内逐章状态推进：后续章节可能缺少前一章刚发生的状态（这是设计选择）。
- 批量输出 JSON 解析失败是高频风险：必须有“拆分重试 → 单章兜底”。
- `memory_retrieve` 合批仅解决 embedding 往返；LanceDB/FTS 检索本身仍需逐章执行（可后续再做并发/合并优化）。
- 如需更严格的一致性（逐章推进 world_state + batching），会显著复杂化 prompt 与落库逻辑，建议作为 Phase 2+ 研究项。
