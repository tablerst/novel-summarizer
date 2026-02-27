# Storyteller Tiering 执行文档

- 创建时间：2026-02-27
- 负责人：Copilot
- 状态：Phase 1 ✅ 已完成；Phase 2（Tiering 基础能力）✅ 已完成；Phase 3（并发流水线基础版）✅ 已完成

## 背景与目标

本轮先落地“去重护栏”第一阶段，确保后续 short/medium/long 分层运行不会造成：

1. `full_story.md` 同章重复导出；
2. narration FTS 与向量索引混入同章多版本，导致检索污染。

## 本阶段范围（Phase 1）

- 新增 `latest narration` 查询路径（按 chapter 仅取最新版本）。
- 导出逻辑改为默认只消费 latest narration。
- narration FTS 重建改为只索引 latest narration。
- narration 向量嵌入改为只嵌入 latest narration。
- 补充回归测试，验证“同章多版本不重复导出/索引”。

## 本阶段范围（Phase 2）

- 引入按章 tier 决策与配置（short/medium/long）。
- 在 chapter loop 中注入 `tier` 与 `storyteller_overrides`。
- 节点读取覆盖参数（entity/memory/generate/refine）。
- 增加 `tiered` profile，支持单次运行分层策略。

## 本阶段范围（Phase 3）

- 新增 `prefetch_window` 配置并在 `tiered` profile 默认启用（2）。
- LLM 客户端提供 async 接口（`complete*_async`），通过 `to_thread + semaphore` 限流，减少 event loop 阻塞。
- 节点优先走 async LLM 接口。
- chapter loop 接入预取：提前预取未来 K 章的 `entity_extract + memory_retrieve`，主流程复用预取结果。
- 保持 world state 写入顺序不变（`state_lookup/state_update` 仍按章串行）。

## 任务清单与进度

| ID | 任务 | 状态 | 说明 |
|---|---|---|---|
| 1 | 创建执行追踪文档 | ✅ 已完成 | 本文件 |
| 2 | 实现 narration latest 查询 API | ✅ 已完成 | `storage/narrations/crud.py`, `storage/repo.py` |
| 3 | 导出逻辑切换为 latest | ✅ 已完成 | `export/markdown.py` |
| 4 | 索引构建切换为 latest | ✅ 已完成 | `storage/narrations/crud.py`, `embeddings/service.py` |
| 5 | 补充去重相关测试 | ✅ 已完成 | `tests/test_narrations_latest.py`, `tests/test_export_markdown.py`, `tests/test_embeddings_service_hybrid.py` |
| 6 | 测试验证与结果记录 | ✅ 已完成 | 全量 `75 passed` |
| 7 | 实现按章 tier 决策模块 | ✅ 已完成 | `storyteller/tiering.py`, `config/schema.py` |
| 8 | 节点支持 tier 覆盖参数 | ✅ 已完成 | `nodes/entity_extract.py`, `nodes/memory_retrieve.py`, `nodes/storyteller_generate.py`, `nodes/refine_narration.py` |
| 9 | service 注入 tier 与 hash | ✅ 已完成 | `storyteller/service.py`, `storyteller/state.py` |
| 10 | tier 相关测试覆盖 | ✅ 已完成 | `tests/test_storyteller_tiering.py` + 节点测试补充 |
| 11 | profile 落地 | ✅ 已完成 | `configs/profiles/tiered.yaml` |
| 12 | 并发流水线基础版落地 | ✅ 已完成 | `llm/factory.py`, `storyteller/service.py`, `storyteller/nodes/*` |
| 13 | 前 10 章实测（tiered） | ✅ 已完成 | `run --input ... --from 1 --to 10 --no-export` |

## 验收标准

- 对同一 `chapter_id` 存在多条 narration 时：
  - `export` 产物只出现该章 1 次；
  - `narrations_fts` 中该章只保留 latest 版本；
  - narration 向量嵌入只处理 latest 版本。

## 变更日志

### 2026-02-27 轮次 1

- 新建执行文档并初始化任务清单。
- 已确认当前重复风险根因：`narrations` 唯一约束包含 `input_hash`，切 profile/tier 会产生同章多版本。

### 2026-02-27 轮次 2（Phase 1 完成）

- 完成 latest narration 全链路落地：
  - `list_latest_narrations_by_book()` 查询 API。
  - storyteller 导出改为 latest。
  - narration 向量嵌入与 FTS 重建改为 latest。
- 修复 FTS `book_id` 参数类型问题（统一使用整数），避免计数/过滤异常。
- 新增并通过去重回归测试。

### 2026-02-27 轮次 3（Phase 2 完成）

- 引入 tiering 配置模型：`StorytellerTieringConfig` / `StorytellerTierProfileConfig`。
- 新增 `storyteller/tiering.py`：
  - `decide_tier()`（关键章启发式）
  - `build_tier_overrides()`（按章参数覆盖）
  - `effective_storyteller_value()`（节点统一读取覆盖值）
  - `has_storyteller_memory_retrieval()`（检索资产预构建判定）
- chapter loop 注入 `tier` 与 `storyteller_overrides`，并将其纳入 `input_hash`。
- 节点支持覆盖参数：entity/memory/generate/refine。
- 新增 `configs/profiles/tiered.yaml`。

### 测试结果（当前）

- 目标回归：`51 passed`
- 全量回归：`78 passed`

### 2026-02-27 轮次 4（Phase 3 基础版完成 + 前10章实测）

- 代码落地：
  - `storyteller.prefetch_window` 配置（默认 0，`tiered.yaml` 设为 2）。
  - `OpenAIChatClient` async 方法族（`complete_async / complete_json_async / complete_structured_async`）。
  - `storytell_book` 新增预取任务池：对未来窗口章节预取实体/记忆结果并复用。
  - `entity_extract`、`memory_retrieve` 增加“预填充短路”以避免重复计算。
- 修复：`embeddings/service.py` 曾出现文件损坏导致缩进错误，已整体重建并恢复。
- 实测命令：
  - `uv run python main.py --profile tiered run --input "data/玄界之门(1-500章).txt" --from-chapter 1 --to-chapter 10 --no-export`
- 实测结果：
  - `Storytell chapters processed`: 10
  - `LLM calls (est)`: 16
  - `Refine calls`: 3
  - `Runtime`: 1264.73s
  - `Cache hit/miss`: 0/16
