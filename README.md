## novel-summarizer

一个面向长篇文本的分层总结 CLI 工具。当前版本已完成 M0–M4：配置加载、章节解析与分块、分层总结、Markdown 导出与引用回溯（向量检索）。

### 快速开始

1. 准备环境变量

- 在项目根目录填好 `.env` 的 `OPENAI_API_KEY`。

2. 查看生效配置

```bash
novel-summarizer config
```

3. 选择 profile

```bash
novel-summarizer --profile fast config
```

4. 导入文本（M1）

```bash
novel-summarizer ingest --input path/to/novel.txt --title "书名" --author "作者"
```

5. 生成摘要（M2/M3，默认导出）

```bash
novel-summarizer summarize --book-id 1
```

6. 仅导出 Markdown（可选）

```bash
novel-summarizer export --book-id 1
```

7. 构建向量索引（M4，可选）

```bash
novel-summarizer embed --book-id 1
```

### 配置说明

- 默认配置位于 `configs/default.yaml`
- 可选 profile 位于 `configs/profiles/`
- 默认输出目录：`./output`（可通过 `app.output_dir` 或 CLI `--output-dir` 覆盖）
- SQLite 数据库默认路径：`./data/novel.db`
- LanceDB 向量库目录：`./data/lancedb`
- chunk/章节摘要会写入 `summaries` 表
- 全书摘要/人物表/时间线/说书稿会写入 `summaries` 表（scope=book）并导出到 `./output/<book_hash>/`
- `summarize.with_citations.enabled=true` 时会自动构建向量索引并注入证据片段
- 向量库访问改用 `langchain_community.vectorstores.LanceDB`
- 说书稿可通过 `summarize.story_words` 启用并控制字数范围

### 数据库与事务（SQLAlchemy）

- 当前数据库层已基于 **SQLAlchemy Async** 重构。
- 应用启动时会初始化数据库连接，并自动检查/创建表结构。
- 业务层通过 `session_scope()` 获取事务能力（异常自动回滚）。

### 存储层代码结构

- 每个表一个子包：`storage/<table>/base.py` 放模型定义，`storage/<table>/crud.py` 放 CRUD 方法。
- 兼容入口：`storage/repo.py` 仍提供聚合式仓储（内部委托到各表 CRUD）。

### 输出产物

- `./output/<book_hash>/book_summary.md`
- `./output/<book_hash>/characters.md`
- `./output/<book_hash>/timeline.md`
- `./output/<book_hash>/story.md`
- SQLite：`books/chapters/chunks/summaries`

### 说明

- 当前仅支持 OpenAI 作为 LLM 提供方。
- 引用功能由 `summarize.with_citations` 控制，启用后会使用 LanceDB 检索证据片段。
- `summarize` 默认导出 Markdown，如需仅生成摘要可加 `--no-export`。
