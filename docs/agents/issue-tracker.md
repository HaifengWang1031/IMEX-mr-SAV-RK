# Issue tracker: Local Markdown

任务与规格保存在当前仓库的 .scratch/ 下。

- 每项工作使用 .scratch/<feature-slug>/。
- 规格保存为 spec.md。
- 每张任务票独立保存为 issues/<NN>-<slug>.md，
  从 01 开始，按依赖顺序编号。
- 本次项目迁移使用 feature-slug：project-architecture。
- 每票包含 What to build、Blocked by、Status 和验收清单。
- 新发布任务的 Status 为 ready-for-agent；
  开始执行设为 claimed，验收完成设为 resolved。
- Blocked by 列出前置票编号与标题；无依赖时明确写 None。
- 只有所有前置票均为 resolved，任务才可开始。
- 评论与执行记录追加到票内的 Comments 小节。
- “发布任务”表示创建本地文件；读取任务须读取完整文件。
- 保留已有任务及执行历史，不覆盖同名任务文件。
