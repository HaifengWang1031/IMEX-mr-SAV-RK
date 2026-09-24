# 历史索引与归档

[legacy-index.json](legacy-index.json) 是原位历史文件的元数据快照。按相对路径检索，记录文件大小与修改时间；不包含大数组，不推断参数、来源或成功状态。实验归属证据见 [实验清单](../docs/experiment-inventory.md)。

在仓库中运行 `python tools/index_legacy_artifacts.py` 可重新生成索引。只替换索引，不移动或修改原始产物。索引范围包括历史产物目录、根目录产物候选，以及 transition 中与源码混放的元数据候选；其中 JSON/Markdown 也可能是配置或说明，不应据索引删除。

只有确认停用的代码和 notebook 才迁入此目录，并在实验清单记录原路径、用途、替代入口及归档原因。2026-09-24 已按用户要求归档旧实验入口，见 legacy/migration.json；各文件字节内容不变，历史大数据仍原位保存。
