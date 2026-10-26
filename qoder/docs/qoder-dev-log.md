# Qoder 开发记录（索引）

本文件是"Qoder 开发记录"的索引：每个阶段做了什么、对应哪张截图或录屏、以及仓库内可直接核对的产物。

会话截图放在 `docs/screenshots/`，录屏放在 `docs/recordings/`，文件名固定为下表"会话证据"列。当前仓库尚未包含这些会话截图（需从 Qoder 会话记录中导出后按名放入，代码无需改动）；表中"仓库内可核对"列的产物均已入库。

| 阶段 | 做了什么 | 会话证据 | 仓库内可核对 |
| --- | --- | --- | --- |
| 需求拆解 | 定下只使用 PowerContext 一种技术：四段式周报、每条正文带 citation、单命令本地启动；明确不接 LLM 润色、不做用户权限 | `docs/screenshots/01-requirements.png` | `README.md` 的「解决的问题」「技术结合说明」 |
| 写 pc_client | 封装 `/v1/atomic-memory/list`、`/v1/atomic-memory/search`、`/v1/context/prepare`；中文分词取前 2 个实词；写 server.json 状态文件 | `docs/screenshots/02-pc-client.png` | `pc_client.py` |
| 调端口冲突 | 8001 metrics 与 8080 Web 端口冲突、8000 可能被旧进程占用；关闭 metrics、启动前探测空闲端口、Ctrl-C 时回收子进程并释放端口 | `docs/screenshots/03-port-conflict.png` | `app.py`、`README.md` 的「已知限制」 |
| 做证据页 | 设计 10 字段事件（ts/path/query/status/hits/items/content_bytes/latency_ms/artifact_ids）写入 `evidence/events.jsonl`，`/evidence` 回答"去掉 PowerContext 这条周报还生成得出来吗" | `docs/screenshots/04-evidence-page.png` | `evidence.py`、`templates/evidence.html`、`samples/screenshots/evidence.png` |
| 发站点 | 作品介绍页：简介、痛点、两张截图、本地运行声明、固定 commit 的仓库链接；公开发布并验证匿名可访问 | `docs/screenshots/05-site-publish.png` | 线上作品页与 `samples/screenshots/` |

补充说明：

- 上述截图只用于证明开发过程在 Qoder 中完成，不包含个人 Qoder UID、账号信息或 API Key。
- 复现方式与开发时一致：在 `qoder/` 目录按 `README.md` 的「快速开始」安装并运行 `app.py`，即可得到与截图相同的首页和 `/evidence` 页面。
