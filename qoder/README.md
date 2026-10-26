# 实验室周报助手（Lab Weekly Report Assistant）

> 2026 Qoder × OceanBase「Build with Qoder」参赛作品。代码位于 PowerContext 仓库 `2026-competition-ecosystem` 分支的 `qoder/` 目录。

## 项目简介

一个本地运行、无需 API Key 的组会周报助手：把实验室的长期上下文（关键决定、风险约束、本周进展、下周计划）写进 PowerContext，一条命令生成四段式周报页面，并附一条可逐行复核的调用证据链。

- 正文与引用全部来自 PowerContext `/v1/context/prepare`，模板只负责排版，不做 LLM 润色
- 检索走本地 SQLite + jieba 分词，不依赖任何外部模型服务
- 单进程启动：`app.py` 拉起 PowerContext Server、生成周报、启动页面，退出时回收子进程

## 解决的问题

组会周报的难点不是排版，而是“上周到底决定了什么、为什么放弃方案 B、显存上限是多少”这些信息散落在聊天记录里：

1. 决策与约束随对话丢失，下一个同学容易再踩同一个坑；
2. 手工整理周报时说不清“这句话出自哪次讨论”；
3. 直接让大模型代写容易编造，还需要 API Key 与联网。

本项目把记忆作为一等公民：按 `decision / constraint / progress / next-step` 四种 kind 存入 PowerContext，周报只做“取回 + 分区 + 附引用”，因此每条内容都能追溯到具体的 artifact 与 revision。

## 快速开始

Windows 验证环境：Windows 11 26x00、Python 3.12，无 Docker、无 WSL、无 API Key。在仓库根目录的 `qoder/` 下执行（本机验证目录为 `D:\qoder\pc-repo\qoder`）：

```bat
cd qoder
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv/Scripts/python.exe app.py
```

启动成功后会打印：

```
[pc]     http://127.0.0.1:8000  reused=False  startup=4822.36 ms
[report] D:\qoder\pc-repo\qoder\out\weekly_report.json
[web]    http://127.0.0.1:8080
```

然后打开：

- 周报首页：http://127.0.0.1:8080/
- 证据链：http://127.0.0.1:8080/evidence

按 Ctrl-C 退出：回收 PowerContext 子进程、释放 8000/8080（8001 未启用），并把 `data/server.json` 的 `running` 改写为 `false`。

> 也可以用 `python -m venv .venv` 建标准 venv；本机 `python` 指向 Microsoft Store 别名，执行时提示 “Python was not found”，因此最终改用 uv 管理的解释器（`uv venv --python 3.12 .venv` 同样可用）。镜像只在安装命令里临时指定，`requirements.txt` 不含镜像地址。

## 验证样例

`data/` 与 `out/` 是运行时目录（已 gitignore）。干净仓库首次运行会自动建库，并写入 `seed/lab_memories.json` 里的 10 条示例记忆，因此一次“黄金运行”的记录也包含 seeding：

- `samples/weekly_report.json`：干净运行的周报 JSON（10 条正文、8 次 prepare、`missing_bodies=0`、`trace` 与 events 完全一致）
- `samples/events.jsonl`：同一次运行的 21 行调用记录（1 次 list 空库 + 10 次 remember + 1 次 list + 1 次 search + 8 次 prepare）

本次干净复现的实测记录：

| 项目 | 结果 |
| --- | --- |
| 依赖安装（清华镜像，uv 缓存命中） | 4 s |
| 依赖安装（PyPI 官方源，本机） | 超过 600 s 仍未完成，已终止 |
| 首次冷启动（空 `data/`，首次建库） | 进程到页面 200：29.12 s；PowerContext 自身 22.15 s |
| 第二次冷启动（空 `data/`，系统缓存已热） | 进程到页面 200：9.64 s；PowerContext 自身 4.82 s |
| 首页 / 证据页 | 均 HTTP 200 |
| 8001 端口 | 全程未被占用（metrics 已关闭） |
| 中断退出 | 退出码 0；8000/8001/8080 全部释放；无 python 残留；`running=false` |

> 自动化环境无法向控制台进程组发送真正的 CTRL_C，验证时发送的是 CTRL_BREAK：两者在 uvicorn 里走同一条“捕获 → 优雅关闭 → 重放信号”路径，`app.py` 对 `SIGINT/SIGBREAK` 都做了接管。

## 技术结合说明

只使用 PowerContext 一种技术，且放在核心链路上，三条链路缺一不可：

1. **骨架** `/v1/atomic-memory/list`：拉全量 active 记忆，按 kind 分成四个固定段落；空库时先写入 seed 再重拉。
2. **焦点** `/v1/atomic-memory/search`：只有给出主题时才调用；问题先压缩成 1–2 个关键词（“我们上周为什么放弃方案B…” → `检查点 放弃`），避免整句触发 FTS 准入门槛而返回 0 条。
3. **正文** `/v1/context/prepare`：所有正文与 citation 的唯一来源。段级用固定两词（`本周 完成 / 决定 方案 / 显存 上限 / 下周 计划`）召回；未被覆盖的条目再用该条正文的前 2 个中文实词补一次（单字、停用词、纯英文丢弃），每条 item 标注 `source: section-prepare | item-prepare`。引用格式为 `atomic-memory/<artifact_id>@<revision>`。

进程模型：`app.py` 用 `subprocess` 启动 `powercontext server run --host 127.0.0.1 --port 8000 --no-env-file`，把 `POWERCONTEXT_HOME` 指向 `data/pc-home`（SQLite），轮询 `/health/ready` 后再启动 Web；端口被占用时向后自动探测；关闭 metrics（默认 8001 会与 HTTP 回退端口相撞）。uvicorn 关停后会重放信号，`app.py` 提前接管 `SIGINT/SIGBREAK`，保证重放时仍能回到 `finally` 回收子进程。

页面上的“同一 artifact 多处引用”是刻意标注：焦点由 search 定位、正文由 prepare 召回，引用同一条记忆属正常，不是重复数据。

## 已知限制

- **Windows 标注 experimental**：仅在 Windows 11 + Python 3.12 上实测；未验证 Linux/macOS，也不提供 Docker/WSL 方案。
- **SQLite 下 `--role api` 不可用**：会以 exit 2 退出（`artifact_processing_role must be 'all'`），因此 Server 一律使用默认的 `all`。
- **无模型时能力降级**：未配置推理模型时 PowerContext 的 prompt 类能力全部 disabled，检索降级为 FTS/text；周报完全由模板渲染，不做 LLM 润色。FTS 在查询词超过 2 个时要求命中 `max(2, ceil(词数*25%))` 个词，所以所有查询都被压到 2 个词。
- **国内网络建议使用镜像源**：本机用 PyPI 官方源安装 122 个包超过 10 分钟仍未完成；`requirements.txt` 不写镜像（写死会让无镜像环境装不上），镜像只在安装命令里临时指定。
- **依赖核对**：新装 venv 的 `uv pip freeze`（见 `req-lock.txt`）与探针 venv 的 `pip freeze` 逐包比对，122 个包版本完全一致；差异只有包名分隔符（如 `pydantic_core` / `pydantic-core`）与新 venv 自带的 `pip`。
- **周报只在启动时生成一次**：修改 `seed/lab_memories.json` 或记忆后需重启 `app.py`，页面不做热更新。
- **单人本地使用**：没有登录、权限与多用户隔离，服务只监听 `127.0.0.1`。

## 开源归属

- [PowerContext](https://github.com/oceanbase/powercontext) 1.4.0（Apache-2.0）：核心依赖，仅通过其 HTTP API 使用，未修改上游源码。
- jieba 0.42.1（MIT）：查询关键词切分。
- FastAPI（MIT）、uvicorn（BSD-3-Clause）、Jinja2（BSD-3-Clause）：本地页面与模板渲染。
- `seed/lab_memories.json` 为自拟示例数据，不含真实实验记录或个人信息；未复用其他项目的代码或数据。
