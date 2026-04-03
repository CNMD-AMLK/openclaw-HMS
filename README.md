# HMS — Hierarchical Memory Scaffold v4.0.0

> **让 AI 拥有真正的记忆 — OpenClaw 原生插件，分层记忆管理系统，模拟人类认知全过程**

## 🧠 核心理念

v1 使用硬编码词典做情感分析和实体提取。
v2 彻底重构：**所有认知分析全部交给 LLM**，系统只负责架构和调度。
v3 进化：**多档位上下文窗口支持 + Embedding 预过滤降本 60-70%**。
**v4 重大升级：OpenClaw 原生插件 + 重构性回忆 + 梦境整合引擎 + 创造性联想 + 记忆覆盖机制。**

## 🚀 快速开始

### 安装

```bash
cd /path/to/openclaw-HMS
bash setup.sh
```

**零外部依赖**：HMS 自带本地文件存储，不依赖任何第三方记忆插件。

- 有 OpenClaw `memory-lancedb` 插件 → 自动对接
- 没有 → 自动降级到本地文件存储（`cache/` 目录）
- 有 `graph-memory` 插件 → 知识图谱自动兼容

### 作为 OpenClaw 插件加载（v4 推荐方式）

```python
from hms import HMSPlugin

plugin = HMSPlugin()
info = plugin.register({})  # 注册到 OpenClaw

# 插件自动提供以下工具：
# - hms_perceive     : 即时感知 + 记忆上下文
# - hms_collide      : 碰撞检测
# - hms_recall       : 重构性回忆
# - hms_consolidate  : 触发整合周期
# - hms_context_inject: 完整记忆注入上下文
```

### 配置敏感参数

Setup 后会在 `hms/` 目录下自动生成 `.env` 文件：

```bash
cp .env.example hms/.env
nano hms/.env
```

**必填项：**

| 变量 | 说明 | 示例 |
|------|------|------|
| `HMS_GATEWAY_URL` | OpenClaw Gateway 地址 | `http://127.0.0.1:18789` |
| `HMS_GATEWAY_TOKEN` | Gateway 认证令牌 | `b69815cbda4476f...` |
| `HMS_LLM_MODEL` | LLM 模型名称 | `openclaw` |

### CLI 用法（向后兼容）

```bash
python -m hms received "你好" --tier 256k
python -m hms process_pending
python -m hms consolidate
python -m hms forget
python -m hms health
```

### 健康检查

```bash
python -m hms health
```

## 🔑 v4 新增功能

### 重构性回忆 (Reconstructive Recall)
不是精确检索记忆，而是像人类一样**重建记忆**——根据检索到的碎片和当前上下文，由 LLM 综合重建最可能的答案。

```python
from hms.scripts.reconstructive_recall import ReconstructiveRecaller
recaller = ReconstructiveRecaller()
result = recaller.recall("我们上次聊了什么？", context=your_perception)
# result: {answer: "...", confidence: 0.75, label: "reconstructed"}
```

### 梦境整合引擎 (Dream Engine)
灵感来自人类睡眠中的记忆整合。梦境引擎在后台随机游走记忆图谱，发现远距离关联，产出洞察：

```python
from hms.scripts.dream_engine import DreamEngine
engine = DreamEngine()
report = engine.run_dream_cycle()
# 产出写入 cache/insights/ 目录
```

### 创造性联想 (Creative Association)
在不同主题之间发现非显而易见的连接，支持多跳图遍历和 LLM 评估：

```python
from hms.scripts.creative_assoc import CreativeAssociator
assoc = CreativeAssociator()
result = assoc.creative_link("编程", "音乐")
# 发现通过图谱多跳连接的两个领域
```

### 记忆覆盖机制 (MemoryOverwriter)
当新证据与旧信念矛盾时，**标记为"已被替代"而非删除**——保留历史上下文，降低旧置信度：

```python
from hms.scripts.forgetting import MemoryOverwriter
ow = MemoryOverwriter()
result = ow.handle_conflict(old_belief, new_evidence)
# result.action: "superseded" | "keep_old" | "no_conflict" | "downgraded"
```

## ⚙️ v4 精简配置

config.json 从 50+ 项精简到 10 项关键配置：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `perception_mode` | string | `lite` | 感知模式（lite/full） |
| `context_tier` | string | `auto` | 上下文档位（auto/32k/128k/256k/1m） |
| `embedding_model` | string | `local` | Embedding 模型（local/sentence-transformers） |
| `token_budget_daily` | number | `50000` | 每日 token 预算 |
| `collision_threshold` | number | `0.7` | 碰撞检测阈值 |
| `consolidate_interval` | string | `0 3 * * *` | 整合 cron 表达式 |
| `forget_interval` | string | `0 4 * * 0` | 遗忘 cron 表达式 |
| `gateway_url` | string | `http://127.0.0.1:18789` | Gateway 地址 |

其余配置使用代码内置的合理默认值——**零配置开箱即用**。

## 🏗️ 架构概要

```
用户消息 → [HMSPlugin.on_message] → [即时感知-启发式] → 检索 + 重建
    ↓                                                          ↓
    │                                                   重构性回忆
    ↓                                                          ↓
助手回复 → [写入 pending 队列] → [心跳触发 process_pending]
    ↓
[Embedding预过滤] → [深度分析-LLM] → 碰撞检测 → 存储
    ↓
[定时巩固-LLM] → 梦境引擎 → 创造性联想 → 压缩 → 指纹
    ↓
[覆盖机制] → 遗忘引擎 → 更新衰减状态
```

## ⚠️ 注意

- LLM 调用会产生 token 消耗，建议配置 `token_budget_daily`
- Embedding 缓存会占用磁盘空间，可在 `cache/` 目录清理
- 梦境引擎会在 `cache/insights/` 产出洞察文件
- 推荐安装 `sentence-transformers` 以获得更高质量的 Embedding

## 📁 项目结构

```
openclaw-HMS/
├── README.md
├── VERSION              → v4.0.0
├── CHANGELOG.md         → v4 变更日志
├── setup.sh
├── requirements.txt
├── .env.example
├── hms/
│   ├── __init__.py      → 包入口 (v4: 暴露 HMSPlugin)
│   ├── __main__.py      → CLI 入口 (向后兼容)
│   ├── plugin.py        → v4 新增: OpenClaw 原生插件
│   ├── plugin_manifest.json → v4 新增: 插件清单
│   ├── config.json      → v4 精简配置 (~10 项)
│   ├── hooks/
│   │   └── __init__.py  → Cron/Skill 接口 (向后兼容)
│   ├── scripts/
│   │   ├── __init__.py  → 暴露所有模块
│   │   ├── memory_manager.py  → 统一调度 (v4: 工具注入优先)
│   │   ├── reconstructive_recall.py  → v4 新增: 重构性回忆
│   │   ├── dream_engine.py           → v4 新增: 梦境整合引擎
│   │   ├── creative_assoc.py         → v4 新增: 创造性联想
│   │   ├── forgetting.py             → v4 新增: MemoryOverwriter
│   │   ├── config_loader.py          → v4: 内置完整默认值
│   │   └── ... (其他 v3 模块不变)
└── integration/
    ├── openclaw_config_example.json
    ├── heartbeat_example.md
    └── cron_example.yaml
```

## 🔑 v4 vs v3 关键区别

| 维度 | v3 | v4 |
|------|----|----|
| 插件接口 | CLI + hooks 模块 | **OpenClaw 原生插件 (HMSPlugin)** |
| 回忆方式 | 精确检索 | **重构性回忆 (LLM 综合重建)** |
| 记忆整合 | 仅定时压缩 | **梦境引擎 + 创造性联想** |
| 信念冲突 | 覆盖/删除 | **superseded 标记 + 渐进降级** |
| 配置文件 | 50+ 项 | **~10 项 (其余内置默认值)** |
| MemoryAdapter | tool → HTTP → stub 三层 | **直接工具注入 + HTTP 可选** |
| 外部依赖 | 需要 memory-lancedb 或 graph-memory | **零外部依赖，完全自给自足** |
