# HMS — Hierarchical Memory Scaffold v2.0

> **让 AI 拥有真正的记忆 — 通过 LLM 驱动的认知记忆系统实现无限上下文**

## 🧠 核心理念

v1 使用硬编码词典做情感分析和实体提取——这就像用字典去理解人类思维。
v2 彻底重构：**所有认知分析全部交给 LLM**，系统只负责架构和调度。

## 🏗️ 架构概要

```
用户消息 → [即时感知-LLM] → 检索相关记忆 → 注入上下文
    ↓
助手回复 → [异步队列]
    ↓
[深度分析-LLM] → 碰撞检测 → 存储
    ↓
[定时巩固-LLM] → 摘要压缩 → 主题时间线 → 认知指纹
```

### 三层无限上下文压缩

```
原始对话 (无限)
    ↓  每N轮压缩
对话摘要 (紧凑)
    ↓  每周聚合
主题时间线 (极简)
    ↓  持续维护
认知指纹 (常驻)
```

**认知指纹** = 用户的思维模式、偏好、性格特征的动态画像
始终在 ~2000 token 以内，却能代表数月的交互历史。

## 📁 结构

```
hms-v2/
├── README.md
├── setup.sh
├── hms/
│   ├── config.json           # 系统配置
│   ├── prompts/              # LLM 提示词模板
│   │   ├── perceive.txt      # 感知分析提示
│   │   ├── collide.txt       # 碰撞检测提示
│   │   ├── consolidate.txt   # 巩固压缩提示
│   │   └── fingerprint.txt   # 认知指纹更新提示
│   ├── cache/                # 运行时缓存
│   │   ├── beliefs.json
│   │   ├── cognitive_fingerprint.json   # 🆕 认知指纹
│   │   ├── topic_timelines.json         # 🆕 主题时间线
│   │   ├── compression_history.json     # 🆕 压缩历史
│   │   ├── decay_state.json
│   │   ├── active_context.md
│   │   └── pending_processing.jsonl
│   ├── hooks/                # OpenClaw 钩子
│   ├── logs/
│   └── scripts/
│       ├── __init__.py
│       ├── models.py         # 数据结构 (保持不变)
│       ├── llm_analyzer.py   # 🆕 LLM 分析核心 (替代词典)
│       ├── perception.py     # 感知引擎 (v2 — LLM驱动)
│       ├── collision.py      # 碰撞引擎 (v2 — 语义级)
│       ├── context_manager.py # 上下文管理 (v2 — 流式压缩)
│       ├── forgetting.py     # 遗忘引擎 (优化)
│       ├── consolidation.py  # 巩固引擎 (v2 — LLM摘要)
│       ├── memory_manager.py # 统一调度 (v2)
│       └── test_e2e.py       # 端到端测试
```

## 🔑 v2 vs v1 关键区别

| 维度 | v1 | v2 |
|------|----|----|
| 情感分析 | 硬编码词典 ~50词 | LLM 语义理解 |
| 实体提取 | 正则 X总/经理 | LLM 深度NER |
| 意图检测 | 关键词匹配 | LLM 意图推理 |
| 碰撞检测 | 关键词重叠 | LLM 语义关联 |
| 概念抽象 | 频率聚类 | LLM 概念归纳 |
| 上下文 | 固定token分配 | 流式三层压缩 |
| 无限上下文 | ❌ | ✅ 认知指纹 |

## ⚙️ 工作原理

### 1. 即时感知 (< 1s, 同步)
收到消息后立即调用 LLM 做轻量分析（实体+情感+意图），
注入上下文让助手回复有记忆基础。

### 2. 深度分析 (异步)
回复完成后，将完整对话对写入 pending queue。
cron 任务批量调用 LLM 做深度分析：
- 信念提取与置信度评估
- 与已有记忆的碰撞检测
- 新实体/关系发现

### 3. 定时巩固 (每天凌晨3点)
- 选择高优先级记忆进行 LLM 回放
- 将连续对话压缩为摘要
- 更新主题时间线
- 发现深层语义关联

### 4. 认知指纹 (持续更新)
每次巩固后更新"认知指纹"——
一个紧凑的 JSON，捕获用户的：
- 思维模式和决策风格
- 核心偏好和价值观
- 情感触发点
- 长期目标和关注点

## 🔧 安装

```bash
cd hms-v2
bash setup.sh
```

## 📋 依赖

- Python 3.10+
- OpenClaw 已安装
- memory-lancedb-pro 插件
- graph-memory 插件

## ⚠️ 注意

- LLM 调用会产生少量 token 消耗
- 建议在 config.json 中配置 `llm_budget_tokens_per_day` 控制日消耗
- 智能遗忘功能会清理低价值记忆，请先备份
