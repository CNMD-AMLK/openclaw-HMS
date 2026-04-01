# HMS — Hierarchical Memory Scaffold v3.0.2

> **让 AI 拥有真正的记忆 — 通过 LLM 驱动的认知记忆系统实现无限上下文**

## 🧠 核心理念

v1 使用硬编码词典做情感分析和实体提取——这就像用字典去理解人类思维。
v2 彻底重构：**所有认知分析全部交给 LLM**，系统只负责架构和调度。
v3 进化：**多档位上下文窗口支持 + Embedding 预过滤降本 60-70%**。
v3.0.2 修复：**Gateway集成 + 配置化 + 代码去重 + 提示词优化**。

## 🏗️ 架构概要

```
用户消息 → [即时感知-LLM] → 检索相关记忆 → 注入上下文
    ↓
助手回复 → [异步队列]
    ↓
[Embedding预过滤] → [深度分析-LLM] → 碰撞检测 → 存储
    ↓
[定时巩固-LLM] → 摘要压缩 → 主题时间线 → 认知指纹
```

### 多档位上下文窗口

| 档位 | 上下文 | 适用场景 |
|------|--------|----------|
| 32k | 32,000 tokens | 低成本/简单对话 |
| 128k | 128,000 tokens | 中等复杂度 |
| 256k (默认) | 256,000 tokens | 高复杂度深度记忆 |
| 1M | 1,048,576 tokens | 超长会话/全历史回溯 |

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
始终在 ~4000 token 以内，却能代表数月的交互历史。

## 📁 结构

```
hms/
├── README.md
├── setup.sh
├── config.json              # 系统配置 (支持多档位)
├── VERSION                 # 版本号
├── prompts/                # LLM 提示词模板
│   ├── perceive.txt        # 感知分析提示
│   ├── collide.txt         # 碰撞检测提示
│   ├── consolidate.txt     # 巩固压缩提示
│   └── fingerprint.txt     # 认知指纹更新提示
├── cache/                  # 运行时缓存
│   ├── beliefs.json
│   ├── cognitive_fingerprint.json   # 认知指纹
│   ├── topic_timelines.json         # 主题时间线
│   ├── compression_history.json     # 压缩历史
│   ├── decay_state.json
│   ├── embedding_cache.json         # v3 新增: Embedding 缓存
│   └── pending_processing.jsonl
├── hooks/                  # OpenClaw 钩子 (v3.0.2 新增)
│   └── __init__.py         # 钩子接口实现
├── logs/
└── scripts/
    ├── __init__.py
    ├── models.py           # 数据结构
    ├── llm_analyzer.py      # LLM 分析核心 (含 circuit breaker)
    ├── embed_cache.py      # v3 新增: Embedding 预过滤
    ├── file_utils.py       # v3 新增: 文件锁/原子写入
    ├── perception.py        # 感知引擎
    ├── collision.py         # 碰撞引擎 (含 Embedding 预过滤)
    ├── context_manager.py   # 上下文管理
    ├── forgetting.py        # 遗忘引擎
    ├── consolidation.py     # 巩固引擎 (含 Embedding 聚类)
    ├── memory_manager.py   # 统一调度
    └── test_e2e.py         # 端到端测试 (40 tests)
```

## 🔑 v3 vs v2 关键区别

| 维度 | v2 | v3 |
|------|----|----|
| 上下文窗口 | 固定 32k | 4 档位可调 (32k/128k/256k/1M) |
| LLM 调用成本 | 全量调用 | Embedding 预过滤，降本 60-70% |
| 并发安全 | 无锁 | fcntl 文件锁 + 原子写入 |
| 错误处理 | 固定重试 | 指数退避 + Circuit Breaker |
| Token 估算 | len//2 | 中英文区分估算 |
| 测试覆盖 | 28 tests | 40 tests + Mock 集成 |

## 📋 v3.0.2 更新内容 (2026-04-01)

### 🔧 关键修复
- **Gateway集成**：LLM调用改为通过OpenClaw Gateway，不再直连外部API
- **配置化**：Gateway地址支持配置文件和环境变量 `OPENCLAW_GATEWAY_URL`
- **代码去重**：`forgetting.py` 统一使用 `models.py` 的 `DecayState.calculate_strength()`
- **提示词优化**：`perceive.txt` 添加4个few-shot示例，提升LLM输出稳定性
- **Fallback修复**：`_fallback_compress()` 现在正确提取实体和主题
- **OpenClaw集成**：新增 `hooks/` 目录实现钩子接口

### 📁 新增文件
- `hms/hooks/__init__.py` - OpenClaw钩子实现

### ⚙️ 配置变更
- `config.json` 新增 `gateway_url` 配置项

### 🐛 问题修复
- 修复LLM调用绕过Gateway的致命缺陷
- 修复Gateway地址硬编码问题
- 修复Fallback压缩创建空集合但不填充的问题
- 改进bare except异常处理（添加日志记录）

## ⚙️ 工作原理

### 1. 即时感知 (< 1s, 同步)
收到消息后：
- Embedding 预过滤：快速筛选相关记忆
- 轻量 LLM 分析（实体+情感+意图）
- 注入上下文让助手回复有记忆基础

### 2. 深度分析 (异步)
回复完成后，将完整对话对写入 pending queue。
- Embedding 相似度预过滤（本地计算，零 API 成本）
- 批量调用 LLM 做深度分析
- 碰撞检测：新记忆 vs 相关记忆

### 3. 定时巩固 (每天凌晨3点)
- 选择高优先级记忆进行 LLM 回放
- 将连续对话压缩为摘要
- Embedding 聚类发现深层关联
- 更新主题时间线和认知指纹
- 同步 decay_state 与 memory store 一致性

### 4. 遗忘引擎 (每周日凌晨4点)
- Ebbinghaus 曲线计算记忆强度
- 动态阈值（重要性 + 记忆类型加权）
- 不死记忆保护（高重要性/高置信度）

## 🚀 快速开始

### 安装
```bash
cd hms
bash setup.sh
```

### 选择上下文档位
```bash
# 默认 256k
python -m hms.scripts.memory_manager received "你好" --tier 256k

# 32k 低成本模式
python -m hms.scripts.memory_manager received "你好" --tier 32k

# 1M 超长会话
python -m hms.scripts.memory_manager received "你好" --tier 1M
```

### 配置
编辑 `config.json` 中的 `context_tiers` 自定义各档位参数。

## ⚠️ 注意

- LLM 调用会产生 token 消耗，建议配置 `llm_budget_tokens_per_day`
- Embedding 缓存会占用磁盘空间，可在 `cache/` 目录清理
- 智能遗忘功能会清理低价值记忆，请先备份
- 推荐安装 `sentence-transformers` 以获得更高质量的 Embedding：
  ```bash
  pip install sentence-transformers
  ```

## 📋 依赖

- Python 3.10+
- OpenClaw 已安装
- memory-lancedb-pro 插件
- graph-memory 插件
- (可选) sentence-transformers 用于高质量 Embedding
