# HMS — Hierarchical Memory Scaffold v3.3.1

> **让 AI 拥有真正的记忆 — 分层记忆管理系统，模拟人类认知过程**

## 🧠 核心理念

v1 使用硬编码词典做情感分析和实体提取——这就像用字典去理解人类思维。
v2 彻底重构：**所有认知分析全部交给 LLM**，系统只负责架构和调度。
v3 进化：**多档位上下文窗口支持 + Embedding 预过滤降本 60-70%**。
v3.0.2 修复：**Gateway集成 + 配置化 + 代码去重 + 提示词优化**。
v3.0.3 修复：**全面代码质量改进 - 异常处理 + 并发安全 + 性能优化**。
v3.0.4 修复：**同步路径阻塞 + 原子队列 + 脏标记批写 + 连接池 + 认知指纹限容 + 情感极性碰撞**。
v3.0.5 修复：**架构对齐 README + 资源清理完善 + 并发安全增强 + 代码去重**。
v3.2.0 重大更新：**中文分词支持 + 记忆去重 + 健康检查 + 情感提取 + 断路器持久化 + pending 队列保护**。
v3.3.0 适配修复：**OpenClaw Gateway API 路径适配 + 认证令牌支持 + 端口/模型配置修正**。

## 🏗️ 架构概要

```
用户消息 → [即时感知-启发式] → 检索相关记忆 → 注入上下文
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
openclaw-HMS/                    # 项目根目录
├── README.md
├── setup.sh
├── config.json                  # 系统配置 (支持多档位)
├── VERSION                      # 版本号
├── requirements.txt             # Python 依赖
├── hms/
│   ├── __init__.py              # 包初始化 (版本号)
│   ├── __main__.py              # v3.0.4 新增: CLI 入口 (python -m hms)
│   ├── logging_config.py        # 日志配置
│   ├── config.json              # 系统配置 (支持多档位)
│   ├── prompts/                 # LLM 提示词模板
│   │   ├── perceive.txt         # 感知分析提示
│   │   ├── collide.txt          # 碰撞检测提示
│   │   ├── consolidate.txt      # 巩固压缩提示
│   │   └── fingerprint.txt      # 认知指纹更新提示
│   ├── cache/                   # 运行时缓存
│   │   ├── beliefs.json
│   │   ├── cognitive_fingerprint.json   # 认知指纹
│   │   ├── topic_timelines.json         # 主题时间线
│   │   ├── compression_history.json     # 压缩历史
│   │   ├── decay_state.json
│   │   ├── embedding_cache.bin          # v3 新增: Embedding 缓存 (二进制)
│   │   └── pending_processing.jsonl
│   ├── hooks/                   # OpenClaw 集成接口
│   │   └── __init__.py          # Cron/Skill 兼容接口 (含 atexit 清理)
│   ├── logs/
│   └── scripts/
│       ├── __init__.py
│       ├── models.py            # 数据结构 (含 Laplace 平滑)
│       ├── utils.py             # 公共工具 (estimate_tokens 等)
│       ├── file_utils.py        # 文件锁/原子写入 (含锁 FD 缓存 + Windows 兼容)
│       ├── llm_analyzer.py      # LLM 分析核心 (含 circuit breaker + Session 池)
│       ├── embed_cache.py       # v3 新增: Embedding 预过滤 (含二进制缓存)
│       ├── perception.py        # 感知引擎 (同步路径仅启发式)
│       ├── collision.py         # 碰撞引擎 (含情感极性判断 + Embedding 预过滤)
│       ├── context_manager.py   # 上下文管理 (含指纹限容 + 原子 pop_all)
│       ├── forgetting.py        # 遗忘引擎 (含脏标记批写)
│       ├── consolidation.py     # 巩固引擎 (含 general fallback + Embedding 聚类)
│       ├── memory_manager.py    # 统一调度 (含 Session 连接池 + 资源清理)
│       └── test_e2e.py          # 端到端测试 (44 tests)
```

## 🔑 v3 vs v2 关键区别

| 维度 | v2 | v3 |
|------|----|----|
| 上下文窗口 | 固定 32k | 4 档位可调 (32k/128k/256k/1M) |
| LLM 调用成本 | 全量调用 | Embedding 预过滤，降本 60-70% |
| 并发安全 | 无锁 | fcntl 文件锁 + 原子写入 + 锁 FD 缓存 |
| 错误处理 | 固定重试 | 指数退避 + Circuit Breaker |
| Token 估算 | len//2 | 中英文区分估算 |
| 测试覆盖 | 28 tests | 44 tests + Mock 集成 |

## 📋 v3.2.0 更新内容 (2026-04-01)

### 🔴 P0 — 关键修复

#### 1. 中文分词支持
- `utils.py`: 新增 `tokenize()` 函数，支持 jieba 分词（自动检测可用）
- 对中文文本使用 jieba 分词，ASCII 文本使用空格分割，混合文本使用 char bigram fallback
- `consolidation.py`: `replay_memory()` 中的 Jaccard 相似度计算从 `text.split()` 改为 `tokenize(text)`
- `embed_cache.py`: `CharNGramEncoder.encode()` 对中文文本使用 jieba token 作为特征
- **修复了中文相似度判断完全失效的致命 bug**

#### 2. 记忆去重机制
- `memory_manager.py`: 新增 `MemoryAdapter.store_with_dedup()` 方法
- 存储前通过 embedding 相似度检查（默认阈值 0.95），高度相似的更新而非新增
- `config.json`: 新增 `dedup_similarity_threshold` 配置项
- **防止同一事实被多次存储，节省召回配额**

#### 3. 系统健康检查
- `llm_analyzer.py`: 新增 `health_check()` 方法，验证 Gateway 连通性和 chat API 可用性
- `memory_manager.py`: 新增 `MemoryAdapter.health_check()` 和 `MemoryManager.health_check()`
- CLI 新增 `python -m hms health` 命令
- **启动时可验证所有依赖是否正常**

### 🟡 P1 — 功能改进

#### 4. 情感时刻提取（Fallback 路径）
- `consolidation.py`: `_fallback_compress()` 新增 `_EMOTION_PATTERNS` 情感关键词检测
- 支持高唤醒度情感（太/很/非常 + 情感词）、重复标点（!!/！！）、正/负面情感词
- **Fallback 模式下认知指纹的情感维度不再为空**

#### 5. 否定词检测改进
- `consolidation.py`: 新增 `_has_negation()` 函数，使用正则避免"不错"/"没问题"等误判
- `collision.py`: 碰撞检测中的否定判断引入 `tokenize()` 提高准确性
- **降低情感极性误判率**

#### 6. Token 预算耗尽日志
- `llm_analyzer.py`: 预算耗尽时输出 WARNING 日志，明确标注降级原因
- **用户可感知系统何时退化到启发式模式**

#### 7. 断路器状态持久化
- `llm_analyzer.py`: 新增 `_save_circuit_breaker_state()` / `_load_circuit_breaker_state()`
- 断路器状态保存到 `cache/circuit_breaker.json`，进程重启后恢复
- **避免每次重启都重新经历故障周期**

#### 8. Pending 队列大小限制
- `context_manager.py`: 新增 `max_pending_size` 配置（默认 1000）
- `enqueue()` 超限时自动丢弃最旧的 20% 条目并输出 WARNING
- **防止 cron 配置错误导致磁盘占满**

### 🟢 P2 — 优化项

#### 9. 压缩摘要动态数量
- `context_manager.py`: `_format_compressed_summaries()` 移除硬编码 `[-5:]` 限制
- 改为传入所有时间窗口内的摘要，由 `truncate_to_tokens()` 按预算截断
- **更充分利用上下文预算**

#### 10. Clustering 全量比较
- `embed_cache.py`: `cluster_by_similarity()` 移除 `i+30` 相邻限制
- 改为全量比较（上限 5000 对），避免漏掉远距离相似项
- **提高关系发现准确率**

#### 11. 1M 档位预算调整
- `config.json`: 1M 档位调整各层比例，`compressed_summaries_ratio` 从 0.03 降至 0.02
- `recent_turns_ratio` 从 0.40 降至 0.35，更合理分配预算

#### 12. 新增 jieba 依赖
- `requirements.txt`: 添加 `jieba>=0.42.1` 作为核心依赖
- 中文分词质量显著提升，fallback embedding 语义区分度改善

## 📋 v3.2.0 更新内容 (2026-04-01)

### 🔴 P0 — 关键修复

#### 1. 中文分词支持
- `utils.py`: 新增 `tokenize()` 函数，支持 jieba 分词（自动检测可用）
- 对中文文本使用 jieba 分词，ASCII 文本使用空格分割，混合文本使用 char bigram fallback
- `consolidation.py`: `replay_memory()` 中的 Jaccard 相似度计算从 `text.split()` 改为 `tokenize(text)`
- `embed_cache.py`: `CharNGramEncoder.encode()` 对中文文本使用 jieba token 作为特征
- **修复了中文相似度判断完全失效的致命 bug**

#### 2. 记忆去重机制
- `memory_manager.py`: 新增 `MemoryAdapter.store_with_dedup()` 方法
- 存储前通过 embedding 相似度检查（默认阈值 0.95），高度相似的更新而非新增
- `config.json`: 新增 `dedup_similarity_threshold` 配置项
- **防止同一事实被多次存储，节省召回配额**

#### 3. 系统健康检查
- `llm_analyzer.py`: 新增 `health_check()` 方法，验证 Gateway 连通性和 chat API 可用性
- `memory_manager.py`: 新增 `MemoryAdapter.health_check()` 和 `MemoryManager.health_check()`
- CLI 新增 `python -m hms health` 命令
- **启动时可验证所有依赖是否正常**

### 🟡 P1 — 功能改进

#### 4. 情感时刻提取（Fallback 路径）
- `consolidation.py`: `_fallback_compress()` 新增 `_EMOTION_PATTERNS` 情感关键词检测
- 支持高唤醒度情感（太/很/非常 + 情感词）、重复标点（!!/！！）、正/负面情感词
- **Fallback 模式下认知指纹的情感维度不再为空**

#### 5. 否定词检测改进
- `consolidation.py`: 新增 `_has_negation()` 函数，使用正则避免"不错"/"没问题"等误判
- `collision.py`: 碰撞检测中的否定判断引入 `tokenize()` 提高准确性
- **降低情感极性误判率**

#### 6. Token 预算耗尽日志
- `llm_analyzer.py`: 预算耗尽时输出 WARNING 日志，明确标注降级原因
- **用户可感知系统何时退化到启发式模式**

#### 7. 断路器状态持久化
- `llm_analyzer.py`: 新增 `_save_circuit_breaker_state()` / `_load_circuit_breaker_state()`
- 断路器状态保存到 `cache/circuit_breaker.json`，进程重启后恢复
- **避免每次重启都重新经历故障周期**

#### 8. Pending 队列大小限制
- `context_manager.py`: 新增 `max_pending_size` 配置（默认 1000）
- `enqueue()` 超限时自动丢弃最旧的 20% 条目并输出 WARNING
- **防止 cron 配置错误导致磁盘占满**

### 🟢 P2 — 优化项

#### 9. 压缩摘要动态数量
- `context_manager.py`: `_format_compressed_summaries()` 移除硬编码 `[-5:]` 限制
- 改为传入所有时间窗口内的摘要，由 `truncate_to_tokens()` 按预算截断
- **更充分利用上下文预算**

#### 10. Clustering 全量比较
- `embed_cache.py`: `cluster_by_similarity()` 移除 `i+30` 相邻限制
- 改为全量比较（上限 5000 对），避免漏掉远距离相似项
- **提高关系发现准确率**

#### 11. 1M 档位预算调整
- `config.json`: 1M 档位调整各层比例，`compressed_summaries_ratio` 从 0.03 降至 0.02
- `recent_turns_ratio` 从 0.40 降至 0.35，更合理分配预算

#### 12. 新增 jieba 依赖
- `requirements.txt`: 添加 `jieba>=0.42.1` 作为核心依赖
- 中文分词质量显著提升，fallback embedding 语义区分度改善

## 📋 v3.3.0 更新内容 (2026-04-01)

### 🔴 P0 — OpenClaw Gateway API 适配

#### 1. Chat Completions 路径修正
- `llm_analyzer.py`: `/api/v1/chat/completions` → `/v1/chat/completions`
- OpenClaw Gateway 的 OpenAI 兼容端点在 `/v1/` 而非 `/api/v1/`
- **修复前所有 LLM 调用返回 404，修复后正常对话**

#### 2. Health Check 路径修正
- `llm_analyzer.py`: `/api/v1/health` → `/health`
- `memory_manager.py`: `/api/v1/health` → `/health`
- OpenClaw Gateway 的健康检查在根路径

#### 3. Gateway 认证令牌支持
- `llm_analyzer.py`: 新增 `gateway_token` 配置，自动设置 `Authorization: Bearer` 头
- `memory_manager.py`: 同步新增 `gateway_token` 支持
- **没有 token 时内部 API 返回 401 Unauthorized**

#### 4. 默认端口和模型修正
- `config.json`: `gateway_url` 从 `3578` 改为 `18789`
- `config.json`: `llm_model` 从 `__current__` 改为 `openclaw`
- `config.json`: 新增 `gateway_token` 字段
- `memory_manager.py`: 默认端口从 `3578` 改为 `18789`
- **修复前 Connection Refused 或 Invalid model 错误**

## 📋 v3.0.4 更新内容 (2026-04-01)

### 🔴 P0 — 必须修复

#### 1. 同步路径 LLM 阻塞修复
- `perception.py`: 同步路径 (`on_message_received`) 不再调用阻塞式 LLM
- 默认走启发式 fallback，LLM 分析全部移至异步队列 (`process_pending`)
- 用户体感延迟从 30s+ 降至 < 1s

#### 2. pending_queue 原子读写
- `context_manager.py`: 新增 `pop_all_pending()` 原子操作
- 读取 + 截断在同一锁内完成，防止并发重复处理
- `memory_manager.py`: `process_pending()` 改用 `pop_all_pending()`

#### 3. forgetting.py 脏标记批量写盘
- `update_on_access()` / `update_on_reinforce()` 不再每次全量写盘
- 引入 `_dirty` 标记，由调用方在批次结束时统一 `flush()`
- 高频召回场景下 I/O 降低 90%+

#### 4. hooks/__init__.py 集成接口重写
- 移除不存在的 `openclaw hook register` 命令引用
- 改为 Cron + Skill/Plugin 双模式集成文档
- 新增 `process_pending()` / `consolidate()` / `forget()` 导出函数

### 🟡 P1 — 建议修复

#### 5. MemoryAdapter 连接池
- 使用 `requests.Session()` 替代每次新建 TCP 连接
- 启用 HTTP keep-alive，批量处理时减少握手开销

#### 6. estimate_tokens() 去重
- 已从 `llm_analyzer.py` 和 `context_manager.py` 统一提取至 `utils.py`

#### 7. fallback_perceive 正则优化
- 使用 `re.match()` 替代裸 `in` 判断
- 闲聊检测加入长度限制 (`len(msg) <= 8`)，避免误判
- 意图识别使用更精确的正则模式

#### 8. consolidation fallback 通用主题
- 无法匹配预定义主题时自动标记为 `"general"`
- 避免非技术对话全部归入空主题

#### 9. 不死记忆保护加强
- `_is_immortal()` 增加证据数量门槛 (`evidence_count >= 3`)
- 新增 `update_confidence_laplace()` 方法，使用 Laplace 平滑 `(ev+1)/(total+2)`
- 防止单条证据即变为不死记忆

#### 10. collision 情感极性判断
- `_heuristic_collision()` 加入正/负面情感词表
- 情感极性相反时标记为矛盾而非强化
- 显著降低误报率

### 🟢 P2 — 优化项

#### 11. 认知指纹限容
- `update_fingerprint()` 为每个列表设置上限 (默认 10)
- 超出时淘汰最旧条目，防止无限膨胀

#### 12. retrieval_top_k 与 collision cap 统一
- `llm_analyzer.py` 移除硬编码 `[:10]` 截断
- 现在处理所有召回的记忆，不再丢弃后 20 条

#### 13. file_utils.py 锁 FD 缓存
- `_get_lock_fd()` 缓存锁文件描述符，进程生命周期内复用
- 减少高频场景下的 `os.open()` / `os.close()` 开销

#### 14. test_e2e.py 新增测试
- 新增 `pop_all_pending` / `fingerprint_cap` / `dirty_flag` / `fallback_general` 测试
- 总计 44 tests (原 40)

#### 15. 新增 __main__.py
- 支持 `python -m hms received "消息"` 直接调用
- 无需再写完整路径 `python -m hms.scripts.memory_manager`

---

## 📋 v3.0.3 更新内容 (2026-04-01)

### 🔧 异常处理改进
- 消除所有 bare except，替换为具体异常类型或日志记录
- 17处异常处理添加 `logger.debug` 记录
- `consolidation.py`: `except Exception` → 具体类型 + 日志
- `forgetting.py`: `except Exception` → `json.JSONDecodeError/ValueError`
- `memory_manager.py`: 7处裸异常全部添加详细日志
- `context_manager.py`: `except Exception` → `ValueError`
- `collision.py`: 添加异常日志记录

### 🔒 并发安全增强
- `context_manager.py`: `get_pending_count()` 添加 `file_lock` 保护
- `embed_cache.py`: `save_cache()` 使用 `file_lock` + `atomic_write_json`
- 所有文件读写操作统一使用锁机制

### ⚡ 性能优化
- `embed_cache.py`: 添加 `max_cache_size` (10000) 限制
- 实现 LRU 淘汰机制，防止内存无限增长
- 提取 `estimate_tokens` 到公共 `utils.py` 消除代码重复

### 🧹 代码质量
- 移除未使用的 `import requests` (memory_manager.py)
- 所有模块统一添加 `logging` 支持
- 修复 `forget()` 方法逻辑缺陷 (memory_manager.py)
- 修复 `embed_cache` 并发写入数据损坏问题

---

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
- 启发式分析（实体+情感+意图），不调用 LLM
- 检索相关记忆并更新访问计数
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
- 不死记忆保护（高重要性/高置信度 + 证据门槛）

## 🚀 快速开始

### 安装
```bash
cd hms
bash setup.sh
```

### 选择上下文档位
```bash
# 默认 256k
python -m hms received "你好" --tier 256k

# 32k 低成本模式
python -m hms received "你好" --tier 32k

# 1M 超长会话
python -m hms received "你好" --tier 1M
```

### 配置
编辑 `config.json` 中的 `context_tiers` 自定义各档位参数。

### Cron 集成
```bash
# 每分钟处理待分析队列
openclaw cron add --schedule "* * * * *" --command "python -m hms process_pending"

# 每天凌晨3点巩固
openclaw cron add --schedule "0 3 * * *" --command "python -m hms consolidate"

# 每周日凌晨4点遗忘
openclaw cron add --schedule "0 4 * * 0" --command "python -m hms forget"
```

### Skill/Plugin 集成
```python
from hms.hooks import on_message_received, on_message_sent

# 在 skill handler 中调用
ctx = on_message_received(user_message)
on_message_sent(user_message, assistant_reply)
```

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
