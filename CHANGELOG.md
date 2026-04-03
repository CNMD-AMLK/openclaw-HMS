# CHANGELOG — HMS (Hybrid Memory System)

## v4.0.0 (2026-04-04) — OpenClaw Native Plugin + Cognitive Enhancements

**作者**: CNMD-AMLK

### 🔴 Breaking Changes
- **配置精简**: `hms/config.json` 从 50+ 项精简到 ~10 项；旧配置文件仍可兼容，多余项被忽略
- **MemoryAdapter 接口**: 优先使用 `tool_impl` 参数注入工具函数，旧的工具尝试逻辑变为可选 fallback
- **forgetting.py**: `MemoryOverwriter` 类新增，`evaluate_all()` 方法新增 `overwriter` 可选参数

### 🟡 New Features

#### 1. OpenClaw 原生插件 (HMSPlugin)
- **新建**: `hms/plugin.py` — 完整的 OpenClaw 插件接口
  - `register(ctx)`: 注册工具到 OpenClaw
  - `on_message(msg, ctx)`: 自动路由用户/助手消息
  - `on_heartbeat(ctx)`: 心跳维护 (pending 处理 + 状态刷新)
  - 提供 5 个工具: `hms_perceive`, `hms_collide`, `hms_recall`, `hms_consolidate`, `hms_context_inject`
- **新建**: `hms/plugin_manifest.json` — OpenClaw 插件清单

#### 2. 重构性回忆 (Reconstructive Recall)
- **新建**: `hms/scripts/reconstructive_recall.py`
  - `recall(query, context)`: 不是精确检索，而是基于碎片重建
  - 片段检索 + 上下文加权 + LLM 综合合成
  - 结果标记: `extracted` / `reconstructed` / `inferred`
  - 内置 LRU 缓存避免重复合成

#### 3. 梦境整合引擎 (Dream Engine)
- **新建**: `hms/scripts/dream_engine.py`
  - `run_dream_cycle()`: 随机游走记忆图谱发现远距离关联
  - `_find_distant_associations()`: embedding cosine < 0.3 但有间接路径
  - `_generate_insight()`: LLM 判断关联是否有意义
  - `_clean_fragments()`: 清理孤立/低质量记忆碎片
  - `_save_insight()`: 写入 `cache/insights/` 目录

#### 4. 创造性联想 (Creative Associator)
- **新建**: `hms/scripts/creative_assoc.py`
  - `creative_link(topic_a, topic_b)`: 远距离跳跃联想
  - `_graph_hop(start, max_hops)`: 通过图谱多跳路径
  - `_evaluate_link(path)`: LLM 判断关联是否有意义
  - `generate_insights()`: 自动生成创意洞察报告

#### 5. 记忆覆盖机制 (Memory Overwriter)
- **修改**: `hms/scripts/forgetting.py` 新增 `MemoryOverwriter` 类
  - `handle_conflict(old_belief, new_evidence)`: 冲突处理
  - `_supersede(old, new)`: 标记为 superseded 而非删除
  - `_downgrade_confidence(belief)`: 降低旧信念置信度
  - `_detect_conflict(text_a, text_b)`: 启发式冲突检测

#### 6. 配置默认值系统
- **修改**: `hms/scripts/config_loader.py`
  - 内置 40+ 项默认值，config.json 只需配置关键项
  - 环境变量自动覆盖敏感配置 (gateway_token, gateway_url)

#### 7. 集成示例
- **新建**: `integration/` 目录
  - `openclaw_config_example.json` — openclaw.json 配置示例
  - `heartbeat_example.md` — HEARTBEAT.md 集成指南
  - `cron_example.yaml` — cron 调度配置

### 🟢 Improvements
- **hms/__init__.py**: 暴露 HMSPlugin 入口点
- **hms/scripts/__init__.py**: 暴露所有 v4 新模块
- **版本号**: 所有位置统一更新到 4.0.0
- **向后兼容**: CLI 接口 (`python -m hms received "..."`) 完全保留
- **hooks/__init__.py**: 保持不变，向后兼容 v3 调用方式

### 🐛 Bug Fixes
- config.json 精简后，Config.get() 保证所有 config key 都有默认值，不会 KeyError
- MemoryAdapter 的 HTTP fallback 从 try/catch 链改为可注入的工具优先模式
- forgetting.py 的 `evaluate_all()` 不再因缺少 overwriter 参数而崩溃

### 📝 Documentation
- README.md: 全面重写，展示 v4 新功能和架构
- CHANGELOG.md: 首次创建
- plugin_manifest.json: OpenClaw 插件元数据

---

## v3.6.1 (2026-04-01)

### 🐛 Bug Fixes
- 添加 .env.example 模板
- 增强 setup.sh 自动检测
- 修复 .gitignore 忽略 config.json

## v3.6.0 (2026-04-01)

### ⚡ Optimizations
- Token 消耗优化
- 全面代码质量提升
- 断路器持久化
- 速率限制
- health_check

## v3.3.0 (2026-04-01)

### 🔴 P0 — OpenClaw Gateway API 适配
- Chat Completions 路径: `/api/v1/chat/completions` → `/v1/chat/completions`
- Health Check 路径: `/api/v1/health` → `/health`
- Gateway 认证令牌支持
- 默认端口: 3578 → 18789

## v3.2.0 (2026-04-01)

### 🔴 P0 — 关键修复
- 中文分词支持 (jieba)
- 记忆去重机制
- 系统健康检查

## v3.0.x (2026-04-01)

### 🔄 早期版本
- v3.0.2: Gateway 集成 + 配置化 + 提示词优化
- v3.0.3: 异常处理 + 并发安全 + 性能优化
- v3.0.4: 同步路径阻塞 + 原子队列 + 脏标记批写
- v3.0.5: 架构对齐 + 资源清理 + 并发安全

## v2.x

### 🔄 LLM 驱动
- 所有认知分析交由 LLM 处理

## v1.x

### 🔄 初始版本
- 硬编码词典做情感分析和实体提取
