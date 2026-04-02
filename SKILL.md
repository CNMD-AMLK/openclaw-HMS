# HMS — Hierarchical Memory Scaffold

> OpenClaw Skill: 分层记忆管理系统

## 功能

HMS 为 OpenClaw 提供类人记忆能力，包括：

- **感知分析**：从对话中提取实体、情感、意图
- **碰撞检测**：发现记忆间的矛盾和关联
- **分层上下文**：认知指纹 + 主题时间线 + 压缩摘要
- **记忆遗忘**：基于 Ebbinghaus 遗忘曲线的自动遗忘
- **认知指纹**：动态维护用户画像

## 使用方式

### 方式一：Skill 直接调用（推荐）

在你的 OpenClaw Skill/Plugin 中导入 HMS：

```python
from hms.hooks import on_message_received, on_message_sent

# 用户消息到达时
result = on_message_received(user_message, session_id)
# result 包含 perception + retrieved_memories + context

# 助手回复完成后
on_message_sent(user_message, assistant_reply, session_id)
```

### 方式二：CLI 直接调用

```bash
# 处理用户消息
python -m hms received "用户消息"

# 处理队列中的待分析消息
python -m hms process_pending

# 每日巩固（建议每天 3AM）
python -m hms consolidate

# 每周遗忘（建议每周日 4AM）
python -m hms forget

# 健康检查
python -m hms health
```

### 方式三：Cron 定时调度

```bash
# 每分钟处理待分析消息
openclaw cron add --schedule "* * * * *" --command "python -m hms process_pending"

# 每天凌晨 3 点巩固
openclaw cron add --schedule "0 3 * * *" --command "python -m hms consolidate"

# 每周日凌晨 4 点遗忘
openclaw cron add --schedule "0 4 * * 0" --command "python -m hms forget"
```

## 配置

1. 复制 `.env.example` 为 `.env`：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env`，填入你的 Gateway token：
   ```
   OPENCLAW_GATEWAY_URL=http://127.0.0.1:18789
   HMS_GATEWAY_TOKEN=your-token-here
   ```

3. 运行健康检查确认配置正确：
   ```bash
   python -m hms health
   ```

## 依赖

- Python 3.10+
- `requests`（必需）
- `jieba`（可选，中文分词质量提升）
- `sentence-transformers`（可选，高质量 embedding）

## 架构

```
用户消息 → [即时感知-启发式] → 检索相关记忆 → 注入上下文
    ↓
助手回复 → [异步队列]
    ↓
[Embedding预过滤] → [深度分析-LLM] → 碰撞检测 → 存储
    ↓
[定时巩固-LLM] → 摘要压缩 → 主题时间线 → 认知指纹
```
