# HMS — Hierarchical Memory Scaffold

> OpenClaw Skill: 分层记忆管理系统

## 功能

HMS 为 OpenClaw 提供类人记忆能力，包括：

- **感知分析**：从对话中提取实体、情感、意图
- **碰撞检测**：发现记忆间的矛盾和关联
- **分层上下文**：认知指纹 + 主题时间线 + 压缩摘要
- **记忆遗忘**：基于 Ebbinghaus 遗忘曲线的自动遗忘
- **认知指纹**：动态维护用户画像
- **重构性回忆**：像人类一样重建记忆（v4 新增）
- **梦境整合**：后台持续发现远距离关联（v4 新增）
- **创造性联想**：多跳图谱发现创意连接（v4 新增）
- **记忆覆盖**：新证据替代旧矛盾（v4 新增）

## 使用方式

### 方式一：OpenClaw 原生插件（v4 推荐）

```python
from hms import HMSPlugin

plugin = HMSPlugin()
# 注册后自动提供 hms_perceive/collide/recall/consolidate/context_inject 工具
```

### 方式二：Skill 直接调用

```python
from hms.hooks import on_message_received, on_message_sent

# 用户消息到达时
result = on_message_received(user_message, session_id)

# 助手回复后
on_message_sent(user_message, assistant_reply, session_id)
```

### 方式三：CLI 直接调用

```bash
python -m hms received "用户消息"
python -m hms process_pending
python -m hms consolidate
python -m hms forget
python -m hms health
```

## 依赖

- Python 3.10+
- `requests`（必需）
- `jieba`（可选，中文分词质量提升）
- `sentence-transformers`（可选，高质量 embedding）

**零外部记忆依赖**：HMS 自带本地文件存储，不依赖 memory-lancedb 等外部插件。

## 架构

```
用户消息 → [HMSPlugin.on_message] → [即时感知-启发式] → 检索
    ↓                                          ↓
    │                                    重构性回忆 (LLM 综合重建)
    ↓                                          ↓
助手回复 → [写入 pending] → [心跳触发 process_pending]
    ↓
[Embedding预过滤] → [深度分析-LLM] → 碰撞检测 → 存储
    ↓
[定时巩固] → 梦境引擎 → 创造性联想 → 压缩 → 指纹
    ↓
[覆盖机制] → 遗忘引擎 → 更新衰减状态
```
