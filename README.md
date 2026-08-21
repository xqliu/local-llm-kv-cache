# Local LLM KV Cache

这套方案为本机的 Pi、Zed 编程 agent 增加两级缓存：

1. 同一会话的内存热缓存，继续对话时直接复用 llama.cpp slot。
2. 项目稳定前缀的磁盘冷缓存，代理或 llama 重启后，新会话可以恢复 system prompt 和工具 schema 的 KV 状态。

当前入口是 `127.0.0.1:18082`，上游仍是原来的 llama.cpp 服务 `127.0.0.1:8080`。llama 主服务没有被替换或重启。

详细架构和 Mermaid 设计图见：[DESIGN.md](./DESIGN.md)。

## Quick start

要求：Python 3.10+，以及已经运行并开启 slot save/restore 的 llama.cpp server。

```bash
git clone https://github.com/xqliu/local-llm-kv-cache.git ~/.local/share/local-llm-kv-cache
mkdir -p ~/.config/systemd/user
cp ~/.local/share/local-llm-kv-cache/local-llm-kv-cache.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now local-llm-kv-cache.service
curl -fsS http://127.0.0.1:18082/health
```

如果 llama.cpp 不在 `127.0.0.1:8080`，修改用户 unit 中的 `PI_LLAMA_UPSTREAM`。Pi 和 Zed 的 provider URL 需要指向 `http://127.0.0.1:18082/v1`。

## 解决的问题

编程 agent 的第一条消息通常包含：

- coding-agent system prompt；
- 项目规则和 `AGENTS.md` 内容；
- 工具定义和 JSON schema；
- 当前模型的 chat template 参数。

这些内容在同一个项目中大部分稳定，但每次新建 session 时，传统请求会重新 prefill 全部 prompt。缓存代理把稳定前缀和会话尾部拆开处理，避免每次从零开始。

## 架构

```text
Pi / pi-acp / Zed
          |
          v
127.0.0.1:18082  cache_proxy.py
          |
          +-- 内存 session/prefix slot 映射
          +-- ~/.llama-slot-cache/*.bin
          |
          v
127.0.0.1:8080  llama.cpp llama-server
```

代理只监听 loopback，没有暴露到局域网或公网。

## Cache key

### Prefix key

代理首先构造稳定前缀对象：

```text
messages：从开头开始，连续保留 role=system/developer 的消息
```

遇到第一个 `user`、`assistant` 或 `tool` 消息就停止。除此之外还加入这些字段：

```text
model
tools
tool_choice
chat_template_kwargs
chat_template_args
enable_thinking
reasoning_effort
reasoning_format
response_format
json_schema
grammar
add_generation_prompt
continue_final_message
parallel_tool_calls
```

然后使用排序后的 canonical JSON 计算：

```text
prefix_key = SHA256(canonical_json(prefix_payload))
```

实现位置：[cache_core.py](./cache_core.py)。

### Snapshot 文件名

磁盘文件名还会加入缓存版本、缓存类型和身份：

```text
snapshot_key = SHA256("2" + kind + identity + prefix_key)
```

文件格式：

```text
local-llm-session-<hash>.bin
local-llm-prefix-<hash>.bin
```

其中：

- `session` 的 identity 是 Pi/Zed 的 session ID；
- `prefix` 的 identity 是固定字符串 `prefix`，所以同一个项目的不同 session 可以共享 prefix 文件；
- 版本号 `2` 用来让旧格式缓存整体失效。

用户消息、assistant 历史和 tool result 不参与 prefix key。这是有意设计：它们属于会话动态尾部，应该由 llama.cpp 的 common-prefix cache 处理。

## 请求生命周期

### 1. 收到请求

代理识别 session affinity：

1. `X-Session-Affinity`
2. `X-Pi-Session-Id`
3. `X-Client-Request-Id`
4. 请求体中的 `session_id` 或 `prompt_cache_key`
5. 没有显式 ID 时，使用 `anonymous-<prefix_key>`

图片和其他非文本多模态请求不做磁盘快照，只转发 `cache_prompt=true`。

### 2. 查找缓存

查找顺序如下：

| 层级 | 条件 | 动作 |
|---|---|---|
| 同 session 热缓存 | session ID、prefix key 相同，slot idle，token 数一致 | 直接复用内存 slot，不读盘 |
| 跨 session 热 prefix | prefix key 相同，slot idle，token 数一致 | 复用内存 slot 的共同前缀 |
| 磁盘 session 快照 | `local-llm-session-*.bin` 存在且 restore 成功 | 恢复具体会话 |
| 磁盘 prefix 快照 | `local-llm-prefix-*.bin` 存在且 restore 成功 | 恢复项目稳定前缀 |
| 缓存未命中 | 上述条件都不满足 | 完整 prefill |

实际请求会被加上：

```json
{
  "cache_prompt": true,
  "id_slot": 1
}
```

### 3. 保存 session

成功响应返回后，代理通过 llama.cpp 的 slot save API 保存：

```text
当前 slot -> local-llm-session-<hash>.bin
```

同时更新内存中的 session 和 prefix slot 映射。

### 4. 后台生成纯 prefix

如果项目还没有纯 prefix 文件，代理等待前台请求结束，然后在安全的空闲 slot 中发送一个内部请求：

```json
{
  "messages": "只保留 system/developer/tools",
  "add_generation_prompt": false,
  "cache_prompt": false,
  "n_predict": 0,
  "stream": false
}
```

这个请求的 slot 会被保存为 `local-llm-prefix-<hash>.bin`。它不会抢占活跃 session 的 slot；没有安全的空闲 slot 时会等待或超时放弃。

纯 prefix 快照很重要。Qwen3.8 是混合 GDN 架构，不能可靠地把“包含旧 user/assistant 历史的完整快照”直接当作所有新 session 的 prefix。纯 prefix 恢复后，llama.cpp 才能把新用户消息作为 suffix 继续计算。

## 缓存不是答案缓存

这个系统缓存的是：

```text
KV(prefix) + 当前 slot 状态
```

不是：

```text
问题 -> 答案
```

因此，相同的 system prompt 或相同的第一个 prefix，只表示模型可以从相同的隐藏状态开始计算。不同的 user suffix、工具结果、采样参数或随机数，仍然可以产生不同答案。

即使完整请求完全相同，也只有在模型版本、推理参数、seed、采样器和硬件计算路径都一致时，才适合期待完全一致。当前 llama-server 没有固定 `--seed`，默认使用随机 seed；`temperature`、`top_p`、`top_k`、`seed` 和 `max_tokens` 不属于 prefix key，因为它们控制生成阶段，不改变已缓存的 prompt 状态。

需要做可重复性检查时，应显式固定相同的 `seed` 和全部采样参数，并比较完整请求的输出。`cached_tokens > 0` 只证明 prompt token 被复用，不证明回答一定相同。

## 什么时候会失效

以下变化会生成新的 prefix key：

- system/developer prompt 文本、顺序、空格或换行变化；
- tools 或工具 JSON schema 变化；
- model ID 变化；
- thinking 或 chat template 参数变化；
- response format、grammar、JSON schema 变化；
- `add_generation_prompt`、`continue_final_message` 等模板行为变化。

以下情况也会导致磁盘缓存未命中：

- session ID 改变：具体 session 快照不同，但仍可能命中同一个 prefix 文件；
- 快照被 12 GiB 上限淘汰；
- 文件损坏或 llama.cpp restore 返回错误；
- 代理或 llama.cpp 的 slot 不可用；
- 多模态请求；
- 缓存版本变化。

restore 失败不会让请求失败。代理会记录 warning，继续使用当前 slot 完整 prefill；成功后可以重新生成 prefix 快照。

### 模型更换注意事项

当前 prefix key 没有包含 GGUF 文件 hash、llama.cpp build hash 和完整启动参数。因此更换模型文件、量化版本、chat template 或关键 KV 配置后，应把旧缓存目录移到备份目录，再让代理重新生成缓存。不要把不同模型版本的 slot snapshot 混用。

## 实测收益

验证使用当前 Qwen3.8-27B IQ4_XS、llama.cpp、RX 7900 XTX 配置：

- 无工具 Pi 请求的完整 prompt 约 5218 tokens；首次请求保存了约 5198-token 的纯 prefix；
- 新建 Pi session 后，日志确认从 `local-llm-prefix-*.bin` restore，wall time 约 2.6 秒；同一代理接口的冷 prefix smoke 已实测返回 `cached_tokens=17`，Pi 请求走的是同一 restore 路径；
- 相比第一次完整 prefill，冷恢复把首条回复的 wall time 降到约 2.6 秒；
- 带 `read` 工具 schema 的测试生成了约 6678-token prefix，说明工具定义也能进入缓存；
- 同一个 session 的第二次请求日志为 `reusing hot session`，没有磁盘 restore；
- Zed 无显式 session header 的直连请求也能按 anonymous prefix affinity 命中 `cached_tokens`。

对于更大的编程 agent prompt，收益主要来自跳过稳定 system prompt、项目规则和工具 schema 的 prefill。能节省多少时间取决于实际 token 数和当前 prompt processing throughput，但跳过的 token 数会直接体现在 `cached_tokens` / `timings.cache_n` 中。

## 成本和边界

- 这是 prompt prefill 优化，不会减少模型权重显存，也不会提高模型 decode 本身的 tokens/s；
- slot snapshot 文件较大，当前验证文件约为每个 150–240 MB，12 GiB 上限约束了可保留的快照数量；
- 首次请求成功后会额外进行一次后台 prefix prefill，这是用一次空闲计算换取之后的新 session 低延迟；
- 缓存请求通过全局 operation lock 串行化，优先保证 slot snapshot 不互相覆盖；家庭场景的低并发下这个取舍是合适的；
- 缓存文件包含本地 prompt/KV 状态，只应保留在本机 loopback 和用户目录，不应把 18082 暴露出去。

## 运行和排查

服务：

```bash
systemctl --user status local-llm-kv-cache.service
systemctl --user is-active local-llm-kv-cache.service
journalctl --user -u local-llm-kv-cache.service -n 100 --no-pager
```

健康检查：

```bash
curl -fsS http://127.0.0.1:18082/health
curl -fsS http://127.0.0.1:18082/v1/models
```

确认真正命中：

```text
代理日志：reusing hot session / reusing hot prefix / restored local-llm-prefix-...
API 响应：usage.prompt_tokens_details.cached_tokens > 0
API timings：timings.cache_n > 0
```

测试：

```bash
cd ~/.local/share/local-llm-kv-cache
python3 -m unittest -v test_cache_core.py test_cache_proxy.py
python3 -m py_compile cache_core.py cache_proxy.py
```

## 当前配置入口

- Pi：`~/.pi/agent/models.json`
- Zed：`~/.config/zed/settings.json`
- systemd 模板：[local-llm-kv-cache.service](./local-llm-kv-cache.service)；当前安装位置是 `~/.config/systemd/user/local-llm-kv-cache.service`
- 代理代码：[cache_proxy.py](./cache_proxy.py)
- key 逻辑：[cache_core.py](./cache_core.py)
- 磁盘缓存：`~/.llama-slot-cache`

## License

Apache-2.0，详见 [LICENSE](./LICENSE)。
