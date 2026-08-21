# Local LLM KV Cache Design

## 1. 目标与边界

本设计针对 Pi、pi-acp、Zed 使用 Qwen3.8-27B 编程 agent 时的首消息延迟。

目标：

- 缓存稳定 system prompt、项目规则和工具 schema 的 prefill 结果；
- 同一 session 继续对话时复用内存 slot；
- 代理或 llama.cpp 重启后，可以从磁盘恢复稳定 prefix；
- 不缓存答案，不改变模型生成和采样逻辑；
- 保持现有 llama.cpp + GGUF + AMD 运行链路。

非目标：

- 不做 response/result cache；
- 不把不同项目或不同模型的 KV 状态混用；
- 不把完整旧对话伪装成跨 session prefix；
- 不改变模型权重、训练行为或 decode 算法。

## 2. 总体架构

~~~mermaid
flowchart LR
    C["Pi / pi-acp / Zed"] -->|OpenAI Chat Completions| P["Cache Proxy<br/>127.0.0.1:18082"]

    subgraph CACHE["Cache Proxy"]
        A["Session affinity"]
        K["Prefix key builder"]
        H["Hot slot maps<br/>session_states / prefix_states"]
        R["Hit selection<br/>hot -> disk -> miss"]
        S["Snapshot manager"]
        A --> K --> R
        H --> R
        R --> S
    end

    P --> CACHE
    S -->|slot save / restore| L["llama.cpp llama-server<br/>127.0.0.1:8080"]
    S -->|shared slot files| D[("~/.llama-slot-cache")]
    L --> G["Qwen3.8-27B<br/>GGUF + MTP"]
~~~

代理只监听 loopback。Pi 和 Zed 指向 18082；原来的 llama.cpp 8080 保持不变。

## 3. 缓存的是什么

稳定 prefix 通常包含：

~~~text
system prompt
developer prompt
项目规则和 AGENTS.md 内容
工具定义和 JSON schema
chat template / thinking 参数
~~~

动态 suffix 通常包含：

~~~text
当前 user 消息
assistant 历史
tool result
当前任务的临时上下文
~~~

~~~mermaid
flowchart LR
    A["Stable prefix<br/>system / developer / tools"] --> B["KV + GDN state<br/>可持久化"]
    B --> C["Dynamic suffix<br/>user / assistant / tool"]
    C --> D["Decode<br/>生成当前回答"]

    style A fill:#d9f2d9,stroke:#397a3c
    style B fill:#d9e8ff,stroke:#3569a8
    style C fill:#fff1cc,stroke:#a87900
    style D fill:#ffe0e0,stroke:#a33a3a
~~~

缓存的是 prefix 对应的模型状态，不是 D 阶段的答案。

## 4. Cache key 设计

### 4.1 Prefix key

代理从 messages 开头连续保留 role 为 system/developer 的消息，遇到第一个 user、assistant 或 tool 消息就停止。

同时加入这些字段：

~~~text
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
~~~

然后计算：

~~~text
prefix_key = SHA256(canonical_json(prefix_payload))
~~~

canonical JSON 使用：

- ensure_ascii=false；
- sort_keys=true；
- 紧凑 separators；
- UTF-8 编码。

~~~mermaid
flowchart TD
    Q["Original request body"] --> M["Read messages from the beginning"]
    M --> T{"system/developer?"}
    T -->|yes| KEEP["Keep message"]
    KEEP --> T
    T -->|no| STOP["Stop at first dynamic message"]
    KEEP --> F["Add stable request fields"]
    STOP --> F
    F --> J["Canonical JSON"]
    J --> H["SHA-256"]
    H --> K["prefix_key"]
~~~

### 4.2 Snapshot 文件名

~~~text
snapshot_key = SHA256("2" + kind + identity + prefix_key)
~~~

文件格式：

~~~text
local-llm-session-<hash>.bin
local-llm-prefix-<hash>.bin
~~~

- session 的 identity 是 Pi/Zed session ID；
- prefix 的 identity 是固定字符串 prefix；
- 版本号 2 用于让旧格式缓存整体失效。

因此，换 session 会错过具体 session 快照，但仍可能命中同项目的 prefix 快照。

## 5. 请求命中顺序

~~~mermaid
flowchart TD
    START["Chat completion request"] --> MEDIA{"Has image or non-text media?"}
    MEDIA -->|yes| BYPASS["cache_prompt=true<br/>forward without disk snapshot"]
    MEDIA -->|no| ID["Resolve session affinity"]
    ID --> KEY["Build prefix_key"]
    KEY --> HOTS{"Hot session matches?"}
    HOTS -->|yes| HS["Reuse hot session slot"]
    HOTS -->|no| HOTP{"Hot prefix matches?"}
    HOTP -->|yes| HP["Reuse hot prefix slot"]
    HOTP -->|no| DS{"Disk session snapshot exists?"}
    DS -->|yes and restore succeeds| DSR["Restore session snapshot"]
    DS -->|no or restore fails| DP{"Disk prefix snapshot exists?"}
    DP -->|yes and restore succeeds| DPR["Restore prefix snapshot"]
    DP -->|no or restore fails| MISS["Use idle slot and full prefill"]

    HS --> SEND["cache_prompt=true + id_slot"]
    HP --> SEND
    DSR --> SEND
    DPR --> SEND
    MISS --> SEND
    SEND --> RESP["Generate current response"]
    RESP --> SAVE["Save current session snapshot"]
    SAVE --> SEED{"Prefix file missing?"}
    SEED -->|yes| BG["Background pure-prefix seed"]
    SEED -->|no| END["Finish"]
    BG --> END
    BYPASS --> END
~~~

### 5.1 Hot session

代理内存中保存：

~~~text
session_id -> slot_id + prefix_key + n_tokens
~~~

只有以下条件同时满足时才直接复用：

- session ID 相同；
- prefix key 相同；
- llama slot 仍然 idle；
- 当前 token 数等于上次保存的 token 数。

### 5.2 Hot prefix

新 session 没有 hot session 时，可以按照相同 prefix key 复用仍在内存中的 slot。llama.cpp 收到新的完整 prompt 后，通过 cache_prompt=true 找共同 token 前缀，只处理变化的 suffix。

### 5.3 Disk prefix

代理重启后内存映射会消失，但 prefix 文件仍然存在。代理调用 llama.cpp：

~~~text
POST /slots/{id}?action=restore
{"filename": "local-llm-prefix-....bin"}
~~~

KV 二进制由 llama.cpp 从 --slot-save-path 读取，代理不解析 KV 文件。

## 6. 冷启动时序

~~~mermaid
sequenceDiagram
    participant C as Pi / Zed
    participant P as Proxy 18082
    participant L as llama.cpp 8080
    participant D as Disk cache

    C->>P: New session chat request
    P->>P: Build session_id and prefix_key
    P->>L: GET /slots
    L-->>P: Find idle slot
    P->>L: POST /slots/id?action=restore
    L->>D: Read local-llm-prefix snapshot
    D-->>L: KV + recurrent state
    L-->>P: Restore complete
    P->>L: Chat request with cache_prompt=true, id_slot
    L-->>P: Stream generated answer
    P-->>C: Forward answer
    P->>L: Save full session snapshot
    L->>D: Write local-llm-session snapshot

    opt Prefix file was missing
        P->>P: Wait for a safe idle slot
        P->>L: Pure prefix request, n_predict=0
        L-->>P: Prefix prefill complete
        P->>L: Save prefix snapshot
        L->>D: Write local-llm-prefix snapshot
    end
~~~

磁盘恢复有 I/O 成本，但避免重新执行完整 prefix prefill。新请求的 suffix 仍然正常执行。

## 7. 同 session 动态追加

~~~mermaid
sequenceDiagram
    participant C as Pi
    participant P as Proxy
    participant M as Hot slot map
    participant L as llama.cpp

    C->>P: Turn 1
    P->>L: Full prompt, cache_prompt=true
    L-->>P: Answer 1
    P->>L: Save session snapshot
    P->>M: session_id -> slot_id

    C->>P: Turn 2 with appended user message
    P->>M: Check session_id, prefix_key, slot state
    M-->>P: Hot session hit
    P->>L: Full conversation + new suffix
    L->>L: Reuse common prefix, process suffix
    L-->>P: Answer 2
    P->>L: Save updated session snapshot
    P->>M: Update token count
~~~

Turn 2 不会直接返回 Turn 1 的答案。它仍然经过 prompt matching 和新的 decode。

## 8. 后台 prefix seed 的安全边界

~~~mermaid
flowchart TD
    R["Successful response"] --> WAIT["Wait 2 seconds"]
    WAIT --> FRONT{"Foreground request waiting?"}
    FRONT -->|yes| RETRY["Wait and retry"]
    FRONT -->|no| LOCK{"Proxy operation lock available?"}
    LOCK -->|no| RETRY
    LOCK -->|yes| SLOT["Find idle slot"]
    SLOT --> RESERVED{"Slot belongs to active session?"}
    RESERVED -->|yes| RETRY
    RESERVED -->|no| SEED["Run pure prefix request"]
    SEED --> SAVE["Save local-llm-prefix snapshot"]
    SAVE --> DONE["Release slot and finish"]
    RETRY --> WAIT
~~~

prefix seed 使用：

~~~json
{
  "messages": "只保留 system/developer/tools",
  "add_generation_prompt": false,
  "cache_prompt": false,
  "n_predict": 0,
  "stream": false
}
~~~

它不会抢占活跃 session 的 slot。没有安全空闲 slot 时会等待或超时放弃。

## 9. 缓存不是答案缓存

~~~text
缓存：KV(prefix) + GDN/recurrent state
不缓存：logits、assistant answer、下一次 RNG 结果
~~~

回答可以抽象为：

~~~text
answer = Decode(KV(prefix), dynamic_suffix, sampling_parameters, RNG)
~~~

因此：

- 相同 prefix、不同 user 消息，答案可以不同；
- temperature、top-p、top-k、seed、max_tokens 不参与 prefix key；
- 相同完整请求也只有在模型、采样参数、seed 和推理路径都一致时，才适合期待完全一致；
- cached_tokens > 0 只说明 prompt token 被复用，不说明回答被缓存。

## 10. 失效和降级

~~~mermaid
flowchart TD
    R["Restore requested"] --> OK{"Restore succeeded?"}
    OK -->|yes| HIT["Use restored state"]
    OK -->|no| WARN["Log warning"]
    WARN --> FULL["Use idle slot and full prefill"]
    FULL --> SAVE["Save new session snapshot"]
    SAVE --> RESEED["Regenerate prefix snapshot"]
~~~

会生成新 prefix key 的变化：

- system/developer prompt 文本、顺序、空格或换行变化；
- tools 或工具 JSON schema 变化；
- model ID 变化；
- thinking 或 chat-template 参数变化；
- grammar、JSON schema、response format 变化。

不会改变 prefix key、但会改变最终回答的变化：

- user 消息变化；
- assistant/tool 历史变化；
- temperature、top-p、top-k、seed、max_tokens 变化。

restore 失败只会降级为普通 prefill，不会直接让用户请求失败。

当前 key 没有包含 GGUF 文件 hash、llama.cpp build hash 和完整启动参数。更换模型文件、量化版本、chat template 或关键 KV 配置后，应先备份并换名旧缓存目录，再重新生成 snapshot。

## 11. 收益与成本

实测：

- 无工具 Pi 请求约 5218 prompt tokens；
- 后台生成约 5198-token prefix snapshot；
- 新 session 冷恢复 wall time 约 2.6–3.4 秒；
- 同 session 后续请求使用 hot session，不读磁盘；
- 带 read 工具 schema 的 prefix 约 6678 tokens，也能正常 seed；
- direct smoke 已返回过 cached_tokens > 0。

收益来自跳过稳定 system prompt、项目规则和工具 schema 的 prefill。收益大小可以直接从 cached_tokens 和 timings.cache_n 观察。

成本：

- 不减少模型权重显存；
- 不提高 decode tokens/s；
- snapshot 约 150–240 MB/个，缓存上限 12 GiB；
- 首次请求完成后会额外做一次后台 prefix prefill；
- 缓存请求使用全局 operation lock，优先保证 snapshot 不互相覆盖；
- 缓存目录包含本地 prompt/KV 状态，只保留在本机用户目录。

## 12. 关键文件

| 文件 | 作用 |
|---|---|
| [cache_core.py](./cache_core.py) | prefix payload、SHA-256 key、slot request helper |
| [cache_proxy.py](./cache_proxy.py) | HTTP proxy、hot slot、save/restore、prefix seed |
| [test_cache_core.py](./test_cache_core.py) | key 和 request helper 测试 |
| [test_cache_proxy.py](./test_cache_proxy.py) | proxy save/restore/hot slot 测试 |
| [README.md](./README.md) | 操作说明、命中判断和收益摘要 |
| [local-llm-kv-cache.service](./local-llm-kv-cache.service) | 用户级 systemd 服务模板 |

运行检查：

~~~bash
curl -fsS http://127.0.0.1:18082/health
systemctl --user status local-llm-kv-cache.service
journalctl --user -u local-llm-kv-cache.service -n 100 --no-pager
~~~
