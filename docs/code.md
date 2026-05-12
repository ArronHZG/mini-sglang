
## 附录 A：核心源码解读

> 以下代码片段均来自 mini-sglang 源码，保留了原始实现逻辑，帮助理解各模块的核心机制。

### A.1 核心数据结构 (`core.py`)

#### SamplingParams - 采样参数

```python
# 文件: python/minisgl/core.py
@dataclass
class SamplingParams:
    temperature: float = 0.0    # 温度，<=0 表示贪婪解码
    top_k: int = -1             # Top-K 采样，-1 表示不限制
    top_p: float = 1.0          # Top-P (核) 采样
    ignore_eos: bool = False    # 是否忽略 EOS token
    max_tokens: int = 1024      # 最大生成 token 数

    @property
    def is_greedy(self) -> bool:
        """判断是否为贪婪解码模式"""
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
```

#### Req - 请求状态

```python
# 文件: python/minisgl/core.py
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor        # 输入 token 序列（CPU 上）
    table_idx: int                  # 在全局 token 表中的索引
    cached_len: int                 # 已缓存的 prefix 长度（Radix Cache 命中时 > 0）
    output_len: int                 # 期望输出的 token 数量
        uid: int                      # 全局唯一请求 ID
    sampling_params: SamplingParams # 采样参数
    cache_handle: BaseCacheHandle   # KV 缓存句柄

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len

    @property
    def remain_len(self) -> int:
        """剩余可生成长度"""
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """本次需要处理的长度 = device_len - cached_len"""
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """完成一个 token 的生成，更新长度"""
        self.cached_len = self.device_len
        self.device_len += 1

    @property
    def can_decode(self) -> bool:
        """是否可以继续 decode"""
        return self.remain_len > 0
```

#### Context - 全局上下文（单例模式）

```python
# 文件: python/minisgl/core.py
@dataclass
class Context:
    page_size: int                          # KV Cache 页大小
    page_table: torch.Tensor = field(init=False)       # 页表
    attn_backend: BaseAttnBackend = field(init=False)  # 注意力后端
    moe_backend: BaseMoeBackend = field(init=False)    # MoE 后端
    kv_cache: BaseKVCachePool = field(init=False)      # KV 缓存池
    _batch: Batch | None = field(default=None, init=False)

    @contextmanager
    def forward_batch(self, batch: Batch):
        """上下文管理器：设置当前 batch，forward 完成后自动清理"""
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None
```

### A.2 推理引擎 (`engine/engine.py`)

#### Engine 初始化流程

```python
# 文件: python/minisgl/engine/engine.py
class Engine:
    def __init__(self, config: EngineConfig):
        # 1. 初始化 CUDA 设备和通信
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        self.stream = torch.cuda.Stream()
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        # 2. 初始化通信 (NCCL / PyNCCL)
        self.tp_cpu_group = self._init_communication(config)

        # 3. 创建模型（meta 设备，不占显存）并加载权重
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # 4. 根据剩余显存计算 KV Cache 页数
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        self.ctx.kv_cache = create_kvcache_pool(...)

        # 5. 初始化页表、注意力后端、MoE 后端、采样器
        self.ctx.page_table = torch.zeros((max_running_req + 1, aligned_max_seq_len), ...)
        self.ctx.attn_backend = create_attention_backend(...)
        self.sampler = Sampler(self.device, vocab_size)

        # 6. 初始化 CUDA Graph 捕获
        self.graph_runner = GraphRunner(stream=self.stream, model=self.model, ...)
```

#### Engine.forward_batch - 核心前向传播

```python
# 文件: python/minisgl/engine/engine.py
def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
    assert torch.cuda.current_stream() == self.stream
    with self.ctx.forward_batch(batch):          # 设置全局上下文的当前 batch
        if self.graph_runner.can_use_cuda_graph(batch):
            logits = self.graph_runner.replay(batch)   # CUDA Graph 路径
        else:
            logits = self.model.forward()              # 普通 Eager 路径

    for req in batch.reqs:
        req.complete_one()                       # 更新每个请求的长度状态

    next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
    next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)  # 异步拷贝到 CPU
    copy_done_event = torch.cuda.Event()
    copy_done_event.record(self.stream)         # 记录拷贝完成事件
    return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

### A.3 CUDA Graph 管理 (`engine/graph.py`)

#### GraphRunner - CUDA Graph 捕获与重放

```python
# 文件: python/minisgl/engine/graph.py
class GraphRunner:
    def _capture_graphs(self, max_seq_len, vocab_size, model):
        """为不同的 batch size 预捕获 CUDA Graph"""
        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)
        for bs in sorted(self.graph_bs_list, reverse=True):
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()  # 捕获这次 forward
            if pool is None:
                pool = graph.pool()  # 复用 CUDA Graph handle 以减少内存
            self.graph_map[bs] = graph

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        """仅 decode 阶段且 batch size 在捕获范围内时使用 CUDA Graph"""
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        """重放已捕获的 CUDA Graph（仅需复制输入 + replay，无需重新执行 kernel）"""
        self.buffer.copy_from(batch)           # 将 batch 数据复制到捕获缓冲区
        g = self.graph_map[batch.padded_size]  # 选择对应 BS 的 graph
        self.attn_backend.prepare_for_replay(batch)
        g.replay()                              # 重放！
        return self.buffer.logits[: batch.size]
```

#### CUDA Graph Batch Size 策略

```python
# 文件: python/minisgl/engine/graph.py
def _determine_cuda_graph_bs(cuda_graph_bs, cuda_graph_max_bs, free_memory):
    """根据可用显存决定 CUDA Graph 捕获的 batch size 列表"""
    if cuda_graph_bs is not None:
        return cuda_graph_bs
    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        cuda_graph_max_bs = 256 if free_memory_gb > 80 else 160  # H200 vs 其他
    if cuda_graph_max_bs < 1:
        return []
    # 小 BS 更密集（decode 常见），大 BS 每 8 一个
    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    # 结果: [1, 2, 4, 8, 16, 24, 32, 40, ...]
```

### A.4 采样器 (`engine/sample.py`)

```python
# 文件: python/minisgl/engine/sample.py
@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        """从请求列表中提取批量采样的参数"""
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)  # 全贪婪模式
        # 准备 temperature, top_k, top_p 张量
        temperatures = make_device_tensor([max(0.0 if p.is_greedy else p.temperature, 1e-6) for p in params], ...)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    def sample(self, logits: Tensor, args: BatchSamplingArgs) -> Tensor:
        """执行采样：贪婪或随机"""
        if args.temperatures is None:  # 全部贪婪解码
            return torch.argmax(logits, dim=-1)
        return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)


def sample_impl(logits, temperatures, top_k, top_p):
    """使用 FlashInfer 进行高效概率采样"""
    import flashinfer.sampling as sampling
    probs = sampling.softmax(logits, temperatures, ...)
    if top_k is not None and top_p is not None:
        return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)
    elif top_k is not None:
        return sampling.top_k_sampling_from_probs(probs, top_k)
    elif top_p is not None:
        return sampling.top_p_sampling_from_probs(probs, top_p)
    return sampling.sampling_from_probs(probs)
```

### A.5 调度器 (`scheduler/scheduler.py`)

#### Scheduler 主循环 - Overlap Scheduling

```python
# 文件: python/minisgl/scheduler/scheduler.py
@torch.inference_mode()
def run_forever(self) -> NoReturn:
    """调度器主入口：普通模式 or Overlap 模式"""
    if ENV.DISABLE_OVERLAP_SCHEDULING:
        while True:
            self.normal_loop()       # 串行：调度 → 执行 → 后处理
    else:
        data = None
        while True:
            data = self.overlap_loop(data)  # 并行：GPU 执行 N 的同时 CPU 准备 N+1


def overlap_loop(self, last_data):
    """
    Overlap 调度的核心：
    - GPU 正在执行第 N 个 batch 时，CPU 同时准备第 N+1 个 batch
    - 返回 ongoing_data 给下一次迭代处理结果
    """
    # 1. 接收新消息（非阻塞）
    for msg in self.receive_msg(blocking=...):
        self._process_one_msg(msg)

    # 2. 调度下一个 batch（在 CPU stream 上）
    forward_input = self._schedule_next_batch()

    # 3. 在 Engine 的 GPU stream 上执行当前 batch
    ongoing_data = None
    if forward_input is not None:
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)  # 等待 CPU 调度完成
            ongoing_data = (forward_input, self._forward(forward_input))

    # 4. 处理上一轮的结果（与步骤 3 的 GPU 执行并行）
    self._process_last_data(last_data)
    return ongoing_data
```

#### PrefillAdder - 分块预填充添加

```python
# 文件: python/minisgl/scheduler/prefill.py
class PrefillAdder:
    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        """尝试将一个待处理请求加入 prefill batch"""
        if self.token_budget <= 0:
            return None

        # 如果是 chunked request 的延续，直接复用之前的 handle
        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(pending_req, chunked_req.cache_handle,
                                      chunked_req.table_idx, chunked_req.cached_len)

        # 尝试分配资源：匹配 Radix Cache + 检查预算
        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(pending_req, cache_handle, table_idx,
                                      cache_handle.cached_len)
        return None

    def _add_one_req(self, pending_req, cache_handle, table_idx, cached_len) -> Req:
        """实际创建 Req 或 ChunkedReq"""
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)  # 关键：受 budget 限制
        is_chunked = chunk_size < remain_len             # 如果一次装不下，创建 ChunkedReq
        CLS = ChunkedReq if is_chunked else Req
        # 将 input_ids 对应部分拷贝到 GPU token_pool
        device_ids = self.token_pool[table_idx, slice(cached_len, cached_len + chunk_size)]
        device_ids.copy_(pending_req.input_ids[slice(cached_len, cached_len + chunk_size)].pin_memory())
        return CLS(input_ids=pending_req.input_ids[:cached_len + chunk_size],
                   table_idx=table_idx, cached_len=cached_len, ...)
```

#### CacheManager - 缓存管理

```python
# 文件: python/minisgl/scheduler/cache.py
class CacheManager:
    def cache_req(self, req: Req, *, finished: bool) -> None:
        """
        请求完成后缓存/释放：
        - 将完成的序列插入 Radix Cache（供后续请求复用）
        - 释放不再需要的 KV Cache 页面
        """
        insert_ids = req.input_ids[:req.cached_len]
        page_indices = self.page_table[req.table_idx, :req.cached_len]
        old_handle = req.cache_handle

        # 插入前缀到 Radix Tree
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        self.unlock(old_handle)  # 解锁旧 handle

        # 释放已被其他请求缓存的部分（避免内存泄漏）
        self._free(page_indices[old_handle.cached_len : cached_len])

        if finished:  # 请求完成，释放尾部
            self._free(page_indices[new_handle.cached_len :])
        else:  # 更新 handle 继续使用
            req.cache_handle = new_handle
            self.lock(new_handle)
```

### A.6 模型前向传播 (`models/llama.py`)

```python
# 文件: python/minisgl/models/llama.py
class LlamaForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = LlamaModel(config)          # Transformer 编码器
        self.lm_head = ParallelLMHead(...)       # 并行输出头

    def forward(self) -> Tensor:
        """从全局 context 获取当前 batch 的 input_ids，执行完整前向传播"""
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


class LlamaModel(BaseOP):
    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed_tokens.forward(input_ids)     # Token Embedding
        residual = None
        for layer in self.layers.op_list:             # N 层 Decoder
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]      # 最终 RMSNorm


class LlamaDecoderLayer(BaseOP):
    def forward(self, x, residual=None) -> Tuple[Tensor, Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)   # Pre-Attention RMSNorm
        x = self.self_attn.forward(x)                              # Multi-Head Attention + RoPE
        x, residual = self.post_attention_layernorm.forward(x, residual)  # Post-Attention RMSNorm
        x = self.mlp.forward(x)                                    # Gated MLP (SiLU)
        return x, residual                                          # 残差连接由 norm fused 处理
```

#### Attention Layer - QKV 投影 + RoPE + 注意力计算

```python
# 文件: python/minisgl/layers/attention.py
class AttentionLayer(StateLessOP):
    def forward(self, qkv: Tensor) -> Tensor:
        ctx = get_global_ctx()
        # 1. 拆分 QKV（已融合为单一投影）
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        # 2. 可选的 QK Norm
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        # 3. 应用旋转位置编码 (RoPE)
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        # 4. 调用注意力后端（FlashInfer 等）计算 attention output
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return o.view(-1, self.qo_attn_dim)


# GatedMLP - 门控 MLP (SwiGLU)
# 文件: python/minisgl/models/utils.py
class GatedMLP(BaseOP):
    def forward(self, x: Tensor) -> Tensor:
        gate_up = self.gate_up_proj.forward(x)   # [bs, seq, 2 * intermediate]
        y = self.act_fn(gate_up)                 # SiLU(gate) * up (fused)
        return self.down_proj.forward(y)         # AllReduce 行并行
```

### A.7 FlashInfer 注意力后端 (`attention/fi.py`)

```python
# 文件: python/minisgl/attention/fi.py
class FlashInferBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        self.kvcache = get_global_ctx().kv_cache
        # workspace buffer 用于 FlashInfer 内部临时存储
        self.float_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        # 分别为 prefill 和 decode 创建 wrapper
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(self.float_workspace_buffer, kv_layout="NHD")
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(self.float_workspace_buffer, ...)

    def forward(self, q, k, v, layer_id, batch) -> Tensor:
        """FlashInfer 注意力核心：store KV → 获取 cache → run attention"""
        metadata = batch.attn_metadata
        self._initialize_metadata_once(metadata)  # 懒初始化 plan
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)  # 写入 KV 到 paged cache
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)  # 执行注意力

    def prepare_metadata(self, batch: Batch) -> None:
        """准备 FlashInfer 所需的元数据：cu_seqlens, indices, page_table 等"""
        reqs = batch.padded_reqs
        seqlens_q = [req.extend_len for req in reqs]   # query 序列长度
        seqlens_k = [req.device_len for req in reqs]    # key 序列总长度
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k).cumsum_(dim=0)
        # 根据阶段选择不同的 cu_seqlens_q 计算方式
        if max_seqlen_q == 1:  # decode: 每个 request 只有一个 query token
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1)
        elif all(l == 0 for l in cached_lens):  # prefill 无缓存命中
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # partial cache hit prefill
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q).cumsum_(dim=0)
        # 组装 FIMetadata，选择 prefill 或 decode wrapper
        batch.attn_metadata = FIMetadata(
            ..., wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper)
```

### A.8 KV Cache 管理 (`kvcache/mha_pool.py`, `kvcache/radix_cache.py`)

#### MHAKVCache - Paged KV Cache 存储

```python
# 文件: python/minisgl/kvcache/mha_pool.py
class MHAKVCache(BaseKVCachePool):
    def __init__(self, num_kv_heads, num_layers, head_dim, num_pages, page_size, dtype, device):
        # 核心张量: (2=K/V, num_layers, num_pages, page_size, local_kv_heads, head_dim)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device, dtype=dtype,
        )

    def store_kv(self, k, v, out_loc, layer_id):
        """调用 CUDA kernel 将 K,V 写入 paged kv_cache 的指定位置"""
        from minisgl.kernel import store_cache
        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc, k=k, v=v,
        )
```

#### RadixPrefixCache - 前缀缓存（基数树）

```python
# 文件: python/minisgl/kvcache/radix_cache.py
class RadixPrefixCache(BasePrefixCache):
    def match_prefix(self, input_ids: Tensor) -> MatchResult:
        """在基数树中查找最长公共前缀"""
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: Tensor, indices: Tensor) -> InsertResult:
        """插入新的前缀到基数树"""
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # 有未匹配的后缀需要插入
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> Tensor:
        """LRU 驱逐：优先驱逐最老的可驱逐叶子节点"""
        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)  # 基于 timestamp 的最小堆
        while evicted_size < size:
            node = heapq.heappop(leave_nodes)  # 取出最老的节点
            evicted_indices.append(node.value)
            del parent.children[self.key_fn(node._key)]  # 从树中移除
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)  # 父节点也变成可驱逐的叶子
        return torch.cat(evicted_indices)

    def _tree_walk(self, input_ids: Tensor) -> Tuple[RadixTreeNode, int]:
        """沿树遍历，找到最长匹配的前缀节点"""
        prefix_len = 0
        node = self.root_node
        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            match_len = align_down(node.get_match_len(input_ids[prefix_len:]), self.page_size)
            prefix_len += match_len
            if match_len != node.length:  # 部分匹配，需要分裂节点
                node = node.split_at(match_len)
                return node, prefix_len
            node.timestamp = time.monotonic_ns()  # 更新 LRU 时间戳
        return node, prefix_len
```

### A.9 MoE 混合专家 (`layers/moe.py`, `moe/fused.py`)

#### MoELayer - MoE 前向传播

```python
# 文件: python/minisgl/layers/moe.py
class MoELayer(BaseOP):
    def forward(self, hidden_states: Tensor, router_logits: Tensor):
        ctx = get_global_ctx()
        # 委托给 MoE 后端执行核心计算
        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj, w2=self.down_proj,
            gating_output=router_logits, topk=self.top_k, ...
        )
        if self.tp_size > 1:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)  # TP AllReduce
        return final_hidden_states
```

#### FusedMoe - 融合 MoE Kernel 实现

```python
# 文件: python/minisgl/moe/fused.py
class FusedMoe(BaseMoeBackend):
    def forward(self, hidden_states, w1, w2, gating_output, topk, ...) -> Tensor:
        # 1. TopK 路由选择
        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk, renormalize)
        # 2. 融合 Expert 计算
        return fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids, ...)


def fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids, ...) -> Tensor:
    """Fused MoE 的完整计算流程"""
    # Step 1: Token 对齐 - 将同一 expert 的 token 排列到连续 block
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        curr_topk_ids, config["BLOCK_SIZE_M"], E  # E = num_experts
    )

    # Step 2: Stage 1 - W1 × x (第一层线性)
    fused_moe_kernel_triton(curr_hidden_states, w1, intermediate_cache1,
                            curr_topk_weights, curr_topk_ids, sorted_token_ids, expert_ids, ...)

    # Step 3: SiLU 激活 (fused: SiLU(gate) * up)
    FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)

    # Step 4: Stage 2 - W2 × act (第二层线性)
    fused_moe_kernel_triton(intermediate_cache2, w2, intermediate_cache3,
                            curr_topk_weights, curr_topk_ids, sorted_token_ids, expert_ids, ...)

    # Step 5: 加权求和归约 - 合并所有 expert 的输出
    moe_sum_reduce_triton(intermediate_cache3, out_hidden_states)
    return out_hidden_states
```

### A.10 API 服务 (`server/api_server.py`)

#### OpenAI 兼容 API 路由

```python
# 文件: python/minisgl/server/api_server.py
@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()

    # 1. 构造 prompt（支持 messages 格式）
    prompt = [msg.model_dump() for msg in req.messages] if req.messages else req.prompt

    # 2. 分配唯一 UID
    uid = state.new_user()

    # 3. 发送 TokenizeMsg 到 Tokenizer 进程（通过 ZMQ）
    await state.send_one(TokenizeMsg(
        uid=uid, text=prompt,
        sampling_params=SamplingParams(
            ignore_eos=req.ignore_eos, max_tokens=req.max_tokens,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
        ),
    ))

    # 4. 流式返回 SSE 响应（OpenAI 格式）
    return StreamingResponse(
        state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
        media_type="text/event-stream",
    )


async def stream_chat_completions(self, uid: int):
    """将 Detokenizer 回复转换为 OpenAI SSE 格式流式输出"""
    first_chunk = True
    async for ack in self.wait_for_ack(uid):  # 等待 tokenizer 进程的回复
        delta = {}
        if first_chunk:
            delta["role"] = "assistant"   # 第一个 chunk 包含 role
            first_chunk = False
        if ack.incremental_output:
            delta["content"] = ack.incremental_output
        yield f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()
        if ack.finished:
            break
    # 发送结束标记
    yield f"data: {{'choices': [{{'delta': {{}}, 'finish_reason': 'stop'}}]}}\n\n".encode()
    yield b"data: [DONE]\n\n"
```

### A.11 消息协议 (`message/`)

```python
# 文件: python/minisgl/message/tokenizer.py
@dataclass
class TokenizeMsg(BaseTokenizerMsg):    # Frontend → Tokenizer: 请求编码文本
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams

@dataclass
class DetokenizeMsg(BaseTokenizerMsg):  # Scheduler → Tokenizer: 请求解码 token
    uid: int
    next_token: int
    finished: bool

@dataclass
class AbortMsg(BaseTokenizerMsg):       # Frontend → Tokenizer: 中止请求
    uid: int


# 文件: python/minisgl/message/backend.py
@dataclass
class UserMsg(BaseBackendMsg):          # Tokenizer → Scheduler: 携带 token 的用户请求
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams

@dataclass
class AbortBackendMsg(BaseBackendMsg):  # Frontend → Scheduler: 中止请求
    uid: int

@dataclass
class ExitMsg(BaseBackendMsg):         # 系统信号: 退出
    pass
```