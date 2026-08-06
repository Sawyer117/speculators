# 技术备忘录

与 DSV4-DSpark 流水线文档分开的独立write-up。放在 `docs/adaptive-spec-graph-memo` 分支上,
便于下载,不污染工作分支。

## `adaptive-spec-graph-mode-ascend.pptx`

**自适应投机解码的图模式代价 —— 昇腾 NPU 上的真实瓶颈在哪(通用模型与 DSV4 两条路线)。**
7 页,全中文。重新生成:`python3 make_adaptive_spec_memo.py --out <file>.pptx`(需 `python-pptx`)。

自适应投机按置信度给每个请求分配 K_i,只验证值得验证的 token;设备图则要求形状、地址、执行路径固定。
备忘录回答的不是"能不能省 token"(一定能),而是**为适配图付出的额外开销是否吃掉收益**。

三条结论,均有代码/文档出处:

1. **GPU 参考被高估。** vLLM PR #48692 已关闭未合入,**明确不做在线预算分配**(改用用户配置的
   `num_speculative_tokens_per_batch_size`),仅支持 `FLASH_ATTENTION`,作者自述**测不出加速** ——
   唯一被证实的收益是固定预算下的接受率(3.707,对比静态 K=5 的 3.445 / K=7 的 3.918)。
   它是"变长如何与全图共存"的参考,不是加速证据。

2. **设备侧长度已经存在,但只在一族算子上。** FIA v2(`npu_fused_infer_attention_score_v2`)的
   `actual_seq_qlen` 是 Host 侧 int 数组,vllm-ascend 全部调用点都用 `.tolist()` 构造;而
   DSA / lightning-indexer / SFA(`torch.ops._C_ascend.npu_vllm_*`)的签名就是 `torch.Tensor`,
   调用处传 `query_start_loc[1:].clone()`,全程不过 host。**DSV4-Flash 走 DSA 路径,前置条件已满足;
   通用稠密模型走 FIA 路径,这才是卡住的地方。**

3. **增量代价被误判。** 固定-K 投机早已入图,replay 前逐层更新 seq lengths 的 `graph_task_update`
   开销**已经在付**;动态 K 的真正增量是 **workspace 每轮 miss**(缓存键正是"本轮总 token 数")、
   **图键爆炸**与 **padding**。修法是结构性的:缓存键与图键都改为
   `(batch bucket × 总验证 token bucket)`,捕图阶段一次性预分配。

**推进顺序**:先在 DSV4/DSA 上把动态 K 端到端跑通并量化收益(前置条件最全、最快回答"值不值得"),
结论为正再推动 FIA v2 的接口改动,让通用模型受益。
