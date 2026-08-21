# 昇腾融合算子怎么进上游 —— 问题与方案

供内部讨论。**2026-08-21 大改**:原稿的结论(自建注册表 + 插件点)已被推翻 ——
torch 里本来就有那个算子,而且华为自己已经在别的框架里适配好了。结论在最后。

---

## 1. 起点问题

DSV4 草稿骨干里有一批昇腾融合算子(`npu_grouped_matmul`、`npu_moe_token_permute`/`unpermute`、
`npu_swiglu`),集中在两个文件、共 20 处 `torch_npu` 调用。**上游 speculators 不会收它们。**

## 2. 判例:同样解决昇腾问题,一个合了一个关了

| PR | 做法 | 结果 |
|---|---|---|
| **#775** Enable Ascend NPU training | `src/` 里 import `torch_npu.contrib.transfer_to_npu` 垫片 | **两天后关闭,未合**(+17/−1) |
| **#589** selectable attention backend | 不塞昇腾代码,加一个**可选后端**(sdpa/eager)让昇腾去选 | **已合**(+234/−14) |

上游自己的设备处理也一致:全仓 `src/` **0 处 `torch_npu` 直接调用、0 处 `is_cuda` 分支**,
统一 `torch.accelerator.current_accelerator()`。⟹ **可移植的进上游,厂商的留外面。**

## 3. ★ 关键发现:torch 里早就有这个算子

```
torch 2.12  aten::_grouped_mm(Tensor self, Tensor mat2, Tensor? offs=None,
                              Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor

分派表:  CUDA ✓   Meta ✓   Autograd[alias] ✓   CompositeExplicitAutograd[alias] ✓
```

CPU 上直接可用(前向 ✓ 反向 ✓,bf16 相对误差 3.1e-3)。**通用回落对任何后端成立** ——
NPU 上现在就能跑,只是走通用分解而非融合算子,所以慢。

### 3.1 反向不用我们写,而且同样被加速

`grad_fn = GroupedMmBackward0`。trace 反向实际派发的算子(带形状):

```
前向     x[64,32] @ w[3,32,64]                             -> out[64,64]
反向 b1  transpose(x, -2,-1)         [64,32] -> [32,64]              xᵀ
     b2  _grouped_mm(xᵀ, grad)       [32,64]@[64,64] -> [3,32,64]    dw = xᵀ @ grad
     b3  transpose(w, -2,-1)         [3,32,64] -> [3,64,32]          wᵀ
     b4  _grouped_mm(grad, wᵀ)       [64,64]@[3,64,32] -> [64,32]    dx = grad @ wᵀ
```

**与我们手写的那两条一模一样**,只是这份由 PyTorch 写在 `derivatives.yaml`、codegen 成
`GroupedMmBackward0`、由 Meta 维护和测试。而且公式是**用别的 aten 算子表达的**,
所以设备无关:里面的 `_grouped_mm` 会各自再分派一次 ⟹ **注册前向 = 前反向一起加速。**

⟹ **模型定义里不再需要 `torch.autograd.Function`**,只要:

```python
out = torch._grouped_mm(x, w, offs=offs)     # 一行,零厂商字样,前反向都快
```

原因是 Autograd 键在分派链上**位于后端键之上**:建图在 Autograd 层完成(设备无关),
我们注册的实现在 PrivateUse1 层,autograd 根本看不见它。
我们现在之所以写了 `autograd.Function`,只是因为调的是 torch **不认识**的 `npu_grouped_matmul`。

## 4. 昇腾没有对齐这个契约(源头核实)

`Ascend/op-plugin` master `6b46b28`(2026-08-20),`op_plugin/config/op_plugin_functions.yaml`
共 **1512 个 func 条目**:

```
_scaled_grouped_mm       ✓  op_api: [v2.7, newest]      ← 量化,MoE 推理
_scaled_grouped_mm_v2    ✓  op_api: [v2.10, newest]
_grouped_mm              ✗  零处                        ← bf16,MoE 训练
npu_grouped_matmul       ✓  op_api: [v2.1, newest]      ← 自有算子,2.1 就有
```

★ **不缺 kernel**:两个适配器调的是**完全同一组底层算子**——

```
ScaledGroupMmKernelNpuOpApi.cpp (641行) → aclnnGroupedMatmul / V4 / V5 / WeightNz
GroupedMatmulKernelNpuOpApi.cpp (434行) → aclnnGroupedMatmul / V4 / V5 / WeightNz   交集全中
```

缺的只是**第三个签名转换层**。而 `_scaled_grouped_mm` 是 **2026-06-04** 才加的
(`527235b`,+12 yaml / +1600 C++ / +1200 测试)—— 说明往 aten 契约对齐这件事今年才开始,
**且先做的是推理那条**。

## 5. ★★ 而华为自己已经做了,只是做在框架侧

`torchtitan-npu/torchtitan_npu/ops/_grouped_mm.py`,文件头 `Copyright (c) 2026 Huawei Technologies`:

```python
@torch.library.impl("aten::_grouped_mm", "PrivateUse1")
def _(self, mat2, offs, bias=None, out_dtype=None):
    # output = x @ w                      [n_tokens, IN] @ [n_experts, IN, OUT]
    # dx     = grad @ w.transpose(-1,-2)  [n_tokens, OUT] @ [n_experts, OUT, IN]
    # dw     = x.T @ grad                 [IN, n_tokens] @ [n_tokens, OUT]   ← 沿 n_tokens 归约
    split_along_k = self.ndim == 2 and mat2.ndim == 2
    return torch_npu.npu_grouped_matmul(
        [self], [mat2], group_list=offs.to(torch.int64), group_list_type=0,
        split_item=2, group_type=(2 if split_along_k else 0),
        bias=[bias] if bias is not None else None, output_dtype=out_dtype)[0]
```

**全文只有前向,没有一行反向。** 而且注释点出了那个坑:**`dw` 沿 `n_tokens` 归约,
必须 `group_type=2`**,靠 `split_along_k` 判别。写死 `group_type=0` 会把 `dw` 算错。

### 为什么没人把它推回 op-plugin

三个框架,三种各自打补丁:

```
MindSpeed        自建 C++ 扩展 npu_gmm/npu_gmm_backward(连 torch_npu 的算子都不用)
                 21 个 GMM 文件、41 处 npu_grouped_matmul、0 处 _grouped_mm
torchtitan-npu   框架侧注册 aten::_grouped_mm(30 行)
speculators(我们) 直调 torch_npu + 手写 autograd.Function
```

**缺口是真的,但性质不是"没人能做",是"每个框架各自打补丁,没人补公共那层"** —— 公地悲剧。
根因:昇腾上做 MoE 训练的主流是 Megatron/MindSpeed,**从不经过 aten 那一层**,所以没人喊疼。

## 6. torch 官方那边:什么都不用提

扩展点已完备且被证明可用(torchtitan-npu 就是证据)。**torch 缺的是零,活全在厂商侧。**

唯一可能的 ask 是"把 `_grouped_mm` 转正"(去下划线),但 **torchtitan(Meta 自己的)
里 `_grouped_mm`/`_scaled_grouped_mm` 出现 17 处**,一方代码就在用私有 API,先例充分,不必先要转正。

## 7. 反向融合:我们不需要

`torch_npu` 的 grouped matmul 一族**没有任何 `_backward`**。MindSpeed 那个
`npu_gmm_backward_fusion` 来自它**自建的 C++ 扩展**,不是 torch_npu 的。

走 aten 唯一损失的是 MindSpeed 的这一手:

```python
npu_groupmatmul_add_fp32(x, grad, group_list, weight.main_grad)   # dw 直接累加进 fp32 主梯度
```

★ 但那依赖 Megatron 的 `main_grad` 手工梯度累加。**我们用 FSDP2 + MixedPrecisionPolicy,
没有 main_grad,该融合不适用 ⟹ 走 aten 对我们零损失。**

## 8. 结论与动作

```
短期(今天就能做)  抄 torchtitan-npu 那 30 行进桥接 → 模型里改成 torch._grouped_mm
                    删掉 _NpuGroupedMatmul(前向 + 手写反向)、删掉 device.type 硬门控
验证(一张卡,几分钟) parity(重点打 dw 那一路,group_type=2 是关键)+ 微基准拿到 N 倍
中期                向 op-plugin 提 issue 请求下沉:
                      · 底层 aclnnGroupedMatmul 现成,同目录有 ScaledGroupMm 模板
                      · torchtitan-npu 已在框架侧验证语义(附文件)
                      · 附我们的实测 N 倍 + 真实 MoE 训练场景
                    ⚠ 门槛在流程不在技术:UT(参照 +1200 行)、多芯片矩阵、内部评审;
                      外部贡献者历史上没做过算子新增(301 提交中 34 个公共邮箱,
                      全是 format/docs/小 bug/版本兼容)
长期                另两个算子(npu_moe_token_permute / npu_swiglu)在 aten 里
                    没有对应物,属于那 370 个自有算子。用它们仍需自写 autograd.Function,
                    或用原生 torch 表达 + 交给 inductor 融合。
```

★ 一句话:**不是"怎么把昇腾算子塞进 speculators",是"把 `_grouped_mm` 加进昇腾已经在做的
aten 对齐工作里"。** 补一次,speculators / vLLM / torchtitan / 任何走原生 torch 的 MoE 训练
在昇腾上全部白拿,而且各自代码里零厂商字样。

---

## 附:原稿被推翻的部分

原稿提议在 speculators 里自建一个 kernel 注册表 + entry-point 插件点。**不需要了** ——
torch 的 dispatcher 就是注册表,`torch_npu:_autoload` 就是插件点(`import torch` 时
torch 自己按 entry-point 把 torch_npu 拉起来,它再把 1490 个 aten 实现注册进 PrivateUse1)。
我们已写的 `kernels.py::discover_plugins` / `npu_bridge.py` 保留但降级为过渡设施。
