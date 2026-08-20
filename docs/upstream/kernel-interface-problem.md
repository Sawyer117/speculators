# 昇腾融合算子怎么进上游 —— 问题与方案

一页纸,供内部讨论。结论在最后。

---

## 问题

我们的 DSV4 草稿骨干里有一批昇腾融合算子(`npu_grouped_matmul`、`npu_moe_token_permute`
/`unpermute`、`npu_swiglu`),集中在两个文件、共 20 处 `torch_npu` 调用。

**上游 speculators 不会收它们。** 这不是猜测,有判例。

## 判例:同样是解决昇腾问题,一个合了一个关了

| PR | 做法 | 结果 |
|---|---|---|
| **#775** Enable Ascend NPU training | `src/` 里 best-effort import `torch_npu.contrib.transfer_to_npu`,让 `torch.cuda.*` 原样跑 | **两天后关闭,未合**(+17/−1) |
| **#589** selectable attention backend | 不塞昇腾代码,**加一个可选后端**(sdpa/eager),让 flex attention 不可用的平台去选 | **已合**(+234/−14) |

两者解决的是同一类问题(昇腾缺某个能力),**区别只在形式**:加厂商分支 vs 加通用选项。

而上游自己的设备处理也一致:全仓 `src/` 里 **0 处 `torch_npu` 直接调用、0 处 `is_cuda` 分支**,
统一走 `torch.accelerator.current_accelerator()`。它假定后端把自己注册进 torch 的设备抽象,
然后只写一份代码。

⟹ **边界很清楚:可移植的进上游,厂商的留外面。**

## 那算子放哪儿?两种放法,差别比看上去大

### 放法一:只收接口

上游收一个"算子接口 + torch 参考实现",昇腾算子住在我们的包里。
昇腾用户装完 speculators 之后,**还要自己找到桥接模块、手写一行 import**。

问题不在"多装一个包" —— **在于忘了装的时候,程序完全正确,只是悄悄地慢**,
而且没有任何东西会告诉他。

### 放法二:收一个插件点(推荐)

上游收的不是接口,而是一个 **entry-point 组**:

```toml
# 昇腾侧的包自己声明,上游不需要知道它存在
[project.entry-points."speculators.kernels"]
ascend = "speculators_ascend.bridge"
```

```python
# 上游 kernels.py 里 ~15 行,扫描并导入;被导入的包自己调 register_kernel
def discover_plugins(group="speculators.kernels") -> list[str]: ...
```

于是昇腾用户 **`pip install` 一次就完事** —— 不写 import,不改代码,模型和训练脚本里
不出现任何厂商名字。没装就走 torch 参考,并且启动时明确打印
`kernels: active=torch (no accelerator bridge installed)`。

**这段代码里没有出现任何厂商名。** 没装插件的机器上它什么都不加载,行为与现在完全一致 ——
这正是它能上游、而 `import torch_npu` 不能的原因。

## 这个形状为什么站得住

我们已经在用的 `kernels.py` 本来就是这个形状(注册表 + 纯 torch 参考 + 调用时解析),
它的设计目标写在文件头上:

> **Torch reference is the source of truth.** 注册的 kernel 要对着它做 parity 校验;
> **零 kernel 注册时模型依然正确。**

★ 这一句是关键:它让昇腾算子从**依赖**变成**可插拔的加速**。
维护者不需要为它负责,也**不需要有昇腾机器才能改这份代码**。

## 我们要付的代价(别自欺)

- 桥接包要自己发布、维护、跟注册表契约的版本对齐
- 要维护对着 torch 参考的 parity 测试
- 上游改了注册表我们就得跟。缓解办法:**把 op 集合保持极小**(现在只有 3 个)

## 那还值得上游吗?值得,但理由不是算子

**模型定义和 EP 机制是可移植的,只有算子是厂商的。**
上游收走可移植的那部分,昇腾用户就只背一层薄加速层;
否则要背整个 fork,而 fork 会随上游漂移。

而且这就是 **vLLM 生态本来的样子**:昇腾用户用 vLLM 时本来就要装 `vllm-ascend`
(一个树外平台插件),vLLM 内核里零昇腾算子。多装一个薄桥接是同一个形状,不是委屈。

## 具体切分账(那 545 行)

| | 行数 | 去处 |
|---|---:|---|
| `moe_ep.py`(专家并行 all-to-all) | 177 | **上游** —— 零 `torch_npu`,纯 `torch.distributed` |
| `kernels.py` + 插件点 | ~110 | **上游** —— 纯通用,而且是最有价值的一块 |
| `_grouped_matmul_torch`(参考实现兼 parity oracle) | ~30 | **上游** |
| `moe_grouped_gemm.py` 的 npu 部分 | ~200 | 桥接包 |
| `moe_compile.py` | 120 | 桥接包 |

**约 340 行可上游,205 行留桥接。**

## 现状与欠账

已实现:`npu_bridge.py`(唯一允许 import `torch_npu` 的模块,CPU 上是 no-op、幂等、不抛异常)
+ `kernels.py::discover_plugins()`。CPU 上已验证:无插件时回落 torch 参考,模拟注册后正确切换。

⚠ **还欠一步,要硬件**:`moe_grouped_gemm.py:108` 还留着一条绕过注册表的硬门控
`if x.device.type == "npu"`。改成走 `get_kernel` 之后,那两个文件里的 `torch_npu`
才能真正搬进桥接。这一步得在昇腾上验一遍,不是纸面重构。
