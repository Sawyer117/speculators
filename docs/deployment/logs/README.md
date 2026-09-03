# 训练日志存档

## `faithful_ep_20260804_165215.redacted.log.xz`

DSV4-Flash DSpark 草稿模型 5 个 epoch 主训练的**完整原始日志**,不是摘要、不是抽样。

```
原始        265,539,558 字节 (253 MB) · 3,255,603 行
xz -9        11,452,832 字节 (10.9 MB)
训练步       124,480(= 5 × 24,896)
MoE 记录     18,672 条(3 层 × 6,224 次)
```

```bash
xz -d -k faithful_ep_20260804_165215.redacted.log.xz
python examples/ascend_npu_dflash/analyze_train_run.py \
    faithful_ep_20260804_165215.redacted.log --out ./analysis
```

分析器会输出 `loss.png` / `accept_len.png` / `acceptance.png` / `position_acc.png` /
`confidence.png` / `timing.png` / `moe_experts.png`,以及稳态耗时、尖峰检测、NaN 定位的文字报告。

### 已做的脱敏

内部账号被替换为占位符,**共 77 处**(71 处完整 + 6 处被日志折行截断的):

```
a00652497  ->  <USER_A>
n84449292  ->  <USER_B>
```

除此之外**逐字节未改**。发布前扫描过:无凭据、无 IP、无 endpoint 主机名。

### ⚠️ 别用 grep 蒸馏这份日志

rich logger 把**一步的指标折成约 26 个物理行**。`grep global_step=` 只会抓到携带它的那一行,
`train/loss` / `train/accept_len` / `profile/step_ms` 全部丢失 —— 实测 124,480 条变 0 条。
要拆就用 `examples/ascend_npu_dflash/split_train_log.py`,它先把折行记录重组再解析。

### 相关工具

| 脚本 | 用途 |
|---|---|
| `analyze_train_run.py` | 读日志出图 + 文字报告 |
| `split_train_log.py` | 按指标 family 拆成小 CSV(loss / accept / timing / moe_load …) |
| `archive_log_push.sh` | 网关限制单次请求体时,按**记录边界**切片 + 逐片推送 + sha256 校验 |
