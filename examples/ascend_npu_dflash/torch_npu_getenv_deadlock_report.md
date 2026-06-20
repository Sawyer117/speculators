# torch_npu `patch_getenv` 逐次日志导致 fork 子进程死锁

**一句话**:torch_npu 对 `os.environ.get` 的逐次 INFO 日志,会在 fork 出的子进程
(如 PyTorch DataLoader worker)里,与继承来的**非 fork-safe 日志锁**相撞,造成**永久
死锁**;日常还会满屏刷 `get env ...`。建议**默认关闭 / 降级到 DEBUG / 用具名 logger**。

> English issue body below — usable as-is for a GitHub issue on the torch_npu repo.

---

## TL;DR

`patch_getenv.py` wraps `os.environ.get` to emit a logging record (INFO) on **every**
call. In any process that forks worker subprocesses (PyTorch `DataLoader` with
`num_workers > 0`, default `fork` start method on Linux), this turns a rare race into a
**near-certain hang**: the forked child inherits a logging handler whose lock was held by
a thread that does not exist in the child, and the **first patched `os.environ.get` in the
child blocks forever** on that lock.

## Environment (please fill in exact versions)

```bash
python -c "import torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__)"
python -c "import torch_npu, os.path as p; print(p.join(p.dirname(torch_npu.__file__), 'utils', 'patch_getenv.py'))"
```

- torch_npu: `<fill>`  (log shows `patch_getenv.py:15`)
- torch: `<fill>`   CANN: `<fill>`   NPU: `<fill>`
- OS: Linux, Python 3.11

## Causal chain

1. `DataLoader(num_workers>0)` forks worker processes (Linux default = `fork`).
2. `fork()` copies only the calling thread; **other threads vanish in the child**, but the
   memory (and any locks they held) is copied as-is. A lock held at fork time is inherited
   **locked with no owner** in the child.
3. The parent's root logger has a handler whose `emit()` takes a lock. If that lock is held
   at the moment of fork (e.g. a `rich` `RichHandler` + a `tqdm.rich` Live display on a TTY
   hold the rich `Console` lock almost continuously), the child inherits it dead-locked.
4. In the child, `patch_getenv` fires on the next `os.environ.get(...)` → `logging.info(...)`
   → handler `emit()` → tries to acquire the inherited (dead) lock → **blocks forever**.
5. The worker never returns a batch → main loop blocks in `DataLoader.__next__` → other
   distributed ranks eventually HCCL-timeout. It is a **hang, not a crash** — hard to
   diagnose without `py-spy dump`.

## Why this is a torch_npu trigger (not merely a downstream bug)

- Python's **own** logging locks are fork-safe — `logging` reinitializes handler locks in
  the child via `os.register_at_fork`. A plain `StreamHandler` therefore does **not**
  deadlock across fork.
- The hang only becomes **guaranteed** because `os.environ.get` — an extremely hot path —
  emits a log record inside the just-forked child, in arbitrary low-level code, **before**
  any downstream fork hook can run. High frequency + arbitrary call site = the amplifier
  that converts a microsecond race window into a certainty.

## Minimal reproduction

Any training that:
- runs on a **TTY** (so `rich`/`tqdm.rich` renders a Live display and holds the console
  lock continuously — piping stdout to a file hides the bug), and
- uses a `DataLoader` with `num_workers > 0`, and
- has a `RichHandler` (or any handler taking a non-fork-safe lock during `emit`) on the
  root logger.

→ hangs within the first few steps (random step, depending on which forked worker caught
the lock held and when its turn comes up in the prefetch rotation).

## Requested fix (priority order)

1. **Do not emit on every `os.environ.get` by default.** Gate it behind an explicit opt-in
   env var (default OFF). This alone removes the spam *and* the deadlock trigger.
2. If kept, log via a **dedicated named logger at DEBUG** (e.g.
   `logging.getLogger("torch_npu.getenv").debug(...)`), not the root logger at INFO — so it
   is silent under normal config, and downstream can `setLevel(WARNING)` to make the call
   short-circuit in `isEnabledFor()` **before any handler lock is touched**.
3. Optionally make the patch fork-aware (skip emitting in a just-forked child), or avoid
   globally patching `os.environ.get` at all (it is a very hot path).

## Note for downstream

The downstream project (speculators) is adding a fork-time logging mute as
**defense-in-depth** (`os.register_at_fork(after_in_child=...)` to drop handlers in forked
workers). That addresses the symptom generically; the **clean root-cause fix is on the
torch_npu side** per the above.
