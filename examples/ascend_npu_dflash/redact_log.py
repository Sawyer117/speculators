#!/usr/bin/env python3
"""Strip internal account ids out of a training log before it goes to a PUBLIC repo.

The DSV4-DSpark logs are full of absolute paths -- checkpoint dirs, conda envs, HS dump
roots -- and every one of them carries the box account name. `Sawyer117/speculators` is
public, so these have to go before the archive is pushed.

TWO FORMS, AND HANDLING ONLY THE FIRST IS A SILENT LEAK:

  * intact       ``.../home/a00652497/dspark_austin/run/...``
  * WRAPPED by the rich logger, split across two physical lines, the tail landing after
    the next line's indent::

        ... Saving checkpoint to                          trainer.py:722
                            /home/a006
                            52497/dspark_austin/run/...

    Measured on faithful_ep_20260804_165215.log: **65 intact, 6 wrapped**. A plain
    ``sed s/a00652497//g`` would have published those 6.

Streams line by line with a one-line lookahead, so a 253 MB log costs no memory.

USAGE
    redact_log.py --scan <log>                     # report ids found; exit 3 if any
    redact_log.py <src> <dst> [id=<PLACEHOLDER> ...]

With no explicit pairs, discovered ids are assigned ``<USER_A>``, ``<USER_B>``, ... in
first-appearance order, which keeps the mapping stable for a re-run of the same log.
"""

from __future__ import annotations

import re
import string
import sys

# Ascend box accounts look like one letter + 8 digits (a00652497, n84449292). Anchored to
# a path so ordinary hex/hashes in the log are not mistaken for account names.
ID = re.compile(r"/home/([a-z]\d{8})\b")
# ★ 2026-09-22:凭据。`http://user:pass@host:port` —— 代理密码。这不是假想:
#   `determinism_sweep.sh` 的 `export "$@"` 在无参数时会打印【整个环境】,于是
#   `HTTPS_PROXY=http://<user>:<pass>@...` 原样进了 ~/det_sweep.log。账号名只是尴尬,
#   密码是事故,所以它先于账号名被替换掉。
CRED = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)(?P<user>[^\s:/@]+):(?P<pw>[^\s@/]+)@")
# 以及 `PASSWORD=`/`TOKEN=`/`SECRET=` 这类裸赋值(shell 的 declare -x 会这么打)。
#   ⚠ 关键词前面允许有前缀:`GITHUB_TOKEN=` 里 `\btoken` 匹配不上 —— 下划线是单词字符,
#     `_` 和 `t` 之间没有词边界。所以写成 `\w*(?:token|…)`,测试用例里就是这么漏掉一条的。
SECRET_KV = re.compile(
    r"(?i)(\w*(?:pass(?:word|wd)?|token|secret|api[_-]?key))(\s*[=:]\s*\"?)([^\s\"']{4,})")


def scrub_secrets(line: str) -> tuple[str, int]:
    """把凭据换成占位符。返回 (新行, 命中数)。和账号名映射无关,永远执行。"""
    n = 0

    def _cred(m: re.Match) -> str:
        nonlocal n
        n += 1
        return f"{m.group('scheme')}{m.group('user')}:<REDACTED>@"

    line, k = CRED.subn(_cred, line)
    n += 0 * k

    def _kv(m: re.Match) -> str:
        nonlocal n
        n += 1
        return f"{m.group(1)}{m.group(2)}<REDACTED>"

    line, _ = SECRET_KV.subn(_kv, line)
    return line, n


# A line ending in `/home/<prefix>` means the rest of the id is on the NEXT line.
TAIL = re.compile(r"(/home/)([a-z]\d{0,8})([ ]*)$")


def scan(path: str) -> list[str]:
    """Return the account ids present in ``path``, in first-appearance order."""
    seen: dict[str, None] = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            for hit in ID.findall(line):
                seen.setdefault(hit, None)
    return list(seen)


def scan_partial(path: str) -> int:
    """Count lines ending in a TRUNCATED ``/home/<partial-id>``.

    Verifying with :func:`scan` alone is not enough and quietly reports success on a file
    that still leaks: when the logger wraps inside the id, neither half is a complete id,
    so the full-id regex sees nothing while ``/home/a006`` + ``52497/...`` still sits in
    the output in plain sight. Found exactly this way while testing the guard.
    """
    n = 0
    with open(path, errors="replace") as fh:
        for line in fh:
            if TAIL.search(line.rstrip("\n")):
                n += 1
    return n


def redact(src: str, dst: str, mapping: dict[str, str]) -> tuple[int, int, int]:
    """Rewrite ``src`` to ``dst`` with ``mapping`` applied.

    Returns (intact, wrapped, creds). Credentials are scrubbed unconditionally --
    they do not depend on any discovered account id.
    """
    intact = wrapped = creds = 0
    with open(src, errors="replace") as fh, open(dst, "w") as out:
        pending: str | None = None  # suffix to strip off the NEXT line
        for raw in fh:
            line, c = scrub_secrets(raw)
            creds += c
            if pending:
                stripped = line.lstrip()
                indent = line[: len(line) - len(stripped)]
                if stripped.startswith(pending):
                    line = indent + stripped[len(pending) :]
                    wrapped += 1
                pending = None

            for real, fake in mapping.items():
                if real in line:
                    intact += line.count(real)
                    line = line.replace(real, fake)

            m = TAIL.search(line.rstrip("\n"))
            if m:
                prefix = m.group(2)
                for real, fake in mapping.items():
                    if prefix and real.startswith(prefix) and prefix != real:
                        pending = real[len(prefix) :]
                        line = line[: m.start(2)] + fake + m.group(3) + "\n"
                        break
            out.write(line)
    return intact, wrapped, creds


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    if args[0] == "--scan":
        found, partial = scan(args[1]), scan_partial(args[1])
        ncred = 0
        with open(args[1], errors="replace") as fh:
            for line in fh:
                _, c = scrub_secrets(line)
                ncred += c
        if not found and not partial and not ncred:
            print("未发现账号 ID,也没有凭据")
            return 0
        msg = []
        if ncred:
            msg.append(f"★ 发现 {ncred} 处凭据(URL 密码 / PASSWORD= / TOKEN=)")
        if found:
            msg.append("发现账号 ID: " + " ".join(found))
        if partial:
            msg.append(f"另有 {partial} 行以被折断的 /home/<id> 结尾")
        print(" · ".join(msg))
        return 3

    if len(args) < 2:
        print(__doc__)
        return 1
    src, dst, pairs = args[0], args[1], args[2:]

    if pairs:
        mapping = dict(p.split("=", 1) for p in pairs)
    else:
        mapping = {
            real: f"<USER_{letter}>"
            for real, letter in zip(scan(src), string.ascii_uppercase, strict=False)
        }
    if not mapping:
        print("未发现账号 ID —— 原样复制")
    for real, fake in mapping.items():
        print(f"  {real} -> {fake}")

    intact, wrapped, creds = redact(src, dst, mapping)
    print(f"完整替换 {intact} 处 · 折行替换 {wrapped} 处  ->  {dst}")
    if creds:
        print(f"★ 另擦掉 {creds} 处凭据(URL 里的密码 / PASSWORD= / TOKEN= 之类)")

    # Prove it: a second scan of the OUTPUT must come back clean, for BOTH forms. Checking
    # only complete ids here printed "复查通过" on a file with 61 surviving wrapped ids.
    left, left_partial = scan(dst), scan_partial(dst)
    if left or left_partial:
        if left:
            print(f"!! 输出里仍有完整 ID: {' '.join(left)}", file=sys.stderr)
        if left_partial:
            msg = f"!! 输出里仍有 {left_partial} 行以折断的 /home/<id> 结尾"
            print(msg, file=sys.stderr)
        return 3
    print("复查通过:完整 ID 与折行 ID 都已清除")
    return 0


if __name__ == "__main__":
    sys.exit(main())
