# Curated run-log excerpts

Whole step records cut out of a full training log, redacted, and committed on purpose.
`*.log` is ignored repo-wide; this directory is the one exemption
(`!docs/deployment/logs/*.log` in `.gitignore`), so excerpts land here and nowhere else.

A full run log is 100–250 MB and belongs in the run directory on the box, not in git.
What belongs here is the part someone else has to read: the step windows that carry a
result, plus the launcher banner that says which commit and recipe produced them.

## Size: small enough to open is not small enough to push

The training box's network caps ONE HTTP request at ~100 KB, and `git push` sends the
whole pack as a single POST — so the limit lands on the **push**, not on the file. A
2.6 MB excerpt committed fine, packed to 193 KB, and came back `HTTP 403`.

**This is that box, not GitHub and not everywhere.** From an unrestricted machine, 84 MB
of raw shards went in one push in 6.4 s. GitHub's own limit is 100 MB per *file*. So the
per-part loop matters when pushing from the box; elsewhere, `git add <dir> && git commit
&& git push` is enough.

Keep the raw excerpt **under ~1 MB** (git's delta+zlib on this kind of log runs ~13x).
`extract_log_steps.py` checks the size it produced and tells you which way to go. For a
66k-step run, `--every 2000 --last 40` is ~70 records and lands well inside one push.

Anything larger goes through `archive_log_push.sh` (`pack` / `push` / `verify`), which
cuts at record boundaries and does **one part per commit, one push per commit**,
resumable.

## Making one

```bash
E=examples/ascend_npu_dflash
L=<run>/faithful_ep_<TS>.log

python $E/extract_log_steps.py "$L" /tmp/cut.log --every 100 --last 300
python $E/redact_log.py /tmp/cut.log docs/deployment/logs/faithful_ep_<TS>.log
git add docs/deployment/logs/faithful_ep_<TS>.log && git commit && git push
```

- `extract_log_steps.py` keeps **whole records**. Do not substitute `grep global_step=`:
  the logger wraps one step across ~26 physical lines and only the last carries that
  token, so grepping it drops loss, accept_len and step_ms.
- `redact_log.py` is **not optional** — this fork is public and every absolute path in a
  log carries the box account id. It handles ids the logger split across two lines, which
  a plain `sed` would miss.
- Name the file after the run (`faithful_ep_<TS>.log`) so it joins up with that run's
  `provenance.txt` and its row in the worklog's asset ledger.

For a full archive rather than an excerpt, use `archive_log_push.sh` (`pack` / `push` /
`verify`), which splits at record boundaries and pushes one part per commit.
