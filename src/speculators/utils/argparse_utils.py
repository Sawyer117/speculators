"""Small, reusable argparse helpers.

Upstream deleted this module in #841, which replaced ``scripts/train.py``'s argparse
with a pydantic ``TrainConfig``. This fork still owns ~40 DSV4/DSpark-only flags that
have no schema fields yet, so it keeps the argparse entry point and therefore this
helper. Delete it again once those flags move into ``train/config/schema.py``.
"""

import argparse
from collections.abc import Iterable


def explicitly_provided_dests(
    parser: argparse.ArgumentParser,
    dests: Iterable[str],
    argv: list[str] | None = None,
) -> set[str]:
    """Return the subset of ``dests`` whose option actually appeared on the CLI.

    Re-parses ``argv`` into a throwaway namespace pre-seeded with a unique sentinel
    per dest. argparse skips applying an action's default when the attribute is already
    present on the namespace, and only overwrites it when the option is provided -- so a
    value that merely *equals* the default is still detected as explicitly provided
    (unlike comparing the parsed value against ``parser.get_default(dest)``).

    :param parser: The parser to re-run.
    :param dests: Argparse destination names to check (e.g. ``"num_layers"``).
    :param argv: Arguments to re-parse; defaults to ``None`` (i.e. ``sys.argv``). Pass
        the same ``argv`` the parser was originally invoked with so the detected
        destinations match the arguments actually parsed.
    :return: The subset of ``dests`` that were explicitly provided.
    """
    sentinels = {dest: object() for dest in dests}
    namespace = argparse.Namespace(**sentinels)
    parser.parse_args(args=argv, namespace=namespace)
    return {
        dest
        for dest, sentinel in sentinels.items()
        if getattr(namespace, dest) is not sentinel
    }
