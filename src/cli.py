"""Compatibility entry point for ``python -m cli``.

New code should import :mod:`litmus_link.cli` or use the installed
``litmus-link`` command.
"""

from litmus_link.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
