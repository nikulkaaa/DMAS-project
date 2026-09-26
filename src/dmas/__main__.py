"""Allow ``python -m dmas <command>``, equivalent to ``dmas <command>``."""

from dmas.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
