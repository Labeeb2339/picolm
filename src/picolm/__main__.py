"""Allow ``python -m picolm`` to invoke the CLI."""

from picolm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
