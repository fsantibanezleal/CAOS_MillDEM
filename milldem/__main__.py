"""Enable ``python -m milldem`` as an alias for the ``milldem`` console script."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
