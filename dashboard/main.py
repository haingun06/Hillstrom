import sys

from hillstrom_app.ui.main_window import HillstromIntegrated, main  # noqa: F401
from hillstrom_app.ui.theme import apply_light_theme  # noqa: F401

if __name__ == '__main__':
    sys.exit(main())
