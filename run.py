import sys
import os

# add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from vocapp.app import main


if __name__ in {"__main__", "__mp_main__"}:
    main()
