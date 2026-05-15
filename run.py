import sys
import os

# add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from vocapp.app import do_round  # <-- your actual game function


if __name__ == "__main__":
    print("Starting VocApp...\n")
    result = do_round()
    print("\nRound finished!")
    print(result)
