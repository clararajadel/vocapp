import os

# Override with environment variable if set:
# export VOCAPP_VOCAB=/path/to/your/file.csv
VOCAB_FILE = os.environ.get(
    "VOCAPP_VOCAB",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "verben_praepositionen.csv")
)

LEARNING_RATE = 0.5
ACTIVE_WORDS = 18
TRIES_PER_ROUND = 20
