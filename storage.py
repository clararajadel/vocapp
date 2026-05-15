import pandas as pd
import os

VOCAB_FILE = "/home/clara/Documents/vocapp/data/verben_praepositionen.csv"

DATA_DIR = os.path.dirname(VOCAB_FILE)

MEMORY_DIR = os.path.join(DATA_DIR, "memory")

memory_file = os.path.join(
    MEMORY_DIR,
    os.path.basename(VOCAB_FILE).replace(".csv", "_memory.csv")
)


def ensure_dirs():
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)


def load_vocab():
    return pd.read_csv(VOCAB_FILE)


def load_memory(memory_cols, vocab_len):
    if not os.path.exists(memory_file):
        memory_dict = {col: [0.0] * vocab_len for col in memory_cols}
        memory_dict["learnt"] = [0.0] * vocab_len

        memory = pd.DataFrame(memory_dict)

        # activate some initial words
        import random
        indices = random.sample(range(vocab_len), min(18, vocab_len))
        memory.loc[indices, memory_cols[0]] = 1

        memory.to_csv(memory_file, index=False)
        return memory

    memory = pd.read_csv(memory_file)

    # extend memory if vocab grows
    if vocab_len > len(memory):
        rows = vocab_len - len(memory)
        new_rows = pd.DataFrame({col: [0.0] * rows for col in memory.columns})
        memory = pd.concat([memory, new_rows], ignore_index=True)

    return memory


def save_memory(memory):
    memory.round(2).to_csv(memory_file, index=False)
