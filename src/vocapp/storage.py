import pandas as pd
import os
from vocapp.config import VOCAB_FILE

def get_memory_file():
    data_dir = os.path.dirname(VOCAB_FILE)
    memory_dir = os.path.join(data_dir, "memory")

    base = os.path.basename(VOCAB_FILE)
    name, _ = os.path.splitext(base)

    return os.path.join(memory_dir, f"{name}_memory.csv")


def ensure_dirs():
    memory_file = get_memory_file()
    memory_dir = os.path.dirname(memory_file)

    os.makedirs(memory_dir, exist_ok=True)


def load_vocab():
    return pd.read_csv(VOCAB_FILE)


def load_memory(memory_cols, vocab_len):
    memory_file = get_memory_file()
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


def save_memory(memory, memory_file):
    memory.round(2).to_csv(memory_file, index=False)
