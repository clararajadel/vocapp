import random
import numpy as np
import re
import unicodedata
from vocapp.config import LEARNING_RATE, ACTIVE_WORDS, TRIES_PER_ROUND


# ---------------- NORMALIZATION ----------------
def normalize_part(text):
    text = re.sub(r"\(.*?\)", "", text)
    text = text.replace('"', '').replace("'", "")

    result = []
    for char in text:
        if char in "äöüÄÖÜ":
            result.append(char)
        else:
            decomposed = unicodedata.normalize('NFD', char)
            stripped = ''.join(c for c in decomposed if not unicodedata.combining(c))
            result.append(stripped)

    return ''.join(result).strip().lower()


def compare(a, b):
    a_parts = {normalize_part(p) for p in a.split(",") if p.strip()}
    b_parts = {normalize_part(p) for p in b.split(",") if p.strip()}

    return a_parts == b_parts


# ---------------- MEMORY UPDATE ----------------
def update_memory(memory, indices, correct):
    w, e = indices

    step = min(LEARNING_RATE, memory.iloc[w, e])

    # First exercise
    if e == 0:
        if correct:
            memory.iloc[w, e] -= step

            if memory.iloc[w, e + 1] < 1:
                memory.iloc[w, e + 1] += step

        else:
            memory.iloc[w, e] += step

            if memory.iloc[w, e + 1] > 0:
                memory.iloc[w, e + 1] -= step

    # Intermediate exercises
    else:
        if correct:
            memory.iloc[w, e] -= step
            memory.iloc[w, e + 1] += step

        else:
            memory.iloc[w, e] -= step
            memory.iloc[w, e - 1] += step

    if (memory < 0).any().any():
        raise ValueError(
            f"Memory contains negative values! indices: {indices}"
        )

    return memory


# ---------------- SAMPLING ----------------
def sample_words(memory, memory_cols):
    weights = memory[memory_cols].to_numpy()

    total = (weights != 0).sum()
    tries = min(TRIES_PER_ROUND, total)

    if tries == 0:
        return None, None

    flat = weights.ravel().astype(float)
    flat /= flat.sum()

    sample = np.random.choice(
        flat.size,
        size=tries,
        replace=False,
        p=flat
    )

    return np.unravel_index(sample, weights.shape)


# ---------------- PROMOTION ----------------
def add_word(memory):
    total = memory.iloc[:, :-1].sum().sum()

    while total <= ACTIVE_WORDS:
        candidates = memory[(memory == 0).all(axis=1)]
        if len(candidates) == 0:
            break

        idx = candidates.sample(1).index[0]
        memory.iloc[idx, 0] = 1
        total = memory.iloc[:, :-1].sum().sum()


# ---------------- PROGRESS ----------------
def get_progress(memory):
    return memory.iloc[:, -1].mean()
