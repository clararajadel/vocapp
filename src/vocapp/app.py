from vocapp.storage import (
    ensure_dirs,
    load_vocab,
    load_memory,
    save_memory
)

from vocapp.logic import (
    compare,
    update_memory,
    sample_words,
    add_word,
    get_progress
)

import random


def main():
    print("App started")


# ---------------- EXERCISES ----------------
def exercise_1(vocab, w):
    o = vocab.loc[w, "original"]
    t = vocab.loc[w, "translated"]
    ex = vocab.loc[w, "example"]

    wrong = list(vocab[vocab["translated"] != t]["translated"].sample(3))
    options = wrong + [t]
    random.shuffle(options)

    print(f"\nChoose translation for: {o}")
    for i, opt in enumerate(options, 1):
        print(i, opt)

    while True:
        ans = input("1-4: ")
        if ans.isdigit() and 1 <= int(ans) <= 4:
            selected = options[int(ans) - 1]
            break

    if compare(selected, t):
        print("Correct")
        return True
    else:
        print("Wrong:", t)
        print("Example:", ex)
        return False


def exercise_2(vocab, w):
    o = vocab.loc[w, "original"]
    t = vocab.loc[w, "translated"]
    ex = vocab.loc[w, "example"]

    print("Translate:", o)
    ans = input()

    if compare(ans, t):
        print("Correct")
        return True
    else:
        print("Wrong:", t)
        print("Example:", ex)
        return False


def exercise_3(vocab, w):
    o = vocab.loc[w, "original"]
    t = vocab.loc[w, "translated"]
    ex = vocab.loc[w, "example"]

    print("Write original for:", t)
    ans = input()

    if compare(ans, o):
        print("Correct")
        return True
    else:
        print("Wrong:", o)
        print("Example:", ex)
        return False


# ---------------- ROUND ----------------
def do_round():
    ensure_dirs()

    print("1: bidirectional")
    print("2: one-directional")

    direction = input("> ")

    if direction == "1":
        exercises = [1, 2, 3]
        memory_cols = ["priority_1", "priority_2", "priority_3"]
    else:
        exercises = [1, 2]
        memory_cols = ["priority_1", "priority_2"]

    vocab = load_vocab()
    memory = load_memory(memory_cols, len(vocab))

    word_idx, ex_idx = sample_words(memory, memory_cols)

    if word_idx is None:
        return "All learned"

    exercise_map = {
        0: exercise_1,
        1: exercise_2,
        2: exercise_3
    }

    for i, (w, e) in enumerate(zip(word_idx, ex_idx)):
        print(f"\nProgress {i+1}")

        correct = exercise_map[e](vocab, w)
        memory = update_memory(memory, (w, e), correct)

        if not correct:
            input()

    add_word(memory)
    save_memory(memory, get_memory_file())

    return f"Progress: {int(get_progress(memory)*100)}%"


# ---------------- RUN ----------------
if __name__ == "__main__":
    print(do_round())
