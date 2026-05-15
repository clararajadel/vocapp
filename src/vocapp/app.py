from vocapp.storage import (
    ensure_dirs,
    load_vocab,
    load_memory,
    save_memory,
    get_memory_file
)

from vocapp.logic import (
    compare,
    update_memory,
    sample_words,
    add_word,
    get_progress
)

import random
from nicegui import ui


# ---------------- GAME STATE ----------------
game = {
    "vocab": None,
    "memory": None,
    "word_idx": [],
    "ex_idx": [],
    "i": 0,
    "direction": None,

    "current_word": None,
    "current_answer": None,
    "current_options": None,
}


# ---------------- GAME ENGINE ----------------
def run_game(direction: str):
    ensure_dirs()

    game["direction"] = direction

    if direction == "1":
        memory_cols = ["priority_1", "priority_2", "priority_3"]
    else:
        memory_cols = ["priority_1", "priority_2"]

    vocab = load_vocab()
    memory = load_memory(memory_cols, len(vocab))

    word_idx, ex_idx = sample_words(memory, memory_cols)

    if word_idx is None:
        return "All learned"

    game["vocab"] = vocab
    game["memory"] = memory
    game["word_idx"] = word_idx
    game["ex_idx"] = ex_idx
    game["i"] = 0

    return next_question()


def next_question():
    i = game["i"]

    if i >= len(game["word_idx"]):
        add_word(game["memory"])
        save_memory(game["memory"], get_memory_file())
        return f"Progress: {int(get_progress(game['memory']) * 100)}%"

    w = game["word_idx"][i]
    e = game["ex_idx"][i]

    vocab = game["vocab"]

    # ---------------- EXERCISE 1 ----------------
    if e == 0:
        o = vocab.loc[w, "original"]
        t = vocab.loc[w, "translated"]

        wrong = list(
            vocab[vocab["translated"] != t]["translated"].sample(3)
        )
        options = wrong + [t]
        random.shuffle(options)

        game["current_word"] = w
        game["current_answer"] = t
        game["current_options"] = options

        return f"Translate: {o}"

    # ---------------- EXERCISE 2 ----------------
    if e == 1:
        o = vocab.loc[w, "original"]
        t = vocab.loc[w, "translated"]

        game["current_word"] = w
        game["current_answer"] = t

        return f"Translate: {o}"

    return "Not implemented"


# ---------------- ANSWER HANDLING ----------------
def check_answer(user_answer: str):
    w = game["current_word"]
    correct = game["current_answer"]

    result = compare(user_answer, correct)

    game["memory"] = update_memory(game["memory"], (w, 0), result)

    game["i"] += 1

    return next_question()


# ---------------- UI ----------------
def main():
    ui.label("VocApp")

    output = ui.label("Choose mode")
    container = ui.column()

    # ---------------- RENDER ----------------
    def render():
        container.clear()

        if game["vocab"] is None:
            return

        if game["i"] >= len(game["word_idx"]):
            output.set_text(next_question())

            with container:
                ui.button("Restart", on_click=reset)

            return

        w = game["word_idx"][game["i"]]
        e = game["ex_idx"][game["i"]]

        vocab = game["vocab"]

        # ---------------- EXERCISE 1 UI ----------------
        if e == 0:
            o = vocab.loc[w, "original"]
            options = game["current_options"]

            output.set_text(f"Translate: {o}")

            with container:
                for opt in options:
                    ui.button(opt, on_click=lambda o=opt: answer(o))

        # ---------------- EXERCISE 2 UI ----------------
        elif e == 1:
            o = vocab.loc[w, "original"]

            output.set_text(f"Translate: {o}")

            with container:
                inp = ui.input("Your answer")

                ui.button(
                    "Submit",
                    on_click=lambda: answer(inp.value)
                )

        else:
            output.set_text("Not implemented")

            with container:
                ui.button("Next", on_click=next_step)

    # ---------------- ACTIONS ----------------
    def start_bidirectional():
        run_game("1")
        render()

    def start_one_directional():
        run_game("2")
        render()

    def answer(value):
        msg = check_answer(value)
        output.set_text(msg)
        render()

    def next_step():
        game["i"] += 1
        render()

    def reset():
        game["vocab"] = None
        game["i"] = 0
        container.clear()
        output.set_text("Choose mode")

    # ---------------- START BUTTONS ----------------
    ui.button("Bidirectional", on_click=start_bidirectional)
    ui.button("One-directional", on_click=start_one_directional)

    ui.run()


# ---------------- RUN ----------------
if __name__ in {"__main__", "__mp_main__"}:
    main()
