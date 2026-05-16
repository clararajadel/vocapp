from flask import Flask, render_template, request, session, redirect, url_for
import random
import os

from vocapp.storage import ensure_dirs, load_vocab, load_memory, save_memory, get_memory_file
from vocapp.logic import compare, update_memory, sample_words, add_word, get_progress

app = Flask(__name__)
app.secret_key = "vocapp-secret-key-change-in-production"


# ---------------- HOME ----------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------- START ROUND ----------------
@app.route("/start", methods=["POST"])
def start():
    direction = request.form.get("direction", "1")

    if direction == "1":
        memory_cols = ["priority_1", "priority_2", "priority_3"]
    else:
        memory_cols = ["priority_1", "priority_2"]

    ensure_dirs()
    vocab = load_vocab()
    memory = load_memory(memory_cols, len(vocab))

    word_idx, ex_idx = sample_words(memory, memory_cols)

    if word_idx is None:
        return render_template("result.html", message="All words learned!", progress=100, score=None)

    session["direction"] = direction
    session["memory_cols"] = memory_cols
    session["word_indices"] = word_idx.tolist()
    session["ex_indices"] = ex_idx.tolist()
    session["current"] = 0
    session["score"] = {"correct": 0, "wrong": 0}
    session["total"] = len(word_idx)

    return redirect(url_for("question"))


# ---------------- QUESTION ----------------
@app.route("/question")
def question():
    if "word_indices" not in session:
        return redirect(url_for("index"))

    current = session["current"]
    total = session["total"]

    if current >= total:
        return redirect(url_for("finish"))

    vocab = load_vocab()
    memory_cols = session["memory_cols"]
    memory = load_memory(memory_cols, len(vocab))

    w = session["word_indices"][current]
    e = session["ex_indices"][current]

    original = vocab.loc[w, "original"]
    translated = vocab.loc[w, "translated"]
    example = vocab.loc[w, "example"] if vocab.loc[w, "example"] else None
    note = vocab.loc[w, "note"] if "note" in vocab.columns and vocab.loc[w, "note"] else None

    options = None
    if e == 0:
        wrong = list(vocab[vocab["translated"] != translated]["translated"].sample(3))
        options = wrong + [translated]
        random.shuffle(options)

    exercise_labels = {
        0: ("multiple_choice", f"Choose the correct translation for:"),
        1: ("free_text",       f"Translate:"),
        2: ("reverse",         f"Write the original word for:"),
    }
    ex_type, prompt = exercise_labels[e]

    display_word = original if e in (0, 1) else translated

    return render_template(
        "question.html",
        current=current + 1,
        total=total,
        score=session["score"],
        ex_type=ex_type,
        prompt=prompt,
        display_word=display_word,
        options=options,
        w=w,
        e=e,
    )


# ---------------- ANSWER ----------------
@app.route("/answer", methods=["POST"])
def answer():
    if "word_indices" not in session:
        return redirect(url_for("index"))

    w = int(request.form.get("w"))
    e = int(request.form.get("e"))
    user_answer = request.form.get("answer", "").strip()

    vocab = load_vocab()
    memory_cols = session["memory_cols"]
    memory = load_memory(memory_cols, len(vocab))

    original = vocab.loc[w, "original"]
    translated = vocab.loc[w, "translated"]
    example = vocab.loc[w, "example"] if vocab.loc[w, "example"] else None
    note = vocab.loc[w, "note"] if "note" in vocab.columns and vocab.loc[w, "note"] else None

    correct_answer = translated if e in (0, 1) else original
    is_correct = compare(user_answer, correct_answer)

    memory = update_memory(memory, (w, e), is_correct)
    save_memory(memory, get_memory_file())

    score = session["score"]
    if is_correct:
        score["correct"] += 1
    else:
        score["wrong"] += 1
    session["score"] = score
    session["current"] = session["current"] + 1

    return render_template(
        "feedback.html",
        is_correct=is_correct,
        user_answer=user_answer,
        correct_answer=correct_answer,
        original=original,
        translated=translated,
        example=example,
        note=note,
        current=session["current"],
        total=session["total"],
        score=score,
    )


# ---------------- FINISH ----------------
@app.route("/finish")
def finish():
    if "word_indices" not in session:
        return redirect(url_for("index"))

    vocab = load_vocab()
    memory_cols = session["memory_cols"]
    memory = load_memory(memory_cols, len(vocab))

    add_word(memory)
    save_memory(memory, get_memory_file())

    progress = int(get_progress(memory) * 100)
    score = session.get("score", {})
    total = session.get("total", 0)

    session.clear()

    return render_template("result.html", progress=progress, score=score, total=total)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
