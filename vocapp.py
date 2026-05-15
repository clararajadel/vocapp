import pandas as pd
import random
import os
import numpy as np
import re
import unicodedata

# ------------------------------
# CONFIGURATION
# ------------------------------
VOCAB_FILE = "/mnt/c/Users/clara/OneDrive/Documents/vocapp/verben_praepositionen.csv"   # Vocabulary file with "original" and "translated"
TRIES_PER_ROUND = 20            # Total tries per round
ACTIVE_WORDS = 18               # Number of words to be learned
LEARNING_RATE = 0.5             # Value to change priority

cache = os.path.join(os.path.dirname(VOCAB_FILE), ".cache")
memory_file = os.path.join(
    cache,
    os.path.basename(VOCAB_FILE).replace(".csv", "_memory.csv")
    )  # File to track user's progress

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def load_data(memory_cols):
    # Load vocabulary
    vocab = pd.read_csv(VOCAB_FILE)
    vocab_len = len(vocab)

    # Check if memory exists
    if not os.path.exists(memory_file):
        # Create memory from scratch
        memory_dict = {}
        
        # Add memory_cols
        for exercise in memory_cols:
            memory_dict[exercise] = [0] * vocab_len
        
        # Add 'learnt' column
        memory_dict["learnt"] = [0] * vocab_len

        memory = pd.DataFrame(memory_dict)

        # Activate initial "active_words" rows
        indices = random.sample(range(vocab_len), min(ACTIVE_WORDS, vocab_len))
        memory.loc[indices, "priority_exercise_1"] = 1

        # Save memory file
        memory.to_csv(memory_file, index=False)
        return vocab, memory

    # Load existing memory
    memory = pd.read_csv(memory_file)
    mem_len = len(memory)

    # If vocab has grown, append missing rows
    if vocab_len > mem_len:
        rows_to_add = vocab_len - mem_len
        new_rows_dict = {col: [0]*rows_to_add for col in memory.columns}
        new_rows = pd.DataFrame(new_rows_dict)
        memory = pd.concat([memory, new_rows], ignore_index=True)
        memory.to_csv(memory_file, index=False)

    return vocab, memory

def update_memory(memory, indices, correct):
    w = indices[0]
    e = indices[1]

    # Update the learning rate if the weight is lower
    step = min(LEARNING_RATE, memory.iloc[w, e])

    # Fisrt exercise: Exercise 1
    if e == 0:
        if correct:
            memory.iloc[w, e] -= step
            if memory.iloc[w, e] < 1:
                memory.iloc[w, e+1] += step
        else:
            memory.iloc[w, e] += step
            if memory.iloc[w, e+1] > 0:
                memory.iloc[w, e+1] -= step
    
    # Intermediate exercises
    else:
        if correct:
            memory.iloc[w, e] -= step
            memory.iloc[w, e+1] += step
        else:
            memory.iloc[w, e] -= step
            memory.iloc[w, e-1] += step

    if (memory < 0).any().any():  # just to check that all works good
        raise ValueError(
            f"Memory contains negative values!"
            f"indices: {indices}"
            )

    return memory

def save_memory(memory):
    memory.round(2).to_csv(memory_file, index=False)

def normalize_part(text):
    text = re.sub(r"\(.*?\)", "", text)  # remove parentheses
    text = text.replace('"', '').replace("'", "")  # remove quotes

    result = []
    for char in text:
        if char in "äöüÄÖÜ":
            result.append(char)
        else:
            decomposed = unicodedata.normalize('NFD', char)
            stripped = ''.join(c for c in decomposed if not unicodedata.combining(c))
            result.append(stripped)

    return ''.join(result).strip().lower()


def compare(selected, translated):
    # split by comma
    parts1 = [normalize_part(p) for p in selected.split(",") if p.strip()]
    parts2 = [normalize_part(p) for p in translated.split(",") if p.strip()]

    return set(parts1) == set(parts2)

def add_word(memory):
    """Check if total priority >= ACTIVE_WORDS, and assign a new word randomly if available."""
    total_priority = memory.iloc[:, :-1].sum().sum()
    while total_priority <= ACTIVE_WORDS:
        # Find words not yet learned
        candidates = memory[(memory == 0).all(axis=1)]
        if len(candidates) > 0:
            new_word = candidates.sample(1).index[0]
            memory.iloc[new_word, 0] = 1
            total_priority = memory.iloc[:, :-1].sum().sum()
        else:
            break

def print_progress_bar(current, total, bar_length=30):
    """
    Prints a progress bar with current progress and try number.
    """
    fraction = current / total
    filled_length = int(bar_length * fraction)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f"\n [{bar}]")


# ------------------------------
# EXERCISES
# ------------------------------
def exercise_1(vocab, word_idx):
    """
    Exercise "Choose the word"
    """
    original = vocab.loc[word_idx, "original"]
    translated = vocab.loc[word_idx, "translated"]
    example = vocab.loc[word_idx, "example"]

    # Pick 3 random wrong options
    wrong_options = list(vocab[vocab["translated"] != translated]["translated"].sample(3))
    options = wrong_options + [translated]
    random.shuffle(options)

    print(f"\nChoose the correct word for: {original}")
    for i, opt in enumerate(options, 1):
        print(f"{i}: {opt}")

    # Prompt user until they give a valid choice
    while True:
        answer = input("Please choose one of the options (1-4): ").strip()
        if answer.isdigit() and 1 <= int(answer) <= 4:
            selected = options[int(answer)-1]
            break
        else:
            print("Invalid input. Type the number corresponding to one of the options (1-4).")

    if compare(selected, translated):
        print("Correct!")
        return True
        
    else:
        print(f"Wrong! The correct answer was: {translated}")
        print("For example:", example)
        return False

def exercise_2(vocab, word_idx):
    """
    Exercise "Write translated"
    """
    original = vocab.loc[word_idx, "original"]
    translated = vocab.loc[word_idx, "translated"]
    example = vocab.loc[word_idx, "example"]

    print(f"\nTranslate this word: {original}")
    answer = input("Your answer: ").strip()
    if compare(answer.lower(), translated.lower()):
        print("Correct!")
        return True
    else:
        print(f"Wrong! The correct answer is: {translated}")
        print("For example:", example)
        return False

def exercise_3(vocab, word_idx):
    """
    Exercise "Write original"
    """
    original = vocab.loc[word_idx, "original"]
    translated = vocab.loc[word_idx, "translated"]
    example = vocab.loc[word_idx, "example"]

    print(f"\nWrite the original word for: {translated}")
    answer = input("Your answer: ").strip()
    if compare(answer.lower(), original.lower()):
        print("Correct!")
        return True
    else:
        print(f"Wrong! The correct answer is: {original}")
        print("For example:", example)
        return False


# ------------------------------
# MAIN ROUND
# ------------------------------
def do_round():

    # Ensure cache folder exists
    if not os.path.exists(cache):
        os.makedirs(cache)

    # Ask user about vocabulary direction
    print("Is this vocabulary bidirectional or one-directional?")
    print("1: Bidirectional (all exercises)")
    print("2: One-directional (ignore exercise 3)")

    while True:
        direction = input("Your choice (1 or 2): ").strip()
        if direction in ["1", "2"]:
            break
        else:
            print("Invalid input. Please type 1 or 2.")

    # Define exercises and memory map based on direction
    if direction == "1":  # bidirectional
        exercises = [1, 2, 3]
        memory_map = {
            1: "priority_exercise_1",
            2: "priority_exercise_2",
            3: "priority_exercise_3"
        }
    else:  # one-directional
        exercises = [1, 2]  # ignore exercise 3
        memory_map = {
            1: "priority_exercise_1",
            2: "priority_exercise_2"
        }

    # Select memory columns for the active exercises
    memory_cols = [memory_map[e] for e in exercises if e in memory_map]

    # Load vocabulary and memory
    vocab, memory = load_data(memory_cols)

    # Get priority weights for sampling words
    weights = memory[memory_cols].to_numpy()

    tries_this_round = min(TRIES_PER_ROUND, (weights != 0).sum())

    if tries_this_round == 0:
        return "All learned. Congratulations!"

    # Sample words for the round
    weights_1d = weights.ravel().astype(float)
    weights_1d /= weights_1d.sum()  # normalize to probabilities
    sample = np.random.choice(
        weights_1d.size,
        size=tries_this_round,
        replace=False,
        p=weights_1d
    )
    word_idx, ex_idx = np.unravel_index(sample, weights.shape)

    # Map exercise indices to functions
    exercise_map = {
        0: exercise_1,
        1: exercise_2,
        2: exercise_3  # will never be called if one-directional
    }

    # Run exercises
    for i, (w, e) in enumerate(zip(word_idx, ex_idx)):
        print_progress_bar(i, tries_this_round)
        ex = exercise_map[e]
        correct = ex(vocab, w)
        memory = update_memory(memory, (w, e), correct)

        if not correct:
            input()

    # Promote new words if needed
    add_word(memory)

    # Save updated memory
    save_memory(memory)
    progress = memory.iloc[:, -1].mean()
    
    return f"Your Learning Progress is: {int(progress*100)} %"

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    msg = do_round()
    print("Round completed!")
    print(msg)

