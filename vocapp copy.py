import pandas as pd
import random
import os
import numpy as np

# ------------------------------
# CONFIGURATION
# ------------------------------
VOCAB_FILE = "/mnt/c/Users/clara/OneDrive/Documents/vocapp/verben_praepositionen.csv"   # Vocabulary file with "original" and "translated"
TRIES_PER_ROUND = 20            # Total tries per round
ACTIVE_WORDS = 30               # Number of words to be learned
LEARNING_RATE = 0.2             # Value to change priority

cache = os.path.join(os.path.dirname(VOCAB_FILE), ".cache")
memory_file = os.path.join(
    cache,
    os.path.basename(VOCAB_FILE).replace(".csv", "_memory.csv")
    )  # File to track user's progress

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def load_vocabulary():
    vocab = pd.read_csv(VOCAB_FILE)
    vocab_len = len(vocab)

    if not os.path.exists(memory_file):
        # Create memory from scratch
        memory = pd.DataFrame({
            "priority_exercise_1": [0]*vocab_len,
            "priority_exercise_2": [0]*vocab_len,
            "priority_exercise_3": [0]*vocab_len,
            "learnt": [0]*vocab_len,
        })

        # Activate initial "ACTIVE_WORDS" rows
        indices = random.sample(range(vocab_len), min(ACTIVE_WORDS, vocab_len))
        memory.loc[indices, "priority_exercise_1"] = 1

        memory.to_csv(memory_file, index=False)
        return vocab, memory

    memory = pd.read_csv(memory_file)
    mem_len = len(memory)

    if vocab_len > mem_len:
        rows_to_add = vocab_len - mem_len
        new_rows = pd.DataFrame({
            "priority_exercise1": [0]*rows_to_add,
            "priority_exercise2": [0]*rows_to_add,
            "priority_exercise3": [0]*rows_to_add,
            "learnt": [0]*rows_to_add,
        })

        # Append missing rows
        memory = pd.concat([memory, new_rows], ignore_index=True)

        # Save updated memory file
        memory.to_csv(memory_file, index=False)

    return vocab, memory

def update_memory(memory, indices, correct):
    w = indices[0]
    e = indices[1]

    # Fisrt exercise: Exercise 1
    if e == 0:
        if correct:
            memory.iloc[w, e] -= LEARNING_RATE
            if memory.iloc[w, e] < 1:
                memory.iloc[w, e+1] += LEARNING_RATE
        else:
            memory.iloc[w, e] += LEARNING_RATE
    
    # Intermediate exercises: Exercise 2, Exercise 3
    else:
        if correct:
            memory.iloc[w, e] -= LEARNING_RATE
            memory.iloc[w, e+1] += LEARNING_RATE
        else:
            memory.iloc[w, e] -= LEARNING_RATE
            memory.iloc[w, e-1] += LEARNING_RATE

    if (memory < 0).any().any():  # just to check that all works good
        raise ValueError(
            f"Memory contains negative values!"
            f"indices: {indices}"
            )

    return memory

def save_memory(memory):
    memory.to_csv(memory_file, index=False)

def add_word(memory):
    """Check if total priority >= ACTIVE_WORDS, and assign a new word randomly if available."""
    total_priority = memory.iloc[:, :-1].sum().sum()
    while total_priority <= ACTIVE_WORDS:
        # Find words not yet learned
        candidates = memory[(memory == 0).all(axis=1)]
        if len(candidates) > 0:
            new_word = candidates.sample(1).index[0]
            memory.loc[new_word, 0] = 1
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

    if selected == translated:
        print("Correct!")
        return True
        
    else:
        print(f"Wrong! The correct answer was: {translated}")
        return False

def exercise_2(vocab, word_idx):
    """
    Exercise "Write translated"
    """
    original = vocab.loc[word_idx, "original"]
    translated = vocab.loc[word_idx, "translated"]

    print(f"\nTranslate this word: {original}")
    answer = input("Your answer: ").strip()
    if answer.lower() == translated.lower():
        print("Correct!")
        return True
    else:
        print(f"Wrong! The correct answer is: {translated}")
        confirm = input("Do you accept your answer as correct? (y/n): ").strip().lower()
        if confirm == "y":
            return True
        else:
            return False

def exercise_3(vocab, word_idx):
    """
    Exercise "Write original"
    """
    original = vocab.loc[word_idx, "original"]
    translated = vocab.loc[word_idx, "translated"]

    print(f"\nWrite the original word for: {translated}")
    answer = input("Your answer: ").strip()
    if answer.lower() == original.lower():
        print("Correct!")
        return True
    else:
        print(f"Wrong! The correct answer is: {original}")
        confirm = input("Do you accept your answer as correct? (y/n): ").strip().lower()
        if confirm == "y":
            return True
        else:
            return False


# ------------------------------
# MAIN ROUND
# ------------------------------
def start_round():

    if not os.path.exists(cache):
        os.makedirs(cache)

    vocab, memory = load_vocabulary()

    # Ask user which exercises to include
    print("Select exercises you want to do (comma separated numbers):")
    print("1: Choose the word")
    print("2: Write transaltion")
    print("3: Write original")
    choices = input("Your choices: ").strip().split(",")
    choices = [int(c.strip()) for c in choices]

    # Select words based on priority
    memory_map = {
        1: "priority_exercise_1",
        2: "priority_exercise_2",
        3: "priority_exercise_3"
    }
    memory_cols = [memory_map[c] for c in choices if c in memory_map]
    
    weights = memory[memory_cols].to_numpy()
    weights_1d = weights.ravel().astype(float)
    weights_1d /= weights_1d.sum()    # normalize to probabilities (sum = 1)
    
    sample = np.random.choice(
        weights_1d.size, 
        size=TRIES_PER_ROUND,
        replace=False,       # or True if needed
        p=weights_1d
    )

    word_idx, ex_idx = np.unravel_index(sample, weights.shape)

    # Start interactive round
    exercise_map = {
        0: exercise_1,
        1: exercise_2,
        2: exercise_3
    }                     # indices start from 0 but exercises from 1

    for i, (w, e) in enumerate(zip(word_idx, ex_idx)):
        print_progress_bar(i, TRIES_PER_ROUND)
        ex = exercise_map[e]
        correct = ex(vocab, w)
        memory = update_memory(memory, (w,e), correct)

    # Check if a new word should be promoted
    add_word(memory)

    save_memory(memory)
    print("\nRound completed! Progress saved.")

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    start_round()

