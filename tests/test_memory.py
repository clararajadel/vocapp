import pandas as pd
from vocapp.logic import update_memory
import vocapp.config as config

memory = pd.DataFrame({
        "p1": [0.5, 0.5],
        "p2": [0.5, 0.5],
        "learnt": [0.0, 0.0]
    })

def test_memory_update_correct_answer():
    lr = config.LEARNING_RATE

    updated = update_memory(memory.copy(), (0, 0), True)

    step = min(lr, memory.iloc[0, 0])

    expected_p1 = memory.iloc[0, 0] - step
    expected_p2 = memory.iloc[0, 1] + step

    assert updated.iloc[0, 0] == expected_p1
    assert updated.iloc[0, 1] == expected_p2


def test_memory_update_wrong_answer():
    lr = config.LEARNING_RATE

    updated = update_memory(memory.copy(), (0, 1), False)

    step = min(lr, memory.iloc[0, 1])

    expected_p1 = memory.iloc[0, 0] + step
    expected_p2 = memory.iloc[0, 1] - step

    assert updated.iloc[0, 0] == expected_p1
    assert updated.iloc[0, 1] == expected_p2
