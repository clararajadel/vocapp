import os
import pandas as pd
import tempfile

from vocapp import storage


def test_save_and_load_memory_roundtrip():
    df = pd.DataFrame({
        "p1": [0.2, 0.5],
        "p2": [0.3, 0.1],
        "learnt": [0.0, 1.0]
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "memory.csv")

        # save
        storage.save_memory(df, file_path)

        # load
        loaded = pd.read_csv(file_path)

        # check equality
        pd.testing.assert_frame_equal(df.round(2), loaded.round(2))


def test_save_does_not_change_shape():
    df = pd.DataFrame({
        "p1": [0.1],
        "p2": [0.2],
        "learnt": [0.0]
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "memory.csv")

        storage.save_memory(df, file_path)
        loaded = pd.read_csv(file_path)

        assert loaded.shape == df.shape
        assert list(loaded.columns) == list(df.columns)
