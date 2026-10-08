"""Project command-line shortcuts for the Modal data and training jobs."""

from __future__ import annotations

import subprocess


def prepare_data() -> None:
    """Run the configured Modal dataset preparation job."""
    subprocess.run(
        [
            "modal",
            "run",
            "prepare_modal_data.py",
            "--dataset-name",
            "bentrevett/multi30k",
            "--artifact-name",
            "multi30k",
        ],
        check=True,
    )


def train_model() -> None:
    """Run the configured Modal Seq2Seq GPU training job."""
    subprocess.run(
        [
            "modal",
            "run",
            "train_modal.py",
            "--artifact-name",
            "multi30k",
            "--epochs",
            "30",
            "--batch-size",
            "32",
        ],
        check=True,
    )


def infer_model() -> None:
    """Run the configured Modal CPU inference check."""
    subprocess.run(
        ["modal", "run", "infer_modal.py"],
        check=True,
    )
