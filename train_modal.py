"""Train the Seq2Seq model on a Modal GPU using prepared Volume artifacts."""

from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import modal

VOLUME_NAME = os.environ.get("SEQ2SEQ_MODAL_VOLUME", "mini-seq2seq-data")
VOLUME_MOUNT_PATH = "/data"
ARTIFACTS_PATH = Path(VOLUME_MOUNT_PATH) / "artifacts"
MODELS_PATH = Path(VOLUME_MOUNT_PATH) / "models"
GPU_TYPE = os.environ.get("SEQ2SEQ_MODAL_GPU", "A10G")

app = modal.App("mini-seq2seq-train")
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.1",
        "datasets==5.1.0",
        "spacy==3.8.16",
    )
    .add_local_python_source("model", "train", "utils")
)


def _validate_name(name: str, label: str) -> None:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"{label} must be a single directory or file name")


def _collate_batch(batch: list[dict[str, Any]]) -> tuple[Any, Any]:
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from utils import PAD

    sources = [torch.tensor(example["src_ids"], dtype=torch.long) for example in batch]
    targets = [torch.tensor(example["trg_ids"], dtype=torch.long) for example in batch]
    return (
        pad_sequence(sources, padding_value=PAD),
        pad_sequence(targets, padding_value=PAD),
    )


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT_PATH: data_volume},
)
def train_on_gpu(
    artifact_name: str = "multi30k",
    run_name: str = "",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    grad_clip: float = 10.0,
    hidden_size: int = 512,
    embed_size: int = 256,
    patience: int = 5,
    seed: int = 42,
) -> str:
    import spacy
    import torch
    from datasets import DatasetDict, load_from_disk
    from model import Decoder, Encoder, Seq2Seq
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from train import evaluate as evaluate_epoch
    from train import train as train_epoch

    if not run_name:
        run_name = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    _validate_name(artifact_name, "artifact_name")
    _validate_name(run_name, "run_name")
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    data_volume.reload()

    artifact_path = ARTIFACTS_PATH / artifact_name
    manifest = json.loads(
        (artifact_path / "manifest.json").read_text(encoding="utf-8")
    )
    vocabulary = json.loads(
        (artifact_path / "vocab.json").read_text(encoding="utf-8")
    )
    for language, expected_lang in (("de", "de"), ("en", "en")):
        model_name = manifest["tokenizers"][language]
        tokenizer_path = artifact_path / "tokenizers" / model_name
        tokenizer = spacy.load(str(tokenizer_path))
        if tokenizer.lang != expected_lang:
            raise ValueError(
                f"Expected {expected_lang!r} tokenizer at {tokenizer_path}, "
                f"got {tokenizer.lang!r}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not provide a CUDA-enabled GPU")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    loaded_dataset = load_from_disk(str(artifact_path / "dataset"))
    if not isinstance(loaded_dataset, DatasetDict):
        raise TypeError("Expected the prepared data to contain named splits")
    dataset = loaded_dataset
    train_iter = DataLoader(
        cast(TorchDataset, dataset["train"]),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )
    val_iter = DataLoader(
        cast(TorchDataset, dataset["validation"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )
    test_iter = DataLoader(
        cast(TorchDataset, dataset["test"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )

    de_size = len(vocabulary["de_tokens"])
    en_size = len(vocabulary["en_tokens"])
    encoder = Encoder(de_size, embed_size, hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, en_size, n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).to(device)
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=2,
    )

    model_dir = MODELS_PATH / artifact_name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"{run_name}.pt"
    if checkpoint_path.exists():
        raise FileExistsError(
            f"Checkpoint {checkpoint_path} already exists; choose another run_name"
        )
    best_val_loss = math.inf
    no_improve = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_epoch(
            seq2seq,
            optimizer,
            train_iter,
            en_size,
            grad_clip,
            device,
        )
        val_loss = evaluate_epoch(seq2seq, val_iter, en_size, device)
        scheduler.step(val_loss)
        print(
            f"[Epoch:{epoch}] val_loss:{val_loss:5.3f} "
            f"| val_pp:{math.exp(val_loss):5.2f}"
        )

        if val_loss < best_val_loss:
            checkpoint = {
                "model_state_dict": seq2seq.state_dict(),
                "model_config": {
                    "source_vocab_size": de_size,
                    "target_vocab_size": en_size,
                    "embed_size": embed_size,
                    "hidden_size": hidden_size,
                    "encoder_layers": 2,
                    "decoder_layers": 1,
                },
                "source_tokens": vocabulary["de_tokens"],
                "target_tokens": vocabulary["en_tokens"],
                "epoch": epoch,
                "validation_loss": val_loss,
            }
            temporary_path = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, checkpoint_path)
            data_volume.commit()
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            print(f"Saved best checkpoint to {checkpoint_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop after {epoch} epochs without validation improvement")
                break

    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    seq2seq.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss = evaluate_epoch(seq2seq, test_iter, en_size, device)
    summary_path = model_dir / f"{run_name}.json"
    summary_path.write_text(
        json.dumps(
            {
                "artifact_name": artifact_name,
                "checkpoint": str(checkpoint_path),
                "best_epoch": best_epoch,
                "validation_loss": best_val_loss,
                "test_loss": test_loss,
                "gpu": torch.cuda.get_device_name(device),
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    data_volume.commit()
    print(f"[TEST] loss:{test_loss:5.2f}")
    print(f"Saved training summary to {summary_path}")
    return str(checkpoint_path)


@app.local_entrypoint()
def main(
    artifact_name: str = "multi30k",
    run_name: str = "",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    grad_clip: float = 10.0,
    hidden_size: int = 512,
    embed_size: int = 256,
    patience: int = 5,
    seed: int = 42,
) -> None:
    if not run_name:
        run_name = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    checkpoint_path = train_on_gpu.remote(
        artifact_name,
        run_name,
        epochs,
        batch_size,
        learning_rate,
        grad_clip,
        hidden_size,
        embed_size,
        patience,
        seed,
    )
    print(f"Training checkpoint saved in Modal Volume at {checkpoint_path}")
