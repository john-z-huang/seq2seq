"""Run a short German-to-English inference check on a Modal CPU worker."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import modal

VOLUME_NAME = os.environ.get("SEQ2SEQ_MODAL_VOLUME", "mini-seq2seq-data")
VOLUME_MOUNT_PATH = "/data"
DEFAULT_CHECKPOINT = "run-20261008T101910930522Z.pt"
MODELS_PATH = Path(VOLUME_MOUNT_PATH) / "models" / "multi30k"
DATASET_PATH = Path(VOLUME_MOUNT_PATH) / "artifacts" / "multi30k" / "dataset"

app = modal.App("mini-seq2seq-infer")
data_volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "datasets==5.1.0")
    .add_local_python_source("model", "utils")
)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=10 * 60,
    volumes={VOLUME_MOUNT_PATH: data_volume},
)
def infer(
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    max_tokens: int = 50,
    sample_count: int = 10,
    seed: int = 42,
) -> list[dict[str, str]]:
    import torch
    from datasets import load_from_disk
    from model import Decoder, Encoder, Seq2Seq
    from utils import EOS, PAD, SOS

    if (
        not checkpoint_name.endswith(".pt")
        or Path(checkpoint_name).name != checkpoint_name
    ):
        raise ValueError("checkpoint_name must be a .pt filename without directories")
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least 2")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")

    data_volume.reload()
    checkpoint_path = MODELS_PATH / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found in the Modal Volume: {checkpoint_path}"
        )
    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(
            f"Prepared dataset not found in the Modal Volume: {DATASET_PATH}"
        )

    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["model_config"]
    encoder = Encoder(
        config["source_vocab_size"],
        config["embed_size"],
        config["hidden_size"],
        n_layers=config["encoder_layers"],
        dropout=0.5,
    )
    decoder = Decoder(
        config["embed_size"],
        config["hidden_size"],
        config["target_vocab_size"],
        n_layers=config["decoder_layers"],
        dropout=0.0,
    )
    model = Seq2Seq(encoder, decoder)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = load_from_disk(str(DATASET_PATH))["test"]
    if len(test_dataset) == 0:
        raise ValueError("The prepared test split is empty")
    sample_indices = random.Random(seed).sample(
        range(len(test_dataset)), k=min(sample_count, len(test_dataset))
    )
    source_tokens = checkpoint["source_tokens"]
    target_tokens = checkpoint["target_tokens"]

    def decode(ids: list[int], vocabulary: list[str]) -> str:
        words: list[str] = []
        for token_id in ids:
            if token_id == EOS:
                break
            if token_id not in (PAD, SOS):
                words.append(vocabulary[token_id])
        return " ".join(words)

    results: list[dict[str, str]] = []
    print(f"Checkpoint: {checkpoint_path}")
    print(
        f"Training checkpoint: epoch={checkpoint['epoch']}, "
        f"validation_loss={checkpoint['validation_loss']:.3f}"
    )
    print(
        f"Randomly selected {len(sample_indices)} examples from the test split (seed={seed})"
    )
    for sample_number, sample_index in enumerate(sample_indices, start=1):
        example = test_dataset[sample_index]
        source_ids = example["src_ids"]
        target_ids = example["trg_ids"]
        source = torch.tensor(source_ids, dtype=torch.long).unsqueeze(1)
        with torch.inference_mode():
            predicted_ids = (
                model(source, max_len=max_tokens, sos=SOS)
                .argmax(dim=2)
                .squeeze(1)
                .tolist()
            )
        source_text = decode(source_ids, source_tokens)
        prediction = decode(predicted_ids[1:], target_tokens)
        reference = decode(target_ids, target_tokens)
        results.append(
            {"source": source_text, "prediction": prediction, "reference": reference}
        )
        print(f"[{sample_number}/{len(sample_indices)}] DE:   {source_text}")
        print(f"             EN:   {prediction}")
        print(f"             REF:  {reference}")
    return results


@app.local_entrypoint()
def main(
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    max_tokens: int = 50,
    sample_count: int = 10,
    seed: int = 42,
) -> None:
    infer.remote(checkpoint_name, max_tokens, sample_count, seed)
