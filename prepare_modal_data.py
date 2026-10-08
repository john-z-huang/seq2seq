"""Prepare tokenized Multi30k data and spaCy models in a Modal Volume."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import modal

VOLUME_NAME = os.environ.get("SEQ2SEQ_MODAL_VOLUME", "mini-seq2seq-data")
VOLUME_MOUNT_PATH = "/data"
HF_CACHE_PATH = Path(VOLUME_MOUNT_PATH) / ".cache" / "huggingface"
ARTIFACTS_PATH = Path(VOLUME_MOUNT_PATH) / "artifacts"
TOKENIZER_NAMES = {
    "de": "de_core_news_sm",
    "en": "en_core_web_sm",
}

app = modal.App("mini-seq2seq-prepare")
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("datasets==5.1.0", "spacy==3.8.16")
    .add_local_python_source("utils")
)


def _validate_name(name: str) -> None:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("artifact_name must be a single directory name")


def _tokenize_split(
    split: Any,
    de_nlp: Any,
    en_nlp: Any,
    batch_size: int,
) -> dict[str, list[list[str]]]:
    de_tokens: list[list[str]] = []
    en_tokens: list[list[str]] = []
    for batch in split.iter(batch_size=batch_size):
        de_tokens.extend(
            [
                [token.text.lower() for token in document]
                for document in de_nlp.pipe(
                    batch["de"],
                    batch_size=batch_size,
                    disable=de_nlp.pipe_names,
                )
            ]
        )
        en_tokens.extend(
            [
                [token.text.lower() for token in document]
                for document in en_nlp.pipe(
                    batch["en"],
                    batch_size=batch_size,
                    disable=en_nlp.pipe_names,
                )
            ]
        )
    return {"de_tokens": de_tokens, "en_tokens": en_tokens}


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    volumes={VOLUME_MOUNT_PATH: data_volume},
)
def prepare_data(
    dataset_name: str = "bentrevett/multi30k",
    artifact_name: str = "multi30k",
) -> str:
    _validate_name(artifact_name)
    os.environ["HF_HOME"] = str(HF_CACHE_PATH)
    os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_PATH / "datasets")
    HF_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    import spacy
    from datasets import Dataset, DatasetDict, load_dataset
    from spacy.cli import download as download_spacy_model
    from utils import EOS, SOS, Vocab

    artifact_path = ARTIFACTS_PATH / artifact_name
    manifest_path = artifact_path / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_name") != dataset_name:
            raise FileExistsError(
                f"{artifact_path} already contains {manifest.get('dataset_name')!r}; "
                "choose another artifact_name"
            )
        required_paths = (
            artifact_path / "dataset",
            artifact_path / "vocab.json",
            *(
                artifact_path / "tokenizers" / model
                for model in TOKENIZER_NAMES.values()
            ),
        )
        if not all(path.exists() for path in required_paths):
            raise RuntimeError(f"Prepared artifacts in {artifact_path} are incomplete")
        print(f"Using existing prepared data at {artifact_path}")
        return str(artifact_path)
    if artifact_path.exists():
        raise FileExistsError(
            f"{artifact_path} exists without a complete manifest; "
            "choose another artifact_name"
        )

    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(prefix=f".{artifact_name}-", dir=ARTIFACTS_PATH)
    )
    try:
        tokenizer_path = staging_path / "tokenizers"
        tokenizer_path.mkdir()
        nlp_by_language: dict[str, Any] = {}
        for language, model_name in TOKENIZER_NAMES.items():
            download_spacy_model(model_name)
            nlp = spacy.load(model_name)
            nlp.to_disk(tokenizer_path / model_name)
            nlp_by_language[language] = nlp

        loaded_dataset = load_dataset(dataset_name)
        if not isinstance(loaded_dataset, DatasetDict):
            raise TypeError("Expected the Hugging Face dataset to contain named splits")
        raw_dataset = loaded_dataset
        data_volume.commit()
        tokenized_splits: dict[str, Dataset] = {}
        for split_name in ("train", "validation", "test"):
            split = raw_dataset[split_name]
            token_columns = _tokenize_split(
                split,
                nlp_by_language["de"],
                nlp_by_language["en"],
                batch_size=128,
            )
            tokenized_splits[split_name] = Dataset.from_dict(token_columns)

        de_counter: Counter[str] = Counter()
        en_counter: Counter[str] = Counter()
        for example in tokenized_splits["train"]:
            de_counter.update(example["de_tokens"])
            en_counter.update(example["en_tokens"])

        de_vocab = Vocab(de_counter, min_freq=2)
        en_vocab = Vocab(en_counter, max_size=10000)
        encoded_splits: dict[str, Dataset] = {}
        for split_name, split in tokenized_splits.items():
            encoded_splits[split_name] = Dataset.from_dict(
                {
                    "src_ids": [
                        de_vocab.encode(tokens) for tokens in split["de_tokens"]
                    ],
                    "trg_ids": [
                        en_vocab.encode(tokens) for tokens in split["en_tokens"]
                    ],
                }
            )

        encoded_dataset = DatasetDict()
        for split_name, split in encoded_splits.items():
            encoded_dataset[split_name] = split
        encoded_dataset.save_to_disk(str(staging_path / "dataset"))
        (staging_path / "vocab.json").write_text(
            json.dumps(
                {
                    "de_tokens": de_vocab.itos,
                    "en_tokens": en_vocab.itos,
                    "special_token_ids": {"sos": SOS, "eos": EOS},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (staging_path / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset_name": dataset_name,
                    "splits": {
                        split_name: len(split)
                        for split_name, split in encoded_dataset.items()
                    },
                    "tokenizers": TOKENIZER_NAMES,
                    "vocab_sizes": {
                        "de": len(de_vocab),
                        "en": len(en_vocab),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        staging_path.rename(artifact_path)
        data_volume.commit()
    finally:
        if staging_path.exists():
            shutil.rmtree(staging_path)

    print(f"Prepared dataset, vocabularies, and tokenizers at {artifact_path}")
    return str(artifact_path)


@app.local_entrypoint()
def main(
    dataset_name: str = "bentrevett/multi30k",
    artifact_name: str = "multi30k",
) -> None:
    artifact_path = prepare_data.remote(dataset_name, artifact_name)
    print(f"Prepared Modal Volume artifact: {artifact_path}")
