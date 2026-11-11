"""Download a pretraining corpus, train a BPE tokenizer on it, and write token bins.

    python scripts/prepare_pretrain.py --dataset tinystories
    python scripts/prepare_pretrain.py --dataset fineweb-edu --max-tokens 1000000000

Writes <out-dir>/{train.bin, val.bin, tokenizer.json, meta.json}. Each document is
prefixed with <|endoftext|>, which doubles as the start token at fine-tuning time.
"""

import argparse
import faulthandler
import itertools
import json
import os
import sys
import tempfile
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slm.tokenizer import EOT_ID, load_tokenizer, train_tokenizer  # noqa: E402

DATASETS = {
    "tinystories": {"vocab_size": 8192},
    "fineweb-edu": {"vocab_size": 16384},
}
DOWNLOADS = os.path.join(tempfile.gettempdir(), "fineweb-edu")  # outside the output folder


def text_stream(dataset: str, split: str):
    """Yield documents one at a time without loading the corpus into memory."""
    from datasets import load_dataset

    if dataset == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="train" if split == "train" else "validation")
        for row in ds:
            yield row["text"]
    else:
        # FineWeb-Edu has no validation split; the caller carves one off the front. Each ~2 GB
        # parquet file is downloaded whole, read in chunks, and deleted once read to the end
        # (a file only partly read is kept for the next pass). Streaming it with datasets
        # crashed (segfault) on Kaggle.
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        for i in range(14):
            path = hf_hub_download("HuggingFaceFW/fineweb-edu", f"sample/10BT/{i:03d}_00000.parquet",
                                   repo_type="dataset", local_dir=DOWNLOADS)
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=1024, columns=["text"]):
                yield from batch.column(0).to_pylist()
            pf.close()
            os.remove(path)


def beat(path: str | None, every_s: float = 60.0, _last=[0.0]):
    """Touch the heartbeat file (at most once a minute) so other sessions waiting
    for this data can tell the build is still alive."""
    if path and time.time() - _last[0] > every_s:
        with open(path, "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
        _last[0] = time.time()


def write_bin(tok, texts, path: str, max_tokens: int | None, batch_docs: int = 1024, total_docs=None, heartbeat=None) -> tuple[int, int]:
    """Encode texts in batches and append uint16 ids to path. Returns (tokens, docs)."""
    n_tokens = n_docs = 0
    with open(path, "wb") as f, tqdm(total=total_docs, desc=os.path.basename(path), unit="doc") as bar:
        while True:
            batch = list(itertools.islice(texts, batch_docs))
            if not batch:
                break
            ids = []
            for enc in tok.encode_batch(batch):
                ids.append(EOT_ID)
                ids.extend(enc.ids)
            arr = np.array(ids, dtype=np.uint16)
            if max_tokens is not None and n_tokens + len(arr) > max_tokens:
                arr = arr[: max_tokens - n_tokens]
            arr.tofile(f)
            beat(heartbeat)
            n_tokens += len(arr)
            n_docs += len(batch)
            bar.update(len(batch))
            bar.set_postfix(tokens=f"{n_tokens / 1e6:.1f}M")
            if max_tokens is not None and n_tokens >= max_tokens:
                break
    return n_tokens, n_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--out-dir", default=None, help="default: data/<dataset>")
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--tokenizer-docs", type=int, default=200_000, help="documents used to learn BPE merges")
    ap.add_argument("--tokenizer", default=None, help="reuse an existing tokenizer.json instead of training one")
    ap.add_argument("--max-tokens", type=int, default=None, help="cap on training tokens (FineWeb-Edu)")
    ap.add_argument("--val-docs", type=int, default=5_000, help="FineWeb-Edu documents held out for validation")
    ap.add_argument("--heartbeat", default=None, help="file to touch every minute while working")
    args = ap.parse_args()
    faulthandler.enable()  # a crash in native code then still leaves a traceback in the log
    beat(args.heartbeat)

    out_dir = args.out_dir or os.path.join("data", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    vocab_size = args.vocab_size or DATASETS[args.dataset]["vocab_size"]
    tok_path = os.path.join(out_dir, "tokenizer.json").replace("\\", "/")  # portable across OSes
    fineweb = args.dataset == "fineweb-edu"
    # FineWeb-Edu docs 0..val_docs-1 are validation; the tokenizer learns from the docs after them.
    skip = args.val_docs if fineweb else 0

    if args.tokenizer:
        tok = load_tokenizer(args.tokenizer)
        tok.save(tok_path)
    elif os.path.exists(tok_path):
        print(f"reusing {tok_path}")
        tok = load_tokenizer(tok_path)
    else:
        print(f"training a {vocab_size}-token BPE tokenizer on {args.tokenizer_docs:,} documents")
        sample = itertools.islice(text_stream(args.dataset, "train"), skip, skip + args.tokenizer_docs)
        tok = train_tokenizer(sample, vocab_size, tok_path)
    assert tok.get_vocab_size() <= 65536, "uint16 bins need a vocabulary of at most 65,536"

    if fineweb:
        stream = text_stream(args.dataset, "train")
        val_tokens, val_docs = write_bin(tok, itertools.islice(stream, args.val_docs), os.path.join(out_dir, "val.bin"), None, total_docs=args.val_docs, heartbeat=args.heartbeat)
        train_tokens, train_docs = write_bin(tok, stream, os.path.join(out_dir, "train.bin"), args.max_tokens, heartbeat=args.heartbeat)
    else:
        val_tokens, val_docs = write_bin(tok, text_stream(args.dataset, "val"), os.path.join(out_dir, "val.bin"), None, heartbeat=args.heartbeat)
        train_tokens, train_docs = write_bin(tok, text_stream(args.dataset, "train"), os.path.join(out_dir, "train.bin"), args.max_tokens, heartbeat=args.heartbeat)

    meta = {
        "dataset": args.dataset,
        "vocab_size": tok.get_vocab_size(),
        "tokenizer": tok_path,
        "train_tokens": train_tokens,
        "train_docs": train_docs,
        "val_tokens": val_tokens,
        "val_docs": val_docs,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
