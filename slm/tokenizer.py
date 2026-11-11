"""Byte-level BPE tokenizer trained on our own pretraining corpus.

The chat special tokens exist from the start, so instruction tuning never has to
resize the embedding matrix. Their ids are fixed by their order below.
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>"]
EOT_ID, USER_ID, ASSISTANT_ID, END_ID = range(len(SPECIAL_TOKENS))


def train_tokenizer(texts, vocab_size: int, out_path: str) -> Tokenizer:
    """Learn BPE merges from an iterable of strings and save to out_path (JSON).

    Byte-level means the base alphabet is the 256 byte values, so any input,
    including emoji or unseen scripts, encodes without an <unk> token."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer=trainer)
    for i, t in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(t) == i, f"special token {t} got id {tok.token_to_id(t)}"
    tok.save(out_path)
    return tok


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)
