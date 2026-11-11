"""Chat template and supervised fine-tuning examples.

One example is laid out as

    <|endoftext|><|user|>{prompt}<|end|><|assistant|>{response}<|end|>

With mask_prompt=True the loss covers only the response and its closing <|end|>,
so the model learns to answer rather than to imitate user prompts.
"""

import torch

from .tokenizer import ASSISTANT_ID, END_ID, EOT_ID, USER_ID

IGNORE = -100


def prompt_ids(tok, prompt: str) -> list[int]:
    """Tokens the model sees before it starts answering."""
    return [EOT_ID, USER_ID] + tok.encode(prompt).ids + [END_ID, ASSISTANT_ID]


def encode_example(tok, prompt: str, response: str, block_size: int, mask_prompt: bool = True):
    """Return (inputs, targets) as int lists, or None if the example is too long."""
    p = prompt_ids(tok, prompt)
    ids = p + tok.encode(response).ids + [END_ID]
    if len(ids) > block_size + 1:
        return None
    x, y = ids[:-1], ids[1:]
    if mask_prompt:
        # y[j] is the token after x[j]; the first response token is y[len(p) - 1].
        y = [IGNORE] * (len(p) - 1) + y[len(p) - 1 :]
    return x, y


def collate(examples, device: str):
    """Right-pad a list of (x, y) pairs into tensors. Padded targets are ignored."""
    T = max(len(x) for x, _ in examples)
    xs = torch.full((len(examples), T), EOT_ID, dtype=torch.long)
    ys = torch.full((len(examples), T), IGNORE, dtype=torch.long)
    for i, (x, y) in enumerate(examples):
        xs[i, : len(x)] = torch.tensor(x)
        ys[i, : len(y)] = torch.tensor(y)
    return xs.to(device), ys.to(device)
