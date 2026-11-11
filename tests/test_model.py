import itertools
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slm.chat import IGNORE, collate, encode_example, prompt_ids  # noqa: E402
from slm.lora import LoRALinear, apply_lora, merge_lora  # noqa: E402
from slm.model import GPT, GPTConfig, apply_rope, rope_cache  # noqa: E402
from slm.utils import cosine_lr, load_config  # noqa: E402

VARIANTS = [
    dict(pos_emb=p, norm=n, mlp=m, n_kv_head=kv)
    for p, n, m, kv in itertools.product(["learned", "rope"], ["layernorm", "rmsnorm"], ["gelu", "swiglu"], [4, 2])
]


def tiny(**kw):
    base = dict(vocab_size=97, block_size=32, n_layer=2, n_head=4, d_model=32)
    return GPTConfig(**{**base, **kw})


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "-".join(str(x) for x in v.values()))
def test_forward_shapes_and_initial_loss(variant):
    torch.manual_seed(0)
    model = GPT(tiny(**variant))
    x = torch.randint(0, 97, (3, 32))
    y = torch.randint(0, 97, (3, 32))
    logits, loss = model(x, y)
    assert logits.shape == (3, 32, 97)
    # A fresh model should be close to uniform over the vocabulary. (Targets must
    # differ from the inputs: tied embeddings make a fresh model favour echoing
    # its input token, so loss on targets == inputs starts well below log(V).)
    assert abs(loss.item() - math.log(97)) < 0.5


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "-".join(str(x) for x in v.values()))
def test_causal(variant):
    """Changing a future token must not change predictions at earlier positions."""
    torch.manual_seed(0)
    model = GPT(tiny(**variant)).eval()
    x = torch.randint(0, 97, (1, 32))
    x2 = x.clone()
    x2[0, 20] = (x2[0, 20] + 1) % 97
    y = torch.zeros_like(x)
    l1, _ = model(x, y)
    l2, _ = model(x2, y)
    assert torch.allclose(l1[:, :20], l2[:, :20], atol=1e-5)
    assert not torch.allclose(l1[:, 20:], l2[:, 20:], atol=1e-5)


def test_rope_is_relative():
    """q.k after RoPE depends only on the offset between positions."""
    torch.manual_seed(0)
    cos, sin = rope_cache(16, 64, 10000.0)
    q, k = torch.randn(16), torch.randn(16)

    def score(i, j):
        qi = apply_rope(q.view(1, 1, 1, 16), cos[i : i + 1], sin[i : i + 1])
        kj = apply_rope(k.view(1, 1, 1, 16), cos[j : j + 1], sin[j : j + 1])
        return (qi * kj).sum().item()

    assert score(5, 2) == pytest.approx(score(40, 37), abs=1e-4)
    assert score(5, 2) != pytest.approx(score(5, 4), abs=1e-4)


def test_swiglu_matches_gelu_parameter_count():
    gelu = GPT(GPTConfig(d_model=384, n_head=6, mlp="gelu")).num_params()
    swiglu = GPT(GPTConfig(d_model=384, n_head=6, mlp="swiglu")).num_params()
    assert gelu == swiglu  # 8/3 * 384 = 1024 exactly, so no rounding slack


def test_gqa_shrinks_kv_projections():
    mha = GPT(tiny(n_kv_head=4))
    gqa = GPT(tiny(n_kv_head=2))
    assert gqa.blocks[0].attn.k_proj.weight.shape == (16, 32)
    assert mha.num_params() - gqa.num_params() == 2 * 2 * 32 * 16  # layers x (k, v) x d x saved width


def test_lora_starts_identical_and_merges_exactly():
    torch.manual_seed(0)
    model = GPT(tiny(mlp="swiglu")).eval()
    x = torch.randint(0, 97, (2, 16))
    before, _ = model(x, x)
    apply_lora(model, r=4, alpha=8, targets=("q_proj", "v_proj", "up"))
    after, _ = model(x, x)
    assert torch.allclose(before, after, atol=1e-6)  # B = 0 at init
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)

    for m in model.modules():  # pretend training moved B away from zero
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_B, std=0.1)
    adapted, _ = model(x, x)
    merge_lora(model)
    merged, _ = model(x, x)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert torch.allclose(adapted, merged, atol=1e-5)


class FakeTok:
    """Stand-in tokenizer: one id per character, offset past the special tokens."""

    class Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, s):
        return self.Enc([4 + (ord(c) % 90) for c in s])


def test_sft_masking_covers_only_the_response():
    tok = FakeTok()
    x, y = encode_example(tok, "hi", "yo", block_size=64, mask_prompt=True)
    p = prompt_ids(tok, "hi")
    assert len(x) == len(y)
    assert y[: len(p) - 1] == [IGNORE] * (len(p) - 1)
    assert y[len(p) - 1 :] == tok.encode("yo").ids + [3]  # response, then <|end|>
    _, y_all = encode_example(tok, "hi", "yo", block_size=64, mask_prompt=False)
    assert IGNORE not in y_all
    assert encode_example(tok, "x" * 100, "yo", block_size=64) is None

    xs, ys = collate([(x, y), (x[:3], y[:3])], "cpu")
    assert xs.shape == ys.shape == (2, len(x))
    assert (ys[1, 3:] == IGNORE).all()


def test_tiny_model_overfits_one_batch():
    torch.manual_seed(0)
    model = GPT(tiny(pos_emb="rope", norm="rmsnorm", mlp="swiglu", n_kv_head=2))
    opt = model.configure_optimizer(3e-3, 0.0, (0.9, 0.95), "cpu")
    x = torch.randint(0, 97, (4, 32))
    first = None
    for _ in range(150):
        _, loss = model(x[:, :-1], x[:, 1:])
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * first


def test_cosine_lr_schedule():
    assert cosine_lr(0, 1000, 1e-3, 1e-4, 100) == pytest.approx(1e-5)
    assert cosine_lr(99, 1000, 1e-3, 1e-4, 100) == pytest.approx(1e-3)
    assert cosine_lr(1000, 1000, 1e-3, 1e-4, 100) == pytest.approx(1e-4)


def test_config_overrides(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"lr": 0.001, "model": {"n_layer": 6, "mlp": "gelu"}}')
    cfg = load_config({"lr": 1, "seed": 0, "model": {}}, str(path), ["model.mlp=swiglu", "seed=3", "out_dir=out/x"])
    assert cfg == {"lr": 0.001, "seed": 3, "model": {"n_layer": 6, "mlp": "swiglu"}, "out_dir": "out/x"}
