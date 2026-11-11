# What Matters in a Small Language Model

AFML default course project (team TA_ALLOT_01): train a small language model (SLM) from scratch,
instruction-tune it, and measure which design choices matter at small scale. Plain PyTorch, no model
libraries: every component is in [slm/](slm/).

Three experiments on top of pretraining and instruction tuning:

1. **Architecture ablations.** A GPT-2-style baseline against RoPE, RMSNorm, SwiGLU and grouped-query
   attention, swapped in one at a time and all together, at equal tokens (~14M parameters, 100M TinyStories
   tokens, 2 seeds each).
2. **Scaling.** Three LLaMA-style models (9M, 19M, 44M parameters) trained on FineWeb-Edu at 20 tokens per
   parameter.
3. **Instruction tuning.** Full fine-tuning against LoRA, each with and without loss masking on the prompt,
   evaluated on 200 held-out Dolly prompts by format checks and a blind pairwise LLM judge.

## Results

Full tables, failure analysis and caveats: [kaggle_results/report/RESULTS.md](kaggle_results/report/RESULTS.md)
(generated from the raw run outputs in [kaggle_results/out/](kaggle_results/out/)).

**Ablations** (validation loss, nats/token; lower is better; mean of 2 seeds)

| Variant | Params | Val loss | Δ vs baseline |
| --- | --- | --- | --- |
| All four (modern) | 12.6M | 1.699 | −0.091 |
| SwiGLU | 14.0M | 1.723 | −0.066 |
| RoPE | 13.8M | 1.728 | −0.062 |
| Baseline (GPT-2 style) | 14.0M | 1.789 | — |
| RMSNorm | 14.0M | 1.791 | +0.002 |
| GQA (2 KV heads) | 12.8M | 1.800 | +0.011 |

Seed-to-seed spread is at most 0.012. A separate run on another GPU gave the same ranking.

**Scaling** (FineWeb-Edu): 8.6M → 4.267, 19.3M → 3.631, 43.8M → 3.289. A power-law fit gives
loss ∝ N^−0.16, about 10.5% lower loss per doubling of parameters (size and data grow together here).

**Instruction tuning** (43.8M base model, 65K Alpaca + Dolly examples)

| Model | Trainable params | In-format answers | Judge score vs comparison |
| --- | --- | --- | --- |
| Base (pretrained only) | — | 0.0% | — |
| Full fine-tune, masked loss | 43.8M | 84.5% | 0.78 ± 0.03 vs base (127 wins, 2 losses) |
| Full fine-tune, unmasked loss | 43.8M | 83.0% | masked vs unmasked: 0.54 ± 0.04 |
| LoRA (r=16), masked loss | 1.8M | 79.5% | full vs LoRA: 0.55 ± 0.05 |
| LoRA (r=16), unmasked loss | 1.8M | 73.5% | masked vs unmasked: 0.53 ± 0.04 |

Judge score: 1 = always preferred, 0.5 = no preference (95% intervals). Judge: Claude Sonnet 5.5, blind,
each pair judged in both orders, 1,600 verdicts.

## Layout

| Path | What it is |
| --- | --- |
| [slm/model.py](slm/model.py) | Decoder-only transformer. Positional encoding, norm, MLP and attention type are config switches |
| [slm/lora.py](slm/lora.py) | LoRA, written out by hand, with merge-back into plain weights |
| [slm/tokenizer.py](slm/tokenizer.py) | Byte-level BPE trained on our corpus; chat special tokens built in |
| [slm/chat.py](slm/chat.py) | Chat template and response-only loss masking |
| [slm/data.py](slm/data.py), [slm/utils.py](slm/utils.py) | Token-bin batches; config, LR schedule, device setup, checkpoints |
| [scripts/prepare_pretrain.py](scripts/prepare_pretrain.py) | Download corpus, train tokenizer, write token bins |
| [scripts/prepare_sft.py](scripts/prepare_sft.py) | Alpaca-cleaned + Dolly 15k, with 200 held-out Dolly eval prompts |
| [scripts/train.py](scripts/train.py) | Pretraining loop (bf16/fp16, grad accumulation, cosine LR, fixed eval batches, pause/resume) |
| [scripts/sweep.py](scripts/sweep.py) | Runs the ablation or scaling sweep |
| [scripts/sft.py](scripts/sft.py) | Instruction tuning: full or LoRA, masked or unmasked loss |
| [scripts/eval_sft.py](scripts/eval_sft.py) | Answers the held-out prompts; on-format rate |
| [scripts/judge.py](scripts/judge.py) | Blind pairwise LLM judge through the Claude Code CLI; resumable |
| [scripts/summarize.py](scripts/summarize.py) | Results table and loss curves for a sweep |
| [scripts/make_report.py](scripts/make_report.py) | One report from all results: tables, scaling fit, failure modes, examples |
| [scripts/generate.py](scripts/generate.py) | Sample from any checkpoint |
| [scripts/pipeline.py](scripts/pipeline.py), [scripts/claims.py](scripts/claims.py) | Job queue that runs everything on several GPUs at once without duplicate work |
| [kaggle/AFML_SLM_kaggle.ipynb](kaggle/AFML_SLM_kaggle.ipynb) | The notebook that produced our results (built by [scripts/make_kaggle_notebook.py](scripts/make_kaggle_notebook.py)) |
| [configs/](configs/) | Run configs; any value can be overridden as `key=value` on the command line |
| [tests/](tests/) | Causality, RoPE, LoRA, masking, overfitting and job-claim checks |
| [kaggle_results/](kaggle_results/) | Our run outputs (logs, summaries, eval answers, judge verdicts) and the generated report |

## Datasets

All public, downloaded automatically by the prepare scripts:

- Pretraining (ablations): [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- Pretraining (scaling and final model): [FineWeb-Edu, sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (first 1B tokens)
- Instruction tuning: [Alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) and [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (200 Dolly prompts held out for evaluation)

## Reproducing

### Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest tests -q
```

(On Linux/macOS use `.venv/bin/python`.)

### Everything on Kaggle (how our results were produced)

[kaggle/AFML_SLM_kaggle.ipynb](kaggle/AFML_SLM_kaggle.ipynb) runs the whole project as a background run on
Kaggle's two T4 GPUs: data preparation on the CPU while both GPUs train, then fine-tuning and evaluation.
Import it into a new Kaggle notebook, set *Accelerator: GPU T4 x2* and *Internet: On*, then *Save Version →
Save & Run All*. It stops itself before Kaggle's 12-hour limit; if the last cell says `NOT FINISHED YET`,
attach the notebook's own output as input and run again, and it resumes. Our results took three runs
(about 2 + 11 + 1 hours). After editing code, rebuild the notebook with `python make_kaggle_notebook.py`
from `scripts/`.

### Step by step on one GPU

1. `python scripts/prepare_pretrain.py --dataset tinystories`
2. `python scripts/sweep.py ablations --seeds 0 1`, then `python scripts/summarize.py out/ablations`
3. `python scripts/prepare_pretrain.py --dataset fineweb-edu --max-tokens 1000000000`
4. `python scripts/sweep.py scaling`, then `python scripts/summarize.py out/scaling`
5. `python scripts/prepare_sft.py`
6. `python scripts/sft.py --config configs/sft_full.json` and `python scripts/sft.py --config configs/sft_lora.json`;
   add `mask_prompt=false out_dir=...` for the unmasked runs
7. `python scripts/eval_sft.py base=out/scaling/l_s0/ckpt.pt full_masked=out/sft/full_masked/ckpt.pt ...`

### Judge and report

`python scripts/judge.py` needs the [Claude Code CLI](https://www.npmjs.com/package/@anthropic-ai/claude-code)
logged in. Each verdict is an isolated `claude -p` call: no tools, no saved session, safe mode (no user
memory or settings), a fixed rubric as the whole system prompt. `--limit 5` runs a small pilot; re-running
resumes. `python scripts/make_report.py` then rebuilds [RESULTS.md](kaggle_results/report/RESULTS.md).

## Design notes

- **Fair ablations.** All variants share pre-norm blocks, bias-free linear layers and tied embeddings, so a
  run differs from the baseline in one component only. SwiGLU's hidden size is 8/3 × d, which matches
  GELU's parameter count. RoPE drops the learned position table and GQA shrinks the k/v projections. Every
  run reports its exact parameter count.
- **Same validation tokens.** Every run is evaluated on the same fixed validation batches, so loss gaps
  between runs are not sampling noise.
- **Chat tokens from day one.** `<|user|>`, `<|assistant|>` and `<|end|>` are in the tokenizer before
  pretraining, so fine-tuning never resizes the embeddings.
- **Response-only loss.** With masking, only the answer and its closing `<|end|>` are trained. Validation
  loss is always response-only, so masked and unmasked runs are measured on the same quantity.
- **Resumable training.** Training saves its full state and pauses itself before a deadline, so long runs
  survive session limits (Kaggle's 12 hours).
- **Windows.** `torch.compile` needs Triton, which isn't bundled on Windows, so it's off by default
  (`compile=true` enables it where available).

[colab/AFML_SLM.ipynb](colab/AFML_SLM.ipynb) is an earlier Colab runner, used for an independent first run of
the ablations; the Kaggle notebook is the tested path.

## Team

Dhyan Rao, Dhruvagouda Patil, Dhruv Talavat, Avyukth Ranjan.
