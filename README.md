# Neural Image Caption Generator with Soft Attention

PyTorch reproduction of **Show, Attend and Tell: Neural Image Caption Generation with Visual Attention** (Xu et al., 2015) on Flickr8k.

The implementation uses a frozen ImageNet VGG-16 encoder, additive soft attention over `14x14x512` annotation vectors, scalar beta gating, an LSTM decoder with the paper's deep output layer, and doubly stochastic attention regularization.

## Repository Layout

```text
models/
  encoder.py       VGG-16 feature extractor
  attention.py     soft additive attention
  decoder.py       LSTM decoder and deep output layer
model.py           end-to-end encoder/decoder wrapper
code/              compatibility wrappers matching the course layout
utils/
  dataset.py       Flickr8k loading, tokenization, vocabulary, buckets
  metrics.py       BLEU-1..4 without brevity penalty
train.py           training, validation BLEU-4 early stopping
evaluate.py        test-set BLEU table
visualize.py       word-level attention overlays
config.py          hyperparameters and paths
data/              dataset instructions
results/           BLEU logs and attention figures
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Colab Pro, install the requirements after mounting the repo and use the GPU runtime. The first encoder construction downloads the torchvision VGG-16 ImageNet weights if they are not already cached.

Run the regression tests:

```bash
pytest -q
```

## Flickr8k Data

Download Flickr8k from Kaggle and arrange the files as:

```text
data/flickr8k/
  images/  (or Images/)
  captions.txt
  Flickr_8k.trainImages.txt
  Flickr_8k.devImages.txt
  Flickr_8k.testImages.txt
```

The loader validates the standard split sizes: train `6000`, validation `1000`, test `1000`. Captions are lowercased and punctuation is stripped before vocabulary building and BLEU scoring. The vocabulary keeps the top `10000` training words plus the four special tokens.

## Train

```bash
python train.py
```

Outputs:

```text
checkpoints/epoch_XXX.pt
checkpoints/best.pt
results/training_log.csv
```

The best checkpoint is selected by validation BLEU-4, not validation loss. The loss is:

```text
cross_entropy + lambda * mean((1 - sum_t alpha_ti)^2)
```

with `lambda=1.0` by default.

## Evaluate

```bash
python evaluate.py \
  --checkpoint checkpoints/best.pt \
  --split test \
  --results_out results/test_bleu.json
```

This prints BLEU-1 through BLEU-4 next to the Flickr8k paper targets:

```text
Soft attention target: BLEU-1 67.0, BLEU-2 44.8, BLEU-3 29.9, BLEU-4 19.5
```

BLEU is computed from corpus clipped n-gram precision without brevity penalty, matching the project specification.

## Attention Visualizations

Generate six qualitative examples from the test split:

```bash
python visualize.py \
  --checkpoint checkpoints/best.pt \
  --vocab data/flickr8k/vocab.json \
  --data_root data/flickr8k \
  --split test \
  --num_examples 6 \
  --output_dir results/attention_examples
```

Or pass explicit images:

```bash
python visualize.py --images data/flickr8k/images/123456.jpg  # or data/flickr8k/Images/...
```

Each generated word gets a `14x14` attention map upsampled to `224x224`, Gaussian-smoothed, and overlaid on the input image.

---

## Ablation Experiments

The `soft-attention-ablations` branch adds a clean ablation framework. All defaults reproduce the paper exactly; set flags to deviate.

### What each ablation tests

| Experiment | Flag(s) | What it isolates |
|---|---|---|
| `baseline_soft_attention` | *(no flags)* | Paper's exact model — use as the comparison anchor |
| `no_attention_mean` | `--attention_mode none` | Does dynamic attention help vs. a static mean-pooled context? |
| `uniform_attention` | `--attention_mode uniform` | Does *learned* spatial attention help vs. a uniform α=1/L baseline? |
| `no_doubly_stochastic_lambda_0` | `--lambda_weight 0.0` | How much does the Eq. 14 regularisation penalty contribute? |
| `no_beta_gate` | `--no_beta_gate` | Does the learned β scalar gate (paper Sec. 4.2.1) matter? |
| `feature_grid_7x7` | `--feature_grid_size 7` | Does higher spatial resolution (L=196 vs. L=49) affect captioning? |

### Run individual experiments

All outputs go to `outputs/ablations/<name>/`. Each run saves `config.json`, `training_log.csv`, `checkpoints/best.pt`, and `eval_val.json` / `eval_test.json`.

```bash
# a) Baseline (paper's exact model)
python train.py \
  --checkpoint_dir outputs/ablations/baseline_soft_attention/checkpoints \
  --results_dir    outputs/ablations/baseline_soft_attention
python evaluate.py \
  --checkpoint  outputs/ablations/baseline_soft_attention/checkpoints/best.pt \
  --split test \
  --results_out outputs/ablations/baseline_soft_attention/eval_test.json

# b) No-attention mean baseline
python train.py --attention_mode none \
  --checkpoint_dir outputs/ablations/no_attention_mean/checkpoints \
  --results_dir    outputs/ablations/no_attention_mean
python evaluate.py --attention_mode none \
  --checkpoint  outputs/ablations/no_attention_mean/checkpoints/best.pt \
  --split test \
  --results_out outputs/ablations/no_attention_mean/eval_test.json

# c) Uniform-attention baseline
python train.py --attention_mode uniform \
  --checkpoint_dir outputs/ablations/uniform_attention/checkpoints \
  --results_dir    outputs/ablations/uniform_attention
python evaluate.py --attention_mode uniform \
  --checkpoint  outputs/ablations/uniform_attention/checkpoints/best.pt \
  --split test \
  --results_out outputs/ablations/uniform_attention/eval_test.json

# d) No doubly stochastic regularisation (λ=0)
python train.py --lambda_weight 0.0 \
  --checkpoint_dir outputs/ablations/no_doubly_stochastic_lambda_0/checkpoints \
  --results_dir    outputs/ablations/no_doubly_stochastic_lambda_0
python evaluate.py \
  --checkpoint  outputs/ablations/no_doubly_stochastic_lambda_0/checkpoints/best.pt \
  --split test \
  --results_out outputs/ablations/no_doubly_stochastic_lambda_0/eval_test.json

# e) No beta gate
python train.py --no_beta_gate \
  --checkpoint_dir outputs/ablations/no_beta_gate/checkpoints \
  --results_dir    outputs/ablations/no_beta_gate
python evaluate.py --no_beta_gate \
  --checkpoint  outputs/ablations/no_beta_gate/checkpoints/best.pt \
  --split test \
  --results_out outputs/ablations/no_beta_gate/eval_test.json

# f) 7×7 feature grid (L=49)
python train.py --feature_grid_size 7 \
  --checkpoint_dir outputs/ablations/feature_grid_7x7/checkpoints \
  --results_dir    outputs/ablations/feature_grid_7x7
python evaluate.py --feature_grid_size 7 \
  --checkpoint  outputs/ablations/feature_grid_7x7/checkpoints/best.pt \
  --split test \
  --results_out outputs/ablations/feature_grid_7x7/eval_test.json
```

Or use the runner script to print / execute all commands at once:

```bash
python scripts/run_ablations.py           # dry-run: print all commands
python scripts/run_ablations.py --smoke   # fast smoke tests (no GPU needed)
python scripts/run_ablations.py --run     # execute everything (expensive)
```

### Smoke tests

Verify every config builds and forward-passes without crashing:

```bash
pytest tests/test_ablations.py -v
```

### Where results are saved

```text
outputs/ablations/
  <experiment_name>/
    config.json          — args used for this run
    training_log.csv     — per-epoch train loss + val BLEU-1..4
    checkpoints/
      epoch_XXX.pt
      best.pt
    eval_val.json        — BLEU-1..4 + METEOR + loss on val split
    eval_test.json       — BLEU-1..4 + METEOR + loss on test split
```

### Suggested reporting table

| Experiment | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | Val Loss | Notes |
|---|---|---|---|---|---|---|---|
| baseline_soft_attention | | | | | | | Paper target: 67/44.8/29.9/19.5 |
| no_attention_mean | | | | | | | Static mean context |
| uniform_attention | | | | | | | α forced to 1/L |
| no_doubly_stochastic_lambda_0 | | | | | | | λ=0, no Eq.14 penalty |
| no_beta_gate | | | | | | | β forced to 1 |
| feature_grid_7x7 | | | | | | | L=49 instead of 196 |

Fill in from `outputs/ablations/<name>/eval_val.json` and `outputs/ablations/<name>/eval_test.json` after training each experiment.

### Visualization with ablation models

```bash
# Works for all modes except 'none' (which produces a flat/uniform overlay)
python visualize.py \
  --checkpoint outputs/ablations/<name>/checkpoints/best.pt \
  --attention_mode <mode> \       # match training flag
  --feature_grid_size <14 or 7>  # match training flag
  --output_dir results/<name>_attention

# For no_beta_gate
python visualize.py \
  --checkpoint outputs/ablations/no_beta_gate/checkpoints/best.pt \
  --no_beta_gate \
  --output_dir results/no_beta_gate_attention
```

For `attention_mode=none` and `attention_mode=uniform`, visualization still runs but produces uniform/flat overlays (annotated in the figure). This is the correct and expected behaviour — either no spatial attention was used during decoding, or α was forced to be uniform.
