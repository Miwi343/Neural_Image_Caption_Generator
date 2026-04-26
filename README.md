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
  images/
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
python visualize.py --images data/flickr8k/images/123456.jpg
```

Each generated word gets a `14x14` attention map upsampled to `224x224`, Gaussian-smoothed, and overlaid on the input image.
