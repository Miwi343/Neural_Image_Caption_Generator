"""Hyperparameters for the VQA yes/no task."""

# Model dims — match the caption model's encoder output so we can reuse weights
ENCODER_DIM = 512       # VGG annotation vector dimension (fixed)
QUESTION_EMBED_DIM = 512
GRU_DIM = 512
ATTENTION_DIM = 512
DROPOUT = 0.5

# Training
LEARNING_RATE = 3e-4
ENCODER_LR = 1e-4
ENCODER_FINETUNE_EPOCH = 5
BATCH_SIZE = 64
NUM_EPOCHS = 30
PATIENCE = 7
GRAD_CLIP = 5.0
LAMBDA = 1.0            # doubly-stochastic attention regularisation weight

# Vocabulary
VOCAB_SIZE = 15_000     # question vocabulary (larger than caption vocab)

# Data
DATA_ROOT = "data/vqa"
VOCAB_PATH = "data/vqa/vocab.json"
CHECKPOINT_DIR = "checkpoints_vqa"
RESULTS_DIR = "results_vqa"

# Kaggle dataset slug for VQA v2
KAGGLE_DATASET = "jacksoncrow/vqa-v2-yes-no-subset"
# Official VQA v2 annotations on Kaggle (full, if the subset isn't available)
KAGGLE_DATASET_FULL = "ambarish/vqa-v2"

# Maximum question token length (questions longer than this are truncated)
MAX_QUESTION_LEN = 20
