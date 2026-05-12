"""Project hyperparameters for the Flickr8k soft-attention reproduction."""

EMBED_DIM = 512
DECODER_DIM = 512
ATTENTION_DIM = 512
ENCODER_DIM = 512
DROPOUT = 0.5
VOCAB_SIZE = 10_000

LAMBDA = 1.0
GRAD_CLIP = 5.0
LEARNING_RATE = 4e-4
ENCODER_LR = 1e-4          # LR for unfrozen encoder layers (lower than decoder)
ENCODER_FINETUNE_EPOCH = 5  # Start fine-tuning encoder after this many warmup epochs
BATCH_SIZE = 64
NUM_EPOCHS = 50
PATIENCE = 10              # More patience now that LR decay can rescue plateaus
MAX_DECODE_LEN = 50

DATA_ROOT = "data/flickr8k"
VOCAB_PATH = "data/flickr8k/vocab.json"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
