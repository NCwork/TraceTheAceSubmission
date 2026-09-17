# Trace the Ace — LLM-KT Qwen3.5-9B (richer judge)

Predicts whether a student answers their post-tutoring quiz question correctly, given the tutoring-session transcript and the learning objective (competition metric: log loss). A LoRA-tuned Qwen3.5-9B knowledge-tracing model scores each (session, objective) pair, aided by a fine-tuned 2B judge that rates transcript evidence; this build scored 0.5964 on the public leaderboard.

## Setup

1. Download the kt adapter from [kt_adapter](https://kookree-my.sharepoint.com/:u:/g/personal/nicholas_kookree_ai/IQDbeVckH6mbRZ3CI7Qhwf5JAdbVtYpPi4DKhw4jb_Oslf8?e=BjJz29) and extract into `kt_adapter_s0/`
2. Download the model  from [Model](https://kookree-my.sharepoint.com/:u:/g/personal/nicholas_kookree_ai/IQDUgpbg2zMCQYhi04spE25HAQyRByVApTWpRJZbbkKqm20?e=oZxhm3) and extract into `models/`

   ```
   models/
   ├── kt9b_base/        # Qwen3.5-9B base (4 safetensors shards)
   ├── judge2b/          # fine-tuned 2B judge
   ├── minilm/           # sentence-embedding model
   └── kt_adapters.zip   # LoRA adapter (s0/), loaded from the zip at runtime
   ```

   (`kt_adapter_s0/` is a reference copy of the raw LoRA adapter; the code only reads `models/kt_adapters.zip`.)

3. Install dependencies (Python 3.10+):

   ```
   pip install -r requirements.txt
   ```

## Run

Place the input data in a `data/` folder next to `main.py`:

```
data/
├── test_features.csv
├── submission_format.csv
└── test_transcripts/<session_id>.csv   # one transcript per session
```

Then:

```
python3 main.py
```

Predictions are written to `submission.csv`. Requires a CUDA GPU with ~24 GB+ VRAM (9B in BF16 + 2B judge); all models load from the local `models/` dir — no network access needed at runtime.
