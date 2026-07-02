# Stage 0 — Data

Download small, license-friendly text datasets to train on.

## What it does

`download_data.py` fetches:
- **Tiny Shakespeare** (~1 MB) — the classic language-modeling starter corpus.
- **TinyStories** (a streamed subset) — short, simple stories with a small
  vocabulary.

Everything is saved as plain `.txt` in `../data/`.

## Run

```bash
python stage0_data/download_data.py                 # both, 10k stories
python stage0_data/download_data.py --stories 50000 # more stories
python stage0_data/download_data.py --no-shakespeare
```

## Notes

- Add your own `.txt` files to `data/` and later stages will include them
  automatically.
- On an NVIDIA box, download far more data (see `docs/07_moving_to_nvidia.md`).

📖 Background: [`docs/00_fundamentals.md`](../docs/00_fundamentals.md)

➡️ Next: [`stage1_tokenizer/`](../stage1_tokenizer/)
