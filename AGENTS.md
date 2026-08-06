# Project Notes

- Training code, dataset source definitions, benchmark fixtures, and model evaluation belong in this private repository.
- Keep raw downloads and generated intermediate/training/review data under `data/raw/`, `data/interim/`, `data/train/`, and `data/review/`; these paths are ignored and must not be committed.
- Every imported record must carry source ID, source URL, license ID, retrieval timestamp, reading source, and reading confidence.
- Only records from sources cleared for the intended use may be submitted to DeepSeek. Never submit user input or local selection events to an external API.
- Run dataset unit tests with `python -m unittest discover -s tools/dataset/tests -v`.
