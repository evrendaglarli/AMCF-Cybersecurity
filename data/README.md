# AMCF Datasets

## Structure

- `raw/` - Original datasets (placeholder)
- `processed/` - Preprocessed feature matrices and events
- `synthetic/` - Synthetically generated data for testing

## Datasets

### Public Datasets
- **CIC-IDS2017**: 10,000 events, ~5% positive
- **CSE-CIC-IDS2018**: 12,000 events, ~4% positive
- **UNSW-NB15**: 8,000 events, ~6% positive

### Enterprise Corpus
- **enterprise_60day**: 25,000 events sampled from 60-day corpus
- Statistics: 100 endpoints, 50 users, ~5% malicious events

## File Format

All datasets are saved as pickle files containing:
```python
(X_train, y_train, X_val, y_val, X_test, y_test, events_test)
```

- `X_*`: Feature matrices (shape: n_samples × 50)
- `y_*`: Binary labels (0 = benign, 1 = malicious)
- `events_test`: List of Event objects with temporal information

## Generation

Datasets are automatically generated on first run:
```bash
python experiments/run_experiments.py
```

Or manually:
```bash
python data/synthetic/generate_synthetic_data.py
```
```

---

## 15. LICENSE

### `LICENSE`
```
MIT License

Copyright (c) 2025 Evren Daglarli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied, including but not limited to the warranties
of merchantability, fitness for a particular purpose and noninfringement.