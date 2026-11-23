# AMGC-AC

AMGC-AC multi-view clustering demo on the UCI dataset.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python demo.py
```

Optional arguments:

```bash
python demo.py --epochs 500 --seed 1000 --device auto
```

The demo evaluates KMeans after every epoch and reports the epoch with the
highest ground-truth-label ACC. All metrics are taken from the same epoch.
This is an oracle evaluation protocol.
