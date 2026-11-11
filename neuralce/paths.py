"""Repository paths.

The package is normally used from a checkout (``pip install -e .``), so the
repository root is the parent of the ``neuralce`` package.  ``NCE_REPO``
overrides it; ``NCE_RUNS`` points large Monte Carlo output somewhere else
(for example a scratch file system).  Everything below is a plain string.
"""
from __future__ import annotations
import os
from pathlib import Path

REPO = os.environ.get("NCE_REPO") or str(Path(__file__).resolve().parent.parent)
DATA = os.environ.get("NCE_DATA") or os.path.join(REPO, "data")
MODELS = os.environ.get("NCE_MODELS") or os.path.join(REPO, "models")
RUNS = os.environ.get("NCE_RUNS") or os.path.join(REPO, "runs")
RESULTS = os.environ.get("NCE_RESULTS") or os.path.join(REPO, "results")


def runs_dir(*parts, mkdir=False):
    p = os.path.join(RUNS, *parts)
    if mkdir:
        os.makedirs(p, exist_ok=True)
    return p


def model_path(kind, tag, name="model.pkl"):
    """models/nce/<tag>/model.pkl or models/ce/<system>/<file>."""
    return os.path.join(MODELS, kind, tag, name)
