"""Sentence encoders used for the dense part of matching.

Kept behind a tiny interface so the rest of the code doesn't care which one
is installed:

* ``sbert``  - sentence-transformers (all-MiniLM-L6-v2 by default). Best
  quality, but pulls in torch and downloads a model on first use.
* ``spacy``  - averaged GloVe-style vectors from en_core_web_md/lg. Much
  weaker, but small, CPU-only and installable from pip.
* ``none``   - dense matching switched off; the hybrid score re-normalises
  its weights over whatever is left.

Pick one with the RESUME_ANALYZER_ENCODER env var, or let ``auto`` try them
in that order.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from loguru import logger


class Encoder:
    name = "base"
    # Cosine similarity of two unrelated sentences sits well above zero for
    # most encoders. ``floor`` is subtracted before rescaling to [0, 1] so the
    # dense score uses its full range. Values were eyeballed on the sample
    # data in data/samples, not tuned on the benchmark.
    floor = 0.0

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def rescale(self, sims: np.ndarray) -> np.ndarray:
        return np.clip((sims - self.floor) / (1.0 - self.floor), 0.0, 1.0)


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class SBertEncoder(Encoder):
    name = "sbert"
    floor = 0.15

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or os.getenv("RESUME_ANALYZER_SBERT_MODEL", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)
        self.name = f"sbert:{self.model_name}"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=64))


class SpacyEncoder(Encoder):
    name = "spacy"
    floor = 0.55

    def __init__(self, model_name: str | None = None):
        import spacy

        candidates = [model_name] if model_name else ["en_core_web_md", "en_core_web_lg"]
        last_err: Exception | None = None
        for cand in candidates:
            try:
                self.nlp = spacy.load(cand, disable=["parser", "ner", "lemmatizer", "tagger", "attribute_ruler"])
                self.name = f"spacy:{cand}"
                break
            except OSError as err:
                last_err = err
        else:
            raise OSError(f"No spaCy model with vectors found ({last_err})")
        if not self.nlp.vocab.vectors.shape[0]:
            raise OSError("spaCy model has no word vectors (use en_core_web_md or _lg, not _sm)")

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        rows = []
        for doc in self.nlp.pipe(texts, batch_size=128):
            # Averaging over every token lets "the/and/with" dominate; skip
            # stop words and punctuation when there's anything left.
            toks = [t.vector for t in doc if t.has_vector and not t.is_stop and not t.is_punct]
            rows.append(np.mean(toks, axis=0) if toks else doc.vector)
        return _normalise(np.vstack(rows))


def load_encoder(kind: str | None = None) -> Encoder | None:
    kind = (kind or os.getenv("RESUME_ANALYZER_ENCODER", "auto")).lower()
    if kind == "none":
        return None
    order = ["sbert", "spacy"] if kind == "auto" else [kind]
    for k in order:
        try:
            enc = SBertEncoder() if k == "sbert" else SpacyEncoder()
            logger.info(f"Using {enc.name} encoder")
            return enc
        except Exception as err:  # ImportError, OSError, network errors on model download...
            logger.debug(f"encoder {k} unavailable: {err}")
            if kind != "auto":
                raise
    logger.warning("No sentence encoder available - dense matching disabled")
    return None


@lru_cache(maxsize=4)
def cached_encoder(kind: str = "auto") -> Encoder | None:
    return load_encoder(kind)
