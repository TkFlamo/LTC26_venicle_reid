from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
import json

from torch.utils.data import Sampler


class PKBatchSampler(Sampler[list[int]]):
    """P identities x K samples, preferring different cameras within each identity.

    Optional hard-negative map changes which identities are grouped together, without changing labels.
    """
    def __init__(self, dataset, p: int = 16, k: int = 4, batches_per_epoch: int | None = None, hard_map: str | Path | None = None, seed: int = 42):
        self.dataset = dataset
        self.p, self.k = int(p), int(k)
        self.seed = int(seed)
        self.by_id = defaultdict(list)
        self.cams = {}
        for i, row in dataset.df.iterrows():
            key = dataset.keys[i]
            self.by_id[key].append(i)
            self.cams[i] = str(row.camera_id)
        self.ids = list(self.by_id)
        self.batches_per_epoch = batches_per_epoch or max(1, len(dataset) // (self.p * self.k))
        self.hard = {}
        if hard_map:
            with open(hard_map, "r", encoding="utf-8") as f:
                self.hard = json.load(f)


    def set_hard_map(self, hard: dict | str | Path | None):
        if hard is None:
            self.hard = {}
        elif isinstance(hard, (str, Path)):
            with open(hard, "r", encoding="utf-8") as f:
                self.hard = json.load(f)
        else:
            self.hard = {str(k): [str(x) for x in v] for k, v in hard.items()}

    def __len__(self):
        return self.batches_per_epoch

    def _sample_k(self, key, rng):
        inds = self.by_id[key]
        by_cam = defaultdict(list)
        for i in inds:
            by_cam[self.cams[i]].append(i)
        cams = list(by_cam)
        rng.shuffle(cams)
        out = []
        for c in cams:
            out.append(rng.choice(by_cam[c]))
            if len(out) == self.k:
                return out
        while len(out) < self.k:
            out.append(rng.choice(inds))
        return out

    def __iter__(self):
        rng = random.Random(self.seed + random.randint(0, 1_000_000))
        for _ in range(self.batches_per_epoch):
            anchor = rng.choice(self.ids)
            chosen = [anchor]
            hard_candidates = [x for x in self.hard.get(anchor, []) if x in self.by_id and x != anchor]
            rng.shuffle(hard_candidates)
            for h in hard_candidates:
                if h not in chosen:
                    chosen.append(h)
                if len(chosen) >= max(2, self.p // 2):
                    break
            remaining = [x for x in self.ids if x not in chosen]
            rng.shuffle(remaining)
            chosen.extend(remaining[: max(0, self.p - len(chosen))])
            batch = []
            for key in chosen[: self.p]:
                batch.extend(self._sample_k(key, rng))
            yield batch
