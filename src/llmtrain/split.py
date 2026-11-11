"""Document-level train/val/test split.

The same document appears in several QA rows (and in groups A, B and AB), so we split
document ids, not rows: every row of a document lands in the same split.
"""

import random

SPLITS = ("train", "val", "test")
RATIOS = (8, 1, 1)


def assign_splits(doc_ids, seed: int) -> dict:
    """Map each unique doc_id to "train", "val" or "test" in an 8/1/1 ratio.

    The result depends only on the set of ids and the seed, not on row order.
    """
    ids = sorted(set(doc_ids))
    random.Random(seed).shuffle(ids)
    total = sum(RATIOS)
    n_train = round(len(ids) * RATIOS[0] / total)
    n_val = round(len(ids) * RATIOS[1] / total)
    bounds = {"train": ids[:n_train], "val": ids[n_train : n_train + n_val], "test": ids[n_train + n_val :]}
    return {doc_id: split for split, members in bounds.items() for doc_id in members}
