"""Document-level train/val/test split.

The same document appears in several QA rows (and in groups A, B and AB), so we split
document ids, not rows: every row of a document lands in the same split.
"""

import random

# Split name -> relative size. Order matters: ids are dealt out in this order.
RATIOS = {"train": 8, "val": 1, "test": 1}
SPLITS = tuple(RATIOS)


def assign_splits(doc_ids, seed: int) -> dict:
    """Map each unique doc_id to a split name in the RATIOS proportions.

    The result depends only on the set of ids and the seed, not on row order.
    The last split takes whatever rounding leaves over.
    """
    ids = sorted(set(doc_ids))
    random.Random(seed).shuffle(ids)
    total = sum(RATIOS.values())
    split_of, start = {}, 0
    for i, (split, ratio) in enumerate(RATIOS.items()):
        end = len(ids) if i == len(RATIOS) - 1 else start + round(len(ids) * ratio / total)
        split_of.update((doc_id, split) for doc_id in ids[start:end])
        start = end
    return split_of
