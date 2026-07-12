"""Canonical subject-level train/val/test split, shared by the supervised baselines
and (for a matched comparison) by the re-scored TiRex-2 evaluation.

Subject-level (not window-level) so no subject leaks across splits. Deterministic given
the seed — the SAME split is reproduced everywhere it is imported.
"""
from __future__ import annotations
import numpy as np


def subject_split(caseids, caseid_to_subject, seed: int = 0, fracs=(0.6, 0.2, 0.2)):
    """Map every caseid -> 'train' | 'val' | 'test' via a seeded subject-level partition.

    caseids: iterable of caseid strings (as they appear in the windows).
    caseid_to_subject: dict caseid -> subjectid (falls back to caseid if missing).
    Returns dict caseid -> split label.
    """
    assert abs(sum(fracs) - 1.0) < 1e-6, "fracs must sum to 1"
    subj_of = {str(c): str(caseid_to_subject.get(str(c), str(c))) for c in caseids}
    subjects = sorted(set(subj_of.values()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(subjects))
    n = len(subjects)
    n_tr = int(round(fracs[0] * n)); n_va = int(round(fracs[1] * n))
    label_of_subj = {}
    for rank, idx in enumerate(perm):
        s = subjects[idx]
        label_of_subj[s] = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")
    return {str(c): label_of_subj[subj_of[str(c)]] for c in caseids}
