# Comparison vs trained foils (scaffold)

Numbers sourced from `notes/RELATED_WORK.md` (single source of truth). **"Ours" cells stay as ledger
placeholders** (`[N] [H] [X%] [Y%] [Z]`) until Phase 3 computes them — do not fill here manually; the
ledger in `notes/PAPER_TARGET.md` is authoritative. This is a framing scaffold, not a results table.

| Dimension | Zhu et al. 2026 (PLOS Med) | Kapral et al. 2024 (eClinMed) | Ours (TiRex-2) |
|---|---|---|---|
| **Paradigm** | Trained (Transformer) | Trained (TFT, from scratch) | **Zero-shot foundation model, no training** |
| **Training cohort** | 319,699 cases (Nanjing) | 73,009 patients (Vienna) | none (off-the-shelf) |
| **Covariate handling** | **None** (vital signs only; assumes signs capture drug effects) | Medication **past/observed only** (cannot use post-origin interventions) | **Future-known infusion trajectory** over the horizon |
| **Task** | Binary IOH classification | Continuous MAP forecast + binary hypotension | Probabilistic MAP forecast + binary hypotension |
| **Horizons** | 5 / 10 / 15 min | 7 min continuous; binary 1/3/5/7 min | up to `[H]` min ({1,3,5,10}) |
| **VitalDB role** | External validation (n=5,260; 10-case alert sim) | External test (n=5,065) | **Primary**, selected for infusion cases |
| **Covariate contribution** | not tested (asserted implicit) | qualitative propofol-omission sidebar (Fig 3a) | **paired, per-horizon ablation with CIs** = `[X%]` |
| **Headline metric** | internal AUC 0.904/0.892/0.882 (5/10/15) | MAE 4/7 mmHg int/ext; AUROC 0.933/0.919 (mean); 5-min 0.909/0.903 | forecast: `[X%]`, `[Y%]`; hypotension AUROC `[Z]` |

**Read:** our contribution is the bottom-two rows read together — the *conjunction* of zero-shot +
future-known conditioning, quantified. Neither foil occupies that cell.
