# Hypotension AUROC — ours vs supervised foils (cases115_covpressor)

Ours = **zero-shot** TiRex-2 (M1, drug covariate), held-out test, case-clustered 95% CI. Foils = **trained** models; Kapral (TFT) & Zhu (Transformer) both use VitalDB as an *external* set. Foil numbers from `notes/RELATED_WORK.md`. Caveat: event definitions/cohorts are not identical across studies — this is an indicative reference, not a matched benchmark (discuss in paper).

| Horizon | Ours M1 AUROC [95% CI] | Kapral internal | Kapral external | Zhu (external) |
|--:|:--|:--|:--|:--|
| 1 min | 0.974 [0.949, 0.990] | — | — | — |
| 3 min | 0.944 [0.916, 0.966] | — | — | — |
| 5 min | 0.911 [0.882, 0.939] | 0.909 | 0.903 | 0.904 |
| 7 min | 0.883 [0.844, 0.917] | 0.88 | 0.867 | — |
| 10 min | 0.840 [0.793, 0.880] | — | — | 0.892 |
| 15 min | 0.824 [0.778, 0.867] | — | — | 0.882 |

Kapral also reports an overall (horizon-averaged) hypotension AUROC of 0.933 internal / 0.919 external.
