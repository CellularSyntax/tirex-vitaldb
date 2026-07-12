# Hypotension AUROC — ours vs supervised foils (all2873)

Ours = **zero-shot** TiRex-2 (M1, drug covariate), held-out test, case-clustered 95% CI. Foils = **trained** models; Kapral (TFT) & Zhu (Transformer) both use VitalDB as an *external* set. Foil numbers from `notes/RELATED_WORK.md`. Caveat: event definitions/cohorts are not identical across studies — this is an indicative reference, not a matched benchmark (discuss in paper).

| Horizon | Ours M1 AUROC [95% CI] | Kapral internal | Kapral external | Zhu (external) |
|--:|:--|:--|:--|:--|
| 1 min | 0.985 [0.983, 0.987] | — | — | — |
| 3 min | 0.955 [0.950, 0.959] | — | — | — |
| 5 min | 0.930 [0.924, 0.935] | 0.909 | 0.903 | 0.904 |
| 7 min | 0.908 [0.902, 0.914] | 0.88 | 0.867 | — |
| 10 min | 0.881 [0.875, 0.888] | — | — | 0.892 |
| 15 min | 0.854 [0.846, 0.862] | — | — | 0.882 |

Kapral also reports an overall (horizon-averaged) hypotension AUROC of 0.933 internal / 0.919 external.
