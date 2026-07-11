# Hypotension AUROC — ours vs supervised foils (n300_s1)

Ours = **zero-shot** TiRex-2 (M1, drug covariate), held-out test, case-clustered 95% CI. Foils = **trained** models; Kapral (TFT) & Zhu (Transformer) both use VitalDB as an *external* set. Foil numbers from `notes/RELATED_WORK.md`. Caveat: event definitions/cohorts are not identical across studies — this is an indicative reference, not a matched benchmark (discuss in paper).

| Horizon | Ours M1 AUROC [95% CI] | Kapral internal | Kapral external | Zhu (external) |
|--:|:--|:--|:--|:--|
| 1 min | 0.990 [0.987, 0.993] | — | — | — |
| 3 min | 0.957 [0.942, 0.968] | — | — | — |
| 5 min | 0.937 [0.923, 0.949] | 0.909 | 0.903 | 0.904 |
| 7 min | 0.916 [0.899, 0.931] | 0.88 | 0.867 | — |
| 10 min | 0.886 [0.863, 0.906] | — | — | 0.892 |
| 15 min | 0.856 [0.831, 0.877] | — | — | 0.882 |

Kapral also reports an overall (horizon-averaged) hypotension AUROC of 0.933 internal / 0.919 external.
