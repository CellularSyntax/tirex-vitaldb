# Hypotension AUROC — ours vs supervised foils (all2873_covrate)

Ours = **zero-shot** TiRex-2 (M1, drug covariate), held-out test, case-clustered 95% CI. Foils = **trained** models; Kapral (TFT) & Zhu (Transformer) both use VitalDB as an *external* set. Foil numbers from `notes/RELATED_WORK.md`. Caveat: event definitions/cohorts are not identical across studies — this is an indicative reference, not a matched benchmark (discuss in paper).

| Horizon | Ours M1 AUROC [95% CI] | Kapral internal | Kapral external | Zhu (external) |
|--:|:--|:--|:--|:--|
| 1 min | 0.985 [0.983, 0.987] | — | — | — |
| 3 min | 0.953 [0.948, 0.958] | — | — | — |
| 5 min | 0.927 [0.921, 0.932] | 0.909 | 0.903 | 0.904 |
| 7 min | 0.905 [0.899, 0.912] | 0.88 | 0.867 | — |
| 10 min | 0.881 [0.874, 0.888] | — | — | 0.892 |
| 15 min | 0.856 [0.848, 0.863] | — | — | 0.882 |

Kapral also reports an overall (horizon-averaged) hypotension AUROC of 0.933 internal / 0.919 external.
