# glada.augmentation

| File | Purpose |
|---|---|
| `rawboost.py` | `RawBoost`: linear/non-linear convolutive noise, impulsive noise, and stationary noise applied to raw waveforms. The only augmentation used in the reported results. |
| `musan_rir.py` | `MusanAugmenter` (additive noise), `RIRAugmenter` (reverberation), and `CombinedAugmenter`. Used in `phase_aug_validation`; off by default. |
| `__init__.py` | Re-exports the augmenters. |
