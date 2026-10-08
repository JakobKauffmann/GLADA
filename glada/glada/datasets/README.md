# glada.datasets

| File | Purpose |
|---|---|
| `audio.py` | `load_audio`: reads audio with `soundfile`, resamples to 16 kHz mono, and crops or pads to 4 s. Every dataset class uses it. |
| `asvspoof.py` | ASVspoof 2019 LA: `ASVspoofDatasetWithSpeaker` (training, returns speaker IDs for the speaker adversary) and `ASVspoofEvalDataset`. |
| `asvspoof2021.py` | `ASVspoof2021Dataset` for the 2021 LA and DF evaluation sets. |
| `itw.py` | `InTheWildDataset`, used two ways: unlabelled target domain during training and labelled evaluation set. |
| `wavefake.py` | `WaveFakeDataset`: six vocoder subsets plus LJSpeech as bona fide. |
| `codecfake.py` | `CodecFakeDataset` for CodecFake+ neural-codec attacks. |
| `loader.py` | `create_dataloaders` (paired source/target loaders) and `create_balanced_sampler` (corrects the ASVspoof bona fide/spoof imbalance). |
| `__init__.py` | Re-exports the classes above. |
