# SANED
《Shared Affective Neural Representation Learning in OPM-MEG for Naturalistic Emotion Decoding》

## Install

```bash
cd SANED
conda activate env
python -m pip install -r requirements.txt
```

## Prepare data

```text
data/source_cache/
├── sentence_manifest.tsv
├── roi_metadata.tsv
├── sub-01_source_roi.npy
└── ...
```

## Train

```bash
python -m saned.train --source-cache-root data/source_cache --output-root outputs
```

## Download Checkpoint and Inference

Download path: https://pan.baidu.com/s/10iW4tKR7qniV3ppCi_TrMg?pwd=zbd1

Download and save as `checkpoints/SANED.pt`。

```bash
python -m saned.predict \
  --checkpoint checkpoints/SANED.pt \
  --source-cache-root data/source_cache \
  --output outputs/predictions.tsv
```

