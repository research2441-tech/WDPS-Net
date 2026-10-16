# WDPS-Net: Wavelength-Stepped State-Space Learning with Degradation-Prompted Expert Routing for Blind Cross-Sensor Hyperspectral Image Denoising

Reference implementation of the experiments in the accompanying manuscript. Everything runs on CPU with PyTorch.
All files are in one folder; run the commands from that folder.

## 1. Environment

```
python >= 3.10, torch >= 2.1, numpy, scipy, scikit-learn, h5py, tifffile, imagecodecs, spectral, matplotlib, pillow
pip install bm3d bm4d hyde-images pytorch_wavelets "setuptools<81"   # model-based baselines (BM4D, FastHyDe, L1HyMixDe)
```

## 2. Data

Download the public cubes listed in `prep_data.py` (Pavia University, Indian Pines 200/220 bands, Salinas, Salinas-A, Cuprite,
San Diego, Jasper Ridge, Salton Sea, Botswana, HYDICE Urban, Washington DC Mall crop, MUUFL Gulfport (64 and 72 bands),
Houston 2013/2018 (48 bands), Trento, APEX, SAMSON, the hylite FENIX test cube, CAVE, KAIST/TSA-Net scenes, EnMAP Potsdam)
into `./data` (variable `D` at the top of `prep_data.py`) with the sub-folder names used there, then:

```
python prep_data.py          # -> cubes/*.npz (scaled cubes + band-centre wavelengths) and cubes/meta.json
python wdps_data.py          # -> testsets/*.npz: 39 fixed 64x64 crops x 6 corruption cases (234 items)
```

Band centres are read from the files when available and otherwise built from the nominal sensor grid and the documented
band-removal list (see `prep_data.py`).

## 3. Files

| File | Content |
|---|---|
| `wdps_data.py` | sensor registry, leave-one-sensor-out folds, degradation simulator (cases C1-C6), fixed test sets |
| `wdps_models.py` | WDPS-Net: wavelength codes, clean-free degradation prompts, routed experts, unmixing-subspace prior, wavelength-stepped selective recurrences, spatial selective scans, uncertainty head |
| `baselines.py` | band-agnostic re-implementations: SqueezeNet-HSI, QRNN3D, SST-style Transformer, PromptIR-3D, SSM base |
| `classical.py` | BM4D, FastHyDe, L1HyMixDe (HyDe library), LRMR |
| `train.py` | source-domain training (one fold, one seed) |
| `evaluate.py` | MPSNR, MSSIM, SAM, ERGAS, band-wise uncertainty on held-out and cross-scene items |
| `adapt.py` | label-free adaptation to the held-out sensor (pseudo-clean re-degradation + unmixing consistency) |
| `downstream.py` | real-noise Indian Pines classification (SVM), MUUFL Gulfport and Cuprite ACE target detection |
| `analysis_runs.py` | noise-intensity sweep, inference-time sensitivity, routing analysis, seed-ensemble calibration |
| `efficiency.py` | parameters, operation counts, latency, peak memory |
| `res_utils.py`, `make_tables.py` | aggregation, Wilcoxon/Holm/Cliff/Friedman statistics, tables |
| `make_figs.py`, `fig_arch.py`, `fig_sensors.py`, `fig_style.py` | figures |
| `make_jobs.py`, `runner.py` | the complete experiment queue and a two-worker job runner |
| `wdps_metrics.py` | metric definitions |

## 4. Reproducing the study

```
python make_jobs.py               # writes jobs.txt (all trainings, evaluations, adaptations and analyses)
python runner.py jobs.txt 2       # two workers; each job is skipped when its result file exists
python make_tables.py             # results/summary.json and results/tables.json
python efficiency.py              # run alone on an otherwise idle machine (latency is measured)
python make_figs.py fig3 fig4 fig5 fig6 fig7 fig8 fig9
python fig_arch.py; python fig_sensors.py
```

Run all commands from this folder; `ensure_runner.sh` restarts the queue after an interruption.

Single runs:

```
python train.py --model WDPS --fold ROSIS --seed 0 --steps 800
python evaluate.py ckpt/WDPS_ROSIS_s0.pt
python adapt.py ckpt/WDPS_ROSIS_s0.pt 150
```

Folds: `ROSIS`, `HYDICE`, `Hyperion`, `VNIR-CASI/AISA`, `Natural`. Ablation flags of `train.py`: `--wave index`, `--no_gap`,
`--no_prompt`, `--no_route`, `--no_subspace`, `--no_spec`; `--gap_learned 0` fixes the gap exponent at one (pure wavelength stepping).
`python adapt.py <ckpt> 150 0` runs the label-free adaptation without the unmixing-consistency term.

## 5. Notes

* Every learned model is trained for 800 iterations with batches of six 24x24 crops, seeds 0, 1 and 2.
* Noise realizations of the test items are fixed by deterministic seeds, so all methods see identical inputs.
* The re-implemented baselines follow the published designs at reduced width; they are not the official checkpoints.
