# InfoSign

Code for **Bridging the Translational Gap for Multimodal Foundation Models via Task-Specific Information Signatures**.

InfoSign combines nine AI-READI data sources for clinical prediction and examines their contributions through modality ablation and concept interventions.

| Experiment | Evaluation |
|---|---|
| **Full Clinical model** | Downstream classification, basic-only clinical inputs at inference, and 11 modality ablations |
| **Basic Clinical model** | Downstream classification and the same 11 modality ablations |
| **Mixed CBM** | Full/masked-input classification, predicted-concept restoration, and measured-concept supplementation |
| **Selected-concept CBM** | New bottleneck using only validation μ + 2σ concepts, compared under full and masked clinical inputs |

## Setup and data

Use Python 3.12 and a compatible [CUDA PyTorch/torchvision build](https://pytorch.org/get-started/locally/), then install the dependencies:

```bash
git clone https://github.com/chili-lab/InfoSign.git
cd InfoSign
python -m pip install -r requirements.txt
```

**Participant data are not included.** Obtain **AI-READI v3.0.0** from the [official dataset page](https://doi.org/10.60775/fairhub.3) using its [download instructions](https://docs.aireadi.org/docs/3/preliminary/azure_storage/azure-download). The study uses 2,280 participants and the provided `recommended_split` assignments.

Place the dataset under `data/aireadi/`, preserving all manifests and referenced files:

```text
data/aireadi/
├── participants.tsv
├── clinical_data/              # person, measurement, observation,
│                              # condition_occurrence, procedure_occurrence CSVs
├── cardiac_ecg/
├── environment/
├── retinal_flio/
├── retinal_oct/
├── retinal_octa/
├── retinal_photography/
├── wearable_activity_monitor/
└── wearable_blood_glucose/
```

For another data location, edit [`config.yaml`](config.yaml): set `preprocessing.dataset_root` and `study.clinical_dir`, and keep both `source_root` settings pointed at the same prepared-input directory. Personal overrides can be saved in `config.local.yaml` and passed with `--config config.local.yaml`.

## Run

From the repository root:

```bash
# Prepare labels, native inputs, train-only scalers, and shared image features.
python -u prepare.py --devices 0 1 2 3

# Train and evaluate all experiments on validation and test sets.
python -u run.py --devices 0 1 2 3
```

Use `--devices 0` for one GPU. Preparation preserves participant splits and caches features from a frozen ImageNet-pretrained ResNet-18. Weights download on first use; offline users can set `preprocessing.retinal_weights_path`. Already prepared inputs can be used directly with `run.py`.

Completed units are reused when rerunning the same command. Use `prepare.py --dry-run` to preview preparation, or `--stage cache` to resume after native preprocessing. Use a new output directory when changing experiment settings. To regenerate reports without training:

```bash
python run.py --config outputs/study_without_observation/runtime.yaml --stage report
```

## Experiments

The six outcomes are **MoCA < 26**, **broad cardiovascular conditions** (hypertension, dyslipidemia, circulation problems, retinal vascular occlusion, or other heart issues), **major cardiovascular events** (myocardial infarction or stroke), and histories of **osteoporosis**, **kidney problems**, and **lung problems**.

Full Clinical uses eligible structured clinical features and eight nonclinical modalities. Basic Clinical retains 23 encoded basic patient features and the same eight modalities. Each model is evaluated after removing each of the nine sources individually, all retinal sources together, or both wearable sources together: **11 ablations**.

Mixed CBM freezes the Full model and trains a shared concept predictor/classifier on paired full and masked representations. **Masking removes nonbasic clinical features and their availability indicators**, retaining basic information and the other eight modalities. From this masked baseline:

- **Predicted restoration** replaces one Concept ID's channels with the same participant's full-input concept predictions, probing internal concept dependence. Restoring all predicted IDs simultaneously reproduces full-input CBM predictions.
- **Measured supplementation** supplies available, encodable recorded values for learnable concept items, estimating the potential value of additional clinical records. Missing or unencodable items retain masked predictions.

Channels sharing a Concept ID are changed together. [Concept definitions](infosign/cbm/concepts.json) retain eligible Measurement, Condition, Device, and Procedure concepts with task-specific outcome exclusions. Observation-domain concepts are excluded, but `observation.csv` remains required for labels and eligible concepts stored there.

Training uses seed 42 and training-only minority oversampling. Validation/test retain natural class proportions; checkpoints and fixed classification thresholds are validation-selected. Reports include AUROC, positive-class F1, balanced accuracy, Brier scores, and 10,000 paired, outcome-stratified bootstrap resamples. Concept IDs exceeding the task/intervention/split mean ΔAUROC + 2 population SD are highlighted. Test ranking is exploratory, not a significance test; intervals do not adjust for selection or training-seed variability.

The selected-concept CBM is trained after those diagnostics. For each task, μ and σ are computed only from validation predicted-restoration ΔAUROC, and a Concept ID is kept only when its validation gain is strictly above μ + 2 population SD. A new concept layer and outcome MLP are then trained on the frozen Full Clinical backbone using only those concepts; there is no bypass around the concept layer. The saved all-concept Mixed CBM is left unchanged. Test gains do not choose the concept list. The robustness comparison is the paired drop from full inputs to masked inputs for the selected-concept CBM, the all-concept Mixed CBM, and the Full Clinical model.

```bash
python -u run.py --stage signature
python -u run.py --stage signature-report
```

## Outputs and code

Prepared data are stored in `outputs/preprocessed/` and `outputs/prepared/`. Experiment reports are under `outputs/study_without_observation/tables/`, with validation and test versions:

| Output | Contents |
|---|---|
| `SUMMARY_test.md` | Classification, modality ablations, and predicted restoration |
| `SUMMARY_CONCEPT_RECOVERY_test.md` | Both concept interventions and selected IDs |
| `MODALITY_ABLATION_{full,basic}_test.csv` | All modality removals |
| `{PREDICTED,MEASURED}_CONCEPT_RECOVERY_{ALL,SELECTED}_test.csv` | All individual-ID effects and selected concepts |
| `MIXED_CBM_PERFORMANCE_test.csv`, `ALL_CONCEPTS_RESTORED_test.csv` | Full/masked CBM performance and simultaneous predicted restoration |
| `concept_diagnostics/` | Concept quality, probability diagnostics, and observed-label-subset results |
| `SUMMARY_SIGNATURE_test.md` | Selected-concept CBM performance and full-to-masked drops |
| `SIGNATURE_CBM_test.csv` | Performance under both inputs and paired masking losses |
| `SIGNATURE_CONCEPTS.csv` | Concept IDs frozen by validation μ + 2σ |

| Location | Purpose |
|---|---|
| `prepare.py`, `run.py`, `config.yaml` | Preparation, experiment execution, and settings |
| `preprocessing/` | Clinical, ECG, sensor, and retinal preprocessing |
| `infosign/encoders.py`, `infosign/train.py` | Modality encoders, fusion, and Full/Basic training |
| `infosign/cbm/` | Concept supervision, Mixed CBM, interventions, and reports |
| `infosign/analysis.py`, `infosign/stats.py`, `infosign/report.py` | Ablations, statistics, and reporting |
