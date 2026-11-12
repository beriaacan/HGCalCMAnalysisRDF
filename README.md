# HGCal common-mode analysis

Derive and apply linear and DNN common-mode noise corrections to HGCal Nano ROOT
files from `Jun2026TB` and `Cassette` campaigns. Analysis choices live in YAML files.
**Submit the complete analysis with [`workflow.py`](#submit-a-complete-htcondor-workflow):**
it creates the required HTCondor jobs, manages their dependencies and starts each
stage automatically when its inputs are ready. One submission can cover several
analysis configs and share their common preparation and training.

Use `steer.py` when you want to run explicit steps locally instead.

## Setup

Use Bash on CERN Linux 9 with CVMFS and access to the CMS input data. Create the
CMSSW environment and install the central dependencies first:

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc13
scram project CMSSW CMSSW_20_1_0_pre2
cd CMSSW_20_1_0_pre2/src
eval "$(scram runtime -sh)"
scram-venv
eval "$(scram runtime -sh)"

git clone https://gitlab.cern.ch/hgcal-dpg/hgcal-comm.git HGCalCommissioning
git clone https://gitlab.cern.ch/hgcal-dpg/calibrations.git HGCalCommissioning/Calibrations
scram b -j 4
```

From that same `src` directory, clone this repository separately:

```bash
git clone https://github.com/reimersa/HGCalCMAnalysisRDF.git HGCalCommissioning/Analysis/HGCalCMAnalysis
cd HGCalCommissioning/Analysis/HGCalCMAnalysis
python3 -m pip install -r requirements.txt
./setup_training.sh
```

Keep the checkout at this location. Training and DNN application automatically use
the installed Python environment; CUDA execution requires a compatible GPU and a
driver supporting CUDA 12.6. CPU execution is also supported.

In each fresh shell:

```bash
cd /path/to/CMSSW_20_1_0_pre2/src/HGCalCommissioning/Analysis/HGCalCMAnalysis
source setup.sh
voms-proxy-init --voms cms --valid 192:00
```

The proxy is needed for remote data access and batch submission. For HTCondor,
keep the checkout, environments and submission/state directories on shared AFS.
Input discovery requires access to the relevant EOS mount.

## Configure an analysis

Copy a supplied config and edit its data productions and output storage:

| Config | Purpose |
| --- | --- |
| [jun2026.yaml](configs/jun2026.yaml) | June test-beam derivation, training and application |
| [cassette_118787.yaml](configs/cassette_118787.yaml) | All 53 modules of Run118787; no trigger-time or ADC cuts |
| [cassette_118787_apply_118786.yaml](configs/cassette_118787_apply_118786.yaml) | Same source models applied to Run118786 |

One analysis YAML describes both the source models (`derivation`) and target data
(`application`). For example:

```yaml
# Available steps: pedestals, prepare, training_inputs, analytic, analytic_allinputs,
#   training_cache, train, submit_train, apply, plots, multimodule_plot, export.
steps: [prepare]
methods: [uncorrected, analytic, analytic_allinputs, dnn]
output_dir: "${HOME}/hgcal_rdf"

derivation:
  campaign: Jun2026TB
  runs: [118151, 118153]
  modules: [ML_F3WC_IH0180, ML_F3WC_IH0182]
  nano_production: REPLACE_SOURCE_PRODUCTION
  pedestal_run: 118151
  selection: trigtime
  trigger_time_min: -2
  trigger_time_max: 5
  adc_max: 10
  validation_split_fraction: 0.2
  train_event_fraction: 1.0
  validation_event_fraction: 1.0

application:
  campaign: Jun2026TB
  runs: [118179]
  modules: all
  nano_production: REPLACE_TARGET_PRODUCTION
  pedestal_run: 118151
  selection: full
  device: auto

epochs: 200
batch_events: 1024
training_device: auto
training_resume: true
```

Replace both production placeholders. Choose an `output_dir` with sufficient space
for ROOT files, typically your EOS storage. Relative paths are YAML-relative;
environment variables and `~` expand. Inspect the resolved settings before running:

```bash
./steer.py --config configs/my_analysis.yaml --show
```

The common options are explained below. The commented
[analysis template](configs/template.yaml) lists **every supported option and
default**, including advanced fitting, plotting and performance settings. Omitted
options use those defaults; unknown and duplicate keys are errors.

| Data option, inside each block | Meaning |
| --- | --- |
| `campaign` | `Jun2026TB` or `Cassette`; selects the default Nano root, module map and calibration search directory. |
| `runs` | One run or a list. Multiple application runs are combined in plots; use separate configs for separate run plots. |
| `modules` | Ordered names, `all`, `ld` or `hd`. Derivation chooses training/linear modules. Application `all` means exactly the derivation set; lists or `ld`/`hd` can select another target set. |
| `nano_production` | Production directory below `Run<run>`. Pin it explicitly; a glob such as `'*'` is accepted only if exactly one production matches per run. |
| `pedestal_run` | Calibration source run, independently chosen for derivation and application. |
| `selection` | `full` or `trigtime`. The latter uses `trigger_time_min < time < trigger_time_max`; events from that block's pedestal run bypass the trigger cut. |
| `adc_min`, `adc_max` | Optional bounds on pedestal-subtracted target ADC; other-channel predictors remain uncut. Upper-only means `ADC <= adc_max`; two-sided bounds are open. Omit for no cut. |
| `validation_split_fraction` | Derivation only; default `0.2` gives an 80%/20% train/validation split, shared across modules. No independent test set. |
| `train_event_fraction`, `validation_event_fraction` | Derivation only; fractions retained within each partition. Both `0.5` means about 40%/10% of all events. Default `1.0` uses both partitions fully. |
| `split_seed` | Derivation seed for reproducible event membership; distinct from the DNN initialization/shuffle `training_seed`. |
| `device` | Application DNN device: `auto`, `cpu` or `cuda`. |
| `allow_unknown_modules` | Application only; explicitly permit DNN transfer to modules outside its training set. Requires combined training and registered target module IDs. |

Campaign defaults are relative to the central `HGCalCommissioning` checkout:

| Campaign | Nano root | Module map | Calibration directory |
| --- | --- | --- | --- |
| `Jun2026TB` | `/eos/cms/store/group/dpg_hgcal/tb_hgcal/2026/TestBeam` | `Calibrations/TB2026/maps/modulelocator_v4.txt` | `Calibrations/TB2026/calib` |
| `Cassette` | `/eos/cms/store/group/dpg_hgcal/tb_hgcal/2025/Test` | `Calibrations/P5/maps/modulelocator_P5v9.txt` | `Calibrations/P5/calib` |

Inputs resolve to `<Nano root>/Run<run>/<production>/NANO_<run>_*.root`.
Optional `nano_root`, or alternatively a complete `input_pattern` containing
`{run}`, supports other layouts. `module_map` and `calibration_file` override the
campaign defaults. Separate pedestal productions can be set with
`pedestal_nano_production` or `pedestal_input_pattern`; otherwise they use that
block's main input pattern. Each block resolves its resources independently.

Without `calibration_file`, the code looks recursively for one matching central
`level0_calib_<pedestal_run>.json`. If absent, it uses the local target
`calibrations/<campaign>/level0_calib_<pedestal_run>.json`. Ambiguous central matches
and filename/run mismatches are errors. `pedestals` produces only missing local
files; it refuses production when matching central constants exist and never
overwrites existing constants. Skip this step when calibrations already exist.

| Top-level option | Meaning |
| --- | --- |
| `steps` | Default ordered steps for local execution; `--steps` replaces them. |
| `methods` | `uncorrected` baseline, `analytic` CM-only linear regression, `analytic_allinputs` all-input linear regression, and/or `dnn`. Include the baseline for noise comparisons. |
| `output_dir` | ROOT caches, models, predictions, numerical statistics and exports. Figures stay in this checkout's `plots/`. |
| `training_grouping` | `combined` (default): one DNN for the derivation modules. `per_module`: one DNN per module, applied to that same module name. Linear models are always module-specific. |
| `epochs`, `batch_events` | Maximum epochs and full module-events per optimizer batch. Early stopping may finish sooner. |
| `learning_rate`, `weight_decay`, `sample_weighting` | AdamW learning rate/decay and loss weighting (`none` or `source_run_channel`). |
| `training_device`, `training_resume` | Device (`auto`, `cpu`, `cuda`) and resume from the last completed epoch. Completed matching models are reused; `false` refuses incomplete existing training. |
| `training_input_backend` | `numeric_cache` (default) reads a prebuilt compact cache; `root` streams prepared ROOT files each epoch. |
| `cpus`, `condor_*` | Local processing threads and batch CPU/GPU, memory, job-flavour and submission settings; see the template. |
| `plot_groups`, `multimodule_plot_groups` | Select per-module or detector-wide figure families. `plot_mip_fit: false` skips the optional spectrum mixture fits. |
| `n_coherent_noise_model`, `plot_noise_fit_variants` | Noise-decomposition rank and fit variants: `independent` fits both components; `fixedD_uncorrected` fixes incoherent variances to the uncorrected fit. |
| `multimodule_noise_range`, `multimodule_performance_range` | Fixed map colour scale and performance-axis limits; values outside the display remain in the saved statistics. |

Model identities are automatic. To apply an existing model to another run or
campaign, keep `derivation` and training settings unchanged and edit `application`.
For DNN transfer to different modules use combined training and, if needed,
`methods: [uncorrected, dnn]`: linear models cannot transfer between module names.
New module IDs can be registered explicitly with
`python3 -m utils.module_registry --registry data/module_registry.json --add MODULE_NAME`;
never renumber existing IDs.

The fixed DNN inputs use calibration-subtracted ADC and CM, train-only internal
z-scores for ADC/CM, and fixed geometry/index/activity features. Targets are
pedestal-subtracted ADC. Normalization is saved with the model and stays fixed
during application; no preprocessing options need coordinating between configs.

## Run locally

`steer.py` runs exactly the listed steps, in order; **it does not add prerequisites**.

| Step | Result |
| --- | --- |
| `pedestals` | Produce missing local calibration JSONs using central calibration tools. |
| `prepare` | Reusable per-module ROOT inputs for both derivation and application runs; calibration subtraction and central measurement validity, before analysis cuts. |
| `training_inputs` | Shared training membership, selections, target counts and feature definitions. |
| `analytic` | CM-only linear predictors from retained training events. |
| `analytic_allinputs` | All-input linear predictors, excluding each target's own ADC. |
| `training_cache` | Compact numeric event arrays for the default DNN backend. |
| `train` | Local CPU/GPU training, validation, checkpoints and loss curves. |
| `submit_train` | Submit training alone to HTCondor; returns immediately. |
| `apply` | Apply the configured methods to the application data and save compact ROOT predictions/corrected values. |
| `plots` | Individual-module matrices, spectra, noise and channel/cell-area comparisons. |
| `multimodule_plot` | Detector-wide summaries; calculate missing covariances/noise fits without needing `plots` first. |
| `export` | Copy original Nanos and append correction branches; originals remain untouched. |

For a first local run with all methods and existing calibrations:

```bash
./steer.py --config configs/my_analysis.yaml --steps prepare training_inputs analytic analytic_allinputs training_cache
./steer.py --config configs/my_analysis.yaml --steps train apply plots multimodule_plot
./steer.py --config configs/my_analysis.yaml --steps export   # optional
```

For application to a new target with existing models:

```bash
./steer.py --config configs/my_analysis.yaml --steps prepare apply multimodule_plot
```

For training-only HTCondor submission, set `condor_gpus: 1` for GPU or `0` for CPU,
with a compatible `training_device`, and run `--steps submit_train` after building
the training inputs/cache. Do not append application steps to that command; use
the complete workflow below if they should follow automatically.

| Additional `steer.py` option | Meaning |
| --- | --- |
| `--config FILE` | Required analysis YAML. |
| `--steps STEP ...` | Replace the YAML's ordered list for this invocation. |
| `--show` | Print effective/defaulted settings and paths without processing or submitting. |
| `--list-models` | List stored DNNs, recipes and completion status; runs no steps. |
| `-h`, `--help` | Show command-line help. |

## Submit a complete HTCondor workflow

Use this interface to submit the whole analysis and let HTCondor manage the
sequence, rather than running or submitting each stage yourself. Keep the analysis
settings in ordinary analysis YAMLs; a workflow YAML lists those configs and the
final results you want.

For example, request plots, multimodule summaries and exports for two application
runs using the same derivation/training recipe:

```yaml
analyses:
  - config: ../configs/cassette_118787.yaml
    steps: [plots, multimodule_plot, export]
  - config: ../configs/cassette_118787_apply_118786.yaml
    steps: [plots, multimodule_plot, export]
max_jobs: 4
max_io_jobs: 2
max_gpu_jobs: 1
```

| Workflow YAML option | Meaning |
| --- | --- |
| `analyses` | Analysis config paths, relative to the workflow YAML, with optional `steps` goals. Omit goals to use that analysis's `steps`. |
| `max_jobs` | Total submitted/active worker-job limit (default 8). |
| `max_io_jobs`, `max_gpu_jobs` | CPU-stage and GPU-stage job limits (defaults 2 and 1). Queued jobs count, so a waiting GPU job occupies its category slot. |
| `work_dir` | Frozen submissions and logs; default `workdir_condor/workflows` in this checkout. |
| `state_dir` | Shared producer/completion records; default `.workflow_state` in this checkout. Use one shared AFS directory across submissions for coordination. |
| `retries` | Additional attempts per failed node (default 0). |
| `resources` | Optional per-step overrides: `cpus`, `gpus`, `memory_gb`, `job_flavour`. Otherwise use the analysis's `condor_*` settings. |

The planner works backwards from those goals: it checks calibrations, adds missing
preparation, training inputs/cache and model derivation/training, then application
and the requested final stages. Identical prerequisites across configs run once;
verified existing products are reused. In this example, the DNN is trained once
and applied separately to both runs.

Each stage runs as a separate job. HTCondor's DAGMan dependency manager starts a
job only after all its prerequisites succeed. Independent jobs can run in
parallel, subject to the limits above and available cluster resources. Plotting
and export can follow application in parallel; multimodule summaries wait for
the corresponding plots when both are requested. Use `train`, not `submit_train`,
if training itself is a workflow goal: the manager submits that job for you.

[workflows/template.yaml](workflows/template.yaml) shows all workflow options.
With the environment and proxy ready, inspect and submit the entire workflow:

```bash
./workflow.py --config workflows/cassette.yaml --show
./workflow.py --config workflows/cassette.yaml --submit
./workflow.py --config workflows/cassette.yaml --status
```

After `--submit`, no further commands are needed to advance the stages. You can
close the terminal or disconnect from lxplus: DAGMan continues managing the jobs
on the HTCondor infrastructure. Use `--status` later to see completed, running,
queued and waiting stages. A failed stage blocks its dependent jobs; independent
branches can continue.

| `workflow.py` option | Meaning |
| --- | --- |
| `--config FILE` | Workflow YAML for planning, preparation, submission or latest-request status. |
| `--show` | Inspect the plan and reuse; default action. No processing/submission. |
| `--prepare` | Freeze settings/code/input inventories and generate job files; do not submit. |
| `--submit` | Reuse a matching preparation or create one, then submit. Refuses duplicate submission of the same request. |
| `--status [DIR]` | With `--config`, monitor its latest submitted request; pass a directory instead for a specific request. |
| `--refresh-plots` | Request fresh figures when planning/preparing/submitting; numerical caches remain reusable. |
| `--json` | Machine-readable plan/status output. |
| `--submit-directory DIR` | Submit a specific prepared request; omit `--config`. |
| `--resume DIR` | Resume a stopped frozen workflow; omit `--config`. The original DAG/workers must have exited. |
| `--release DIR --node NAME --owner-stopped` | Release abandoned workflow ownership after verifying its DAG/workers stopped. Does not remove other stage locks. |
| `-h`, `--help` | Show command-line help. |

An optional `--prepare` before `--submit` needs no directory copy/paste. To create
another request after a previous submission, run `--prepare` again, then `--submit`
with the same options. Submitted requests keep frozen settings: YAML edits do not
change active jobs, and source-code changes can invalidate later stages. Keep code
unchanged until the workflow finishes.

Logs are under the printed submission directory: each node has
`job.<cluster>.<proc>.out/.err`, `pre.log` and `post.log`; the manager log is
`workflow.dag.dagman.out`. `condor_q` also shows queued/running jobs.

## Results and reuse

| Product | Where to look |
| --- | --- |
| Prepared ROOT inputs | `<output_dir>/<campaign>/prepared/` |
| Training membership/numeric arrays | `<output_dir>/<campaign>/training/` |
| Linear/DNN models | `<output_dir>/models/<derivation campaign>/` by default |
| Predictions, numerical plot caches and exports | `<output_dir>/<application campaign>/evaluation/` |
| Individual-module figures | `plots/<application campaign>/Run<run>/<module>/...` |
| Multimodule figures | `plots/<application campaign>/Run<run>/multimodule/...` |
| Training curves | `plots/<derivation campaign>/training/...` |

Figure paths continue with `pedestals_from_Run<run>/<source-model campaign>/`
`<selection>/variants/rdf2_<ID>/`. Automatic IDs distinguish data/model recipes;
rerunning the same variant overwrites its figures. Completion manifests and
checksums protect cached numerical products. Matching completed work is reused;
incompatible or damaged products stop reuse. `--show` and stage output give paths.

Export adds `HGCDigi_adcCorrection_<method>`, `HGCDigi_adcCorrected_<method>` and
`HGCDigi_adcCorrectionValid_<method>` branches. Exported corrected ADC is **raw ADC
minus prediction**, retaining the pedestal offset; compact application files hold
pedestal-subtracted corrected ADC. Export does not extend saved prediction coverage:
rejected/unavailable digis receive NaNs and validity zero. Files for a run are
together under `nano_exports/<method-set>/Run<run>/`.

## Common problems

| Symptom | Action |
| --- | --- |
| Python/ROOT import errors | Start a fresh shell, source `setup.sh`, and use `python3`. Install the provided requirements in the CMSSW venv. |
| Missing inputs or ambiguous production | Check run, production, EOS access and proxy; select one production explicitly. |
| Missing calibration/model/cache | Run the required local steps, or let `workflow.py` include them. Do not run `pedestals` when central constants already exist. |
| CUDA unavailable or job idle/held | Check GPU allocation, `training_device`/application `device`, resources, proxy, `condor_q` and job logs. |
| Existing lock/producer | Check the recorded host/PID and scheduler before recovery. Never remove a live owner's lock. |
| Provenance/checksum mismatch | Preserve the affected product and investigate; do not edit manifests to bypass checks. |
| Slow plotting | Spectrum fits and vector matrix PDFs can take time. Use `multimodule_plot` alone for detector-wide summaries. |

