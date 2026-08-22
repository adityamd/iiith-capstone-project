# Three-model inference dashboard

This Streamlit app compares balanced Logistic Regression, weighted Histogram Gradient Boosting,
and the three-seed PyTorch MLP from `checkpoint1_EDA_v3.ipynb`. All three models use the same
54 leakage-safe predictors available at the 25% course checkpoint.

## Prerequisites

- Windows with the existing Conda environment named `MLDL`.
- The OULAD CSV files in a folder named `open+university+learning+analytics+dataset` in the
  repository directory or one of its parent directories.
- The packages in `dashboard/requirements.txt` installed in MLDL.

Expected dataset files:

```text
open+university+learning+analytics+dataset/
├── assessments.csv
├── courses.csv
├── studentAssessment.csv
├── studentInfo.csv
├── studentRegistration.csv
├── studentVle.csv
└── vle.csv
```

Install or refresh the dependencies with:

```powershell
conda activate MLDL
python -m pip install -r dashboard\requirements.txt
```

Verify that the active interpreter and core packages are available:

```powershell
conda activate MLDL
python -c "import sys, sklearn, torch, streamlit; print(sys.executable); print(sklearn.__version__, torch.__version__, streamlit.__version__)"
```

The interpreter path should contain `anaconda3\envs\MLDL`.

## Start the dashboard

### Recommended launcher

From the repository root:

```powershell
.\dashboard\run_dashboard.ps1
```

The launcher looks for MLDL under the current user's Anaconda/Miniconda installation and the common
system-wide Conda locations. It adds the MLDL Windows DLL directories to `PATH` before starting
Streamlit. Open the local URL printed in the terminal, normally <http://localhost:8501>.

For an MLDL environment in a different location, supply its interpreter explicitly:

```powershell
$env:MLDL_PYTHON = 'D:\path\to\envs\MLDL\python.exe'
.\dashboard\run_dashboard.ps1
```

Stop the dashboard with `Ctrl+C` in the terminal.

### Launch from an activated environment

From the repository root, activate MLDL first so its Windows DLL folders are available:

```powershell
conda activate MLDL
python -m streamlit run dashboard\app.py
```

## What is loaded during inference

The dashboard does not execute notebooks or retrain models. At startup it loads:

| Artifact | Contents |
|---|---|
| `dashboard/artifacts/dashboard_bundle.joblib` | Fitted preprocessor, Logistic Regression, HGB, schema, test predictions, metrics, curves, curated cases and reference values |
| `dashboard/artifacts/mlp_states.joblib` | NumPy exports of the three trained neural-network state dictionaries |
| `dashboard/artifacts/benchmark_metrics.csv` | Human- and machine-readable common test metrics |
| `dashboard/artifacts/benchmark_summary.json` | Artifact version, dataset fingerprint, feature counts and metric summary |

Each inference button follows the same path:

```text
54 raw checkpoint features
        → saved preprocessing pipeline
        → 93 encoded/scaled inputs
        → selected trained model
        → withdrawal probability
        → decision at the displayed threshold
```

Logistic Regression and HGB use their fitted scikit-learn estimators. The neural-network button
runs an equivalent NumPy evaluation-mode forward pass over the exported PyTorch weights and
averages probabilities from seeds 42, 43 and 44. Artifact-parity tests verify these probabilities
against the trained model outputs.

## Rebuild artifacts

Rebuild artifacts when they are missing or after changing feature engineering, preprocessing,
the split, model configuration, or training dependencies:

```powershell
conda activate MLDL
python -m dashboard.train_dashboard_models
```

The training command locates `open+university+learning+analytics+dataset` in the current directory
or a parent directory, regenerates the grouped benchmark, and writes the bundle, MLP states, metrics,
and summary to `dashboard/artifacts`.

Training reproduces the v3 25%-checkpoint feature engineering, keeps every enrollment belonging
to a student in one partition, fits final models on training plus evaluation data after settings
are frozen, and reports all three models on the common grouped test partition. The historical test
partition appeared in earlier project analysis, so these results should be described as a
consistent held-out comparison rather than previously unseen external validation.

## Tests

```powershell
conda activate MLDL
python -m unittest discover dashboard\tests -v
```

Expected result:

```text
Ran 4 tests
OK
```

The tests cover curated-case selection, artifact schema, saved-model probability parity, and
feature-group sensitivity output.

## Manual acceptance test

1. Select each curated case from the sidebar.
2. Run the three models individually and confirm that results persist side by side.
3. Use **Run all** and verify the probability comparison chart appears.
4. Change the shared threshold and confirm case decisions reset and can be recalculated.
5. Enable what-if editing, change an engagement or assessment input, and rerun the models.
6. Open **Performance** and verify the metric table, PR/ROC curves, calibration plot and subgroup audit.
7. Open **Prediction differences** and verify pairwise agreement and probability distributions.
8. Open **Case gallery** and confirm the five evidence-selected examples are available.

The most direct disagreement demonstration is **HGB catches a withdrawal others miss**. The saved
case produces approximately 11% Logistic Regression risk, 75% HGB risk and 23% neural-network risk.

## Threshold interpretation

The benchmark threshold is fixed at `0.50`. The dashboard slider is a case-level what-if control and
does not alter the published test metrics.

Changing the slider can change the displayed class for a selected case, but it does not rewrite
the metric table, confusion counts, curves, or saved test predictions.

## Troubleshooting

### `torch` or `c10.dll` fails to load

Use `dashboard\run_dashboard.ps1` or activate MLDL before starting Streamlit. Directly invoking
`MLDL\python.exe` without its DLL directories on `PATH` can prevent PyTorch from loading on Windows.
The live dashboard does not import PyTorch, but artifact rebuilding does.

### Dashboard artifacts are missing

Run:

```powershell
conda activate MLDL
python -m dashboard.train_dashboard_models
```

### Dataset folder is not found

Confirm the folder name is exactly `open+university+learning+analytics+dataset` and that it is in
the repository directory or a parent directory. Alternatively, specify it explicitly:

```powershell
conda activate MLDL
python -m dashboard.train_dashboard_models --data-dir "C:\path\to\open+university+learning+analytics+dataset"
```

### Port 8501 is already in use

Start Streamlit on another port:

```powershell
conda activate MLDL
python -m streamlit run dashboard\app.py --server.port 8502
```
