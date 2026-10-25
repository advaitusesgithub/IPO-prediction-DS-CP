# IPO Listing Prediction

## Run the app

Install Python 3.13 (64-bit) and Git, then clone this repository:

```text
git clone https://github.com/advaitusesgithub/IPO-prediction-DS-CP.git
cd IPO-prediction-DS-CP
```

On **Windows Command Prompt**, run:

```bat
start_app.cmd
```

The launcher creates `.venv-win` and installs the pinned app dependencies on
first use. Internet is needed for that installation. WSL and Docker are not
required. On **Linux or macOS**, create the environment first:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-app.txt
bash start_app.sh
```

Open **http://localhost:8501**. Enter a historical IPO name, for example **NSDL**
or **Hyundai**, and press **Predict**, or choose one from the historical list.
Keep the terminal open; press Ctrl+C to stop the server.

Prepared data and trained models are included, so no dataset downloads or
retraining are needed to run the demo. This version supports historical IPOs;
live web lookup for new IPOs is not implemented. Predictions and reported
accuracy are historical evaluation results, with limitations described below.

For notebooks, tests, and report generation, install the full environment with
`python -m pip install -r requirements.lock.txt` inside your virtual environment.
The minimal app dependencies above are enough for the website.

S0 listing-opening-price regression and positive-listing-gain classification.
All runtime data, code, configuration and generated results live inside this
directory. Paths are derived from the package location; its enclosing folder
name is not used in application code or configuration.

## Current status

The final modeling audit is complete. Read `docs/final_model_audit.md` and the
executed `notebooks/04_model_audit_and_refinement.ipynb`. It compares 48 model
candidates over eight time-ordered folds and tests on 17 unused 2026 IPOs.
On those same new IPOs, price MAE improves from INR 68.97 to INR 50.95 through
additional training history; unchanged refitted methods achieve the same result.
Announced cap still wins at INR 30.67. Classifier accuracy remains 29.41%, equal
to always Apply, and useful Skip detection is not established. No tuning was
performed after viewing the new-data scores. Original 2025 records are preserved.

The app now also shows these 17 cases as **2026 audit predictions**. Their
decision threshold is 0.45; original results retain 0.5. Different evaluation
periods and retrospective training illustrations must not be pooled as one score.

Parts 1–4 are complete. Prepared inputs and temporal memberships are fixed;
development models have been compared and their choices frozen. The executed
notebooks retain processing, modeling and final evaluation outputs. The final
models were fitted on 149 IPOs and evaluated on 103 reserved IPOs.

Reserved price MAE is INR 49.86 (announced cap alone: INR 44.95). Classifier
accuracy is 68.0%, identical to always Apply, with zero Skip predictions.
90% accuracy and useful Skip detection have not been demonstrated.
Launch the historical local demo with:

```bash
.venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8501
```

See `docs/run_demo.md` and `notebooks/03_final_evaluation_and_demo.ipynb`.

## Submission materials

Open `submission/START_HERE.md` for the final report, presentation, folder guide,
demonstration sequence and ZIP. From Windows Command Prompt, enter this folder
using its Windows path and run `start_app.cmd`; it uses a separate native Windows
environment, without requiring WSL. From Bash, run `bash start_app.sh`. No folder name is hardcoded.

The ZIP excludes `.venv` and runtime caches. After extracting it, recreate the
Python environment using the locked dependencies before starting the app.
`scripts/package_submission.py` rebuilds the report, slides and ZIP from the
recorded results; it does not refit models or alter evaluation results.

Across 90 development validation IPOs, the training-median-premium price baseline
wins with MAE INR 94.45. The selected Random Forest classifier achieves 77.8%
accuracy, matching always Apply and identifying no non-positive listings at the
fixed threshold. A useful Apply/Skip distinction has **not** been established.
Read `docs/part2_interpretation.md` before presenting these results.

## Start here

The frontend has been redesigned: enter an IPO name and press Predict, then
read a results table. A separate historical collection supports browsing.
It supports 103 held-out, 90 validation and 59 clearly labeled retrospective
training illustrations. See `docs/frontend_update.md`. Stop the old Streamlit
process before running the updated launcher on port 8501.

Open `notebooks/01_input_validation_and_setup.ipynb` for input processing and
`notebooks/02_development_modeling.ipynb` for the executed model comparisons.
Read `docs/input_contract.md` for definitions, `docs/part2_findings.md` for
quantitative findings and `docs/build_plan.md` for the four-part sequence.

From this directory, use the installed isolated environment:

```bash
.venv/bin/python -m ipo_prediction.prepare
.venv/bin/python -m pytest
.venv/bin/python scripts/execute_notebook.py
```

To reproduce the fixed Part 2 comparison and its notebook:

```bash
.venv/bin/python -m ipo_prediction.train
.venv/bin/python scripts/execute_notebook.py 02_development_modeling.ipynb
.venv/bin/python -m pytest --junitxml=outputs/part2/test_results.xml
```

The modeling command refuses to overwrite a different frozen selection.
Re-running repeats the fixed search; it is not permission to tune on final
results. Model specifications, threshold and input/code hashes are recorded in
`config/selected_models.json`. Production fitting and the reserved 103-row
evaluation were completed in Part 3. The 12 saved fold models remain development
artifacts; the two final pipelines are in `models/final/`.

Notebook execution starts a local Jupyter kernel. Choose this directory's
`.venv/bin/python` as the notebook interpreter in your editor. The execution
script explicitly uses the interpreter that launches it.

## Files

| Location | Contents |
|---|---|
| `data/prepared/` | Fixed model matrix, IPO master and labels |
| `data/supporting/` | Compact financial, manager and market source tables |
| `data/splits/` | Frozen development, validation and reserved memberships |
| `data/provenance/` | Bundled input checksums |
| `config/` | Source schema, split contract, fixed search and selected specifications |
| `ipo_prediction/` | Reusable loaders, split verification and preprocessing |
| `notebooks/` | Executed processing/readiness evidence |
| `outputs/part1/` | Readiness reports and diagnostic development matrices |
| `outputs/part2/` | Full comparisons, fold/per-IPO predictions, audits and tests |
| `figures/part2/` | Five model comparison and diagnostic charts |
| `outputs/part3/` | Reserved predictions, separate outcomes, metrics and evaluation receipt |
| `models/final/` | Two final pipelines with exact training IDs and settings |
| `figures/part3/` | Reserved price and confusion-matrix plots |
| `reports/prior_preparation/` | Earlier cleaning and source reconciliation evidence |
| `reports/eda/`, `figures/eda/` | Imported descriptive EDA evidence |
| `tests/` | Label-maturity, leakage, preprocessing and relocation checks |
| `docs/` | Input definitions, scope and implementation sequence |
| `models/validation/` | Selected and best-learned challenger fold pipelines |
| `app/` | Custom HTML/CSS/JavaScript frontend and Python API |

The imported EDA is historical evidence. Its gain-regression suggestions have
been superseded by direct opening-price regression in `config/evaluation.json`.
The current configuration and input contract govern subsequent model work.
No original heavy source archive or sibling directory is read by runtime code.

## Rebuild the environment or rename the directory

The installed environment uses Python 3.14. Exact resolved versions are in
`requirements.lock.txt`; direct dependencies are listed in `requirements.in`.
For a fresh compatible Python environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m ipo_prediction.prepare
.venv/bin/python -m pytest
```

Virtual environments contain generated absolute paths and are not portable.
After moving/renaming the directory, recreate `.venv` at the new location and
select its interpreter. Project code/config/data require no path edits.
If the operating system lacks venv/ensurepip support, install its Python venv
package first. On Windows-native Python, use `.venv\Scripts\python.exe`.

Large historical raw-document extraction is not rerun by this bundle. Its
audit reports retain source names and hashes; all required model inputs are
already bundled. Preserve source license/attribution requirements.
