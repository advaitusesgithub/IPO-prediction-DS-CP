# File guide

| Folder/file | Purpose |
|---|---|
| `submission/` | Report, presentation, guides, manifest and final ZIP |
| `notebooks/01_input_validation_and_setup.ipynb` | Executed input checks and fold-specific preprocessing |
| `notebooks/02_development_modeling.ipynb` | Executed 19-candidate comparison and frozen model selection |
| `notebooks/03_final_evaluation_and_demo.ipynb` | Executed final evaluation and demo handoff |
| `notebooks/04_model_audit_and_refinement.ipynb` | Executed tuning audit and comparison on 17 unused 2026 IPOs |
| `data/refinement/` | Small imported source snapshots, eligibility audit and new input snapshots |
| `outputs/refinement/`, `models/refined/` | Full development tuning, new-data results and refined models |
| `data/prepared/` | 253-row predictor matrix, IPO master and separate labels |
| `data/supporting/` | Compact financial, manager and market tables |
| `data/provenance/` | Bundled input checksums |
| `data/splits/` | Exact training, validation, excluded and reserved memberships |
| `reports/prior_preparation/` | Historical cleaning/source reconciliation evidence |
| `reports/eda/`, `figures/eda/` | Earlier descriptive analysis |
| `config/source_schema.json` | Original 52 allowed predictor columns |
| `config/features.json` | Core 12 and optional 7 financial predictors |
| `config/model_search.json` | Prespecified finite model search |
| `config/selected_models.json` | Selected models, threshold, code/input hashes |
| `ipo_prediction/data.py`, `splits.py`, `preprocessing.py` | Loading, time splits and training-only preprocessing |
| `ipo_prediction/models.py`, `metrics.py`, `train.py` | Estimators, metrics and development comparisons |
| `ipo_prediction/evaluate.py` | Frozen final fit and receipt-based evaluation reuse |
| `models/validation/` | 12 development fold pipelines; not final models |
| `models/final/` | Price and classification pipelines fitted on 149 IPOs |
| `outputs/part1/` | Input readiness results |
| `outputs/part2/` | Candidate metrics, held-out development predictions and audits |
| `outputs/part3/` | Reserved predictions, outcomes, metrics and evaluation receipt |
| `outputs/part4/` | Final test and delivery checks |
| `app/main.py` | Optional launcher for the custom frontend; use app/server.py and app/static/ |
| `tests/` | Time-order, leakage, model reload and application tests |
| `scripts/` | Notebook execution and document/archive generation |
| `requirements.lock.txt` | Exact installed dependency versions |
| `start_app.cmd`, `start_app.sh` | Native Windows and Linux/Bash launchers |

The ZIP contains the same working files and deliverables, without itself,
the installed environment, temporary files or caches. Its manifest records
relative paths and SHA-256 hashes; the manifest excludes itself to avoid a
circular checksum. Source paths in historical audits are provenance references,
not dependencies on sibling directories.
