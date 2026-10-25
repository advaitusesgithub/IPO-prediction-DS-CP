# IPO Listing Prediction — Delivery Guide

## Read and present

- `IPO_Listing_Prediction_Report.pdf`: report ready for reading or printing.
- `IPO_Listing_Prediction_Report.docx`: editable report.
- `IPO_Listing_Prediction_Presentation.pptx`: presentation with speaker notes.
- `report.md`: plain-text report with links to evidence.
- `DEMO_AND_VIVA.md`: demonstration sequence and explanation prompts.
- `FOLDER_GUIDE.md`: project files and their purpose.
- `IPO_Listing_Prediction.zip`: portable source, data and deliverable bundle.
- `MANIFEST.csv`: checksums and sizes of the files included in the ZIP.

All four implementation parts are delivered. This does not mean the initial
90% accuracy aspiration was achieved. Reserved classification accuracy is
68.0%, equal to always Apply, with no Skip predictions. Price MAE is INR 49.86,
versus INR 44.95 for announced cap. Present these measured results honestly.

The final refinement is also included. On 17 unused 2026 IPOs, more training
history reduces price MAE from INR 68.97 to INR 50.95; unchanged refitted methods
achieve the same result. Announced cap wins at INR 30.67. Both classifiers score
29.41% and predict Apply throughout. Do not compare scores across different test
periods or present this audit as a successful 90%-accurate model. See the fourth
executed notebook and `docs/final_model_audit.md`.

## Start on the existing Windows computer

In Windows Command Prompt, change to the project folder using its Windows
path. You can also open that folder in File Explorer, type `cmd` in the address
bar and press Enter. Then run:

```cmd
start_app.cmd
```

Stop any old Streamlit process with Ctrl+C first. The launcher derives its location automatically and uses the native Windows
`.venv-win` environment. It does not need a hardcoded folder name. Open the local address
printed by Uvicorn, normally http://localhost:8501. Keep the terminal open;
press Ctrl+C to stop. Do not paste Linux `/mnt/...` paths into Windows `cd`.

The Windows launcher uses the local `.venv-win` environment and does not invoke
WSL. If the environment is missing, it creates it using the installed Windows
Python and `requirements-app.txt`. First setup needs internet. Linux/WSL still
uses the separate `.venv` environment.

From a WSL/Bash terminal opened in this folder:

```bash
bash start_app.sh
```

## Set up after extraction or renaming

The ZIP has one neutral top-level folder, `ipo_listing_prediction`. It contains
all required runtime inputs. Open a WSL/Linux terminal in that directory:

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m ipo_prediction.evaluate
.venv/bin/python scripts/verify_submission.py
.venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8501
```

The full notebook environment uses Python 3.14 on Linux/WSL. The app also supports a separate native Windows Python environment with the same model library versions. Use the same Python version
for the exact locked dependency set. If `venv`/`ensurepip` is unavailable, install
the venv support package for that Python version. Dependencies require internet
when installing; the app itself reads local files and needs no live data API.

Virtual environments contain absolute paths. Recreate `.venv` after moving or
renaming; application paths, model files and configuration require no edits.
Select the new `.venv/bin/python` interpreter in the editor.

## Verify

```bash
.venv/bin/python -m pytest --junitxml=outputs/part4/test_results.xml
.venv/bin/python -m ipo_prediction.evaluate
```

The evaluation command verifies the recorded evaluation and reuses it. It does
not fit or tune models again. Use the saved executed notebooks for review;
their outputs are included. To rerun a notebook:

```bash
.venv/bin/python scripts/execute_notebook.py 03_final_evaluation_and_demo.ipynb
```

This starts a local Jupyter kernel. Do not modify sealed data, frozen settings
or evaluation files to obtain a better score. The extraction of the original
large source PDFs is not part of runtime reproduction; prepared tables and
historical extraction audits are bundled instead. Original source licensing
and attribution still apply.

## Rebuild delivery documents

```bash
.venv/bin/python scripts/package_submission.py
```

This refreshes presentation files and the archive from the recorded outputs.
Machine-specific environment and runtime caches are deliberately excluded.

## Windows setup without WSL

Run `start_app.cmd` from Command Prompt. It creates `.venv-win` using your
installed Windows Python (3.12 or newer) if necessary, then installs only the
pinned application requirements. The original Linux environment and saved
models are not modified. Python 3.13 is used for the native Windows app.

To check the environment without starting the server:

```cmd
start_app.cmd --check
```

For a manual setup:

```cmd
py -3 -m venv .venv-win
.venv-win\Scripts\python.exe -m pip install -r requirements-app.txt
start_app.cmd
```

Both virtual environments are excluded from the submission ZIP.
