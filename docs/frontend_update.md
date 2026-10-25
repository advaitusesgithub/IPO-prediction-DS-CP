# Historical prediction interface

The active frontend is custom HTML, CSS and JavaScript, served by a small
Starlette/Uvicorn API. Streamlit is no longer the application server. The models,
selected threshold and recorded 2025 evaluation remain unchanged.

## User flow

1. Type an IPO name or stock symbol, then press **Predict** or Enter.
2. Read estimated listing opening price, Apply/Skip signal, raw positive-gain
   probability and announced price band in a results table.
3. Alternatively, use **See historical IPOs prediction** and its local filter.
   Clicking a company runs the same prediction flow.
4. Request the recorded actual outcome separately, or download the result.
5. Open **About the model** for measured performance and limitations.

No prediction is displayed until requested. Empty, unknown and ambiguous names
have inline guidance. Example chips fill the search input; Predict remains the
explicit action. Unknown names do not trigger web lookup or fabricated outputs.

## Coverage and honest labels

- 103 reserved IPOs: exact saved held-out predictions.
- 90 development validation IPOs: exact saved fold-validation predictions.
- 59 initial training IPOs: inference using the existing final pipelines,
  marked **Training illustration**. These are retrospective examples, not
  independent historical forecasts or additional test results.
- One boundary IPO: explicitly unavailable; no time-valid forecast is invented.

The search catalog now has 270 names, with 269 supported predictions: the original 253 names plus 17 separately labeled 2026 audit examples. See docs/final_model_audit.md for the new-data results. Symbol lookup
includes exchange/profile symbols and a reviewed NSDL acronym alias. Partial
matches prompt a choice when more than one company matches.

## Implementation

- `app/server.py`: same-origin API and local static assets.
- `app/static/index.html`, `styles.css`, `app.js`: responsive interface.
- `ipo_prediction/inference.py`: local lookup, saved predictions and labeled
  training illustration inference. No fitting or live network retrieval.
- `start_app.cmd` and `start_app.sh`: derive their directory and launch Uvicorn.
- `tests/test_prediction_api.py`: lookup, errors, provenance and saved results.
- `scripts/check_frontend.py`: actual Chromium desktop/mobile interaction checks.

Browser screenshots and checks are stored in `outputs/ui/`. Playwright and its
browser are only for testing; running the app needs no browser automation tools.

## Restart after upgrading

Stop the old Streamlit terminal with Ctrl+C. In Windows Command Prompt opened
inside the project directory, run:

```cmd
start_app.cmd
```

From WSL/Bash:

```bash
bash start_app.sh
```

Open http://localhost:8501 and hard-refresh if the old page is cached. The
direct command is:

```bash
.venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8501
```

Live IPO research is a later extension and is not enabled by this redesign.
