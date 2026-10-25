# Run the IPO prediction interface

Stop any old app with Ctrl+C, then run `start_app.cmd` from Windows Command
Prompt inside this directory, or `bash start_app.sh` from WSL/Bash. Open
http://localhost:8501. Keep the terminal open; use Ctrl+C to stop.

Enter an IPO name or symbol and press **Predict**. Results appear in a table.
Alternatively, choose from **See historical IPOs prediction**. Use the optional
actual-outcome action for comparison; **About the model** contains measured
performance and limitations. See `docs/frontend_update.md` for coverage and
source labels.

The interface uses local historical data only. It can search 270 names and
show 269 predictions: 103 original held-out, 90 development validation, 59 explicitly
labeled training illustrations and 17 new 2026 audit predictions. The excluded boundary IPO has no forecast.
Training illustrations are not independent historical forecasts. Nothing is
fitted by the app, and unknown names do not initiate web lookup.

The evaluated 103-row model accuracy remains 68.0%, identical to always Apply,
with no Skip predictions. Price MAE remains INR 49.86 versus INR 44.95 for
announced cap. Probabilities are uncalibrated; costs and allotment are not modeled.

Direct WSL/Bash command:

```bash
.venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8501
```

After moving or renaming the directory, recreate `.venv` using the dependency
lock as described in submission/START_HERE.md. Project paths need no edits.
The active server is Uvicorn; the old Streamlit command is no longer used.
Windows startup no longer invokes WSL; `.venv-win` is separate from `.venv`.
