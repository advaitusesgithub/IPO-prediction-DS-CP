"""Launch using a project-local Windows environment; no WSL dependency."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / ".venv-win"
PYTHON = ENVIRONMENT / "Scripts/python.exe"


def launch():
    if os.name != "nt":
        print("Use bash start_app.sh on Linux/WSL.")
        return 1
    # A moved virtual environment is rebuilt using the system Python launcher.
    ready = False
    if PYTHON.exists():
        probe = subprocess.run([str(PYTHON), "-c", "import uvicorn,starlette,pandas,numpy,sklearn,pyarrow,joblib"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ready = probe.returncode == 0
    if not ready:
        if sys.prefix != sys.base_prefix:
            return subprocess.call(["py", "-3", str(Path(__file__).resolve())])
        if sys.version_info < (3, 12):
            print("Python 3.12 or newer is required. Install a current Windows Python and run start_app.cmd again.")
            return 1
        print("Preparing the local Windows app environment. First setup requires internet.", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(ENVIRONMENT)], check=True)
        subprocess.run([str(PYTHON), "-m", "pip", "install", "-r", str(ROOT / "requirements-app.txt")], check=True)
    # Ensure the pickle/runtime versions match instead of silently using system libraries.
    check = "import importlib.metadata,json; print(json.dumps({n:importlib.metadata.version(n) for n in ['scikit-learn','pandas','numpy','joblib']}))"
    actual = json.loads(subprocess.check_output([str(PYTHON), "-c", check], text=True))
    expected = json.loads((ROOT / "config/selected_models.json").read_text())["versions"]
    if actual != expected:
        print("Restoring the recorded model library versions...", flush=True)
        subprocess.run([str(PYTHON), "-m", "pip", "install", "-r", str(ROOT / "requirements-app.txt")], check=True)
    if "--check" in sys.argv:
        probe = "from ipo_prediction.inference import lookup; r,c=lookup('NSDL'); assert c==200; print('Native Windows environment and saved-model prediction verified.'); print(r['ipo']['name'])"
        return subprocess.call([str(PYTHON), "-c", probe], cwd=ROOT)
    print("Starting IPO Listing Prediction on http://localhost:8501", flush=True)
    print("Keep this window open. Press Ctrl+C to stop.", flush=True)
    return subprocess.call([str(PYTHON), "-m", "uvicorn", "app.server:app",
                            "--host", "127.0.0.1", "--port", "8501"], cwd=ROOT)


if __name__ == "__main__":
    try:
        raise SystemExit(launch())
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Startup failed: {error}")
        print("See submission/START_HERE.md, Windows setup.")
        raise SystemExit(1)
