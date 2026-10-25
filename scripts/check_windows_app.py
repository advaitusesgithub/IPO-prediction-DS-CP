"""Check the native app on an ephemeral local port, then stop the test server."""
from pathlib import Path
import json
import math
import socket
import sys
import threading
import time
import urllib.request

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import uvicorn
import pandas as pd

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
port = sock.getsockname()[1]
server = uvicorn.Server(uvicorn.Config("app.server:app", host="127.0.0.1", port=port, log_level="warning"))
thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
thread.start()
try:
    deadline = time.monotonic() + 45
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(.1)
    assert server.started, "Native API server did not start"
    url = f"http://127.0.0.1:{port}"
    assert urllib.request.urlopen(url, timeout=15).status == 200
    request = urllib.request.Request(url + "/api/predict", data=json.dumps({"name": "NSDL"}).encode(),
                                     headers={"Content-Type": "application/json"})
    result = json.load(urllib.request.urlopen(request, timeout=45))
    saved = pd.read_parquet(root / "outputs/part3/heldout_predictions.parquet").set_index("ipo_id").loc[result["ipo"]["id"]]
    assert math.isclose(result["prediction"]["opening_price"], saved.predicted_open_inr, abs_tol=1e-12)
    assert math.isclose(result["prediction"]["positive_probability"], saved.positive_gain_probability, abs_tol=1e-12)
    refined_available=(root/'outputs/refinement/refined_2026_predictions.parquet').exists()
    if refined_available:
        fresh=pd.read_parquet(root/'outputs/refinement/refined_2026_predictions.parquet').iloc[0]
        request=urllib.request.Request(url+'/api/predict',data=json.dumps({'name':fresh.symbol}).encode(),headers={'Content-Type':'application/json'})
        refined=json.load(urllib.request.urlopen(request,timeout=45))
        assert refined['source']['kind']=='external'
        assert math.isclose(refined['prediction']['opening_price'],fresh.predicted_open_inr,abs_tol=1e-12)
        assert refined['prediction']['threshold']==.45
    report = {"native_windows": sys.platform == "win32", "python_version": sys.version.split()[0],
              "http_server_started": True, "homepage_loaded": True,
              "api_prediction_matches_recorded_result": True, "refinement_sample_verified":refined_available,"wsl_required": False}
    (root / "outputs/windows").mkdir(exist_ok=True)
    (root / "outputs/windows/startup_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
finally:
    server.should_exit = True
    thread.join(timeout=15)
    sock.close()
