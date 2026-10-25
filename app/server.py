"""Local API and static frontend for IPO Listing Prediction."""
from pathlib import Path
import json
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from ipo_prediction.inference import repository, lookup, outcome
from ipo_prediction.paths import project_root, read_json

STATIC = Path(__file__).resolve().parent / "static"


async def index(request):
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


async def catalog(request):
    data = await run_in_threadpool(repository)
    return JSONResponse({"ipos": data[-1], "supported": sum(x["available"] for x in data[-1])})


async def prediction(request):
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        return JSONResponse({"error": "Please enter a valid IPO name.", "suggestions": []}, status_code=400)
    if not isinstance(body, dict) or not isinstance(body.get("name"), str):
        return JSONResponse({"error": "Please enter an IPO name.", "suggestions": []}, status_code=400)
    data, code = await run_in_threadpool(lookup, body["name"])
    return JSONResponse(data, status_code=code)


async def actual(request):
    data = await run_in_threadpool(outcome, request.path_params["ipo_id"])
    return JSONResponse(data if data else {"error": "IPO not found"}, status_code=200 if data else 404)


async def summary(request):
    data=read_json("outputs/part3/completion_summary.json")
    if (project_root()/'outputs/refinement/summary.json').exists():
        data['refinement']=read_json('outputs/refinement/summary.json')
    return JSONResponse(data)


async def download(request):
    return FileResponse(project_root() / "outputs/part3/heldout_predictions.csv",
                        media_type="text/csv", filename="ipo_listing_predictions.csv")


async def health(request):
    return JSONResponse({"status": "ok"})


app = Starlette(routes=[Route("/", index), Route("/api/ipos", catalog),
    Route("/api/predict", prediction, methods=["POST"]), Route("/api/outcome/{ipo_id}", actual),
    Route("/api/summary", summary), Route("/api/download", download), Route("/health", health),
    Mount("/assets", StaticFiles(directory=STATIC), name="assets")])
