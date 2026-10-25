"""Name lookup and historical prediction access; no web lookup or model fitting."""
from difflib import get_close_matches
from functools import lru_cache
import re
import joblib
import pandas as pd
from .data import get_xy, load_dataset
from .paths import project_root, read_json
from .models import positive_probability


def normalize(value):
    words = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split()
    return " ".join(w for w in words if w not in {"limited", "ltd", "private", "pvt", "ipo"})


@lru_cache(maxsize=1)
def repository():
    root = project_root()
    data = load_dataset()
    master = pd.read_parquet(root / "data/prepared/ipo_master.parquet").set_index("ipo_id")
    membership = pd.read_csv(root / "data/splits/final_membership.csv").set_index("ipo_id")
    heldout = pd.read_parquet(root / "outputs/part3/heldout_predictions.parquet").set_index("ipo_id")
    validation = pd.read_csv(root / "outputs/part2/selected_validation_predictions.csv")
    revised = root / 'outputs/refinement/refined_2026_predictions.parquet'
    if revised.exists():
        extension = pd.read_parquet(root / 'data/refinement/external_2026.parquet').set_index('ipo_id',drop=False)
        predictions = pd.read_parquet(revised).set_index('ipo_id')
        predictions['model_revision'] = 'v2'
        heldout = pd.concat([heldout,predictions])
        extra_master = pd.DataFrame(index=extension.index)
        extra_master['company_name'] = extension.company_name
        extra_master['nse_symbol'] = extension.symbol
        extra_master['profile_symbol'] = extension.symbol
        extra_master['bse_symbol'] = None
        extra_master['listing_year'] = 2026
        extra_master['price_band_low_inr'] = extension.band_low_inr
        extra_master['price_band_high_inr'] = extension.band_high_inr
        cohort = pd.read_csv(root / 'data/refinement/cohort.csv').set_index('issue_id')
        extra_master['venue'] = extra_master.index.map(cohort.venue)
        master = pd.concat([master,extra_master])
        membership = pd.concat([membership,pd.DataFrame({'role':'external_refinement'},index=extension.index)])
        data = pd.concat([data,extension])
    catalog = []
    for ipo_id, row in master.sort_values("company_name").iterrows():
        aliases = [str(row[c]) for c in ['nse_symbol','profile_symbol','bse_symbol'] if pd.notna(row[c])]
        aliases += {'national securities depository': ['NSDL']}.get(normalize(row.company_name), [])
        symbol = aliases[0] if aliases else ""
        role = membership.loc[ipo_id, "role"]
        catalog.append({"id": ipo_id, "name": row.company_name, "symbol": symbol, "aliases": aliases,
                        "year": int(row.listing_year), "available": role != "purged_label_unavailable"})
    return data, master, membership, heldout, validation, catalog


def lookup(query):
    query = query.strip()
    if not query or len(query) > 180:
        return {"error": "Enter an IPO name to get started.", "suggestions": []}, 400
    *_, catalog = repository()
    normalized = normalize(query)
    if not normalized:
        return {"error": "Please enter a company name, such as Hyundai or NSDL.", "suggestions": []}, 400
    exact = [x for x in catalog if query == x["id"] or normalize(x["name"]) == normalized or
             any(normalize(alias) == normalized for alias in x["aliases"])]
    matches = exact or [x for x in catalog if normalized in normalize(x["name"])]
    if len(matches) == 1:
        return predict(matches[0]["id"])
    if len(matches) > 1:
        return {"error": "A few IPOs match that name. Choose the company you mean.", "suggestions": matches[:8]}, 409
    names = {normalize(x["name"]): x for x in catalog}
    close = get_close_matches(normalized, names, n=4, cutoff=.4)
    return {"error": "We couldn’t find that IPO in the historical collection. Try another name or choose from the list.",
            "suggestions": [names[n] for n in close]}, 404


def predict(ipo_id):
    data, master, membership, heldout, validation, _ = repository()
    row = master.loc[ipo_id]
    role = membership.loc[ipo_id, "role"]
    if role == "purged_label_unavailable":
        return {"error": "This IPO was excluded at the evaluation boundary and has no saved time-valid forecast. Please choose another IPO.", "suggestions": []}, 422
    if ipo_id in heldout.index:
        saved = heldout.loc[ipo_id]
        price, probability = float(saved.predicted_open_inr), float(saved.positive_gain_probability)
        source, label = "reserved", "Held-out prediction"
        description = "Saved evaluation prediction. This IPO was not used to fit the models."
        if saved.get('model_revision') == 'v2':
            source, label = 'external', '2026 audit prediction'
            description = 'Saved new-data evaluation from models trained on 2023–2025 IPOs. The audit did not establish useful Apply/Skip detection.'
    else:
        saved = validation.loc[validation.ipo_id.eq(ipo_id)]
        if len(saved):
            price = float(saved.loc[saved.task.eq("regression"), "prediction"].iloc[0])
            probability = float(saved.loc[saved.task.eq("classification"), "prediction"].iloc[0])
            source, label = "validation", "Validation prediction"
            description = "Saved development validation prediction, from models fitted on earlier IPOs."
        else:
            bundles = final_models()
            prices = []
            for task in ["regression", "classification"]:
                bundle = bundles[task]
                x, _ = get_xy(data, [ipo_id], bundle["spec"]["feature_set"])
                prices.append(float(bundle["model"].predict(x)[0]) if task == "regression"
                              else float(positive_probability(bundle["model"], x)[0]))
            price, probability = prices
            source, label = "training", "Training illustration"
            description = "This IPO was used to train the final models. This is a retrospective illustration, not an independent historical forecast."
    threshold = read_json('config/refined_selection.json')['selection']['classification']['threshold'] if source=='external' else .5
    return {"ipo": {"id": ipo_id, "name": row.company_name,
                    "symbol": "" if pd.isna(row.nse_symbol) else row.nse_symbol,
                    "year": int(row.listing_year), "venue": row.venue},
            "prediction": {"opening_price": price, "positive_probability": probability,
                           "signal": "Apply" if probability >= threshold else "Skip", "threshold": threshold,
                           "band_low": float(row.price_band_low_inr),
                           "band_high": float(row.price_band_high_inr)},
            "source": {"kind": source, "label": label, "description": description},
            "cutoff": data.loc[ipo_id, "s0_cutoff_at"].strftime("%d %b %Y · 09:00 IST")}, 200


@lru_cache(maxsize=1)
def final_models():
    return {task: joblib.load(project_root() / f"models/final/{task}.joblib")
            for task in ["regression", "classification"]}


def outcome(ipo_id):
    data, *_ = repository()
    if ipo_id not in data.index:
        return None
    row = data.loc[ipo_id]
    return {"opening_price": float(row.listing_open_inr), "issue_price": float(row.final_issue_price_inr),
            "listing_gain": float(row.target_listing_gain_open)}
