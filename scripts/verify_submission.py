"""Read-only verification of the recorded results and packaged deliverables."""
from pathlib import Path
import csv
import hashlib
import io
import json
import subprocess
import sys
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import nbformat
from docx import Document
from pptx import Presentation

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from ipo_prediction.evaluate import run


def verify():
    run()  # Receipt verification only; fitting/scoring is not repeated.
    test_report=next(p for p in [root/'outputs/refinement/test_results.xml',root/'outputs/ui/test_results.xml',root/'outputs/part4/test_results.xml'] if p.exists())
    suite = ET.parse(test_report).getroot().find("testsuite")
    assert int(suite.get("tests")) >= 26
    assert all(int(suite.get(name)) == 0 for name in ["failures", "errors", "skipped"])
    notebooks = []
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        code = [cell for cell in nb.cells if cell.cell_type == "code"]
        assert code and all(cell.execution_count is not None for cell in code)
        assert not [output for cell in code for output in cell.outputs if output.output_type == "error"]
        notebooks.append({"file": path.name, "executed_code_cells": len(code)})
    assert len(notebooks) >= 3
    destination = root / "submission"
    doc = Document(destination / "IPO_Listing_Prediction_Report.docx")
    text = " ".join(p.text for p in doc.paragraphs)
    assert "90%" in text and "not achieved" in text
    assert len(doc.tables) >= 6 and len(doc.inline_shapes) >= 2
    assert (destination / "IPO_Listing_Prediction_Report.pdf").read_bytes().startswith(b"%PDF")
    slides = Presentation(destination / "IPO_Listing_Prediction_Presentation.pptx")
    assert len(slides.slides) >= 12
    assert all(slide.has_notes_slide for slide in slides.slides)
    for path in destination.glob("*.docx"):
        with ZipFile(path) as z: assert z.testzip() is None
    for path in destination.glob("*.pptx"):
        with ZipFile(path) as z: assert z.testzip() is None
    archive = destination / "IPO_Listing_Prediction.zip"
    with ZipFile(archive) as z:
        assert z.testzip() is None
        prefix = "ipo_listing_prediction/"
        manifest = list(csv.DictReader(io.StringIO(z.read(prefix + "submission/MANIFEST.csv").decode())))
        expected_names = {prefix + row["relative_path"] for row in manifest} | {prefix + "submission/MANIFEST.csv"}
        assert set(z.namelist()) == expected_names
        for row in manifest:
            content = z.read(prefix + row["relative_path"])
            assert len(content) == int(row["bytes"])
            assert hashlib.sha256(content).hexdigest() == row["sha256"]
            assert content == (root / row["relative_path"]).read_bytes(), row["relative_path"]
            assert not {".venv", ".venv-win", ".runtime", "__pycache__", ".pytest_cache"} & set(Path(row["relative_path"]).parts)
    sidecar = (destination / "IPO_Listing_Prediction.zip.sha256").read_text().split()[0]
    assert sidecar == hashlib.sha256(archive.read_bytes()).hexdigest()
    deps = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=True)
    print(deps.stdout.strip())
    result = {"parts_delivered": [1,2,3,4], "tests_passed": int(suite.get("tests")), "test_failures": 0,
              "notebooks": notebooks, "report_formats": ["docx", "pdf", "markdown"],
              "presentation_slides": len(slides.slides), "manifest_and_archive_verified": True,
              "archive_files": len(manifest)+1, "dependencies_consistent": True}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()
