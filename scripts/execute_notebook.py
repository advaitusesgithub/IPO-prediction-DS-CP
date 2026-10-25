"""Execute a project notebook with this interpreter; retain all cell outputs."""
from pathlib import Path
import argparse
import os
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
runtime = root / ".runtime"
for variable, subfolder in [("JUPYTER_RUNTIME_DIR", "jupyter"), ("IPYTHONDIR", "ipython"),
                            ("MPLCONFIGDIR", "matplotlib")]:
    path = runtime / subfolder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(path)
parser = argparse.ArgumentParser()
parser.add_argument("notebook", nargs="?", default="01_input_validation_and_setup.ipynb",
                    help="Notebook filename within notebooks/")
args = parser.parse_args()
path = (root / "notebooks" / args.notebook).resolve()
if path.parent != (root / "notebooks").resolve() or path.suffix != ".ipynb":
    parser.error("Choose a notebook directly inside notebooks/")
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(notebook, km=manager, timeout=600,
                        resources={"metadata": {"path": str(root)}})
client.execute()
nbformat.write(notebook, path)
print("Executed notebook saved with outputs.")
