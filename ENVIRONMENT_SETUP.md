# Environment Setup

This project has been validated in a Windows GPU environment with CUDA-enabled PyTorch.

## Recommended Baseline

- OS: Windows 10/11
- Python: 3.11.x
- GPU: NVIDIA GPU with a working driver
- Verified local setup date: May 11, 2026
- Verified PyTorch build: `torch==2.11.0+cu130`

## Files To Reuse

- Dependency lock file: [`requirements.txt`](./requirements.txt)
- Main entry: [`run_full_project_pipeline.py`](./run_full_project_pipeline.py)
- Scenario config: [`env_config.md`](./env_config.md)

## Clean Setup

Create a fresh virtual environment in the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## GPU Validation

After installation, verify that PyTorch can see CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
```

Expected result in a GPU-ready environment:

- `torch.__version__` includes `+cu130`
- `torch.cuda.is_available()` prints `True`

## Run Checks

Quick import check:

```powershell
.\.venv\Scripts\python.exe -c "import torch, yaml, numpy, networkx, skyfield, simpy, h3, plotly, geopandas, shapely, tzdata, scipy, sklearn, joblib, pandas, sgp4, jplephem, pyproj, pyogrio; print('imports_ok')"
```

Pipeline dry run:

```powershell
.\.venv\Scripts\python.exe run_full_project_pipeline.py --dry-run
```

Full run:

```powershell
.\.venv\Scripts\python.exe run_full_project_pipeline.py
```

## Notes For Teammates

- This repo previously had a copied `.venv` that pointed to another machine's Python path. Do not commit or reuse an existing `.venv` across machines.
- The full pipeline defaults to requiring CUDA. If CUDA is unavailable, the script will fail fast unless `--allow-cpu` is passed.
- CPU fallback is only for debugging or smoke tests. Full runs can be very slow on CPU.
- If a future project version changes the supported CUDA or PyTorch version, update `requirements.txt` first and keep this document in sync.

## Troubleshooting

If `torch.cuda.is_available()` is `False`:

1. Check `nvidia-smi` works in a terminal.
2. Confirm the active interpreter is the repo-local `.venv`.
3. Confirm `pip show torch` reports `2.11.0+cu130` instead of a `+cpu` build.
4. Reinstall with `pip install -r requirements.txt`.

If the virtual environment is broken:

```powershell
Rename-Item .venv .venv_broken
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
