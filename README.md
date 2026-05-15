# Handwritten Text Recognition

A PyTorch project for recognizing handwritten characters (digits, uppercase and lowercase letters) using the [EMNIST Balanced](https://www.nist.gov/itl/products-and-services/emnist-dataset) dataset. A small CNN baseline is implemented in `src/` and trained interactively in Jupyter notebooks.

## Features

- **EMNIST Balanced** — 47 classes (0–9, A–Z, a–z, and a few punctuation marks)
- **`EMNISTDataset`** — loads CSV exports, corrects EMNIST orientation, normalizes pixels to `[0, 1]`
- **`SimpleCNN`** — two convolutional blocks plus a linear classifier (47 outputs)
- **`get_char`** — maps class indices to display characters via the EMNIST label mapping

## Project structure

```
handwritten-text-recognition/
├── data/raw/emnist_balenced/   # EMNIST Balanced CSV files (Git LFS)
├── notebooks/
│   ├── baseline.ipynb          # Train and evaluate the CNN baseline
│   └── test.ipynb              # Quick experiments
├── results/                    # Saved checkpoints (created when training)
├── src/
│   ├── datasets/emnist.py
│   ├── models/cnn.py
│   └── utils/label.py
├── pyproject.toml
└── uv.lock
```

## Requirements

- **Python 3.14+** (see `requires-python` in `pyproject.toml`)
- **[uv](https://docs.astral.sh/uv/)** — dependency and environment management
- **[Git LFS](https://git-lfs.com/)** — required if you clone this repo and want the bundled CSV data

Optional but recommended for GPU training:

- NVIDIA GPU with a [CUDA-enabled PyTorch build](https://pytorch.org/get-started/locally/) (the default `torch` wheel from PyPI may be CPU-only depending on platform)

## Install uv

### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell, or add `~/.local/bin` to your `PATH` if the installer tells you to.

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the terminal so `uv` is on your `PATH`.

Verify:

```bash
uv --version
```

## Setup

Clone the repository and install dependencies with uv. These steps are the same on Linux and Windows; only shell syntax for environment variables differs (see [Run notebooks](#run-notebooks) below).

```bash
git clone https://github.com/shikoyud/handwritten-text-recognition
cd handwritten-text-recognition
```

### 1. Fetch dataset files (Git LFS)

Training CSVs are tracked with Git LFS:

```bash
git lfs install
git lfs pull
```

Expected paths:

- `data/raw/emnist_balenced/emnist-balanced-train.csv`
- `data/raw/emnist_balenced/emnist-balanced-test.csv`

If you are not using the repo’s LFS files, download the **EMNIST Balanced** split from [NIST EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset), extract the CSVs, and place them in `data/raw/emnist_balenced/` (keep the filenames above).

### 2. Create the virtual environment and install packages

From the project root:

```bash
uv sync
```

This creates `.venv` and installs locked dependencies from `uv.lock` (PyTorch, pandas, matplotlib, ipykernel, etc.).

### 3. Run notebooks

Notebooks import modules from `src/` (`datasets`, `models`, `utils`). Set **`PYTHONPATH`** to the `src` folder before starting Jupyter or selecting a kernel in your editor.

### Option A — Cursor / VS Code (recommended)

1. Open the project folder.
2. Run **`uv sync`** if you have not already.
3. Open `notebooks/baseline.ipynb`.
4. Choose the Python interpreter: **`.venv`** inside the project root.
5. In your workspace or user settings, point imports at `src` (example for `.vscode/settings.json`):

   ```json
   {
     "terminal.integrated.env.linux": {
       "PYTHONPATH": "${workspaceFolder}/src"
     },
     "terminal.integrated.env.windows": {
       "PYTHONPATH": "${workspaceFolder}/src"
     }
   }
   ```

   Alternatively, set `PYTHONPATH` in the integrated terminal before running cells (see below).

6. Run all cells in `baseline.ipynb` to train for 5 epochs and save `results/best_model.pth`.

### Option B — Jupyter from the terminal

Install Jupyter for this session (it is not a core project dependency):

```bash
uv run --with jupyter jupyter notebook
```

Set `PYTHONPATH` in the same terminal session **before** launching Jupyter.

**Linux / macOS / Git Bash:**

```bash
export PYTHONPATH="${PWD}/src"
uv run --with jupyter jupyter notebook notebooks/baseline.ipynb
```

**Windows PowerShell:**

```powershell
$env:PYTHONPATH = "$PWD\src"
uv run --with jupyter jupyter notebook notebooks/baseline.ipynb
```

**Windows CMD:**

```cmd
set PYTHONPATH=%CD%\src
uv run --with jupyter jupyter notebook notebooks/baseline.ipynb
```

Open `http://localhost:8888` in your browser if Jupyter does not open automatically.

### Register a Jupyter kernel (optional)

```bash
uv run python -m ipykernel install --user --name=handwritten-text-recognition --display-name="Handwritten Text Recognition"
```

Select that kernel in Jupyter or in your editor’s notebook UI.

## Common uv commands

| Task | Command |
|------|---------|
| Install / update env from lockfile | `uv sync` |
| Add a dependency | `uv add <package>` |
| Run a script with the project env | `uv run python script.py` |
| Run any tool in the venv | `uv run <command>` |
| Upgrade locked packages | `uv lock --upgrade` then `uv sync` |

## Training overview

The baseline notebook (`notebooks/baseline.ipynb`):

1. Loads `emnist-balanced-train.csv` via `EMNISTDataset`
2. Builds a `DataLoader` (batch size 64, shuffled)
3. Trains `SimpleCNN` for 5 epochs with cross-entropy loss
4. Saves the best checkpoint to `results/best_model.pth`

The model outputs **47 logits** per image (28×28 grayscale). Use `get_char(predicted_class)` to turn a class index into a character.

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| `ModuleNotFoundError: No module named 'datasets'` | Set `PYTHONPATH` to the project’s `src` directory (see above). |
| CSV files missing or tiny pointer files | Run `git lfs pull`, or download EMNIST Balanced CSVs manually. |
| `uv: command not found` | Reinstall uv and restart the terminal; confirm `uv` is on `PATH`. |
| Python version error on `uv sync` | Install Python 3.14+; uv can install one with `uv python install 3.14`. |
| Slow training | Install a CUDA build of PyTorch for your platform, or reduce batch size / epochs in the notebook. |
