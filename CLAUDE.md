# FERAL — project instructions for Claude

## Python environment

The right Python environment depends on **which machine you're running on**. Check `uname` at the start of every session:

- **On macOS (`uname` returns `Darwin`) — Peter's local Mac:** always use the `feral` conda environment. Never invoke `python` from the base shell, never create or use a venv, never `pip install` into base.
- **On Linux (e.g., a runpod machine):** use whatever Python is already on the system. There is no `feral` conda env on remote machines — those are throwaway training boxes with their own pre-installed environment. Just call `python` directly.

### macOS (local Mac) workflow

The env was created with:

```
conda create -n feral python=3.11
```

To run a Python command on the Mac, prefix it with `conda run -n feral`:

```
conda run -n feral python tests/generate_synthetic_dataset.py
conda run -n feral python -m unittest tests/tests.py
conda run -n feral pip install <package>
```

If you need to install a new dependency *for local Mac development*, install it into the `feral` env (not base) and then add it to `requirements.txt`.

### Notes on the macOS environment

- **`decord` is installed via `eva-decord`** in the Mac env, because the community fork is the only one that ships Apple Silicon wheels. The upstream `decord` package on PyPI has no arm64 build. Both expose the same `decord` module name, so imports work transparently. **On Linux/runpod, use the real `decord` package** — it has Linux wheels and is what the README assumes.
- `torch` has a minimum floor of `>=2.4` in `requirements.txt` (required by `transformers` 5.x) but is intentionally **not pinned to an exact version** — cloud/HPC environments usually ship a working torch+CUDA, and a hard pin fights that. The README documents the tested torch/CUDA combo.
- The rest of `requirements.txt` is pinned with `==` and chosen to keep the project compatible with **Python 3.10+**. Notably, `pandas` is held at `2.3.x` and `scikit-learn` at `1.7.x` because their next majors (`pandas` 3.0, `scikit-learn` 1.8) require Python ≥ 3.11. If you bump either of those, double-check the new floor.


Do not regenerate fixtures. they are already ok
Use pytest
use twine to publish to pypi
Dont do the following without explicit approval from me:
- git commit
- git push
- publish to pypi
- update python project information
- Readme