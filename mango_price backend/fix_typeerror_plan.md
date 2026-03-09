# Plan to Fix TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'

## 1. Understand the Root Cause
The error `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` occurs on the following line in `prediction_service.py`: 
`def _predict_single_model(...) -> np.ndarray | None:`

This error is happening because the bitwise OR operator `|` is being used for type hinting (Union types, meaning "either `np.ndarray` OR `None`"). This syntax was introduced in **Python 3.10** (PEP 604). The virtual environment `(.venv)` in which you are running the API is using a Python version older than 3.10 (e.g., Python 3.8 or 3.9), which does not understand this syntax.

## 2. Choose a Fix Strategy
To fix this, you have two possible paths:
*   **Strategy A (Recommended):** Update the code to use the standard `typing` module, which is backwards-compatible with older Python versions.
*   **Strategy B:** Upgrade the Python version of your virtual environment to Python 3.10 or newer.

---

## Step-by-Step Execution Plan

### If following Strategy A: Modify the Code for Compatibility
This is the fastest and most robust approach if you don't want to change your system's Python environment.

**Step 1: Locate the target file**
*   Open `/Users/gimhanrajapaksha/PycharmProjects/MangoWebSocket/mango/mango_api/app/services/prediction_service.py` in your IDE.

**Step 2: Update the Imports**
*   At the top of the file, find the `typing` import.
*   Change it to import `Optional` and `Dict` as well: `from typing import Any, Optional, Dict`.

**Step 3: Fix the Union Type (`|`)**
*   Go to the `_predict_single_model` function definition (around line 9).
*   Change the return type hint from `-> np.ndarray | None:` to `-> Optional[np.ndarray]:`.

**Step 4: Fix Generic Collection Types**
*   If your Python version is < 3.9, the lowercase `dict[str, Any]` type hints will also throw an error.
*   Scan the file for `dict[...]` annotations (e.g., `def predict(...) -> dict[str, Any]:`).
*   Replace them with the capitalized `Dict[...]` from the `typing` module.

**Step 5: Check other files**
*   Perform a quick search across the `/app` directory for any other uses of the `|` symbol in type hints and apply the same `Optional` (or `Union`) fix to them.

---

### If following Strategy B: Upgrade Python Environment
If you prefer to keep the modern Python 3.10+ syntax, you need to upgrade the runtime environment.

**Step 1: Verify current Python version**
*   Run `python --version` inside your `(.venv)` to confirm it is `< 3.10`.

**Step 2: Install newer Python**
*   Install Python 3.10, 3.11, or 3.12 on your macOS (e.g., using Homebrew: `brew install python@3.11`).

**Step 3: Recreate the Virtual Environment**
*   Deactivate the current environment: `deactivate`
*   Delete the existing virtual environment folder: `rm -rf .venv`
*   Create a new one using the specific newer Python version: `python3.11 -m venv .venv`
*   Activate the new environment: `source .venv/bin/activate`

**Step 4: Reinstall Dependencies**
*   Install your project requirements again: `pip install -r requirements.txt`

---

## 3. Verification
Once either strategy is complete, you should verify the fix works:
*   Run the command again: `MODEL_DIR=output/models python run.py`
*   Ensure that no `TypeError` gets thrown and the application starts properly.
