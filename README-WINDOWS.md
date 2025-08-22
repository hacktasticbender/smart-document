# Windows Fix - Smart Document QA

This version pins dependencies to Windows-friendly versions.

## Install Steps (Windows PowerShell)

```powershell
# 1. Create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip tools
python -m pip install --upgrade pip setuptools wheel

# 3. Install requirements
pip install -r requirements.txt

# 4. Run app
streamlit run app.py
# or
python -m streamlit run app.py
```

If any error occurs with faiss, uninstall and reinstall pinned version:

```powershell
pip uninstall -y faiss-cpu
pip install faiss-cpu==1.7.4
```
