"""Builds Electricity_Tariff_Band_Forecasting.ipynb from cells.py (+ interpretations.py if present).

Run:  python notebooks/build_notebook.py
Then: jupyter nbconvert --to notebook --execute --inplace notebooks/Electricity_Tariff_Band_Forecasting.ipynb
"""
import re
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
cells = []
INTERP = {}
if (HERE / "interpretations.py").exists():
    exec((HERE / "interpretations.py").read_text())  # defines INTERP dict


def md(s):
    s = s.strip()
    s = re.sub(r"<<INTERP:(\w+)>>", lambda m: INTERP.get(m.group(1), f"*(interpretation '{m.group(1)}' pending)*").strip(), s)
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


exec((HERE / "cells.py").read_text())

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, HERE / "Electricity_Tariff_Band_Forecasting.ipynb")
print(f"wrote {len(cells)} cells")
