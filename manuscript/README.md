# Manuscript build

Every number in the text and every table is computed from `../results` and `../analysis`; nothing is
typed by hand.

```
cd manuscript
python derive_v2.py ../results derived                        # early stopping, selected checkpoint, gains
python make_figures_v2.py ../results ../analysis derived figures
python build_v2.py Access-Template.docx figures FedSAA_IEEE_Access_Manuscript.docx
```

- `derive_v2.py` - validation-based early stopping (`es.csv`), planned contrasts with Holm correction
  (`es_pairs.csv`), validation-selected checkpoint metrics (`cpsel.pkl`) and specialty/client transfer
  gains (`spec_gain_sel.csv`, `client_gain_sel.csv`). The outputs are also committed in `derived/`.
- `make_figures_v2.py` - the five figures (PNG at 600 dpi and PDF), also committed in `figures/`.
- `content_v2.py` - the manuscript text and tables, computed from the files above
  (`content.py`, `content_long.py` and `refs.py` hold reused text and the reference list).
- `build_v2.py` (with helpers in `build_ieee.py`) - fills the IEEE Access Word template. The template is
  not redistributed here; download it from the IEEE Access author resources and pass its path.

Requirements: Python 3, NumPy, pandas, SciPy, matplotlib, python-docx, lxml and pandoc.
Paths can be overridden with `FEDSAA_RESULTS`, `FEDSAA_ANALYSIS` and `FEDSAA_DERIVED`.
