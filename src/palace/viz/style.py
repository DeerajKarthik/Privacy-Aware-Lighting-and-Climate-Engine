"""IEEE figure style: 3.5 in single / 7.16 in double column, 300 dpi, serif font, 8 pt, colour-blind-safe palette."""
import matplotlib as mpl
COL1, COL2 = 3.5, 7.16
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
METHOD_COLORS = {"B0": "#000000", "B1": "#E69F00", "B2": "#0072B2", "B3": "#56B4E9", "B4": "#009E73", "B5": "#CC79A7", "B6": "#999999", "PALACE": "#D55E00"}
def apply():
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": 100, "savefig.dpi": 300, "axes.prop_cycle": mpl.cycler(color=OKABE_ITO), "axes.linewidth": 0.6, "pdf.fonttype": 42})
def save(fig, path):
    from pathlib import Path
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(p.with_suffix(".png"), bbox_inches="tight")
