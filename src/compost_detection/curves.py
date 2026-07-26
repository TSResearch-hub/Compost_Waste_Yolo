"""Courbes d'entraînement : lecture, recousage et tracé des results.csv.

Un run Ultralytics écrit un results.csv (une ligne par epoch : loss et
métriques). Complications gérées ici :

- les en-têtes varient selon les versions (espaces de bourrage) ;
- les composantes de loss dépendent de l'architecture (YOLO : box/cls/dfl,
  RT-DETR : giou/cls/l1) — elles sont détectées, jamais codées en dur ;
- un run repris après crash (--resume sur une VM Colab neuve) produit un csv
  qui repart de l'epoch de reprise : les morceaux sont recousus (stitch) et
  la sauvegarde n'écrase jamais l'historique (copy_results_preserving_history).

Les figures (métriques, loss, comparaison de runs) sont construites ici pour
être réutilisables : CLI scripts/plot_curves.py aujourd'hui, app Streamlit
demain.
"""

import csv
import shutil
from pathlib import Path

import matplotlib

# Rendu fichier uniquement : Jupyter/Streamlit exportent MPLBACKEND=...inline,
# inutilisable hors notebook (piège connu du repo).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colonnes de métriques, identiques YOLO / RT-DETR, avec libellé de figure.
METRIC_LABELS = {
    "metrics/precision(B)": "Précision",
    "metrics/recall(B)": "Rappel",
    "metrics/mAP50(B)": "mAP@50",
    "metrics/mAP50-95(B)": "mAP@50-95",
}

# Palette catégorielle en ordre FIXE (validée daltonisme) et encres du thème.
SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                 "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
TRAIN_COLOR = "#86b6ef"   # même teinte que la série 1, plus claire : courbe train
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def read_results_csv(path):
    """Lit un results.csv -> liste de dicts {colonne: float} (en-têtes nettoyés)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = {}
            for key, value in raw.items():
                if key is None or value is None:
                    continue
                try:
                    row[key.strip()] = float(value)
                except ValueError:
                    continue
            if "epoch" in row:
                rows.append(row)
    return rows


def first_epoch(path):
    """Epoch de la première ligne de données d'un results.csv (None si illisible)."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                for key, value in raw.items():
                    if key is not None and key.strip() == "epoch":
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return None
                return None
    except OSError:
        return None
    return None


def find_results_csvs(run_dir):
    """Les results*.csv d'un dossier de run, du plus ancien au plus récent."""
    return sorted(Path(run_dir).glob("results*.csv"), key=lambda p: p.stat().st_mtime)


def stitch(segments):
    """Recoud les morceaux d'un même run (cas des reprises après crash).

    ``segments`` : listes de lignes (read_results_csv), du plus ancien au plus
    récent — en cas d'epoch en double, le morceau le plus récent gagne.
    Retourne les lignes triées par epoch.
    """
    by_epoch = {}
    for rows in segments:
        for row in rows:
            by_epoch[int(row["epoch"])] = row
    return [by_epoch[e] for e in sorted(by_epoch)]


def load_run(path):
    """Charge un run : dossier (results*.csv recousus) ou chemin d'un csv."""
    path = Path(path)
    csvs = [path] if path.is_file() else find_results_csvs(path)
    if not csvs:
        raise FileNotFoundError(f"aucun results*.csv dans {path}")
    return stitch([read_results_csv(c) for c in csvs])


def loss_columns(rows):
    """Paires (train, val) des composantes de loss présentes dans les lignes.

    Détectées dynamiquement : YOLO logge box/cls/dfl_loss, RT-DETR giou/cls/l1_loss.
    """
    cols = set()
    for row in rows:
        cols.update(row)
    pairs = []
    for train in sorted(c for c in cols if c.startswith("train/") and c.endswith("_loss")):
        val = "val/" + train.split("/", 1)[1]
        pairs.append((train, val if val in cols else None))
    return pairs


def column(rows, name):
    """(epochs, valeurs) d'une colonne, en sautant les lignes où elle manque."""
    xs, ys = [], []
    for row in rows:
        if name in row:
            xs.append(row["epoch"])
            ys.append(row[name])
    return xs, ys


def copy_results_preserving_history(save_dir, dst):
    """Copie les results.* d'un run vers dst sans écraser l'historique.

    Après un --resume sur VM neuve, le results.csv local repart de l'epoch de
    reprise : si le csv déjà sauvegardé commence plus tôt, il est archivé
    (results_avant_epoch_N.csv) au lieu d'être écrasé. stitch() recoud le tout.
    """
    for extra in Path(save_dir).glob("results.*"):
        target = Path(dst) / extra.name
        if extra.name == "results.csv" and target.exists():
            old, new = first_epoch(target), first_epoch(extra)
            if old is not None and new is not None and new > old:
                target.rename(target.with_name(f"results_avant_epoch_{int(new)}.csv"))
        shutil.copy2(extra, target)


# ---------------------------------------------------------------- figures ---

def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_SOFT, fontsize=11, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def _label_end(ax, xs, ys, offset_pts=0):
    """Étiquette la valeur finale d'une courbe (en encre, pas en couleur de série)."""
    if xs:
        ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]),
                    xytext=(5, offset_pts), textcoords="offset points",
                    fontsize=8.5, color=INK, va="center", annotation_clip=False)


def _new_figure(n_rows, n_cols, title):
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.6 * n_cols, 3.3 * n_rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, color=INK, fontsize=13, fontweight="bold")
    return fig, axes


def make_metrics_figure(rows, label):
    """Figure 2x2 : précision, rappel, mAP@50, mAP@50-95 au fil des epochs."""
    fig, axes = _new_figure(2, 2, f"{label} — métriques de validation par epoch")
    for ax, (col, name) in zip(axes.flat, METRIC_LABELS.items()):
        xs, ys = column(rows, col)
        ax.plot(xs, ys, color=SERIES_COLORS[0], linewidth=2)
        _label_end(ax, xs, ys)
        ax.set_ylim(0, 1.02)
        _style(ax, name)
    for ax in axes[-1]:
        ax.set_xlabel("Époque", color=INK_MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def make_losses_figure(rows, label):
    """Une case par composante de loss : train (clair, tireté) vs val (plein).

    Lecture : un écart qui se creuse — val qui remonte pendant que train
    continue de descendre — est la signature du sur-apprentissage.
    """
    pairs = loss_columns(rows)
    fig, axes = _new_figure(1, max(len(pairs), 1), f"{label} — loss par epoch")
    for ax, (train, val) in zip(axes.flat, pairs):
        xs, ys = column(rows, train)
        ax.plot(xs, ys, color=TRAIN_COLOR, linewidth=2, linestyle="--", label="train")
        if val:
            xv, yv = column(rows, val)
            ax.plot(xv, yv, color=SERIES_COLORS[0], linewidth=2, label="val")
            _label_end(ax, xv, yv)
        ax.set_ylim(bottom=0)
        _style(ax, train.split("/", 1)[1])
        ax.set_xlabel("Époque", color=INK_MUTED, fontsize=9)
    axes.flat[0].legend(frameon=False, fontsize=9, labelcolor=INK_SOFT)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def make_compare_figure(runs):
    """Superpose les métriques de plusieurs runs (dict {libellé: lignes})."""
    if len(runs) > len(SERIES_COLORS):
        raise ValueError(f"{len(SERIES_COLORS)} runs maximum par comparaison")
    fig, axes = _new_figure(2, 2, "Comparaison des runs — métriques de validation")
    for ax, (col, name) in zip(axes.flat, METRIC_LABELS.items()):
        series = []
        for color, (label, rows) in zip(SERIES_COLORS, runs.items()):
            xs, ys = column(rows, col)
            ax.plot(xs, ys, color=color, linewidth=2, label=label)
            if xs:
                series.append((xs, ys))
        # étiquettes de fin, décalées verticalement si les courbes finissent proches
        finals = sorted(s[1][-1] for s in series)
        crowded = any(b - a < 0.05 for a, b in zip(finals, finals[1:]))
        for rank, (xs, ys) in enumerate(sorted(series, key=lambda s: s[1][-1])):
            offset = (rank - (len(series) - 1) / 2) * 11 if crowded else 0
            _label_end(ax, xs, ys, offset)
        ax.set_ylim(0, 1.02)
        _style(ax, name)
    for ax in axes[-1]:
        ax.set_xlabel("Époque", color=INK_MUTED, fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    # légende centrée SOUS le titre (en haut à droite elle le chevaucherait)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               frameon=False, fontsize=9, labelcolor=INK_SOFT, ncol=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig
