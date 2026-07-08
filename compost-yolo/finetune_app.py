"""Interface de fine-tuning — pour réentraîner le modèle sans ligne de commande.

Habillage Streamlit des scripts du repo (aucune logique métier ici) :
onglet Dataset -> update_dataset.py, onglet Réentraîner -> retrain.py,
onglet Évaluer -> evaluate.py, onglet Résultats -> lecture de runs/.

Lancement :
    cd compost-yolo && source venv/bin/activate
    streamlit run finetune_app.py
"""

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# ligne d'epoch d'Ultralytics : «   12/50   2.35G   1.234  ... » (n/total puis mémoire)
EPOCH_LINE = re.compile(r"^\s*(\d+)/(\d+)\s+[\d.]+G?\s")

st.set_page_config(page_title="Compost — fine-tuning", layout="wide")
st.title("Fine-tuning du détecteur d'intrus")


# ---------------------------------------------------------------- utilitaires
def stream_command(cmd, log_box, progress_bar=None):
    """Lance une commande, affiche sa sortie en direct ; met à jour la barre de
    progression sur les lignes d'epoch d'Ultralytics. Retourne (code, sortie)."""
    proc = subprocess.Popen([PYTHON] + cmd, cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, "MPLBACKEND": "Agg"})
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        log_box.code("\n".join(lines[-30:]), language=None)
        if progress_bar is not None and (m := EPOCH_LINE.match(line)):
            cur, total = int(m.group(1)), int(m.group(2))
            if total > 1:  # ignore les compteurs de batchs d'éval (1/3...)
                progress_bar.progress(min(cur / total, 1.0),
                                      text=f"Fine-tuning : epoch {cur}/{total}")
    proc.wait()
    return proc.returncode, "\n".join(lines)


def class_names():
    return yaml.safe_load(open(ROOT / "configs/data.yaml", encoding="utf-8"))["names"]


def snapshots():
    root = ROOT / "data/captures"
    return sorted((d for d in root.glob("v*") if d.is_dir()), reverse=True) if root.is_dir() else []


def latest_snapshot():
    latest = ROOT / "data/captures/latest"
    if latest.exists():
        return latest.resolve()
    snaps = snapshots()
    return snaps[0] if snaps else (ROOT / "data/raw/captures"
                                   if (ROOT / "data/raw/captures").is_dir() else None)


def dataset_stats(dataset):
    """(instances par classe, nb images, nb négatives, images par session)."""
    names = class_names()
    per_class, n_img, n_neg, per_session = Counter(), 0, 0, Counter()
    groups = {}
    gcsv = dataset / "groups.csv"
    if gcsv.exists():
        import csv as _csv
        with open(gcsv, encoding="utf-8") as f:
            groups = {r["stem"]: r["group_id"] for r in _csv.DictReader(f)}
    for img in (dataset / "images").iterdir():
        if img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        n_img += 1
        per_session[groups.get(img.stem, "?")] += 1
        lbl = dataset / "labels" / f"{img.stem}.txt"
        lines = [l for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
        if not lines:
            n_neg += 1
        for l in lines:
            per_class[names[int(l.split()[0])]] += 1
    return per_class, n_img, n_neg, per_session


def eval_runs(prefix="eval_"):
    return sorted((d for d in (ROOT / "runs").glob(f"{prefix}*") if d.is_dir()),
                  key=lambda d: d.stat().st_mtime, reverse=True)


def show_eval(run_dir):
    """Affiche les métriques + visuels d'un dossier runs/eval_*."""
    metrics = run_dir / "image_level_metrics.json"
    if metrics.exists():
        m = json.load(open(metrics, encoding="utf-8"))
        il = m["image_level"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Rappel image (intrus signalés)", f"{il['recall']:.2f}" if il["recall"] is not None else "—")
        c2.metric("Précision image", f"{il['precision']:.2f}" if il["precision"] is not None else "—")
        neg = m.get("negatives", {})
        c3.metric("Fausses alertes / image sans intrus",
                  f"{neg['mean_intruder_detections']:.2f}" if neg.get("mean_intruder_detections") is not None else "—")
        if "weights" in m:
            st.caption(f"Modèle évalué : `{m['weights']}`")
    for png, legend in [("per_class_metrics.png", "Métriques par classe (niveau boîte)"),
                        ("confusion_matrices.png", "Confusion par classe (niveau image)"),
                        ("ultralytics_val/confusion_matrix_normalized.png",
                         "Confusion niveau instance (7 classes + background)")]:
        f = run_dir / png
        if f.exists():
            st.image(str(f), caption=legend)


# ------------------------------------------------------------------- onglets
tab_data, tab_train, tab_eval, tab_results = st.tabs(
    ["Dataset", "Réentraîner", "Évaluer", "Résultats"])

# ================================================================= DATASET ==
with tab_data:
    st.subheader("Mettre à jour le dataset")
    st.markdown(
        "Chaque mise à jour crée un **nouveau snapshot figé** (`data/captures/vNNN_date`) "
        "à partir du précédent + les annotations des sources. Les anciens snapshots ne "
        "bougent jamais : chaque entraînement reste reproductible.")
    default_src = str((ROOT.parent / "dataset_recolte").resolve())
    sources_text = st.text_area(
        "Sources d'annotations (une par ligne — un dossier par poste d'annotation)",
        value=default_src, height=80)
    if st.button("Créer un nouveau snapshot", type="primary"):
        sources = [s.strip() for s in sources_text.splitlines() if s.strip()]
        log = st.empty()
        code, _ = stream_command(["scripts/update_dataset.py", "--source"] + sources, log)
        st.success("Snapshot créé.") if code == 0 else st.error("Échec — voir le journal.")

    st.divider()
    st.subheader("Contenu du dataset")
    snaps = snapshots()
    options = snaps or ([latest_snapshot()] if latest_snapshot() else [])
    if not options:
        st.info("Aucun dataset : crée un premier snapshot ci-dessus.")
    else:
        chosen = st.selectbox("Snapshot", options, format_func=lambda d: d.name
                              + (" (latest)" if d == latest_snapshot() else ""))
        per_class, n_img, n_neg, per_session = dataset_stats(chosen)
        c1, c2, c3 = st.columns(3)
        c1.metric("Images", n_img)
        c2.metric("Sans intrus (négatives)", n_neg)
        c3.metric("Instances annotées", sum(per_class.values()))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Instances par classe**")
            st.bar_chart({"instances": {n: per_class.get(n, 0) for n in class_names()}})
        with col2:
            st.markdown("**Images par session**")
            st.bar_chart({"images": dict(per_session)})
        weak = [n for n in class_names() if per_class.get(n, 0) < 30 and n != "Carton"]
        if weak:
            st.warning("Classes sous-représentées (< 30 instances) : **"
                       + ", ".join(weak) + "** — à prioriser à la prochaine collecte.")

# ============================================================= RÉENTRAÎNER ==
with tab_train:
    st.subheader("Réentraîner le modèle (fine-tuning)")
    st.markdown(
        "Enchaîne automatiquement : split (test compost préservé) → **éval avant** "
        "(pré-entraîné) → **fine-tuning** → **éval après** (même test).")
    pretrains = sorted((ROOT / "models").glob("*.pt"))
    if not pretrains:
        st.error("Aucun modèle pré-entraîné dans `models/` — déposer un "
                 "`pretrain_*.pt` (voir README).")
    else:
        pretrain = st.selectbox("Modèle pré-entraîné de départ", pretrains,
                                format_func=lambda p: p.name)
        st.caption(
            "Ce même fichier sert à l'éval « avant » ET de point de départ du "
            "fine-tuning (garanti par retrain.py). L'architecture (YOLO / RT-DETR) "
            "est contenue dans le .pt : le fine-tuning produit forcément le même modèle. "
            "On ne repart jamais d'un fine-tuné précédent.")
        c1, c2, c3, c4 = st.columns(4)
        epochs = c1.number_input("Epochs", 1, 300, 50)
        batch = c2.number_input("Batch", 1, 64, 8,
                                help="baisser à 4 si mémoire GPU insuffisante")
        lr0 = c3.number_input("Learning rate initial", min_value=0.00001, max_value=0.1,
                              value=0.001, step=0.0005, format="%.4f",
                              help="bas (0.001) = départ doux, ne casse pas le pré-entraînement ; "
                                   "le défaut Ultralytics (~0.01) est trop agressif pour un fine-tuning")
        deploy = c4.checkbox("Déployer à la fin",
                             help="copie le best.pt final vers ../weights/best.pt (interface d'annotation)")
        snap = latest_snapshot()
        st.caption(f"Dataset utilisé : `{snap.name if snap else '—'}` (le dernier snapshot)")
        if st.button("Lancer le réentraînement", type="primary"):
            st.info("Compter plusieurs minutes — suivre la barre et le journal ci-dessous.")
            cmd = ["scripts/retrain.py", "--pretrain", str(pretrain),
                   "--epochs", str(int(epochs)), "--batch", str(int(batch)),
                   "--lr0", str(lr0)]
            if deploy:
                cmd.append("--deploy")
            bar = st.progress(0.0, text="Préparation (split + éval avant)...")
            log = st.empty()
            code, out = stream_command(cmd, log, progress_bar=bar)
            if code == 0:
                bar.progress(1.0, text="Terminé")
                st.success("Réentraînement terminé — comparaison avant/après dans le "
                           "journal ci-dessus, détails dans l'onglet **Résultats**.")
            else:
                st.error("Échec — voir le journal.")

# ================================================================= ÉVALUER ==
with tab_eval:
    st.subheader("Évaluer un modèle")
    weights = sorted((ROOT / "models").glob("*.pt"))
    weights += sorted((ROOT / "runs").glob("*_*/weights/best.pt"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if not weights:
        st.info("Aucun modèle disponible (models/ vide et aucun run).")
    else:
        w = st.selectbox("Modèle", weights,
                         format_func=lambda p: p.parent.parent.name + "/best.pt"
                         if p.parent.name == "weights" else f"models/{p.name}")
        test_yaml = ROOT / "data/finetune/captures_test/data.yaml"
        choice = st.radio("Données d'évaluation", [
            "Test compost (mis de côté au dernier split)",
            "Autre data.yaml (ex. nouvelle session, AVANT de l'intégrer au dataset)"])
        data_path = str(test_yaml)
        if choice.startswith("Autre"):
            data_path = st.text_input("Chemin du data.yaml", value=str(test_yaml))
        if st.button("Évaluer", type="primary"):
            if not Path(data_path).exists():
                st.error(f"Introuvable : {data_path} — lance d'abord un réentraînement "
                         "(le split crée le test compost).")
            else:
                log = st.empty()
                code, _ = stream_command(["scripts/evaluate.py", "--weights", str(w),
                                          "--data", data_path, "--split", "test"], log)
                if code == 0:
                    st.success("Évaluation terminée.")
                    show_eval(eval_runs()[0])
                else:
                    st.error("Échec — voir le journal.")

# ================================================================ RÉSULTATS ==
with tab_results:
    st.subheader("Résultats des évaluations")
    evals = eval_runs()
    if not evals:
        st.info("Aucune évaluation pour l'instant.")
    else:
        before = next((e for e in evals if e.name.startswith("eval_pretrain")), None)
        after = next((e for e in evals if e.name.startswith("eval_finetune")), None)
        if before and after:
            st.markdown("#### Avant / après fine-tuning (dernières évals)")
            rows = {}
            for tag, d in (("avant (pré-entraîné)", before), ("après (fine-tuné)", after)):
                m = json.load(open(d / "image_level_metrics.json", encoding="utf-8"))
                il = m["image_level"]
                rows[tag] = {"rappel image": round(il["recall"], 3) if il["recall"] is not None else None,
                             "précision image": round(il["precision"], 3) if il["precision"] is not None else None}
            st.table(rows)
        chosen = st.selectbox("Détail d'une évaluation", evals, format_func=lambda d: d.name)
        show_eval(chosen)
