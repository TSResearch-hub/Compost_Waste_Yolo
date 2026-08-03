"""Interface de fine-tuning — pour réentraîner le modèle sans ligne de commande.

Habillage Streamlit des scripts du repo (aucune logique métier ici) :
onglet Production -> boucle caméra + infer.py (alertes de alert_rules.yaml),
onglet Dataset -> update_dataset.py, onglet Réentraîner -> retrain.py,
onglet Évaluer -> evaluate.py, onglet Résultats -> lecture de runs/.

Lancement (depuis la racine du dépôt) :
    source venv/bin/activate
    streamlit run interfaces/production/finetune_app.py

Sur la Jetson (interface servie sur le réseau local, voir docs/README_jetson.md) :
    streamlit run interfaces/production/finetune_app.py --server.address 0.0.0.0 --server.port 8502
"""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml

from compost_detection.alert import alerting_classes, load_alert_config
from compost_detection.naming import create_run_dir

# Le code de l'interface vit dans interfaces/production/ ; tout ce qu'elle
# manipule (scripts/, runs/, models/, configs/) vit à la racine du dépôt.
ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# ligne d'epoch d'Ultralytics : «   12/50   2.35G   1.234  ... » (n/total puis mémoire)
EPOCH_LINE = re.compile(r"^\s*(\d+)/(\d+)\s+[\d.]+G?\s")
# codes ANSI (couleurs, effacement de ligne) émis par Ultralytics dans ses journaux
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

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
        # barre de progression : ne garder que l'état final (après le dernier \r)
        line = ANSI_RE.sub("", line.rstrip("\n").split("\r")[-1])
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
        with open(gcsv, encoding="utf-8") as f:
            groups = {r["stem"]: r["group_id"] for r in csv.DictReader(f)}
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
        try:
            m = json.load(open(metrics, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            st.warning(f"Métriques illisibles dans {run_dir.name} (évaluation "
                       "interrompue ? disque plein ?) — dossier à supprimer.")
            return
        il = m["image_level"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Rappel image (intrus signalés)", f"{il['recall']:.2f}" if il["recall"] is not None else "—")
        c2.metric("Précision image", f"{il['precision']:.2f}" if il["precision"] is not None else "—")
        neg = m.get("negatives", {})
        c3.metric("Fausses alertes / image sans intrus",
                  f"{neg['mean_intruder_detections']:.2f}" if neg.get("mean_intruder_detections") is not None else "—")
        if "weights" in m:
            src = f"Modèle évalué : `{m['weights']}`"
            if m.get("data"):
                src += f" — données : `{m['data']}`"
            st.caption(src)
    c1, c2 = st.columns(2)
    for col, png, legend in [(c1, "per_class_metrics.png", "Métriques par classe (niveau boîte)"),
                             (c2, "confusion_matrices.png", "Confusion par classe (niveau image)")]:
        f = run_dir / png
        if f.exists():
            col.image(str(f), caption=legend)
    f = run_dir / "ultralytics_val/confusion_matrix_normalized.png"
    if f.exists():
        with st.expander(f"Confusion niveau instance ({len(class_names())} classes + background)"):
            st.image(str(f))


@st.cache_resource(show_spinner="Chargement du modèle...")
def load_model(weights_path):
    """Charge un modèle une seule fois par session (l'import ultralytics est
    lourd, on ne le paie qu'au premier usage de l'onglet Production)."""
    from ultralytics import YOLO
    # .engine = TensorRT (Jetson) : le fichier ne porte pas la tâche, il faut
    # la préciser au chargement ; l'architecture (YOLO / RT-DETR) est figée
    # dans le moteur, la classe YOLO sert de chargeur pour les deux
    if str(weights_path).endswith(".engine"):
        return YOLO(weights_path, task="detect")
    return YOLO(weights_path)


# ------------------------------------------------------------------- onglets
_snap = latest_snapshot()
_deployed = ROOT / "weights" / "best.pt"
st.caption(
    "Dataset courant : " + (f"`{_snap.name}`" if _snap else "aucun")
    + " — modèle déployé : "
    + (f"`weights/best.pt` du {datetime.fromtimestamp(_deployed.stat().st_mtime):%d/%m %H:%M}"
       if _deployed.exists() else "aucun"))

tab_prod, tab_data, tab_train, tab_eval, tab_results = st.tabs(
    ["Production", "Dataset", "Réentraîner", "Évaluer", "Résultats"])

# ================================================================= DATASET ==
with tab_data:
    st.subheader("Mettre à jour le dataset")
    st.caption("Chaque mise à jour crée un snapshot figé (`data/captures/vNNN_date`) — "
               "les anciens ne bougent jamais, chaque entraînement reste reproductible.")
    default_src = str((ROOT / "dataset_recolte").resolve())
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
    st.caption("Split → éval avant → fine-tuning → éval après (même test compost).")
    # v0_* uniquement : models/ contient aussi les fine-tunés v1/v2 (pour
    # l'éval et le déploiement), qui ne doivent jamais servir de départ.
    pretrains = sorted((ROOT / "models").glob("v0_*.pt"))
    if not pretrains:
        st.error("Aucun modèle pré-entraîné dans `models/` — déposer un "
                 "`v0_pretrain_*.pt` (voir models/README.md).")
    else:
        pretrain = st.selectbox(
            "Modèle pré-entraîné de départ", pretrains, format_func=lambda p: p.name,
            help="Seuls les pré-entraînés canoniques de `models/` sont proposés : on ne "
                 "repart jamais d'un fine-tuné précédent (le split change quand le dataset "
                 "grandit, le test aurait déjà été appris). Ce même fichier sert à l'éval "
                 "« avant » et de point de départ du fine-tuning ; l'architecture "
                 "(YOLO / RT-DETR) est contenue dans le .pt.")
        is_rtdetr = "rtdetr" in pretrain.name.lower()
        c1, c2, c3 = st.columns(3)
        epochs = c1.number_input("Epochs", 1, 300, 30,
                                 help="~25-30 suffisent sur nos volumes actuels (les courbes "
                                      "sur-apprennent au-delà) ; best.pt garde le pic de toute façon")
        batch = c2.number_input("Batch", 1, 64, 4 if is_rtdetr else 8,
                                help="RT-DETR : 4 max sur 8 Go de VRAM ; YOLO : 8 passe")
        deploy = c3.checkbox("Déployer à la fin",
                             help="copie le best.pt final vers weights/best.pt "
                                  "(le modèle des interfaces)")
        snap = latest_snapshot()
        st.caption(f"Dataset : `{snap.name if snap else '—'}` (dernier snapshot)"
                   + (" — RT-DETR local : ~15-30 min/epoch, préférer le terminal "
                      "pour un run long (`scripts/retrain.py`)." if is_rtdetr else ""))
        if st.button("Lancer le réentraînement", type="primary"):
            st.info("Progression ci-dessous — garder l'onglet ouvert jusqu'à la fin.")
            cmd = ["scripts/retrain.py", "--pretrain", str(pretrain),
                   "--epochs", str(int(epochs)), "--batch", str(int(batch))]
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
                try:
                    m = json.load(open(d / "image_level_metrics.json", encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                il = m["image_level"]
                neg = m.get("negatives", {})
                rows[tag] = {"rappel image": round(il["recall"], 3) if il["recall"] is not None else None,
                             "précision image": round(il["precision"], 3) if il["precision"] is not None else None,
                             "fausses alertes / img sans intrus":
                                 round(neg["mean_intruder_detections"], 2)
                                 if neg.get("mean_intruder_detections") is not None else None}
            if rows:
                st.table(rows)
                st.caption(f"Runs comparés : `{before.name}` (avant) vs `{after.name}` (après).")
        chosen = st.selectbox("Détail d'une évaluation", evals, format_func=lambda d: d.name)
        show_eval(chosen)

# ================================================================ PRODUCTION ==
# Bloc placé en DERNIER dans le code (mais 1er onglet affiché) : la boucle
# caméra bloque le script tant que la surveillance tourne, les autres onglets
# doivent donc avoir été rendus avant d'y entrer.
with tab_prod:
    st.subheader("Surveillance du compost")
    st.caption("Un intrus au-dessus du seuil de sa classe déclenche une alerte avec consignes "
               "(`configs/alert_rules.yaml` — calibration : `evaluate.py --sweep-thresholds`).")
    thresholds, instructions = load_alert_config(ROOT / "configs/alert_rules.yaml")

    deployed = ROOT / "weights" / "best.pt"
    # .engine en premier : c'est l'export TensorRT local (scripts/export.py
    # --formats engine, à lancer SUR la Jetson) — beaucoup plus rapide que le
    # .pt sur cette machine, mais valable uniquement là où il a été exporté
    prod_weights = sorted((ROOT / "weights").glob("*.engine"))
    prod_weights += [deployed] if deployed.exists() else []
    prod_weights += sorted((ROOT / "models").glob("*.pt"))
    prod_weights += sorted((ROOT / "runs").glob("*_*/weights/best.pt"),
                           key=lambda p: p.stat().st_mtime, reverse=True)

    def prod_label(p):
        if p.parent == ROOT / "weights":
            return (f"déployé (TensorRT) — weights/{p.name}" if p.suffix == ".engine"
                    else "déployé — weights/best.pt")
        if p.parent.name == "weights":
            return p.parent.parent.name + "/best.pt"
        return f"models/{p.name}"

    if not prod_weights:
        st.error("Aucun modèle disponible — déployer un best.pt (onglet Réentraîner, "
                 "case « Déployer ») ou déposer un .pt dans `models/`.")
    else:
        w = st.selectbox("Modèle", prod_weights, key="prod_weights",
                         format_func=prod_label)
        mode = st.radio("Source", ["Caméra en direct", "Tester un fichier (image ou vidéo)"],
                        horizontal=True)
        # réglage à la volée pour les deux modes ; le yaml reste la référence
        # calibrée (un changement pendant la surveillance relance la boucle)
        with st.expander("Seuils par classe (défauts : configs/alert_rules.yaml)"):
            cols = st.columns(2)
            thresholds = {
                name: cols[i % 2].slider(name, 0.05, 0.95, float(default), 0.05,
                                         key=f"prod_thr_{name}")
                for i, (name, default) in enumerate(sorted(thresholds.items()))
            }
        min_conf = min(thresholds.values())

        # ------------------------------------------------------ caméra en direct
        if mode == "Caméra en direct":
            st.session_state.setdefault("prod_running", False)
            st.session_state.setdefault("prod_alerts", [])
            st.session_state.setdefault("prod_dir", None)
            st.session_state.setdefault("last_raw", None)    # dernière frame sans boîtes
            st.session_state.setdefault("last_shown", None)  # la même, avec boîtes
            cam = st.text_input(
                "Caméra : indice (0, 1...) ou URL de flux (http/rtsp)", value="0",
                help="Jetson / Linux : la caméra USB est l'indice 0. Sous WSL la webcam "
                     "n'est en général pas visible : lancer l'app depuis Windows, ou "
                     "utiliser un téléphone en caméra IP et coller son URL.")
            c1, c2, c3 = st.columns(3)
            if c1.button("Démarrer la surveillance", type="primary"):
                st.session_state.prod_running = True
                st.session_state.prod_alerts = []     # nouvelle session = nouveau journal
                st.session_state.prod_dir = None
            if c2.button("Arrêter la surveillance"):
                st.session_state.prod_running = False
            manual_clicked = c3.button(
                "Capture manuelle (à annoter)",
                help="Enregistre l'image affichée même sans alerte — pour un intrus que "
                     "le modèle a raté. L'image brute part dans a_annoter/, la "
                     "surveillance reprend toute seule.")

            status_box = st.empty()
            frame_box = st.empty()
            info_box = st.empty()
            st.markdown("**Journal de la session** — une ligne par capture : alerte, capture "
                        "manuelle, ou image saine prise à chaque fin d'alerte. Brutes pour "
                        "l'annotation : `runs/production_*/a_annoter/` — avec boîtes, pour "
                        "vérifier : `frames/`")
            journal_box = st.empty()
            if st.session_state.prod_alerts:
                journal_box.dataframe(st.session_state.prod_alerts)

            past = sorted((d for d in (ROOT / "runs").glob("production_*") if d.is_dir()),
                          key=lambda d: d.stat().st_mtime, reverse=True)
            if past:
                with st.expander("Sessions de surveillance précédentes"):
                    sel = st.selectbox("Session", past, format_func=lambda d: d.name)
                    acsv = sel / "alerts.csv"
                    if acsv.exists():
                        with open(acsv, encoding="utf-8") as f:
                            st.dataframe(list(csv.DictReader(f)))
                        n_ann = len(list((sel / "a_annoter").glob("*.jpg")))
                        if n_ann:
                            st.caption(f"{n_ann} image(s) brute(s) à annoter dans "
                                       f"`{sel.name}/a_annoter/` — à donner comme source "
                                       "à l'interface d'annotation.")
                        for fr in sorted((sel / "frames").glob("*.jpg"))[:6]:
                            st.image(str(fr), caption=fr.name, width=320)
                    else:
                        st.caption("Aucune alerte enregistrée dans cette session.")

            def prod_save(kind, raw, shown=None, classes="", conf=None):
                """Toute capture passe ici : image brute dans a_annoter/ (à importer
                dans l'interface d'annotation), copie avec boîtes dans frames/ (pour
                vérifier à l'œil), ligne dans alerts.csv (le manifeste de la session)."""
                import cv2
                if st.session_state.prod_dir is None:
                    st.session_state.prod_dir = str(create_run_dir(ROOT / "runs",
                                                                   "production"))
                d = Path(st.session_state.prod_dir)
                (d / "frames").mkdir(exist_ok=True)
                (d / "a_annoter").mkdir(exist_ok=True)
                name = f"{datetime.now():%Hh%M-%Ss}_{kind}.jpg"
                cv2.imwrite(str(d / "a_annoter" / name), raw)
                if shown is not None:
                    cv2.imwrite(str(d / "frames" / name), shown)
                row = {"horodatage": f"{datetime.now():%d/%m %H:%M:%S}", "type": kind,
                       "classes": classes,
                       "confiance max": round(conf, 2) if conf is not None else "",
                       "frame": name}
                st.session_state.prod_alerts.append(row)
                new_csv = not (d / "alerts.csv").exists()
                with open(d / "alerts.csv", "a", newline="", encoding="utf-8") as f:
                    wr = csv.DictWriter(f, fieldnames=row.keys())
                    if new_csv:
                        wr.writeheader()
                    wr.writerow(row)
                journal_box.dataframe(st.session_state.prod_alerts)

            if manual_clicked:
                if st.session_state.last_raw is None:
                    st.warning("Aucune image reçue de la caméra — démarrer la "
                               "surveillance d'abord.")
                else:
                    prod_save("manuelle", st.session_state.last_raw,
                              st.session_state.last_shown)
                    st.toast("Capture enregistrée dans a_annoter/ — "
                             "la surveillance reprend.")

            if st.session_state.prod_running:
                import cv2  # import local : seulement pour la surveillance

                model = load_model(str(w))
                src = int(cam.strip()) if cam.strip().isdigit() else cam.strip()
                if isinstance(src, int) and sys.platform.startswith("linux"):
                    # caméra USB sous Linux (Jetson) : backend V4L2 + flux MJPEG —
                    # sans ça la caméra sert du YUYV brut plafonné à ~5 img/s
                    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    if not cap.isOpened():  # caméra non V4L2 : backend par défaut
                        cap = cv2.VideoCapture(src)
                else:
                    cap = cv2.VideoCapture(src)
                if not cap.isOpened():
                    st.session_state.prod_running = False
                    status_box.error(f"Caméra « {cam} » inaccessible — vérifier l'indice ou "
                                     "l'URL (sous WSL, voir l'aide du champ ci-dessus).")
                else:
                    hold_s = 3.0   # l'alerte reste affichée 3 s après la dernière détection
                    event_open, event_classes, last_trigger = False, set(), 0.0
                    held_boxes = []   # dernières boîtes en alerte, tenues comme l'alerte
                    fps = None
                    try:
                        while st.session_state.prod_running:
                            ok, frame = cap.read()
                            if not ok:
                                status_box.error("Flux caméra interrompu.")
                                break
                            t0 = time.time()
                            raw = frame.copy()   # version sans boîtes, pour l'annotation
                            res = model.predict(frame, conf=min_conf, verbose=False)[0]
                            dets = [(res.names[int(c)], float(cf), [int(v) for v in box])
                                    for c, cf, box in zip(res.boxes.cls, res.boxes.conf,
                                                          res.boxes.xyxy.tolist())]
                            # seules les détections qui atteignent le seuil DE LEUR classe
                            kept = [(n, cf, b) for n, cf, b in dets
                                    if alerting_classes([(n, cf)], thresholds)]
                            # les détections clignotent d'une frame à l'autre : on
                            # redessine les dernières boîtes tant que l'alerte est
                            # tenue, sinon le bandeau ALERTE s'affiche sans boîte
                            if kept:
                                held_boxes = kept
                            for n, cf, (x1, y1, x2, y2) in held_boxes:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, f"{n} {cf:.2f}", (x1, max(y1 - 8, 14)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            # mémorisées pour la capture manuelle : le clic interrompt
                            # la boucle, on sauvegarde alors la dernière image affichée
                            st.session_state.last_raw = raw
                            st.session_state.last_shown = frame
                            now = time.time()
                            if kept:
                                last_trigger = now
                                event_classes |= {n for n, _, _ in kept}
                                if not event_open:  # début d'événement -> journal + snapshot
                                    event_open = True
                                    prod_save("alerte", raw, frame,
                                              classes=", ".join(sorted({n for n, _, _ in kept})),
                                              conf=max(cf for _, cf, _ in kept))
                            elif event_open and now - last_trigger > hold_s:
                                event_open, event_classes, held_boxes = False, set(), []
                                # fin d'alerte = scène saine depuis 3 s : on garde UNE
                                # image négative appariée (même scène sans l'intrus)
                                prod_save("saine", raw)
                            if event_open:
                                status_box.error(
                                    "ALERTE — intrus : " + ", ".join(sorted(event_classes))
                                    + "\n\n" + "\n".join(
                                        f"- {c} : {instructions.get(c, 'retirer cet intrus.')}"
                                        for c in sorted(event_classes)))
                            else:
                                status_box.success("Aucun intrus — flux surveillé.")
                            # cadre rouge = alerte en cours, vert = rien à signaler
                            edge = (0, 0, 255) if event_open else (0, 170, 0)
                            frame = cv2.copyMakeBorder(frame, 12, 12, 12, 12,
                                                       cv2.BORDER_CONSTANT, value=edge)
                            frame_box.image(frame, channels="BGR", width="stretch")
                            dt = time.time() - t0
                            fps = (1 / dt) if fps is None else 0.9 * fps + 0.1 / dt
                            info_box.caption(f"{fps:.1f} images/s — modèle : {w.name}")
                    finally:
                        cap.release()

        # -------------------------------------------------------- fichier ponctuel
        else:
            up = st.file_uploader("Image ou vidéo à analyser",
                                  type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
            if up is not None and st.button("Analyser", type="primary"):
                src_file = Path(tempfile.mkdtemp(prefix="production_")) / up.name
                src_file.write_bytes(up.getbuffer())
                # infer.py lit ses seuils dans un yaml : on lui passe ceux des sliders
                rules_file = src_file.parent / "alert_rules.yaml"
                rules_file.write_text(
                    yaml.safe_dump({"intruder_thresholds": thresholds}, allow_unicode=True),
                    encoding="utf-8")
                log = st.empty()
                code, _ = stream_command(
                    ["scripts/infer.py", "--weights", str(w), "--source", str(src_file),
                     "--alert-rules", str(rules_file), "--conf", str(min_conf)],
                    log)
                if code != 0:
                    st.error("Échec — voir le journal.")
                else:
                    run_dir = max((ROOT / "runs").glob("infer_*"),
                                  key=lambda p: p.stat().st_mtime)
                    recs = json.load(open(run_dir / "detections.json", encoding="utf-8"))
                    n_alert = sum(bool(r.get("alert")) for r in recs)
                    triggered = alerting_classes(
                        [(det["class"], det["confidence"])
                         for r in recs for det in r["detections"]], thresholds)
                    if n_alert:
                        st.error(f"ALERTE — intrus sur {n_alert}/{len(recs)} "
                                 "image(s)/frame(s) : " + ", ".join(triggered))
                        for c in triggered:
                            st.markdown(f"- **{c}** : {instructions.get(c, 'retirer cet intrus.')}")
                    else:
                        st.success(f"Aucun intrus détecté "
                                   f"({len(recs)} image(s)/frame(s) analysée(s)).")
                    for img in sorted(list(run_dir.glob("*.jpg")) + list(run_dir.glob("*.png")))[:6]:
                        st.image(str(img), caption=img.name)
                    for vid in sorted(run_dir.glob("*.mp4")):
                        st.video(str(vid))
                    for vid in sorted(run_dir.glob("*.avi")):
                        st.caption(f"Vidéo annotée (AVI, non lisible dans le navigateur) : `{vid}`")
                    st.caption(f"Détails : `{run_dir}/detections.json`")
