"""Inférence sur une image, un dossier ou une vidéo.

Sauvegarde les images/vidéos annotées et un JSON des détections. Avec
--alert-rules, applique la logique d'alerte et affiche ALERTE/OK par image
(ou par frame pour une vidéo).

Exemples :
    python scripts/infer.py --weights runs/train_xxx/weights/best.pt --source photo.jpg
    python scripts/infer.py --weights best.pt --source dossier/ --alert-rules configs/alert_rules.yaml
"""

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from compost_detection.alert import is_alert, load_alert_rules
from compost_detection.naming import run_name


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="chemin du .pt")
    parser.add_argument("--source", required=True, help="image, dossier ou vidéo")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="seuil de confiance des détections (défaut : 0.25)")
    parser.add_argument("--alert-rules",
                        help="YAML des règles d'alerte (ex. configs/alert_rules.yaml)")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--device")
    args = parser.parse_args()

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    rules = load_alert_rules(args.alert_rules) if args.alert_rules else None

    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.conf, device=device,
                            # chemin absolu : sinon Ultralytics le préfixe par son runs_dir global
                            save=True, project=str(Path(args.runs_dir).resolve()),
                            name=run_name("infer"), stream=True, verbose=False)
    records = []
    n_alerts = 0
    for i, result in enumerate(results):
        names = result.names
        detections = [
            {"class": names[int(cls)], "confidence": round(float(conf), 4),
             "box_xyxy": [round(v, 1) for v in box]}
            for cls, conf, box in zip(result.boxes.cls, result.boxes.conf,
                                      result.boxes.xyxy.tolist())
        ]
        record = {"source": result.path, "frame": i, "detections": detections}
        if rules is not None:
            alert = is_alert([(d["class"], d["confidence"]) for d in detections], rules)
            record["alert"] = alert
            n_alerts += alert
            print(f"  [{'ALERTE' if alert else '  OK  '}] {Path(result.path).name} "
                  f"(frame {i}) — {len(detections)} détection(s)")
        else:
            print(f"  {Path(result.path).name} (frame {i}) — {len(detections)} détection(s)")
        records.append(record)

    # dossier RÉEL utilisé par Ultralytics (qui suffixe le nom en cas de collision)
    out_dir = Path(model.predictor.save_dir)
    with open(out_dir / "detections.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    if rules is not None:
        print(f"\n{n_alerts}/{len(records)} image(s)/frame(s) en ALERTE")
    print(f"Sorties annotées + detections.json dans {out_dir}")


if __name__ == "__main__":
    main()
