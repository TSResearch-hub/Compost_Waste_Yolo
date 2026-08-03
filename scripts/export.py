"""Export du modèle pour le déploiement : ONNX / NCNN (Raspberry Pi, CPU)
ou TensorRT (Jetson, GPU).

Un moteur TensorRT (.engine) est compilé pour LE GPU et LA version de TensorRT
de la machine qui l'exporte : lancer l'export SUR la Jetson, pas sur le PC.
Le fichier est créé à côté du .pt d'origine (weights/best.pt -> weights/best.engine).

Exemples :
    python scripts/export.py --weights runs/train_xxx/weights/best.pt
    python scripts/export.py --weights best.pt --formats ncnn --imgsz 320 --half
    python scripts/export.py --weights weights/best.pt --formats engine --half   # sur la Jetson
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="chemin du .pt à exporter")
    parser.add_argument("--formats", nargs="+", default=["onnx", "ncnn"],
                        choices=["onnx", "ncnn", "engine"],
                        help="formats d'export (défaut : onnx ncnn ; "
                             "engine = TensorRT, à lancer sur la Jetson)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="taille d'image de l'export (défaut : 640 ; "
                             "réduire, ex. 320, accélère l'inférence sur Pi)")
    parser.add_argument("--half", action="store_true",
                        help="poids en FP16 (quand le format le supporte ; "
                             "recommandé pour TensorRT sur Jetson)")
    parser.add_argument("--int8", action="store_true",
                        help="quantification INT8 (quand le format le supporte)")
    parser.add_argument("--device",
                        help="périphérique d'export (défaut : 0 pour engine — "
                             "TensorRT exige le GPU —, celui d'Ultralytics sinon)")
    args = parser.parse_args()

    model = YOLO(args.weights)
    for fmt in args.formats:
        print(f"\nExport {fmt}...")
        kwargs = {"imgsz": args.imgsz, "half": args.half, "int8": args.int8}
        if args.device or fmt == "engine":
            kwargs["device"] = args.device or "0"
        path = model.export(format=fmt, **kwargs)
        print(f"Exporté : {path}")
    print("\nONNX/NCNN : copier le fichier exporté sur la machine cible. "
          ".engine (TensorRT) : utilisable uniquement sur cette machine.")


if __name__ == "__main__":
    main()
