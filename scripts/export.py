"""Export du modèle vers ONNX et/ou NCNN pour le déploiement Raspberry Pi 4 (CPU).

Exemples :
    python scripts/export.py --weights runs/train_xxx/weights/best.pt
    python scripts/export.py --weights best.pt --formats ncnn --imgsz 320 --half
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="chemin du .pt à exporter")
    parser.add_argument("--formats", nargs="+", default=["onnx", "ncnn"],
                        choices=["onnx", "ncnn"],
                        help="formats d'export (défaut : onnx ncnn)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="taille d'image de l'export (défaut : 640 ; "
                             "réduire, ex. 320, accélère l'inférence sur Pi)")
    parser.add_argument("--half", action="store_true",
                        help="poids en FP16 (quand le format le supporte)")
    parser.add_argument("--int8", action="store_true",
                        help="quantification INT8 (quand le format le supporte)")
    args = parser.parse_args()

    model = YOLO(args.weights)
    for fmt in args.formats:
        print(f"\nExport {fmt}...")
        path = model.export(format=fmt, imgsz=args.imgsz,
                            half=args.half, int8=args.int8)
        print(f"Exporté : {path}")
    print("\nCopier le(s) fichier(s) exporté(s) sur le Raspberry Pi pour l'inférence.")


if __name__ == "__main__":
    main()
