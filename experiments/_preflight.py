"""Fast environment check run as job 00 of the overnight orchestrator."""
import pathlib
import sys

print("python     ", sys.version.split()[0])

try:
    import torch
    print("torch      ", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("gpu        ", torch.cuda.get_device_name(0),
              round(props.total_memory / 1e9, 1), "GB")
    else:
        print("WARNING: no CUDA device visible - jobs will run on CPU and be slow")
except Exception as e:
    print("torch      FAILED:", e)

for mod in ("pandas", "numpy", "sklearn", "xgboost", "ultralytics", "pyarrow"):
    try:
        m = __import__(mod)
        print(f"{mod:11s}", getattr(m, "__version__", "ok"))
    except Exception as e:
        print(f"{mod:11s} MISSING: {e}")

need = [
    "experiments/out/manila_segments.parquet",
    "experiments/out/sathorn.parquet",
    "experiments/out/dhakaai_yolo/dataset.yaml",
    "experiments/runs/manila_segments_gru_cls_h1/best.pt",
    "experiments/runs/manila_segments_gru_cls_h2/best.pt",
    "experiments/runs/manila_segments_gru_cls_h4/best.pt",
    "runs/detect/runs/detect/dhakaai_yolov8s/weights/last.pt",
]
missing = 0
for p in need:
    ok = pathlib.Path(p).exists()
    missing += 0 if ok else 1
    print(("OK      " if ok else "MISSING "), p)

cctv = list(pathlib.Path(
    "dataset/bangkok_sathorn-intersection/extracted/cctv").rglob("*_volume_*.csv"))
print(f"cctv csv files: {len(cctv)}")

print("PREFLIGHT", "OK" if missing == 0 else f"{missing} MISSING INPUT(S)")
