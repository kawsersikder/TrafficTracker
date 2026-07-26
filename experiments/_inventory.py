"""Print the artifact inventory at the end of the overnight run."""
import pathlib

for label, pattern in [
    ("paper artifacts", "experiments/out/paper/*"),
    ("transfer CSVs", "experiments/runs/transfer_*.csv"),
    ("parquets", "experiments/out/*.parquet"),
]:
    print(f"\n--- {label} ---")
    for p in sorted(pathlib.Path().glob(pattern)):
        print(f"{p.name:52s} {p.stat().st_size / 1024:9.1f} KB")

print("\n--- yolo runs ---")
for p in sorted(pathlib.Path("runs").rglob("results.csv")):
    rows = p.read_text(errors="ignore").strip().split("\n")
    print(f"{str(p.parent):60s} {len(rows) - 1:4d} epochs")
