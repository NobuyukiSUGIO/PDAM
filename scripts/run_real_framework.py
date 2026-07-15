"""Phase B runner: real-framework backend matrix (§5.4). Writes CSVs."""
import csv, os, time
from pdam.experiments import real_framework

OUT = "results/real_framework"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
tables = real_framework()
for name, rows in tables.items():
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"wrote {name}.csv ({len(rows)} rows)")
print(f"DONE in {(time.time()-t0)/60:.1f} min")
