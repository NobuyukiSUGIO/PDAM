import csv, time
from pdam.experiments import dormancy_sweep
t0=time.time()
rows = dormancy_sweep()
with open("results/real_framework/dormancy.csv","w",newline="") as fh:
    w=csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"DONE {len(rows)} runs in {(time.time()-t0)/60:.1f} min")
for r in rows:
    print(f"{r['backend']:10s} {r['attack_type']} {r['workload']:20s} N={r['dormancy']:>4d}  "
          f"surv={r['survive']} retr={r['retrieve']} full={r['full_persist']} eff={r['effect']}")
