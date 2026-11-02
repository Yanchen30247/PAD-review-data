# Rerun the whole analysis in the order the scripts depend on one another. Run: python run_all.py
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for s in ["run_rq_meta.py", "results_table.py", "results_table_s3.py", "vote_counts.py", "vote_by_study.py", "rq5_chain.py", "sensitivity_runs.py",
          "make_results_figs.py", "make_funnels.py"]:
    print("running", s, flush=True)
    subprocess.run([sys.executable, s], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
print("done")
