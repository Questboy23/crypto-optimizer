import json, subprocess, sys
r = {}
for g in ("1", "2", "3"):
    p = subprocess.run(["uv", "run", "python", "gates.py", g], capture_output=True, text=True)
    r.update(json.loads(p.stdout.strip().splitlines()[-1]))
r["tests"] = "pass" if "failed" not in open("logs/pytest.log").read().splitlines()[-1] else "FAIL"
print(json.dumps(r))
