"""批量运行 demos/01..10，汇总成功/失败。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable   # 用当前解释器，避免硬编码其他项目的 venv

demos = sorted(ROOT.glob("demos/[0-9][0-9]_*.py"))
failed = []
for d in demos:
    print(f"\n{'#'*66}\n# RUN {d.name}\n{'#'*66}")
    r = subprocess.run([PY, str(d)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(ROOT))
    out = (r.stdout or "") + (r.stderr or "")
    print(out[-3000:])
    status = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
    print(f"# => {status}")
    if r.returncode != 0:
        failed.append(d.name)

print(f"\n{'='*66}\nSUMMARY: {len(demos)-len(failed)}/{len(demos)} passed")
if failed:
    print("FAILED:", ", ".join(failed))
    sys.exit(1)
