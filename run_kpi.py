"""Run the KPI selected in configs/run_kpi.yaml."""

from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).parent
RUN_CONFIG = ROOT / "configs" / "run_kpi.yaml"


def main() -> None:
    with RUN_CONFIG.open(encoding="utf-8") as config_file:
        settings = yaml.safe_load(config_file)

    selected = settings["active_kpi"]
    kpi = settings["kpis"].get(selected)
    if kpi is None:
        available = ", ".join(settings["kpis"])
        raise ValueError(f"Unknown active_kpi {selected!r}. Choose from: {available}")

    command = [
        sys.executable,
        str(ROOT / kpi["script"]),
        "--config",
        str(RUN_CONFIG),
        "--kpi",
        selected,
        *kpi.get("args", []),
    ]

    print(f"[INFO] Running KPI: {selected}")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
