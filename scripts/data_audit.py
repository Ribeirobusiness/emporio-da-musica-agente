"""Print data-quality findings for the challenge dataset.

Usage: python scripts/data_audit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emporio_agent.data import audit, load_data  # noqa: E402

if __name__ == "__main__":
    findings = audit(load_data())
    print(f"{len(findings)} findings\n")
    for f in findings:
        print("-", f)
