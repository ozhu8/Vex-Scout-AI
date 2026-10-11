"""Usage:
  python scripts/ingest.py demo                 # synthetic offline event
  python scripts/ingest.py RE-V5RC-26-1234      # real event via RobotEvents
  python scripts/ingest.py RE-V5RC-26-1234 --no-history
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vexscout.demo_data import load_demo  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "demo":
        print("Loaded synthetic event id", load_demo())
    else:
        from vexscout.robotevents import ingest_event
        ingest_event(sys.argv[1], with_history="--no-history" not in sys.argv)
