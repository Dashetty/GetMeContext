import os
import sys
from pathlib import Path

# Make the project root importable when pytest runs from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Keep ChromaDB from attempting telemetry during tests.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
