"""Regenerate the submission and check it is byte-identical to the committed one.

    python -m kronos.check

Writes to a temporary file, validates it with the official checks and compares its SHA-256 with
submission/Kronos.csv. Exit code 0 only if both pass.
"""

import hashlib
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kronos.run import main as run  # noqa: E402

SUBMISSION = os.path.join(ROOT, "submission", "Kronos.csv")


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "Kronos.csv")
        run(out)
        valid = subprocess.call([sys.executable,
                                 os.path.join(ROOT, "evaluation", "validate_submission.py"), out]) == 0
        new, old = sha256(out), sha256(SUBMISSION)
    print("regenerated ", new)
    print("committed   ", old)
    same = new == old
    print("IDENTICAL" if same else "DIFFERENT: regenerate with `python -m kronos.run` and commit")
    sys.exit(0 if (valid and same) else 1)
