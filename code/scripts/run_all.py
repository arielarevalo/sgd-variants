"""Regenerate every figure deterministically and run the sanity checks.

Usage:  uv run scripts/run_all.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgd_variants import check  # noqa: E402
from sgd_variants.figures import FIG_DIR, generate_all  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main():
    check.main()
    print("\ngenerating figures ...")
    outputs, accs = generate_all()
    for p in outputs:
        print(f"  {p.relative_to(REPO)}")
    print("\nfinal accuracies over the full dataset (banknote, no held-out split):")
    for k, v in accs.items():
        print(f"  {k:<9} {v:.4f}")
    # the deck embeds the dark logistic figure
    src = FIG_DIR / "fig-logistic-dark.png"
    dst = REPO / "presentation" / "assets" / "fig-logistic-dark.png"
    shutil.copy(src, dst)
    print(f"\ncopied {src.name} -> {dst.relative_to(REPO)}")


if __name__ == "__main__":
    main()
