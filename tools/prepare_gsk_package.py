"""Render the frozen GSK deployment submission script from the B1S template."""

from __future__ import annotations

import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "src", "script_blend_b1s.py")
OUT = os.path.join(ROOT, "out", "script_blend_gsk.py")
TEMPLATE_SHA = "5a473187e39df8574b22a146731fb1809a01ad8b943a7cba2416f0f852b9ef2b"
OLD_RETURN = "    return np.clip(p + mid + pb, 0, 1)"
NEW_RETURN = "    return np.clip(p + mid + pb + (+0.002515795361), 0, 1)"


def render(text):
    if text.count('f"cat_B1S_cell_s{s}.pkl"') != 1:
        raise ValueError("B1S cell hook is not unique")
    if text.count(OLD_RETURN) != 1:
        raise ValueError("final-output hook is not unique")
    text = text.replace('f"cat_B1S_cell_s{s}.pkl"',
                        'f"cat_GSKDEP_cell_s{s}.pkl"')
    return text.replace(OLD_RETURN, NEW_RETURN)


def main():
    raw = open(TEMPLATE, "rb").read()
    if hashlib.sha256(raw).hexdigest() != TEMPLATE_SHA:
        raise SystemExit("B1S script template hash drift")
    text = render(raw.decode("utf-8"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {OUT}")
    print("cell family B1S -> GSKDEP; E-LB1 final shift retained")


if __name__ == "__main__":
    main()
