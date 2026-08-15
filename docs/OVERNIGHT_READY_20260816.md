# Morning handoff — 2026-08-16

Nothing was submitted overnight. Champion remains B1S8 / LB
`1108.4333490288`.

## Ready probe 1 — submit first

`submissions/elb1_plus_0816.zip`

- fixed final-output delta: `+0.010`
- SHA256: `f0db2cb9a545d5006abc32cf737250d88dd18baba1b1b6433cdfc84bf3a29bfb`
- size: 88,794,030 bytes; 22 members; only `script.py` differs from B1S8
- 245,789-row synthetic inference: 44 seconds
- strong subset-independence worst drift: `0.000e+00`

## Ready probe 2 — submit second

`submissions/elb1_minus_0816.zip`

- fixed final-output delta: `-0.010`
- SHA256: `25cae8c84bb35b13889f890c6990936731a9ace88e497c1f79aeccf786bdb87d`
- size: 88,794,029 bytes; 22 members; only `script.py` differs from B1S8
- 245,789-row synthetic inference: 44 seconds
- strong subset-independence worst drift: `0.000e+00`

Both packages passed ZIP-root/layout, POSIX path, format, finite probability,
range, non-constant output and local server-shape smoke checks. Their public
sample predictions differ by exactly 0.020000 (within floating precision).
Audit evidence is in `out/elb1_manifest.json`, `out/elb1_plus_audit.txt` and
`out/elb1_minus_audit.txt`.

## After both official scores arrive

Do not estimate by eye. Give Codex both scores, or run:

```powershell
.\.venv\Scripts\python.exe tools\elb_quadratic.py --plus PLUS_SCORE --minus MINUS_SCORE
```

The baseline is frozen at `1108.4333490288`. Stop if curvature is non-concave,
`abs(delta_star)>0.02`, or the output exits 2. No final-delta package was built
overnight because doing so before the two scores exist would fabricate the only
value it is meant to contain.

## GSK2 result

GSK2 is **HOLD**, not ready for submission: six-seed fixed-core mean +3.141,
t=4.168, ensemble +3.108, but F=-3.126 violates the pre-registered non-negative
segment gate. Full machine-readable report: `out/gsk2_gate.json`.

