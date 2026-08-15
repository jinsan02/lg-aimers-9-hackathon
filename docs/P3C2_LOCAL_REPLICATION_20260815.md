# P3-C2 independent local replication

This run followed the already-completed primary 5070 gate recorded in commit
`8b5ecd6`. It is an independent corroboration on the current computer, not a
replacement gate and not a second chance to select a result. Cross-machine
point estimates are not pooled.

## Setup and verdict

- Host: `localhost`, NVIDIA GeForce RTX 3070 Ti 8 GB.
- Source: `e340223`, the same P3-C2 implementation used by the primary run.
- Surface: fit through 2022, validation 2023, untouched test 2024.
- Seed: 3; fresh local base and fresh local cell control.
- Result: **FAIL corroborated**. Untouched-2024 fixed core
  `890.294435 -> 885.486834`, delta **-4.807601 BSS**.
- No six-seed extension, variant, package, or submission.

The primary 5070 run remains authoritative at fixed-core delta `-1.222`. Both
machines put the experiment below zero, so the pre-registered FAIL and closure
of P3-C/P3-C2 are unchanged.

## Comparable artifacts

| member | tag | best iteration | val 2023 BSS | untouched 2024 BSS |
| --- | --- | ---: | ---: | ---: |
| fixed base | `P3C2LBASE_base` | 794 | 567.09 | 864.84 |
| fresh cell control | `P3C2LCTL_cell` | 2922 | 614.06 | 888.57 |
| P3-C2 cell | `P3C2LCAND_cell` | 2931 | 602.60 | 879.71 |

All three have 121 ordered features, feature hash `ac03d14a49872b1e`, fit-row
hash `383a417f7f146006`, and identical `870,752 / 245,525 / 253,507`
fit/validation/test contracts. Host, surface, seed, `row_id`, and target arrays
agree elementwise. The only training-relevant difference between the two cell
arms is `p3c2_balanced`.

## Results

| surface/segment | control core | P3-C2 core | delta |
| --- | ---: | ---: | ---: |
| source 2023 | 607.101596 | 599.590605 | -7.510991 |
| untouched 2024 | 890.294435 | 885.486834 | **-4.807601** |
| 2024 first row half | | | -2.704934 |
| 2024 second row half | | | -6.912628 |
| 2024 month <= 6 | 1018.791880 | 1016.585459 | -2.206421 |
| 2024 month > 6 | 701.148602 | 692.940753 | -8.207850 |
| 2024 R league | 888.080907 | 885.096493 | -2.984414 |
| 2024 F league | 581.155687 | 562.669392 | -18.486295 |

Debiased core also stayed negative: `888.119366 -> 885.253458`, delta
`-2.865908`. Cell RMS was `0.005818366`, Pearson `0.992048418`; core
reliability/resolution moved from `0.000031011 / 0.002225462` to
`0.000034615 / 0.002208102`. The mechanism matches the primary run: the
weighting changes routing but loses resolution.

Actual filtered-row weights matched the primary 5070 run exactly:

| fit | n9/n10/n11 | w9/w10/w11 | mass error |
| --- | --- | --- | ---: |
| selection | 323113 / 131681 / 627 | 0.70376927 / 1.72687783 / 1.0 | 0 |
| refit | 407870 / 169500 / 803 | 0.70778679 / 1.70315634 / 1.0 | 0 |

Analytic deweight behaved correctly. The weighted success sum scored
`745.716193`; the frozen deweighted candidate scored `879.710928`, still below
the fresh control. Candidate success-cell predicted masses for 9/10/11 were
`0.336024156 / 0.151548626 / 0.000762900` against actual
`0.332787655 / 0.152378435 / 0.000938830`.

## Integrity and hashes

- P3-C2 contract 11/11 and full suite 17/17 passed before training.
- Champion 14/14 members and blend remained bit-identical.
- Saved candidate fresh-load prediction matched all 253,507 stored predictions
  exactly; reversal, half-frame, and single-row drift were zero.
- Candidate outputs were finite and inside `[0,1]`.

SHA-256 packs:

- base: `b430f422a971d63e0a9a8f59e6e98be7e753e7ae825f5984e539c103f68b91f0`
- control: `46d980ae6026532a49180fbb2ed078291d281cc7f9a3ac80d5a87d05f4d614e6`
- candidate: `8f8a785ed953ed92a026c0044fad9899dfe92885a7ceba5efb579b4d1606ac1e`

Champion B1S8 remains unchanged at LB `1108.4333490288`.
