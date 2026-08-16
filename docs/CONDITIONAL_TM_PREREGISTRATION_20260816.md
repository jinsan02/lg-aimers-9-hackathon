# Conditional Trackman geometry — CPU-only preregistration

Written before the audit result.  This is not a GPU candidate and does not
reopen predicted pitch type, H1, `tm_context`, or `expected-pitch-mix`.

## Fixed representation

- Unit: pitcher-season in official 2019-2024 Trackman, mapped by the frozen
  `pitcher_map2.csv`.
- Seven physical columns: release speed, spin, IVB, HB, extension, release
  height, release side.
- Five historical conditions: batter hand Left/Right and count family
  `balls==3`, `strikes==2`, or neutral.  Each condition's mean minus that
  pitcher-season's global mean is shrunk by `n/(n+50)`.
- The resulting 35 columns are standardized and compressed to **four PCA
  coordinates**.  Four is fixed here; no dimension, shrinkage, condition, or
  physical-column sweep is allowed.
- At a source season S the scaler/PCA sees Trackman seasons `<S`; rows in S use
  Trackman S-1.  The same frozen projector is applied to S+1 using Trackman S.
- Cold pitcher fallback is the zero vector.  Inference remains a frozen
  pitcher lookup and cannot aggregate evaluation rows.

## Gates

All must pass:

1. same-pitcher year-to-year stability is present on at least two recent
   transitions;
2. the four coordinates are not reconstructible from the fixed global TM100
   summary (`5-fold CV R2 < .90` is only a duplicate screen, not evidence);
3. source champion residual fit -> untouched next-season apply has the same
   sign on 2022->2023 and 2023->2024;
4. each frozen rho exceeds the p99 of 400 same-dimension, same-fit/freeze
   pitcher-permutation nulls;
5. neither result is carried by only R/F or one half.

Any failure closes this representation.  There is no rescue by changing PCA
dimension, context definitions, shrinkage, or learner.
