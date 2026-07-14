# Changelog

## v1.0.3

- Add `sfris lib`: starting structure library covering 555 molecular formulas
  (C/H/O/N/F, up to 9 heavy atoms), 2750 GFN2-xTB geometries, top 5 per formula.
  Convenience feature; structures are unrefined and unvalidated, intended only
  as starting geometries.
- Add `sfris --version`.

## v1.0.2

- Fix: with Block 2 (ADVOPT) disabled, `SFRIS_ISOMER.lis` and `SFRIS.sum`
  labeled xTB energies as DFT, and `SFRIS.sum` printed a `Reference (DFT)` line
  duplicating the xTB reference.
- Fix: structures excluded from Block 2 by `AdvOptFilter` fell back to their xTB
  absolute energy, which was then subtracted from the DFT reference, producing
  meaningless relative energies. Such structures are now reported as
  `not refined`.
- Runs in which every structure reaches Block 2 are unaffected; output is
  byte-identical to v1.0.1.

## v1.0.1

- Version described in the manuscript.
