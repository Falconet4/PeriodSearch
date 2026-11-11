# Attribution

## Upstream application

`period_search` is part of Asteroids@home, developed at the Astronomical Institute, Charles
University in Prague:

- https://github.com/AsteroidsAtHome/PeriodSearch
- https://asteroidsathome.net/boinc

This port is derived from that source. Copyright and license terms are unchanged; see
`LICENSE` and `COPYRIGHT` at the repository root.

## IanSteveC's PeriodSearch

The warp-cooperative Curve1 formulation (`src/curve1_warp.h`, selected by `PS_WARP=1`) follows
the approach used in IanSteveC's independent port:

- https://github.com/IanSteveC/PeriodSearch

Two specific ideas are acknowledged:

- **Warp-cooperative Curve1 search** — lane = facet, shared compaction of visible facets,
  column-owned coefficient accumulation, butterfly reduction (`mrqcof_curve1_opt`).
- **Folded `Dg` (`Dgfold`)** — the per-block `Dg` matrix is not materialised; the per-facet
  factor is folded into `gfac`/the geometry pass.

The warp-cooperative pivot search in the LM solve (`src/gauss_errc.hip`) follows the same
approach.

## AMD HIP headers

`src/include/hip/` contains headers from AMD's HIP distribution, used so the build is
self-contained. They carry `SPDX-License-Identifier: MIT` and their original copyright notices.

## AI assistance

The port was developed with assistance from DeepSeek V4.1 Flash.
