# Design and measurement notes

This page collects the performance work behind the port: the cost model that guided it, the
changes that survived measurement, and the directions that were measured and rejected. It is not
a build guide or a changelog.

## Platform and cost model

- gfx1200 (RX 9060 XT, RDNA4), ROCm 10.1.0, FP64 scalar, 128-thread blocks, one context per block.
- A degree-6 model gives `n_coef = 49`, `ma = 57`, `mfit = 55`. The normal equations are
  55x55 (row stride 56) and the solve is 55 dependent pivot steps.
- Reference kernel split on a short-curve workunit (rocprofv3): Curve1 warp kernel 69.0 %,
  `Mrqmin1End` 11.9 %, Curve2 11.5 %, Curve1Last 2.9 %, everything else under 3 %. On
  long-curve workunits the solve falls to about 5 % and Curve1 dominates more.
- The kernels carrying `Cuda...` names are launch wrappers; the real bodies are out-of-line
  device functions. Instruction-level claims have to be made against those symbols.
- Arithmetic floor for the solve: the 55x55 trailing update is `n^3/3 ~ 55,500` FP64 FMAs per
  block. At roughly 8 lanes/cycle/CU and 16 CUs that is ~1.2-1.3 ms per launch of 4,096 blocks,
  against ~4 ms measured. The gap is barriers and memory latency, not arithmetic.

## What was adopted

- One device translation unit, so cross-file device helpers inline; no `-fgpu-rdc`.
- Warp-cooperative Curve1: lane = facet, shared compaction of visible facets, column-owned
  coefficient accumulation, butterfly reduction. Selected at runtime with `PS_WARP=1`.
- Curve points spread evenly over the warps, and one point per Curve1 pass instead of the
  earlier two-point A/B pairing.
- Optional FP32 tail beyond a threshold (`PS_FP32_TAIL`), leaving the FP64 prefix untouched.
- Curve2 shared-memory tiling with a work-balanced triangular mapping and a 2x2 register-blocked
  alpha update.
- Warp-cooperative pivot search; pivot row staged in shared memory; one barrier merged per
  elimination step; the dead L-factor store removed.
- Curve1Last specialised for the three-point constraint curve and run at 64 threads.
- Linear zeroing, flattened matrix copies, 16-byte-aligned geometry staging, dead structure
  members removed.

## What was measured and rejected

| direction | measurement | reason |
|---|---|---|
| Whole 55x55 solve matrix in shared memory | +26.5 % on `Mrqmin1End` | 27 KB of LDS per block cuts occupancy; the faster element access does not pay |
| Warp-per-context solve | +1.7 % and 13 of 2,171 rows changed | the chunked pivot search is not equivalent to the block form |
| 64-thread solve block | +10.4 % | the trailing update uses all four warps; halving the block halves rows in flight |
| 256-thread solve block | +3.4 % | no extra parallelism available; more synchronisation |
| `-ffast-math -fno-finite-math-only` | rejected | non-finite poles; accuracy contract lost |
| Float-seed Newton reciprocal in the geometry | -2.8 % overall, but output changes on 2 of 5 tasks (one 81-degree pole move) | only usable if byte-identity is given up |
| Curve2 tile 64 | +19.8 % | monotonic in the wrong direction; the phase is staging-bound |
| Curve2 alpha staged in shared memory | +19.6 % | extra shared memory costs more occupancy than the traffic saves |
| Four-point `conv()` merge | no change | `conv()`'s barriers were never the cost |
| 2, 3, 5 or 8 Curve1 warps instead of 4 | ±2-4 % from two to five warps with no coherent sign; eight warps +8 to +15 % | the block shape is a plateau between two and five warps, then an occupancy cliff at eight |
| Matrix cores (int8 WMMA) | Curve1 was the only competitive matrix phase, but the best matrix configuration was still about 1.37x the time of this FP64 build; Curve2 and the solve were slower, geometry 2.8-5.3x slower, the rank-1 LU update cost 29-30 % wall clock, a persistent per-context kernel cost 15 % | gfx12 exposes only 16x16x16 int8 WMMA; there is no FP64 matrix path, and the exact-int8 limb scheme multiplies the MAC count |
| Blocked Cholesky in double | 13-15 % slower than the pivoted LU; solve kernel about 4x the LU | the implementation staged the whole matrix in shared memory, the same occupancy trap; the panel-only design was never built |
| Shared-memory LU and wider solve blocks | all slower or neutral | recorded above; no route left inside the current solver |

## Measurement practice

- Warm up before timing: the first run in a session costs about 2 s more (driver module load).
- A standalone BOINC process spends about 2 s failing to reach a client in `boinc_finish()`.
  Subtract it, or compare builds on the same harness rather than quoting absolute short-task times.
- Use interleaved A/B measurements at kernel level. Wall clock on this machine cannot resolve
  differences below about 1 %.
- Rebuild both sides from the same verified tree. A contaminated source once produced a false
  5 % "win".
- Any source change, even an arithmetically inert one, can change codegen under
  `-ffp-contract=fast` and move the last bits. Verify with the identity check.

## Accuracy contract

- Each build is checked by byte-identity against its own reference outputs on all 54 workunits.
  K=40 is additionally compared with FP64 and stays inside the validator's allowances.
- The project validator's per-line test is `|a-b| / (2*(a+b)) <= 0.1` on period, `<= 0.1` on rms
  and `<= 0.5` on chi-square; equal line counts are required and poles are not compared.
- Reordering a reduction can change the last digits and break byte-identity even when the result
  is physically equivalent. Several otherwise attractive changes were rejected for that reason.

## Where the remaining time is

- Curve1 coefficient accumulation: an ablation put it at 19-28 % of a workunit. It is bound by
  LDS/L1 request throughput, not by FMAs; removing it requires changing the reduction order.
- Curve2: about 13 % of a point-heavy workunit, and staging-bound.
- The solve: 11.9 % of a short-curve workunit and 4.7 % of a long one; barrier- and
  serialisation-bound.
