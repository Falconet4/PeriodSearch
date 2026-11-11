# period_search HIP port (gfx1200)

An unofficial HIP/ROCm port of Asteroids@home `period_search`, the lightcurve-inversion
application. This tree builds two production binaries from one source. No prebuilt binaries are
distributed; build them from `src/`.

| binary | default | behaviour |
|---|---|---|
| `period_search_hip_FP64` | `PS_FP32_TAIL=0` | full FP64 in every LM iteration |
| `period_search_hip_K40` | `PS_FP32_TAIL=40` | iterations 1-40 in FP64, later iterations on the FP32 tail |

Either default can be overridden at runtime (`PS_FP32_TAIL=0|40`), so one binary can act as the
other. `build-hip.sh all` additionally builds a multi-architecture pair carrying gfx1200,
gfx1100 and gfx1030; only the gfx1200 code has been executed or measured.

**Tested only on** Linux Mint 22.3 (kernel 7.0.0-42), ROCm 10.1.0 / HIP 7.16.26385, on an AMD
Radeon RX 9060 XT 16 GB (gfx1200, RDNA4). No other operating system, kernel, ROCm version or GPU
was tested. The measurements use BOINC application version 10231.

## Validator tolerances

The project validator compares output line by line. It requires equal line counts and, for every
line (`validator/period_search_validator*.cpp`):

```
|a - b| / (2 * (a + b))  <=  0.1   (period)
                        <=  0.1   (rms)
                        <=  0.5   (chi-square)
```

Any failing line rejects the pair. Poles (`dark`, `lambda`, `beta`) are not part of that test.
The source expression is `|a-b| / (2*(a+b))`. Some earlier notes on this project quote
`|a-b| / ((a+b)/2)`, which is four times larger; the figures below use the source expression.
Every workunit passes under either reading.

## Accuracy

Measured over the 54 real BOINC production workunits in `benchmark/` (234,532 solution rows).
The reference is the OpenCL 10222 outputs in `benchmark/outputs/opencl-10222/`; the K=40
comparison is against this port's FP64 build.

| comparison | rows | exact lines | poles <= 5 deg | poles <= 1 deg |
|---|---:|---:|---:|---:|
| FP64 vs OpenCL 10222 | 234,532 | 98.82 % | 99.97 % | 99.87 % |
| K=40 vs FP64 | 234,532 | 60.57 % | 99.98 % | 99.89 % |
| K=40 vs OpenCL 10222 | 234,532 | 60.26 % | 99.95 % | 99.76 % |

"Exact" means all six output fields are numerically equal. Line counts match on every task.

Worst observed value of the validator expression, and the margin to the allowance it is judged
against:

| comparison | worst period | worst rms | worst chi-square | margin (period / rms / chi) |
|---|---:|---:|---:|---:|
| FP64 vs OpenCL 10222 | 1.089e-03 | 6.478e-03 | 1.295e-02 | 92x / 15x / 39x |
| K=40 vs FP64 | 2.376e-04 | 1.759e-03 | 3.517e-03 | 421x / 57x / 142x |
| K=40 vs OpenCL 10222 | 1.089e-03 | 6.478e-03 | 1.295e-02 | 92x / 15x / 39x |

The allowances are 0.1 / 0.1 / 0.5. No line of any workunit fails in any of the three
comparisons. K=40's own deviation from FP64 uses 1.8 % of the rms allowance and 0.7 % of the
chi-square allowance, so it is buried under the gap that already exists between FP64 and the
reference and does not move the worst case.

## Performance

Wall clock on the reference machine, one task per process, standalone harness, same 54 workunits:

| build | total |
|---|---:|
| `period_search_hip_FP64` | 7,016.6 s |
| `period_search_hip_K40` | 6,072.2 s |
| OpenCL 10222 (reference outputs) | 11,575.1 s |

FP64 is 1.65x faster than 10222 (39 % less wall clock); K=40 is 1.91x faster (47.5 % less).
These are single-workunit times, not BOINC throughput; the client adds its own scheduling
overhead. Per-task values for all three builds are in `benchmark/TIMES.tsv`.

## Requirements

- ROCm/HIP with `hipcc` (built and measured with ROCm 10.1.0).
- BOINC API headers, expected at `/usr/include/boinc` (e.g. `boinc-dev`).
- `python3` (used by `patch_timer_stack.py`).
- A gfx1200 GPU for the single-architecture builds.

## Build

```sh
./build-hip.sh            # FP64 and K=40, single-arch for $BRP_ARCHS (default gfx1200)
./build-hip.sh all        # adds the multiarch pair (gfx1200 + gfx1100 + gfx1030)
BRP_ARCHS=gfx1030 ./build-hip.sh fp64
```

Targets: `fp64`, `k40`, `multiarch`, `multiarch-k40`, `all`.

All device code is compiled as one translation unit and linked with the BOINC client libraries.
No prebuilt binary is distributed; the measurement figures describe the two builds defined by
this source. The link line in `build-hip.sh` assumes the tested Mint/Ubuntu library layout
(`libcurl.so.4`, `libssl.so.3`, `-L/usr/lib/x86_64-linux-gnu`); adjust it for other distributions.

## Run standalone

```sh
mkdir -p run && cd run
cp ../benchmark/inputs/ps_261021_input_118556_46 ps_in
printf '<soft_link>ps_in</soft_link>\n' > period_search_in
PS_STANDALONE=1 ../period_search_hip_FP64
```

The result is written to `period_search_out`. `PS_WARP=0` selects the older per-thread Curve1
path instead of the warp-cooperative one.

`benchmark/run-tasks.sh` runs the whole 54-workunit set; `benchmark/score_vs_reference.py`
scores a run with the validator's formula.

## Deploy under BOINC

BOINC records the application file's size when it reads `app_info.xml`. Stop the client before
replacing the binary, install it as the name `app_info.xml` expects (`Period_Search_HIP`), bump
`version_num`, then start the client. Replacing it live produces `File ... has wrong size` and
every queued task fails without launching.

`PS_FP32_TAIL` cannot be set from `app_info.xml`, so deploy the binary whose compiled-in default
you want.

## Changes vs upstream

The port is a CUDA-to-HIP conversion of the upstream `period_search` source. The adopted changes,
in rough code order:

- Single device translation unit: all device code compiled as one HIP TU, no `-fgpu-rdc`.
- No materialised `Dg` matrix: the per-block buffer is gone and the per-facet factor is folded into `gfac`.
- Fused per-point geometry: matrix, phase function and brightness/derivatives in one pass.
- Warp-cooperative Curve1T (`PS_WARP=1`): lane = facet, shared compaction, column-owned accumulation, butterfly reduction.
- Dedicated warp-only Curve1 kernels (`...Curve1Tw`); the older path remains behind `PS_WARP=0`.
- Build flags: `-O3 -ffp-contract=fast -mcumode`, `-mllvm -amdgpu-stress-vgpr=208`.
- Optional FP32 tail beyond `PS_FP32_TAIL`; the FP64 prefix is unchanged.
- BOINC 7.24 workaround: `patch_timer_stack.py` raises the timer-thread stack to 1 MB.
- Standalone harness (`PS_STANDALONE=1`) and the 54-workunit benchmark set.
- Curve2 shared-memory tiling with a work-balanced triangular mapping.
- Curve2 2x2 register-blocked alpha update; shared-memory reductions in the solve.
- Warp-cooperative pivot search; pivot row staged in shared memory; one barrier merged per step; dead L store dropped.
- Flattened `mrqmin_2_end` copies and `mrqcof_end` symmetrisation; parallel `atry` update.
- Curve1Last specialised for the three-point constraint curve, four points per pass, 64 threads.
- Linear zeroing in `mrqcof_start`; 16-byte-aligned geometry staging.
- Curve points spread evenly over the warps; one point per Curve1 pass instead of A/B pairing.
- Per-lightcurve point limit raised from 2,000 to 20,000; Curve2 tile guard at `ma <= 66`.
- `Areaf` removed (the FP32 tail casts `Area` on the fly); dead `freq_context` members removed.
- Host side: 13 of 14 host source files are byte-identical to upstream; only `period_search_BOINC.cpp` changed.
- `PS_K_DEFAULT` build flag: FP64 and K=40 from one source.

## Repository layout

```
hip/
  src/                     full source; HIP headers are vendored under src/include/hip
  build-hip.sh             builds FP64 and K=40 (and optionally the multiarch pair)
  patch_timer_stack.py     raises BOINC's 32 KB timer-thread stack to 1 MB
  benchmark/               54 BOINC production workunits, outputs and timings; see benchmark/README.md
  docs/                    design and measurement notes
  ATTRIBUTION.md
```

`docs/DESIGN-NOTES.md` covers the cost model, the changes that survived measurement, the
directions that were measured and rejected, and the measurement practice.

## Attribution

This port is based on the upstream Asteroids@home `PeriodSearch` source. It also incorporates
ideas from [IanSteveC's HIP/CUDA port](https://github.com/IanSteveC/PeriodSearch), in particular
the warp-cooperative Curve1 search and the folded `Dg`/`gfac` formulation. See `ATTRIBUTION.md`
for details and links.

The port was developed with AI assistance (DeepSeek V4.1 Flash).

## License

Distributed under the same terms as the upstream `PeriodSearch` repository. See the `LICENSE`
and `COPYRIGHT` files at the repository root.

## Disclaimer

This is an unofficial fork. It is not affiliated with, supported by, or endorsed by
Asteroids@home, the Astronomical Institute of Charles University, or the BOINC project. No
warranty of any kind is provided.
