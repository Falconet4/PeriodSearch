# period_search HIP port (gfx1200)

An unofficial HIP/ROCm port of Asteroids@home `period_search`, the lightcurve-inversion
application. The port lives in [`hip/`](hip/README.md); the upstream application source is
unchanged at the repository root.

**Status:** source only. No prebuilt binaries are distributed; build them from `hip/src`.

**Tested platform (the only one tested):** Linux Mint 22.3, kernel 7.0.0-42, ROCm 10.1.0 /
HIP 7.16.26385, on an AMD Radeon RX 9060 XT 16 GB (gfx1200, RDNA4). No other operating system,
kernel, ROCm version or GPU was tested. The single-architecture builds contain gfx1200 code only;
the optional multiarch pair carries gfx1200 + gfx1100 + gfx1030 code, but only the gfx1200 part
has ever been executed.

**Version:** `v1.0.0-hip`. The measurements use BOINC application version 10231.

## What is here

- [`hip/README.md`](hip/README.md) — build, run, deploy, accuracy, the change list and the two
  build targets.
- [`hip/benchmark/`](hip/benchmark/README.md) — 54 real BOINC production workunits; outputs for
  FP64, K=40 and OpenCL 10222; `TASKS.tsv`, `TIMES.tsv`, runner and scoring script.
- [`hip/docs/DESIGN-NOTES.md`](hip/docs/DESIGN-NOTES.md) — cost model, adopted changes, rejected
  directions, measurement practice.
- [`hip/ATTRIBUTION.md`](hip/ATTRIBUTION.md) — upstream and IanSteveC credits, vendored HIP
  headers.

## Headline

Over the 54 real BOINC production workunits (234,532 solution rows), FP64 matches the
[OpenCL 10222 reference](https://github.com/ahorek/asteroids_prerelease/releases/tag/10222) on
98.82 % of output lines, 99.97 % of poles within 5 degrees and 99.87 % within 1 degree; every
workunit passes the project validator with at least 15x margin on rms and 39x on chi-square. K=40
trades line exactness for speed and stays inside the same allowances.

Wall clock on the tested machine: FP64 7,016.6 s, K=40 6,072.2 s, OpenCL 10222 11,575.1 s —
1.65x faster for FP64 (39 % less wall clock) and 1.91x for K=40 (47.5 % less).
Per-task values are in [`hip/benchmark/TIMES.tsv`](hip/benchmark/TIMES.tsv).

## Build

```sh
cd hip
./build-hip.sh            # FP64 and K=40, gfx1200
./build-hip.sh all        # adds the multiarch pair
```

Requirements and deployment details are in [`hip/README.md`](hip/README.md).

## License and attribution

The fork is distributed under the same terms as the upstream
[Asteroids@home PeriodSearch](https://github.com/AsteroidsAtHome/PeriodSearch) repository; see
`LICENSE` and `COPYRIGHT`. It incorporates ideas from
[IanSteveC's HIP/CUDA port](https://github.com/IanSteveC/PeriodSearch) (the warp-cooperative
Curve1 search and the folded `Dg`/`gfac` formulation); see
[`hip/ATTRIBUTION.md`](hip/ATTRIBUTION.md). The port was developed with AI assistance
(DeepSeek V4.1 Flash).

## Disclaimer

Unofficial. Not affiliated with, supported by, or endorsed by Asteroids@home, the Astronomical
Institute of Charles University, or the BOINC project. No warranty of any kind is provided.
