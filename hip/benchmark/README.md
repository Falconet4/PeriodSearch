# Benchmark set - 54 workunits

Real BOINC production workunits issued by Asteroids@home, kept under their original names, plus
the outputs of the two builds in this repository and of the OpenCL 10222 reference.

## Layout

```
inputs/                    54 workunit inputs, unmodified
outputs/fp64/              54 <task>.out, produced by period_search_hip_FP64
outputs/k40/               54 <task>.out, produced by period_search_hip_K40
outputs/opencl-10222/      54 <task>.out, reference outputs of OpenCL 10222
TASKS.tsv                  task, input filename, set
TIMES.tsv                  wall clock for FP64, K=40 and OpenCL 10222, all 54
run-tasks.sh               runs a binary over the whole set
score_vs_reference.py      scores a run with the validator's formula
SHA256SUMS                 checksums of every file above
```

The 54 workunits are three groups: `validation-23` (the set used for every accuracy figure),
`production-30` (a further production set) and `extra` (`1333_1086`, the largest frequency scan).

The inputs are unmodified workunits issued by Asteroids@home, included so the benchmark is
reproducible; they contain no personal data. The FP64 and K=40 outputs were produced by the
builds in this repository. The OpenCL 10222 outputs come from Ahorek's 10222 release (linked
above) and are included only as a comparison reference.

`outputs/opencl-10222/` is the reference application from Ahorek's 10222 pre-release build
(`period_search_BOINC_10222_linux_opencl_x64`), published at
https://github.com/ahorek/asteroids_prerelease/releases/tag/10222 (tag `10222`, commit
`8683154`, released 2026-10-07). The release asset is `asteroids_aio_linux.zip`, sha256
`2b0ccfaa1cc0c82e0b18b6f3d352ca11eccaa038b5fbdb8cf38d250e1ccf193a`; its OpenCL binary is the
upstream app_dev build (version 102.22.1.1) and embeds the app_dev OpenCL kernels. The
`asteroids_prerelease` repository is release-only (a README and the assets); the application
source is the upstream `PeriodSearch` app_dev code. It is included only as a comparison
reference; it is not part of this port.

## Running

```sh
./run-tasks.sh ../period_search_hip_FP64 /tmp/fp64-run
./run-tasks.sh ../period_search_hip_K40  /tmp/k40-run
```

Each task runs standalone (`PS_STANDALONE=1`) in its own directory with its output saved as
`<task>.out` and the wall clock appended to `times.tsv`. Set `PS_FP32_TAIL` or `PS_WARP` in the
environment to override the binary's defaults.

A standalone BOINC process spends about 2 s failing to reach a client on exit; subtract that when
comparing short tasks, or compare two builds on the same harness rather than quoting absolute
times.

## Scoring

```sh
./score_vs_reference.py /tmp/fp64-run --ref outputs/opencl-10222
./score_vs_reference.py /tmp/k40-run  --ref outputs/fp64
```

The score uses the project validator's per-line test verbatim: equal line counts, and for every
line

```
|a - b| / (2 * (a + b))  <=  0.1   (period)
                        <=  0.1   (rms)
                        <=  0.5   (chi-square)
```

Any failing line rejects the pair. Pole agreement (3-D great-circle angle between `lambda` and
`beta`) is reported separately for information; the validator does not compare poles.

## Timings

`TIMES.tsv` gives the wall clock of all 54 workunits for the two builds here and for the OpenCL
10222 reference. Totals: FP64 7,016.6 s, K=40 6,072.2 s, OpenCL 10222 11,575.1 s. FP64 is
1.65x faster than 10222 (39 % less wall clock); K=40 is 1.91x faster (47.5 % less).

Conditions: standalone harness (`PS_STANDALONE=1`), one task per process, with 24 BOINC CPU tasks
running on the host; RX 9060 XT (gfx1200), Linux Mint 22.3, kernel 7.0.0-42, ROCm 10.1.0. The
FP64 and K=40 columns are the two builds in this repository. A standalone run carries about 2 s
of BOINC teardown, so compare builds rather than absolute times on short tasks.
