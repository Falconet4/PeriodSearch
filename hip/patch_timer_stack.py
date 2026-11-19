#!/usr/bin/env python3
"""Raise BOINC's timer-thread stack to 1 MB.

BOINC API 7.24.1 hardcodes a 32 KB stack in start_timer_thread(); BOINC 8.3
uses 64 KB.  The ROCm HIP runtime loads librocprofiler-register, whose ~30 KB
static TLS makes glibc's minimum thread stack ~47 KB, so the 32 KB build fails
with pthread_create() EINVAL (22).  64 KB is marginal; raise both to 1 MB.
The boinc_disable_timer_thread flag is not honoured by the 7.24 API.
"""
import sys
p = sys.argv[1]
d = bytearray(open(p, "rb").read())
plain32 = bytes.fromhex("4889dfbe00800000")   # mov %rbx,%rdi; mov $0x8000,%esi
plain64 = bytes.fromhex("4889dfbe00000100")   # same with $0x10000 (BOINC 8.3)
big     = bytes.fromhex("4889dfbe00001000")   # same with $0x100000
for plain, label in ((plain32, "32 KB"), (plain64, "64 KB")):
    if plain in d:
        i = d.find(plain)
        d[i+4:i+8] = (0x100000).to_bytes(4, "little")
        open(p, "wb").write(d)
        print("patched BOINC timer-thread stack: %s -> 1 MB" % label)
        break
else:
    if big in d:
        print("BOINC timer-thread stack already 1 MB")
    else:
        sys.exit("ERROR: BOINC timer-thread stack pattern not found; this build would fail under ROCm")
