#!/usr/bin/env python3
"""Raise BOINC's timer-thread stack from 32 KB to 1 MB.

BOINC API 7.24.1 hardcodes a 32 KB stack in start_timer_thread().  The ROCm
HIP runtime loads librocprofiler-register, whose 30 KB static TLS makes glibc's
minimum thread stack ~47 KB, so pthread_create() returns EINVAL (22) and
boinc_init() fails.  The 1 MB stack fixes it.  The boinc_disable_timer_thread
flag is not honoured by this API version.
"""
import sys
p = sys.argv[1]
d = bytearray(open(p, "rb").read())
plain = bytes.fromhex("4889dfbe00800000")   # mov %rbx,%rdi; mov $0x8000,%esi
big   = bytes.fromhex("4889dfbe00001000")   # same with $0x100000
if plain in d:
    i = d.find(plain)
    d[i+4:i+8] = (0x100000).to_bytes(4, "little")
    open(p, "wb").write(d)
    print("patched BOINC timer-thread stack: 32 KB -> 1 MB")
elif big in d:
    print("BOINC timer-thread stack already 1 MB")
else:
    sys.exit("ERROR: BOINC timer-thread stack pattern not found; this build would fail under ROCm")
