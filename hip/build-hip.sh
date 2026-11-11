#!/bin/sh
# Build the HIP port of Asteroids@home period_search.
#
# Usage: ./build-hip.sh [TARGET ...]
#
#   fp64            PS_K_DEFAULT=0,  single-arch for $BRP_ARCHS (default gfx1200)
#   k40             PS_K_DEFAULT=40, single-arch for $BRP_ARCHS
#   multiarch       PS_K_DEFAULT=0,  gfx1200 + gfx1100 + gfx1030
#   multiarch-k40   PS_K_DEFAULT=40, gfx1200 + gfx1100 + gfx1030
#   all             all four
#
# With no target, fp64 and k40 are built for $BRP_ARCHS. The FP32-tail default
# is a compile-time literal in Start.hip; at runtime PS_FP32_TAIL overrides it,
# so either binary can act as the other.
#
# Requires: hipcc (ROCm), BOINC headers in /usr/include/boinc, python3.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/src"
BUILD="$HERE/build"
BOINC_H="-I/usr/include/boinc"
# HIP headers are vendored in src/include/hip so the build is reproducible anywhere.
HIP_H="-I$SRC/include"
MULTI="gfx1200 gfx1100 gfx1030"

if [ $# -eq 0 ]; then
	set -- fp64 k40
fi

build() {
	name="$1"; kdef="$2"; archs="$3"
	OFF=""
	for a in $archs; do
		OFF="$OFF --offload-arch=$a"
	done

	OBJ="$BUILD/obj-$name"
	rm -rf "$OBJ"
	mkdir -p "$OBJ"

	# All device code is compiled as one translation unit (no -fgpu-rdc) so the
	# cross-file device helpers inline.
	DEV="$BUILD/dev-$name.hip"
	: > "$DEV"
	for f in Start blmatrix bright conv curv curve2_CUDA gauss_errc matrix mrqcof mrqmin phasec start_CUDA; do
		echo "#include \"$SRC/$f.hip\"" >> "$DEV"
	done

	hipcc -c -O3 $OFF -pthread -ffp-contract=fast -mcumode \
		-DPS_K_DEFAULT=$kdef \
		-mllvm -amdgpu-stress-vgpr=208 \
		-I"$SRC" $BOINC_H "$HIP_H" -x hip \
		-o "$OBJ/all_dev.o" "$DEV"

	for f in areanorm covsrt dot_product ellfit lubksb ludcmp memory sphfunc trifac period_search_BOINC; do
		hipcc -c -O3 $OFF -pthread -I"$SRC" $BOINC_H "$HIP_H" -x hip \
			-o "$OBJ/$f.o" "$SRC/$f.cpp"
	done

	OUT="$HERE/period_search_hip_$name"
	hipcc $OFF -o "$OUT" "$OBJ"/*.o \
		-Wl,-Bstatic -lboinc_api -lboinc -Wl,-Bdynamic \
		-l:libcurl.so.4 -l:libssl.so.3 -l:libcrypto.so.3 -lz -lm -ldl -lpthread \
		-L/usr/lib/x86_64-linux-gnu

	# BOINC 7.24.x uses a 32 KB timer-thread stack; ROCm's ~30 KB static TLS
	# makes pthread_create() fail with EINVAL. Raise it to 1 MB.
	python3 "$HERE/patch_timer_stack.py" "$OUT"

	rm -rf "$OBJ" "$DEV"
	echo "built $OUT ($archs, PS_K_DEFAULT=$kdef)"
}

for t in "$@"; do
	case "$t" in
		fp64)          build FP64         0  "${BRP_ARCHS:-gfx1200}" ;;
		k40)           build K40         40  "${BRP_ARCHS:-gfx1200}" ;;
		multiarch)     build FP64_multiarch 0  "$MULTI" ;;
		multiarch-k40) build K40_multiarch 40  "$MULTI" ;;
		all)
			build FP64         0  "${BRP_ARCHS:-gfx1200}"
			build K40         40  "${BRP_ARCHS:-gfx1200}"
			build FP64_multiarch 0  "$MULTI"
			build K40_multiarch 40  "$MULTI"
			;;
		*) echo "unknown target: $t" >&2; exit 2 ;;
	esac
done
