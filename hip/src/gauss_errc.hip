#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"

#if defined(__GFX8__) || defined(__GFX9__)
#define PS_WAVE_BITS 6
#else
#define PS_WAVE_BITS 5
#endif


#include <hip/hip_runtime.h>

// LU with partial (row) pivoting + back substitution.  The LM solve only needs
// the solution vector; the full Gauss-Jordan inverse was never read.
__device__ int gauss_errc(freq_context* CUDA_LCC, const int ma)
{
	__shared__ double sh_big[CUDA_BLOCK_DIM];
	__shared__ int sh_irow[CUDA_BLOCK_DIM];
	__shared__ double sh_da[MAX_N_PAR + 1];
	__shared__ double sh_pivinv;
	__shared__ int sh_piv;
	__shared__ int sh_sing;

	const int n = CUDA_mfit;
	const int mf1 = CUDA_mfit1;
	const int tid = threadIdx.x;
	const int warp = tid >> PS_WAVE_BITS;
	const int lane = tid & ((1 << PS_WAVE_BITS) - 1);
	const int nwarp = CUDA_BLOCK_DIM >> PS_WAVE_BITS;

	for (int j = tid + 1; j <= n; j += CUDA_BLOCK_DIM) sh_da[j] = (*CUDA_LCC).da[j];
	if (tid == 0) sh_sing = 0;
	__syncthreads();

	for (int k = 1; k <= n; k++)
	{
		/* partial pivot: largest |a[i][k]|, i = k..n */
		{
			const int i = k + tid;
			double v = 0.0;
			int ir = 0;
			if (i <= n) { v = fabs((*CUDA_LCC).covar[i * mf1 + k]); ir = i; }
			sh_big[tid] = v;
			sh_irow[tid] = ir;
		}
		__syncthreads();
		if (warp == 0)
		{
			/* warp-cooperative pivot search (approach after IanSteveC's
			   PeriodSearch); tie-break preserved (last index wins) */
			int nent = n - k + 1;
			if (nent > CUDA_BLOCK_DIM) nent = CUDA_BLOCK_DIM;
			double big = -1.0;
			int p = 0;
			for (int base = 0; base < nent; base += 32)
			{
				const int t = base + lane;
				const double v = (t < nent) ? sh_big[t] : -1.0;
				double m = v;
				#pragma unroll
				for (int off = 16; off > 0; off >>= 1)
				{
					const double o = __shfl_xor_sync(0xffffffffffffffffULL, m, off);
					m = (o > m) ? o : m;
				}
				const unsigned long long eq = __ballot_sync(0xffffffffffffffffULL, (t < nent) && (v == m));
				if (lane == 0 && eq != 0ULL)
				{
					const int hi = 63 - __clzll(eq);
					if (m >= big) { big = m; p = sh_irow[base + hi]; }
				}
			}
			if (lane == 0)
			{
				sh_piv = p;
				if (big == 0.0) sh_sing = 1;
			}
		}
		__syncthreads();
		if (sh_sing) break;

		const int p = sh_piv;
		if (p != k)
		{
			for (int j = tid + 1; j <= n; j += CUDA_BLOCK_DIM)
			{
				const double t2 = (*CUDA_LCC).covar[k * mf1 + j];
				(*CUDA_LCC).covar[k * mf1 + j] = (*CUDA_LCC).covar[p * mf1 + j];
				(*CUDA_LCC).covar[p * mf1 + j] = t2;
			}
			if (tid == 0) { const double t2 = sh_da[k]; sh_da[k] = sh_da[p]; sh_da[p] = t2; }
		}
		__syncthreads();
		if (tid == 0) sh_pivinv = 1.0 / (*CUDA_LCC).covar[k * mf1 + k];
		/* The reciprocal and the pivot-row staging both read state already ordered
		   by the swap barrier and neither depends on the other, so they share one
		   barrier instead of taking two. */
		{
			__shared__ double sh_row[MAX_N_PAR + 1];
			const int pbase = k * mf1;
			for (int j = k + 1 + tid; j <= n; j += CUDA_BLOCK_DIM)
				sh_row[j] = (*CUDA_LCC).covar[pbase + j];
			__syncthreads();

			const double pivinv = sh_pivinv;
			const double dak = sh_da[k];
			for (int i = k + 1 + warp; i <= n; i += nwarp)
			{
				const double mult = (*CUDA_LCC).covar[i * mf1 + k] * pivinv;
				/* the L factor is not stored: the matrix below the diagonal is never
				   read again - the back substitution uses the upper triangle and the
				   diagonal only */
				if (lane == 0)
					sh_da[i] -= mult * dak;
				const int base = i * mf1;
				for (int j = k + 1 + lane; j <= n; j += (1 << PS_WAVE_BITS))
					(*CUDA_LCC).covar[base + j] -= mult * sh_row[j];
			}
			__syncthreads();
		}
	}

	int ret = sh_sing ? 2 : 0;
	if (ret == 0)
	{
		/* back substitution on the upper triangle, one warp */
		if (warp == 0)
		{
			for (int i = n; i >= 1; i--)
			{
				double sum = 0.0;
				const int base = i * mf1;
				for (int j = i + 1 + lane; j <= n; j += (1 << PS_WAVE_BITS))
					sum += (*CUDA_LCC).covar[base + j] * sh_da[j];
				for (int off = (1 << (PS_WAVE_BITS - 1)); off > 0; off >>= 1) sum += __shfl_down(sum, off, 1 << PS_WAVE_BITS);
				if (lane == 0) sh_da[i] = (sh_da[i] - sum) / (*CUDA_LCC).covar[base + i];
				__syncwarp();
			}
		}
		__syncthreads();
	}

	for (int j = tid + 1; j <= n; j += CUDA_BLOCK_DIM) (*CUDA_LCC).da[j] = sh_da[j];
	if (ret != 0 && tid == 0)
	{
		/* singular factorisation: take NO step (atry = cg).  The partially
		   eliminated RHS in sh_da is not a solution; using it produced a
		   garbage trial step.  This matches the CPU mrqmin behaviour: a
		   singular factorisation takes no step. */
		for (int l = 1; l <= ma; l++)
			if (CUDA_ia[l]) (*CUDA_LCC).atry[l] = (*CUDA_LCC).cg[l];
	}
	__syncthreads();
	return ret;
}
#undef SWAP
/* from Numerical Recipes */
