//#ifndef __CUDACC__
//#define __CUDACC__
//#endif

#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
//#include <cuda_runtime.h>

template<typename APtr>
__device__ void MrqcofCurve2T(freq_context* CUDA_LCC, APtr alpha, double beta[], int inrel, int lpoints)
{
	int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
	const int dstride = CUDA_ma + 1;
	double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;
	int2 xx;


	//precalc thread boundaries
	int tmph, tmpl;
	tmph = lpoints / blockDim.x;
	if (lpoints % blockDim.x) tmph++;
	tmpl = threadIdx.x * tmph;
	lnp1 = (*CUDA_LCC).np1 + tmpl;
	tmph = tmpl + tmph;
	if (tmph > lpoints) tmph = lpoints;
	tmpl++;

	int matmph, matmpl;
	matmph = CUDA_ma / blockDim.x;
	if (CUDA_ma % blockDim.x) matmph++;
	matmpl = threadIdx.x * matmph;
	matmph = matmpl + matmph;
	if (matmph > CUDA_ma) matmph = CUDA_ma;
	matmpl++;

	int latmph, latmpl;
	latmph = CUDA_lastone / blockDim.x;
	if (CUDA_lastone % blockDim.x) latmph++;
	latmpl = threadIdx.x * latmph;
	latmph = latmpl + latmph;
	if (latmph > CUDA_lastone) latmph = CUDA_lastone;
	latmpl++;

	/*   if ((*CUDA_LCC).Lastcall != 1) always ==0
	   {*/
	if (inrel /*==1*/)
	{
		for (jp = tmpl; jp <= tmph; jp++)
		{
			lnp1++;
			const int ib = jp * dstride;
			/* Set the size scale coeff. deriv. explicitly zero for relative lcurves */
			(*CUDA_LCC).dytemp[ib + 1] = 0;

			xx = __ldg(&texsig[lnp1]);
			coef = __hiloint2double(xx.y, xx.x) * lpoints / (*CUDA_LCC).ave;

			double yytmp = (*CUDA_LCC).ytemp[jp];
			coef1 = yytmp / (*CUDA_LCC).ave;
			(*CUDA_LCC).ytemp[jp] = coef * yytmp;

			for (l = 2; l <= CUDA_ma; l++)
				(*CUDA_LCC).dytemp[ib + l] = coef * ((*CUDA_LCC).dytemp[ib + l] - coef1 * (*CUDA_LCC).dave[l]);
		}
	}
	__syncthreads();

	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np1 += lpoints;
	}

	lnp2 = (*CUDA_LCC).np2;
	ltrial_chisq = (*CUDA_LCC).trial_chisq;

	if (CUDA_ia[1]) //not relative
	{
		double beta_acc = 0;
		for (jp = 1; jp <= lpoints; jp++)
		{
			ymod = (*CUDA_LCC).ytemp[jp];

			for (l = matmpl; l <= matmph; l++)
				(*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[jp * dstride + l];
			__syncthreads();

			lnp2++;
			xx = __ldg(&texsig[lnp2]);
			sig2i = 1 / (__hiloint2double(xx.y, xx.x) * __hiloint2double(xx.y, xx.x));

			xx = __ldg(&texWeight[lnp2]);
			wght = __hiloint2double(xx.y, xx.x);

			xx = __ldg(&texbrightness[lnp2]);
			dy = __hiloint2double(xx.y, xx.x) - ymod;

			j = 0;
			//
			double sig2iwght = sig2i * wght;
			//
			if (CUDA_lastone <= blockDim.x)
			{
				/* lastone <= blockDim: every lane owns exactly one column.
				   dyda[tid+1] is invariant across l; hoist it manually because
				   the compiler cannot prove alpha and dyda do not alias. */
				const int tid = threadIdx.x;
				const double myd = (tid < CUDA_lastone) ? (*CUDA_LCC).dyda[tid + 1] : 0.0;
				/* manual 2-way unroll: two independent alpha updates in flight */
				for (l = 1; l + 1 <= CUDA_lastone; l += 2)
				{
					const int j0 = ++j;
					const int j1 = ++j;
					const double wt0 = (*CUDA_LCC).dyda[l] * sig2iwght;
					const double wt1 = (*CUDA_LCC).dyda[l + 1] * sig2iwght;
					if (tid < l)     alpha[j0 * (CUDA_mfit1)+tid + 1] += wt0 * myd;
					if (tid < l + 1) alpha[j1 * (CUDA_mfit1)+tid + 1] += wt1 * myd;
				} /* l */
				for (; l <= CUDA_lastone; l++)
				{
					j++;
					wt = (*CUDA_LCC).dyda[l] * sig2iwght;
					if (tid < l)
						alpha[j * (CUDA_mfit1)+tid + 1] += wt * myd;
				} /* l */
				if (tid < CUDA_lastone)
				{
					const double wtb = myd * sig2iwght;
					beta_acc = beta_acc + dy * wtb;
				}
			}
			else
			{
			/* UNPROFILED FALLBACK.  Reached only when CUDA_lastone > blockDim.x,
			   i.e. more than 64 linear parameters (CURVE2_BLOCK_DIM == 64).  Our
			   models have ~30, so this has never executed on any workunit in this
			   archive.  It re-partitions each row across the block on every l
			   (two integer divisions per l per thread) and serialises the beta
			   update on thread 0, so it is expected to be substantially slower
			   than the branch above.  Nothing here has been measured - if you
			   ever see lastone > 64, treat this path as untuned. */
			for (l = 1; l <= CUDA_lastone; l++)
			{
				j++;
				wt = (*CUDA_LCC).dyda[l] * sig2iwght;
				//				   k = 0;
				//precalc thread boundaries
				tmph = l / blockDim.x;
				if (l % blockDim.x) tmph++;
				tmpl = threadIdx.x * tmph;
				tmph = tmpl + tmph;
				if (tmph > l) tmph = l;
				tmpl++;
				for (m = tmpl; m <= tmph; m++)
				{
					//				  k++;
					alpha[j * (CUDA_mfit1)+m] = alpha[j * (CUDA_mfit1)+m] + wt * (*CUDA_LCC).dyda[m];
				} /* m */
				if (threadIdx.x == 0)
				{
					beta[j] = beta[j] + dy * wt;
				}
			} /* l */
			}
			for (; l <= CUDA_lastma; l++)
			{
				if (CUDA_ia[l])
				{
					j++;
					wt = (*CUDA_LCC).dyda[l] * sig2iwght;
					//				   k = 0;

					for (m = latmpl; m <= latmph; m++)
					{
						//					  k++;
						alpha[j * (CUDA_mfit1)+m] = alpha[j * (CUDA_mfit1)+m] + wt * (*CUDA_LCC).dyda[m];
					} /* m */
					if (threadIdx.x == 0)
					{
						k = CUDA_lastone;
						m = CUDA_lastone + 1;
						for (; m <= l; m++)
						{
							if (CUDA_ia[m])
							{
								k++;
								alpha[j * (CUDA_mfit1)+k] = alpha[j * (CUDA_mfit1)+k] + wt * (*CUDA_LCC).dyda[m];
							}
						} /* m */
						beta[j] = beta[j] + dy * wt;
					}
				}
			} /* l */
			ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
			/* dyda is reused by the next point; make sure every lane is done
			   reading it before the next cooperative load overwrites it. */
			__syncthreads();
		} /* jp */
		if (threadIdx.x < CUDA_lastone)
			beta[threadIdx.x + 1] = beta[threadIdx.x + 1] + beta_acc;
	}
	else //relative ia[1]==0
	{
		double beta_acc = 0;
		const int tid = threadIdx.x;
		if (CUDA_lastone <= blockDim.x && CUDA_lastone <= 63 && CUDA_ma <= 65 && CUDA_lastone == CUDA_lastma)
		{
			/* alpha lives in global memory and every store is write-through to L2.
			   Batch TILE points per read-modify-write to cut the store traffic. */
			const int TILE = 20;
			const int C2S = 67;
			__shared__ double sh_dyda[TILE][C2S];
			const int np2_curve = lnp2;
			int jp0 = 1;
			while (jp0 <= lpoints)
			{
				const int nt = (jp0 + TILE - 1 <= lpoints) ? TILE : (lpoints - jp0 + 1);
				/* One lane per point computes that point's scalars and parks them in
				   the two tile columns that are never coefficients (64/65, since
				   lastone <= 63).  Every thread then reads them as LDS broadcasts. */
				{
					const int t = tid;
					if (t < nt)
					{
						const int jpt = jp0 + t;
						const double ymod0 = (*CUDA_LCC).ytemp[jpt];
						const int lnp = np2_curve + jpt;
						int2 xx2;
						xx2 = __ldg(&texsig[lnp]);
						const double sig2i0 = 1 / (__hiloint2double(xx2.y, xx2.x) * __hiloint2double(xx2.y, xx2.x));
						xx2 = __ldg(&texWeight[lnp]);
						const double wght0 = __hiloint2double(xx2.y, xx2.x);
						xx2 = __ldg(&texbrightness[lnp]);
						sh_dyda[t][64] = sig2i0 * wght0;
						sh_dyda[t][65] = __hiloint2double(xx2.y, xx2.x) - ymod0;
					}
				}
				/* Coalesced transpose staging: groups of TILE lanes read TILE
				   consecutive points for one coefficient, then scatter into LDS.
				   The old mapping read one column per thread, i.e. 32 cache lines
				   per warp; this reads two. */
				{
					/* Point-major dytempT: lane = coefficient; each point row is
					   read as one coalesced run. */
					const int lcol = tid;
					if (lcol <= CUDA_lastone)
					{
						for (int p = 0; p < nt; p++)
							sh_dyda[p][lcol] = (*CUDA_LCC).dytemp[(jp0 + p) * dstride + lcol];
					}
				}
				__syncthreads();
				for (int t = 0; t < nt; t++)
					ltrial_chisq = ltrial_chisq + sh_dyda[t][65] * sh_dyda[t][65] * sh_dyda[t][64];
				/* Work-balanced triangular mapping.  The original gives each lane one
				   column and rows l > tid, so the slowest lane does 7 iterations while
				   the mean is 3.4 - half the issue slots run masked off.  Here the
				   1,539 (row, column) pairs are spread evenly over the block, row-major
				   so a wave's alpha stores stay contiguous.  Row index r = l-1. */
				const int lastone = CUDA_lastone;
				const int lastL   = lastone - 1;
				/* 2x2 register blocking of the triangular alpha update: each thread
				   owns a 2x2 block of (row, column) pairs, so the weight and the two
				   dyda rows are loaded once for four accumulations instead of once per
				   element.  Every element still accumulates over t = 0..nt-1 in the
				   same order, so the arithmetic is unchanged. */
				const int nB   = (lastL + 2) >> 1;
				const int nBk  = (nB * (nB + 1)) >> 1;   /* blocks with bc <= br only */
				for (int blk = tid; blk < nBk; blk += blockDim.x)
				{
					int br = 0, rem = blk;
					while (rem >= br + 1) { rem -= br + 1; br++; }
					const int bc = rem;
					const int r0 = br << 1, c0 = bc << 1;
					double a00 = 0, a01 = 0, a10 = 0, a11 = 0;
					#pragma unroll
					for (int t = 0; t < TILE; t++)
						if (t < nt)
						{
							const double w  = sh_dyda[t][64];
							const double w0 = sh_dyda[t][r0 + 1] * w;
							const double w1 = sh_dyda[t][r0 + 2] * w;
							const double c1 = sh_dyda[t][c0 + 1];
							const double c2 = sh_dyda[t][c0 + 2];
							a00 += w0 * c1;
							a01 += w0 * c2;
							a10 += w1 * c1;
							a11 += w1 * c2;
						}
					const int eL[4] = { r0, r0, r0 + 1, r0 + 1 };
					const int eK[4] = { c0, c0 + 1, c0, c0 + 1 };
					const double ev[4] = { a00, a01, a10, a11 };
					for (int e = 0; e < 4; e++)
					{
						const int pL = eL[e], pK = eK[e];
						const int rowlen = ((pL + 1) < lastone) ? (pL + 1) : lastone;
						if (pL >= 1 && pL <= lastL && pK < rowlen)
							alpha[pL * (CUDA_mfit1) + pK] += ev[e];
					}
				}
				if (tid + 1 < CUDA_lastone)
				{
					double bacc = 0;
#pragma unroll
					for (int t = 0; t < TILE; t++)
						if (t < nt) bacc += sh_dyda[t][65] * (sh_dyda[t][tid + 2] * sh_dyda[t][64]);
					beta_acc = beta_acc + bacc;
				}
				__syncthreads();
				jp0 += nt;
			}
			lnp2 = np2_curve + lpoints;
		}
		else
		{
		for (jp = 1; jp <= lpoints; jp++)
		{
			ymod = (*CUDA_LCC).ytemp[jp];

			for (l = matmpl; l <= matmph; l++)
				(*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[jp * dstride + l];
			__syncthreads();

			lnp2++;

			xx = __ldg(&texsig[lnp2]);
			sig2i = 1 / (__hiloint2double(xx.y, xx.x) * __hiloint2double(xx.y, xx.x));

			xx = __ldg(&texWeight[lnp2]);
			wght = __hiloint2double(xx.y, xx.x);

			xx = __ldg(&texbrightness[lnp2]);
			dy = __hiloint2double(xx.y, xx.x) - ymod;

			j = 0;
			//
			double sig2iwght = sig2i * wght;
			//l==1
			//
			if (CUDA_lastone <= blockDim.x)
			{
				const int tid = threadIdx.x;
				const double myd = (tid + 1 <= CUDA_lastone) ? (*CUDA_LCC).dyda[tid + 1] : 0.0;
				for (l = 2; l + 1 <= CUDA_lastone; l += 2)
				{
					const int j0 = ++j;
					const int j1 = ++j;
					const double wt0 = (*CUDA_LCC).dyda[l] * sig2iwght;
					const double wt1 = (*CUDA_LCC).dyda[l + 1] * sig2iwght;
					if (tid >= 1 && tid < l)     alpha[j0 * (CUDA_mfit1)+tid] += wt0 * myd;
					if (tid >= 1 && tid < l + 1) alpha[j1 * (CUDA_mfit1)+tid] += wt1 * myd;
				} /* l */
				for (; l <= CUDA_lastone; l++)
				{
					j++;
					wt = (*CUDA_LCC).dyda[l] * sig2iwght;
					if (tid >= 1 && tid < l)
						alpha[j * (CUDA_mfit1)+tid] += wt * myd;
				} /* l */
				if (tid + 1 < CUDA_lastone)
				{
					const double wtb = (*CUDA_LCC).dyda[tid + 2] * sig2iwght;
					beta_acc = beta_acc + dy * wtb;
				}
			}
			else
			{
			for (l = 2; l <= CUDA_lastone; l++)
			{
				j++;
				wt = (*CUDA_LCC).dyda[l] * sig2iwght;
				//				   k = 0;
				//precalc thread boundaries
				tmph = l / blockDim.x;
				if (l % blockDim.x) tmph++;
				tmpl = threadIdx.x * tmph;
				tmph = tmpl + tmph;
				if (tmph > l) tmph = l;
				tmpl++;
				//m==1
				if (tmpl == 1) tmpl++;
				//
				for (m = tmpl; m <= tmph; m++)
				{
					//					  k++;
					alpha[j * (CUDA_mfit1)+m - 1] = alpha[j * (CUDA_mfit1)+m - 1] + wt * (*CUDA_LCC).dyda[m];
				} /* m */
				if (threadIdx.x == 0)
				{
					beta[j] = beta[j] + dy * wt;
				}
			} /* l */
			}
			for (; l <= CUDA_lastma; l++)
			{
				if (CUDA_ia[l])
				{
					j++;
					wt = (*CUDA_LCC).dyda[l] * sig2iwght;
					//				   k = 0;

					tmpl = latmpl;
					//m==1
					if (tmpl == 1) tmpl++;
					//
					for (m = tmpl; m <= latmph; m++)
					{
						//k++;
						alpha[j * (CUDA_mfit1)+m - 1] = alpha[j * (CUDA_mfit1)+m - 1] + wt * (*CUDA_LCC).dyda[m];
					} /* m */
					if (threadIdx.x == 0)
					{
						k = CUDA_lastone - 1;
						m = CUDA_lastone + 1;
						for (; m <= l; m++)
						{
							if (CUDA_ia[m])
							{
								k++;
								alpha[j * (CUDA_mfit1)+k] = alpha[j * (CUDA_mfit1)+k] + wt * (*CUDA_LCC).dyda[m];
							}
						} /* m */
						beta[j] = beta[j] + dy * wt;
					}
				}
			} /* l */
			ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
			/* dyda is reused by the next point; make sure every lane is done
			   reading it before the next cooperative load overwrites it. */
			__syncthreads();
		} /* jp */
		}
		if (threadIdx.x + 1 < CUDA_lastone)
			beta[threadIdx.x + 1] = beta[threadIdx.x + 1] + beta_acc;
	}
	/*     } always ==0 /* Lastcall != 1 */

	   /*  if (((*CUDA_LCC).Lastcall == 1) && (CUDA_Inrel[i] == 1)) always ==0
			(*CUDA_LCC).Sclnw[i] = (*CUDA_LCC).Scale * CUDA_Lpoints[i] * CUDA_sig[np]/ave;*/

	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np2 = lnp2;
		(*CUDA_LCC).trial_chisq = ltrial_chisq;
	}
}


__device__ void MrqcofCurve2(freq_context* CUDA_LCC, double* alpha, double beta[], int inrel, int lpoints)
{
	MrqcofCurve2T(CUDA_LCC, alpha, beta, inrel, lpoints);
}

__global__ void __launch_bounds__(64, 8) CudaCalculateIter1Mrqcof1Curve2(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	MrqcofCurve2T(CUDA_LCC, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
}

__global__ void __launch_bounds__(64, 8) CudaCalculateIter1Mrqcof2Curve2(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	MrqcofCurve2T(CUDA_LCC, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
}
