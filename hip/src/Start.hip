#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <math.h>

#include "constants.h"
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
//#include "../../../../../../../Program Files (x86)/Windows Kits/10/Include/10.0.10240.0/ucrt/math.h"
#include <cstdio>
#include "globals.h"
#include "curve1_tmpl.h"

/* Default FP32-tail threshold. 0 = full FP64 for every iteration; 40 = the
   K=40 build. Overridable at runtime with PS_FP32_TAIL, so either binary can
   act as the other. */
#ifndef PS_K_DEFAULT
#define PS_K_DEFAULT 0
#endif

__global__ void CudaCalculatePrepare(int n_start, int n_max, double freq_start, double freq_step)
{
	const auto n = n_start + blockIdx.x;
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

	//zero context
	//	CUDA_CC is zeroed itself as global memory but need to reset between freq TODO
	if (n > n_max)
	{
		(*CUDA_LCC).isInvalid = 1;
		return;
	}
	else
	{
		(*CUDA_LCC).isInvalid = 0;
	}

	(*CUDA_LCC).freq = freq_start - (n - 1) * freq_step;

	/* initial poles */
	(*CUDA_LFR).per_best = 0;
	(*CUDA_LFR).dark_best = 0;
	(*CUDA_LFR).la_best = 0;
	(*CUDA_LFR).be_best = 0;
	(*CUDA_LFR).dev_best = 1e40;
}


__global__ void CudaCalculatePreparePacked(int max_test_periods, double freq_start, double freq_step)
{
	const auto n = 1 + (blockIdx.x % max_test_periods);
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];
	(*CUDA_LCC).isInvalid = 0;
	(*CUDA_LCC).freq = freq_start - (n - 1) * freq_step;
	(*CUDA_LFR).per_best = 0;
	(*CUDA_LFR).dark_best = 0;
	(*CUDA_LFR).la_best = 0;
	(*CUDA_LFR).be_best = 0;
	(*CUDA_LFR).dev_best = 1e40;
}

__global__ void CudaCalculatePreparePole(int m)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid)
	{
		atomicAdd(&CUDA_End, 1);
		(*CUDA_LFR).isReported = 0; //signal not to read result

		return;
	}

	const auto period = 1 / (*CUDA_LCC).freq;

	/* starts from the initial ellipsoid */
	for (auto i = 1; i <= CUDA_Ncoef; i++)
	{
		(*CUDA_LCC).cg[i] = CUDA_cg_first[i];
	}

	(*CUDA_LCC).cg[CUDA_Ncoef + 1] = CUDA_beta_pole[m];
	(*CUDA_LCC).cg[CUDA_Ncoef + 2] = CUDA_lambda_pole[m];

	/* The formulas use beta measured from the pole */
	(*CUDA_LCC).cg[CUDA_Ncoef + 1] = 90 - (*CUDA_LCC).cg[CUDA_Ncoef + 1];

	/* conversion of lambda, beta to radians */
	(*CUDA_LCC).cg[CUDA_Ncoef + 1] = DEG2RAD * (*CUDA_LCC).cg[CUDA_Ncoef + 1];
	(*CUDA_LCC).cg[CUDA_Ncoef + 2] = DEG2RAD * (*CUDA_LCC).cg[CUDA_Ncoef + 2];

	/* Use omega instead of period */
	(*CUDA_LCC).cg[CUDA_Ncoef + 3] = 24 * 2 * PI / period;

	for (auto i = 1; i <= CUDA_Nphpar; i++)
	{
		(*CUDA_LCC).cg[CUDA_Ncoef + 3 + i] = CUDA_par[i];
		//              ia[Ncoef+3+i] = ia_par[i]; moved to global
	}

	/* Lommel-Seeliger part */
	(*CUDA_LCC).cg[CUDA_Ncoef + 3 + CUDA_Nphpar + 2] = 1;
	/* Use logarithmic formulation for Lambert to keep it positive */
	(*CUDA_LCC).cg[CUDA_Ncoef + 3 + CUDA_Nphpar + 1] = log(CUDA_cl);

	/* Levenberg-Marquardt loop */
	// moved to global iter_max,iter_min,iter_dif_max
	//
	(*CUDA_LCC).rchisq = -1;
	(*CUDA_LCC).Alamda = -1;
	(*CUDA_LCC).Niter = 0;
	(*CUDA_LCC).iter_diff = 1e40;
	(*CUDA_LCC).dev_old = 1e30;
	(*CUDA_LCC).dev_new = 0;
	//	(*CUDA_LCC).Lastcall=0; always ==0
	(*CUDA_LFR).isReported = 0;
}


__global__ void CudaCalculatePreparePolePacked(int max_test_periods)
{
	const int m = 1 + blockIdx.x / max_test_periods;
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid)
	{
		atomicAdd(&CUDA_End, 1);
		(*CUDA_LFR).isReported = 0;
		return;
	}

	const auto period = 1 / (*CUDA_LCC).freq;
	for (auto i = 1; i <= CUDA_Ncoef; i++)
		(*CUDA_LCC).cg[i] = CUDA_cg_first[i];
	(*CUDA_LCC).cg[CUDA_Ncoef + 1] = CUDA_beta_pole[m];
	(*CUDA_LCC).cg[CUDA_Ncoef + 2] = CUDA_lambda_pole[m];
	(*CUDA_LCC).cg[CUDA_Ncoef + 1] = 90 - (*CUDA_LCC).cg[CUDA_Ncoef + 1];
	(*CUDA_LCC).cg[CUDA_Ncoef + 1] = DEG2RAD * (*CUDA_LCC).cg[CUDA_Ncoef + 1];
	(*CUDA_LCC).cg[CUDA_Ncoef + 2] = DEG2RAD * (*CUDA_LCC).cg[CUDA_Ncoef + 2];
	(*CUDA_LCC).cg[CUDA_Ncoef + 3] = 24 * 2 * PI / period;
	for (auto i = 1; i <= CUDA_Nphpar; i++)
		(*CUDA_LCC).cg[CUDA_Ncoef + 3 + i] = CUDA_par[i];
	(*CUDA_LCC).cg[CUDA_Ncoef + 3 + CUDA_Nphpar + 2] = 1;
	(*CUDA_LCC).cg[CUDA_Ncoef + 3 + CUDA_Nphpar + 1] = log(CUDA_cl);
	(*CUDA_LCC).rchisq = -1;
	(*CUDA_LCC).Alamda = -1;
	(*CUDA_LCC).Niter = 0;
	(*CUDA_LCC).iter_diff = 1e40;
	(*CUDA_LCC).dev_old = 1e30;
	(*CUDA_LCC).dev_new = 0;
	(*CUDA_LFR).isReported = 0;
}
__global__ void CudaCalculateIter1Begin(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid)
	{
		return;
	}

	(*CUDA_LCC).isNiter = (((*CUDA_LCC).Niter < CUDA_n_iter_max) && ((*CUDA_LCC).iter_diff > CUDA_iter_diff_max)) || ((*CUDA_LCC).Niter < CUDA_n_iter_min);

	if ((*CUDA_LCC).isNiter)
	{
		if ((*CUDA_LCC).Alamda < 0)
		{
			(*CUDA_LCC).isAlamda = 1;
			(*CUDA_LCC).Alamda = CUDA_Alamda_start; /* initial alambda */
		}
		else
			(*CUDA_LCC).isAlamda = 0;
	}
	else
	{
		if (!(*CUDA_LFR).isReported)
		{
			atomicAdd(&CUDA_End, 1);
#ifdef _DEBUG
			/*const int is_precalc = CUDA_Is_Precalc;
			if(is_precalc)
			{
				printf("%d ", CUDA_End);
			}*/
#endif
			(*CUDA_LFR).isReported = 1;
		}
	}

}

__global__ void CudaCalculateIter1Mrqmin1End(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	int block = CUDA_BLOCK_DIM;
	/*gauss_err=*/mrqmin_1_end(CUDA_LCC, CUDA_ma, CUDA_mfit, CUDA_mfit1, block);
}

__global__ void CudaCalculateIter1Mrqmin2End(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqmin_2_end(CUDA_LCC, CUDA_ia, CUDA_ma);
	if (threadIdx.x == 0) (*CUDA_LCC).Niter++;
}

__global__ void CudaCalculateIter1Mrqcof1Start(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_start(CUDA_LCC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta);
}

#if 0 /* legacy, unlaunched */
__global__ void CudaCalculateIter1Mrqcof1Matrix(const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_matrix(CUDA_LCC, (*CUDA_LCC).cg, lpoints);
}
#endif


#if 0 /* legacy, unlaunched */
__global__ void CudaCalculateIter1Mrqcof1Curve1(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_curve1(CUDA_LCC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
}
#endif


__global__ void CudaCalculateIter1Mrqcof1Curve1Last(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_curve1_last(CUDA_LCC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
}

__global__ void CudaCalculateIter1Mrqcof1End(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	const double tc = mrqcof_end(CUDA_LCC, (*CUDA_LCC).alpha);
	if (threadIdx.x == 0) (*CUDA_LCC).Ochisq = tc;
}

__global__ void CudaCalculateIter1Mrqcof2Start(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_start(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da);
}

#if 0 /* legacy, unlaunched */
__global__ void CudaCalculateIter1Mrqcof2Matrix(const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_matrix(CUDA_LCC, (*CUDA_LCC).atry, lpoints);
}
#endif


#if 0 /* legacy, unlaunched */
__global__ void CudaCalculateIter1Mrqcof2Curve1(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_curve1(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
}
#endif


__global__ void CudaCalculateIter1Mrqcof2Curve1Last(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_curve1_last(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
}

__global__ void CudaCalculateIter1Mrqcof2End(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	const double tc = mrqcof_end(CUDA_LCC, (*CUDA_LCC).covar);
	if (threadIdx.x == 0) (*CUDA_LCC).Chisq = tc;
}

__global__ void CudaCalculateIter2(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid)
	{
		return;
	}

	if ((*CUDA_LCC).isNiter)
	{
		const int tid = threadIdx.x;
		__shared__ int s_do;
		if (tid == 0)
			s_do = ((*CUDA_LCC).Niter == 1 || (*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq) ? 1 : 0;
		__syncthreads();
		if (s_do)
		{
			if (tid == 0) (*CUDA_LCC).Ochisq = (*CUDA_LCC).Chisq;
			__syncthreads();

			if ((*CUDA_LCC).Niter == 1)
				curv(CUDA_LCC, (*CUDA_LCC).cg, 1 + tid, CUDA_Numfac);

			double c1 = 0, c2 = 0, c3 = 0;
			for (auto j = 1 + tid; j <= CUDA_Numfac; j += blockDim.x)
			{
				const double a = (*CUDA_LCC).Area[j];
				c1 += a * CUDA_Nor[0][j];
				c2 += a * CUDA_Nor[1][j];
				c3 += a * CUDA_Nor[2][j];
			}
			__shared__ double sc1[CUDA_BLOCK_DIM], sc2[CUDA_BLOCK_DIM], sc3[CUDA_BLOCK_DIM];
			sc1[tid] = c1; sc2[tid] = c2; sc3[tid] = c3;
			__syncthreads();
			for (int s = CUDA_BLOCK_DIM >> 1; s > 0; s >>= 1)
			{
				if (tid < s) { sc1[tid] += sc1[tid + s]; sc2[tid] += sc2[tid + s]; sc3[tid] += sc3[tid + s]; }
				__syncthreads();
			}
			if (tid == 0)
			{
				(*CUDA_LCC).chck[1] = sc1[0];
				(*CUDA_LCC).chck[2] = sc2[0];
				(*CUDA_LCC).chck[3] = sc3[0];
				(*CUDA_LCC).rchisq = (*CUDA_LCC).Chisq - ((*CUDA_LCC).chck[1] * (*CUDA_LCC).chck[1] + (*CUDA_LCC).chck[2] * (*CUDA_LCC).chck[2]+ (*CUDA_LCC).chck[3] * (*CUDA_LCC).chck[3]) * (CUDA_conw_r * CUDA_conw_r);
			}
		}
		if (threadIdx.x == 0)
		{
			(*CUDA_LCC).dev_new = sqrt((*CUDA_LCC).rchisq / (CUDA_ndata - 3));
			/* only if this step is better than the previous,
				1e-10 is for numeric errors */
			if ((*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new > 1e-10)
			{
				(*CUDA_LCC).iter_diff = (*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new;
				(*CUDA_LCC).dev_old = (*CUDA_LCC).dev_new;
			}
			//		(*CUDA_LFR).Niter=(*CUDA_LCC).Niter;
		}
	}
}

__global__ void CudaCalculateFinishPole(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	const int tid = threadIdx.x;
	__shared__ double sh_ta[CUDA_BLOCK_DIM];
	double ta = 0;
	for (auto i = 1 + tid; i <= CUDA_Numfac; i += blockDim.x)
		ta += (*CUDA_LCC).Area[i];
	sh_ta[tid] = ta;
	__syncthreads();
	for (int s = CUDA_BLOCK_DIM >> 1; s > 0; s >>= 1)
	{
		if (tid < s) sh_ta[tid] += sh_ta[tid + s];
		__syncthreads();
	}
	if (tid == 0)
	{
		const double totarea = sh_ta[0];
		const auto sum = ((*CUDA_LCC).chck[1] * (*CUDA_LCC).chck[1]) + ((*CUDA_LCC).chck[2] * (*CUDA_LCC).chck[2]) + ((*CUDA_LCC).chck[3] * (*CUDA_LCC).chck[3]);
		const auto dark = sqrt(sum);
		const auto period = 2 * PI / (*CUDA_LCC).cg[CUDA_Ncoef + 3];
		const auto la_tmp = RAD2DEG * (*CUDA_LCC).cg[CUDA_Ncoef + 2];
		const auto be_tmp = 90 - RAD2DEG * (*CUDA_LCC).cg[CUDA_Ncoef + 1];
		const double dev_cand = (*CUDA_LCC).dev_new;
		const double dev_inc = (*CUDA_LFR).dev_best;
		if ((dev_cand == dev_cand) && ((dev_inc != dev_inc) || (dev_cand < dev_inc)))
		{
			(*CUDA_LFR).dev_best = dev_cand;
			(*CUDA_LFR).per_best = period;
			double ddark = dark / totarea * 100;
			if (isnan(ddark)) ddark = 1.0;
			(*CUDA_LFR).dark_best = ddark;
			(*CUDA_LFR).la_best = la_tmp + (la_tmp < 0 ? 360 : 0);
			(*CUDA_LFR).be_best = be_tmp;
		}
	}
	//debug
	/*	(*CUDA_LFR).dark=dark;
	(*CUDA_LFR).totarea=totarea;
	(*CUDA_LFR).chck[1]=(*CUDA_LCC).chck[1];
	(*CUDA_LFR).chck[2]=(*CUDA_LCC).chck[2];
	(*CUDA_LFR).chck[3]=(*CUDA_LCC).chck[3];*/
}

__global__ void CudaCalculateFinish(void)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if ((*CUDA_LFR).la_best < 0)
		(*CUDA_LFR).la_best += 360;

	if (isnan((*CUDA_LFR).dark_best) == 1)
		(*CUDA_LFR).dark_best = 1.0;
}


/* --- bucketed fast path: private facet scratch sized to the actual facet count --- */
__global__ void __launch_bounds__(128, 4) CudaCalculateIter1Mrqcof1Curve1T(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_curve1T<0>(CUDA_LCC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
}

__global__ void __launch_bounds__(128, 4) CudaCalculateIter1Mrqcof1Curve1Tw(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;
	if (!(*CUDA_LCC).isAlamda) return;
	mrqcof_curve1T<1>(CUDA_LCC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
}

__global__ void __launch_bounds__(128, 4) CudaCalculateIter1Mrqcof2Curve1T(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_curve1T<0>(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
}

__global__ void __launch_bounds__(128, 4) CudaCalculateIter1Mrqcof2Curve1Tw(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;
	mrqcof_curve1T<1>(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
}

__global__ void __launch_bounds__(128, 4) CudaCalculateIter1Mrqcof1Curve1T_fp(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;
	if (!(*CUDA_LCC).isAlamda) return;
	mrqcof_curve1T_fp(CUDA_LCC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
}

__global__ void __launch_bounds__(128, 4) CudaCalculateIter1Mrqcof2Curve1T_fp(const int inrel, const int lpoints)
{
	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	if ((*CUDA_LCC).isInvalid) return;
	if (!(*CUDA_LCC).isNiter) return;
	mrqcof_curve1T_fp(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
}

static int ps_use_fp32_tail(int iter)
{
	static int k = -1;
	if (k < 0) { const char* e = getenv("PS_FP32_TAIL"); k = e ? atoi(e) : PS_K_DEFAULT; }
	return (k > 0 && iter > k) ? 1 : 0;
}


void LaunchCurve1Kernel(const int which, const int grid, const int inrel, const int lpoints, const int iter)
{
	static int wmv = -1;
	if (wmv < 0) { const char* e = getenv("PS_WARP"); wmv = e ? atoi(e) : 1; }
	if (ps_use_fp32_tail(iter))
	{
		if (which == 1) CudaCalculateIter1Mrqcof1Curve1T_fp <<<grid, CUDA_BLOCK_DIM >>> (inrel, lpoints);
		else            CudaCalculateIter1Mrqcof2Curve1T_fp <<<grid, CUDA_BLOCK_DIM >>> (inrel, lpoints);
	}
	else
	{
		if (wmv)
		{
			if (which == 1) CudaCalculateIter1Mrqcof1Curve1Tw <<<grid, CUDA_BLOCK_DIM >>> (inrel, lpoints);
			else            CudaCalculateIter1Mrqcof2Curve1Tw <<<grid, CUDA_BLOCK_DIM >>> (inrel, lpoints);
		}
		else
		{
			if (which == 1) CudaCalculateIter1Mrqcof1Curve1T <<<grid, CUDA_BLOCK_DIM >>> (inrel, lpoints);
			else            CudaCalculateIter1Mrqcof2Curve1T <<<grid, CUDA_BLOCK_DIM >>> (inrel, lpoints);
		}
	}
}
