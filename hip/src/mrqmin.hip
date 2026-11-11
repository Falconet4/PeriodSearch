/* N.B. The foll. L-M routines are modified versions of Press et al.
   converted from Mikko's fortran code

   8.11.2006
*/

#include <hip/hip_runtime.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
#include <stdio.h>

__device__ int mrqmin_1_end(freq_context* CUDA_LCC, const int ma, const int mfit, const int mfit1, const int block)
{
	int j;
	//precalc thread boundaries
	int tmph, tmpl;
	tmph = ma / block;
	if (ma % block) tmph++;
	tmpl = threadIdx.x * tmph;
	tmph = tmpl + tmph;
	if (tmph > ma) tmph = ma;
	tmpl++;
	//
	int brtmph, brtmpl;
	brtmph = mfit / block;
	if (mfit % block) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > mfit) brtmph = mfit;
	brtmpl++;

	if ((*CUDA_LCC).isAlamda)
	{
		for (j = tmpl; j <= tmph; j++)
		{
			(*CUDA_LCC).atry[j] = (*CUDA_LCC).cg[j];
		}

		__syncthreads();
	}

	{
		double* __restrict__ dst = (*CUDA_LCC).covar;
		const double* __restrict__ src = (*CUDA_LCC).alpha;
		const int total = CUDA_mfit * CUDA_mfit;
		for (int e = threadIdx.x; e < total; e += blockDim.x)
		{
			const int jj = e / CUDA_mfit + 1;
			const int kk = e - (jj - 1) * CUDA_mfit + 1;
			dst[jj * mfit1 + kk] = src[jj * mfit1 + kk];
		}
	}
	__syncthreads();
	for (j = 1 + threadIdx.x; j <= CUDA_mfit; j += blockDim.x)
	{
		(*CUDA_LCC).covar[j * mfit1 + j] = (*CUDA_LCC).alpha[j * mfit1 + j] * (1 + (*CUDA_LCC).Alamda);
		(*CUDA_LCC).da[j] = (*CUDA_LCC).beta[j];
	}
	__syncthreads();

	int err_code = gauss_errc(CUDA_LCC, ma);
	if(err_code)
	{
		return err_code;
	}

	//err_code = gauss_errc(CUDA_LCC, CUDA_mfit, (*CUDA_LCC).da);

	//     __syncthreads(); inside gauss

	/* Each thread owns one parameter and derives its own index into da from the
	   same prefix count the serial loop walked.  Every atry[l] is independent. */
	{
		const int l = 1 + threadIdx.x;
		if (l <= ma && CUDA_ia[l])
		{
			int jj = 0;
			for (int q = 1; q <= l; q++)
				if (CUDA_ia[q]) jj++;
			(*CUDA_LCC).atry[l] = (*CUDA_LCC).cg[l] + (*CUDA_LCC).da[jj];
		}
	}
	__syncthreads();

	return err_code;
}

__device__ void mrqmin_2_end(freq_context* CUDA_LCC, int ia[], int ma)
{
	int j, k, l;

	const int tid = threadIdx.x;
	if ((*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
	{
		if (tid == 0) (*CUDA_LCC).Alamda = (*CUDA_LCC).Alamda / CUDA_Alamda_incr;
		/* flatten the mfit x mfit copy over the whole block: the old form had
		   only mfit (55) threads active, each walking a row serially with a
		   stride of mfit1 doubles between lanes (uncoalesced).  Pure copy, so
		   the values and therefore the output are unchanged. */
		for (int e = tid; e < CUDA_mfit * CUDA_mfit; e += blockDim.x)
		{
			const int j = e / CUDA_mfit + 1;
			const int k = e - (j - 1) * CUDA_mfit + 1;
			(*CUDA_LCC).alpha[j * CUDA_mfit1 + k] = (*CUDA_LCC).covar[j * CUDA_mfit1 + k];
		}
		for (int j = 1 + tid; j <= CUDA_mfit; j += blockDim.x)
			(*CUDA_LCC).beta[j] = (*CUDA_LCC).da[j];
		for (l = 1 + tid; l <= ma; l += blockDim.x)
			(*CUDA_LCC).cg[l] = (*CUDA_LCC).atry[l];
	}
	else if (tid == 0)
	{
		(*CUDA_LCC).Alamda = CUDA_Alamda_incr * (*CUDA_LCC).Alamda;
		(*CUDA_LCC).Chisq = (*CUDA_LCC).Ochisq;
	}

	return;
}