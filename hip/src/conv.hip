/* Convexity regularization function

   8.11.2006
*/

#include "hip/hip_runtime.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"

//#define __device

__device__ double conv(freq_context *CUDA_LCC,int nc,int tmpl,int tmph,int brtmpl,int brtmph,int brtmpl2,int brtmph2,int do_dyda)
{
   int i, j,k;
   __shared__ double res[CURVE1LAST_BLOCK_DIM];
   double tmp,dtmp;

   tmp=0;
	j=blockIdx.x*(CUDA_Numfac1)+brtmpl;
   for (i = brtmpl; i <= brtmph; i++,j++)
   {
	   int2 bfr;
		bfr=__ldg(&texArea[j]);
		tmp += __hiloint2double(bfr.y,bfr.x) * CUDA_Nor[nc][i];
   }
   /* Reproduce the CUDA_BLOCK_DIM-thread reduction with fewer threads: this one
      also sums the partial of logical thread (tid + CURVE1LAST_BLOCK_DIM) and adds
      it first - exactly the tree's first step - so every rounding is unchanged. */
   {
      double tmp2 = 0;
      int j2 = blockIdx.x*(CUDA_Numfac1)+brtmpl2;
      for (i = brtmpl2; i <= brtmph2; i++, j2++)
      {
         int2 bfr2;
         bfr2=__ldg(&texArea[j2]);
         tmp2 += __hiloint2double(bfr2.y,bfr2.x) * CUDA_Nor[nc][i];
      }
      tmp = tmp + tmp2;
   }
   res[threadIdx.x]=tmp;
   __syncthreads();
//parallel reduction
	k=CURVE1LAST_BLOCK_DIM>>1;
	while (k>1)
	{
		if (threadIdx.x<k) res[threadIdx.x]+=res[threadIdx.x+k];
		k=k>>1;
		__syncthreads();
	}
	if (threadIdx.x==0)
	{
		tmp = res[0]+res[1];
	}
//parallel reduction end
	__syncthreads();

	int m=blockIdx.x*CUDA_Dg_block+tmpl*(CUDA_Numfac1);
	for (j = tmpl; do_dyda && j <= tmph; j++,m+=(CUDA_Numfac1))
	{
			dtmp=0;
			if (j<=CUDA_Ncoef)
			{
				int mm=m+1;
				for (i = 1; i <= CUDA_Numfac; i++,mm++)
				{
					const double gfi = (*CUDA_LCC).gfac[i];
					dtmp += CUDA_Darea[i] * (gfi * CUDA_Dsph[i][j]) * CUDA_Nor[nc][i];
				}
			}
			(*CUDA_LCC).dyda[j]=dtmp;
	}
	__syncthreads();

   return (tmp);
}
