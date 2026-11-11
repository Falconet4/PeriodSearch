/* Curvature function (and hence facet area) from Laplace series 

   8.11.2006
*/

#include <hip/hip_runtime.h>
#include <math.h>
#include "globals_CUDA.h"


__device__ void curv(freq_context *CUDA_LCC,double cg[],int brtmpl,int brtmph)
{
   int i, m, n, l, k;
   
   double fsum,
          g;


   for (i = brtmpl; i <= brtmph; i += blockDim.x)
   {
      g = 0;
      n = 0;
      for (m = 0; m <= CUDA_Mmax; m++)
        for (l = m; l <= CUDA_Lmax; l++)
		{
            n++;
            fsum = cg[n] * CUDA_Fc[m][i];
            if (m != 0) 
            {
	       n++;
               fsum = fsum + cg[n] * CUDA_Fs[m][i];
            }
            g = g + CUDA_Pleg[l][m][i] * fsum;
        }
      g = exp(g);
      const double area_i = CUDA_Darea[i] * g;
      (*CUDA_LCC).Area[i] = area_i;
      (*CUDA_LCC).gfac[i] = g;

   }
   __syncthreads();
}
