/* slighly changed code from Numerical Recipes
   converted from Mikko's fortran code

   8.11.2006
*/

#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"


/* comment the following line if no YORP */
/*#define YORP*/

__device__ void mrqcof_start(freq_context *CUDA_LCC, double a[],
	      double *alpha, double beta[])
{
   int j,k;
//
    int brtmph,brtmpl;
	brtmpl = 1 + threadIdx.x;
	brtmph = CUDA_Numfac;

   /* N.B. curv and blmatrix called outside bright
      because output same for all points */
   curv(CUDA_LCC,a,brtmpl,brtmph);

   if (threadIdx.x==0)
   {
//   #ifdef YORP
//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
  // #else
      blmatrix(CUDA_LCC,a[CUDA_ma-4-CUDA_Nphpar],a[CUDA_ma-3-CUDA_Nphpar]);
//   #endif
	   (*CUDA_LCC).trial_chisq = 0;
	   (*CUDA_LCC).np = 0;
	   (*CUDA_LCC).np1 = 0;
	   (*CUDA_LCC).np2 = 0;
	   (*CUDA_LCC).ave = 0;
   }

    brtmph=CUDA_mfit/CUDA_BLOCK_DIM;
	if(CUDA_mfit%CUDA_BLOCK_DIM) brtmph++;
	brtmpl=threadIdx.x*brtmph;
	brtmph=brtmpl+brtmph;
	if (brtmph>CUDA_mfit) brtmph=CUDA_mfit;
	brtmpl++;

   /* Linear zeroing of the whole row block: coalesced, no per-element divide.
      It also clears the upper triangle, which nothing reads before mrqcof_end's
      mirror overwrites it.  Rows 0..mfit inclusive, hence the +1. */
   {
      const int tot = (CUDA_mfit + 1) * CUDA_mfit1;
      for (int e = threadIdx.x; e < tot; e += CUDA_BLOCK_DIM)
         alpha[e] = 0;
      for (int jj = 1 + threadIdx.x; jj <= CUDA_mfit; jj += CUDA_BLOCK_DIM)
         beta[jj] = 0;
   }

   __syncthreads(); //pro jistotu
}

__device__ double mrqcof_end(freq_context *CUDA_LCC,double *alpha)
{
   int j,k;

   {
      const int total = CUDA_mfit * CUDA_mfit;
      for (int e = threadIdx.x; e < total; e += blockDim.x)
      {
         const int jj = e / CUDA_mfit + 1;
         const int kk = e - (jj - 1) * CUDA_mfit + 1;
         if (kk < jj) alpha[kk*(CUDA_mfit1)+jj] = alpha[jj*(CUDA_mfit1)+kk];
      }
   }

   return (*CUDA_LCC).trial_chisq;
}

#if 0 /* legacy, unlaunched */
__device__ void mrqcof_matrix(freq_context *CUDA_LCC, double a[], int Lpoints)
{
   matrix_neo(CUDA_LCC, a,(*CUDA_LCC).np, Lpoints);
}
#endif


#if 0 /* legacy, unlaunched */
__device__ void mrqcof_curve1(freq_context *CUDA_LCC, double a[],
	      double *alpha, double beta[],int Inrel,int Lpoints)
{
	int l,k,jp, lnp,Lpoints1=Lpoints+1;
   double lave;
   __shared__ double tmave[CUDA_BLOCK_DIM];

   lnp=(*CUDA_LCC).np;
   lave=(*CUDA_LCC).ave;
//precalc thread boundaries
    int brtmph,brtmpl;
	brtmph=Lpoints/CUDA_BLOCK_DIM;
	if(Lpoints%CUDA_BLOCK_DIM) brtmph++;
	brtmpl=threadIdx.x*brtmph;
	brtmph=brtmpl+brtmph;
	if (brtmph>Lpoints) brtmph=Lpoints;
	brtmpl++;
//

   for (jp = brtmpl; jp <= brtmph; jp++)
    bright(CUDA_LCC,a,jp,Lpoints1,Inrel);

   __syncthreads();

  if (Inrel == 1) {
    int tmph,tmpl;
	tmph=CUDA_ma/CUDA_BLOCK_DIM;
	if(CUDA_ma%CUDA_BLOCK_DIM) tmph++;
	tmpl=threadIdx.x*tmph;
	tmph=tmpl+tmph;
	if (tmph>CUDA_ma) tmph=CUDA_ma;
	tmpl++;
	if (tmpl==1) tmpl++;

	  int ixx;
	  ixx=tmpl*Lpoints1;
	  for (l=tmpl; l <= tmph; l++)
		{
	  //jp==1
			ixx++;
			(*CUDA_LCC).dave[l] = (*CUDA_LCC).dytemp[ixx];
      //jp>=2
			ixx++;
		   for (jp = 2; jp <= Lpoints; jp++,ixx++)
		   {
			(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dytemp[ixx];
		   }

	  }
		tmave[threadIdx.x] = 0;
	   for (jp = brtmpl; jp <= brtmph; jp++) tmave[threadIdx.x] += (*CUDA_LCC).ytemp[jp];
	   __syncthreads();
//parallel reduction
	   k=CUDA_BLOCK_DIM>>1;
	   while (k>1)
	   {
		   if (threadIdx.x<k) tmave[threadIdx.x]+=tmave[threadIdx.x+k];
		   k=k>>1;
		   __syncthreads();
	   }
	   if (threadIdx.x==0) lave=tmave[0]+tmave[1];
//parallel reduction end

  }
	  if (threadIdx.x==0)
	  {
	   (*CUDA_LCC).np=lnp+Lpoints;
	   (*CUDA_LCC).ave=lave;
	  }
}
#endif


__device__ void mrqcof_curve1_last(freq_context *CUDA_LCC, double a[],
	      double *alpha, double beta[],int Inrel,int Lpoints)
{
	int l,jp, lnp;
   double ymod, lave;

   lnp=(*CUDA_LCC).np;
   //
   if (threadIdx.x==0)
   {
	   if (Inrel == 1) /* is the LC relative? */
	   {
		  lave = 0;
		  for (l = 1; l <= CUDA_ma; l++)
		  (*CUDA_LCC).dave[l]=0;
	   }
	   else
		  lave=(*CUDA_LCC).ave;
   }
//precalc thread boundaries
    int tmph,tmpl;
	tmph=CUDA_ma/CUDA_BLOCK_DIM;
	if(CUDA_ma%CUDA_BLOCK_DIM) tmph++;
	tmpl=threadIdx.x*tmph;
	tmph=tmpl+tmph;
	if (tmph>CUDA_ma) tmph=CUDA_ma;
	tmpl++;
//
    int brtmph,brtmpl;
	/* facet split uses the logical CUDA_BLOCK_DIM partition whatever the launch width,
	   so the reduction stays the one the 128-thread version produced */
	brtmph=CUDA_Numfac/CUDA_BLOCK_DIM;
	if(CUDA_Numfac%CUDA_BLOCK_DIM) brtmph++;
	brtmpl=threadIdx.x*brtmph;
	brtmph=brtmpl+brtmph;
	if (brtmph>CUDA_Numfac) brtmph=CUDA_Numfac;
	brtmpl++;
	const int brtmpl2 = brtmpl + CURVE1LAST_BLOCK_DIM*((CUDA_Numfac+CUDA_BLOCK_DIM-1)/CUDA_BLOCK_DIM);
	int brtmph2 = brtmph + CURVE1LAST_BLOCK_DIM*((CUDA_Numfac+CUDA_BLOCK_DIM-1)/CUDA_BLOCK_DIM);
	if (brtmph2>CUDA_Numfac) brtmph2=CUDA_Numfac;

	__syncthreads();


      /* Four points per pass.  The thread->coefficient mapping and the
	 ascending facet order are exactly as before, so every dyda / ymod value
	 is bit-identical; the gain is that Darea[i], gfac[i] and Dsph[i][l] are
	 loaded once for four points instead of once per point. */
      for (int jp0 = 1; jp0 <= Lpoints; jp0 += 4)
      {
	 int nq = Lpoints - jp0 + 1;
	 if (nq > 4) nq = 4;
	 lnp += nq;

	 double ymv[4];
	 for (int q = 0; q < nq; q++)
	    ymv[q] = conv(CUDA_LCC,jp0+q-1,tmpl,tmph,brtmpl,brtmph,brtmpl2,brtmph2,0);

	 if (nq == 3)
	 {
	    for (l=tmpl; l <= tmph; l++)
	    {
	       double d0=0, d1=0, d2=0;
	       if (l <= CUDA_Ncoef)
		  for (int i = 1; i <= CUDA_Numfac; i++)
		  {
		     const double da = CUDA_Darea[i];
		     const double gd = (*CUDA_LCC).gfac[i] * CUDA_Dsph[i][l];
		     d0 += da * gd * CUDA_Nor[jp0-1][i];
		     d1 += da * gd * CUDA_Nor[jp0][i];
		     d2 += da * gd * CUDA_Nor[jp0+1][i];
		  }
	       (*CUDA_LCC).dytemp[(jp0+0)*(CUDA_ma+1) + l] = d0;
	       (*CUDA_LCC).dytemp[(jp0+1)*(CUDA_ma+1) + l] = d1;
	       (*CUDA_LCC).dytemp[(jp0+2)*(CUDA_ma+1) + l] = d2;
	       if (Inrel == 1)
		  (*CUDA_LCC).dave[l] = (((*CUDA_LCC).dave[l] + d0) + d1) + d2;
	    }
	 }
	 else
	 for (l=tmpl; l <= tmph; l++)
	 {
	    double d0=0, d1=0, d2=0, d3=0;
	    if (l <= CUDA_Ncoef)
	       for (int i = 1; i <= CUDA_Numfac; i++)
	       {
		  const double da = CUDA_Darea[i];
		  const double gd = (*CUDA_LCC).gfac[i] * CUDA_Dsph[i][l];
		  d0 += da * gd * CUDA_Nor[jp0-1][i];
		  if (nq > 1) d1 += da * gd * CUDA_Nor[jp0][i];
		  if (nq > 2) d2 += da * gd * CUDA_Nor[jp0+1][i];
		  if (nq > 3) d3 += da * gd * CUDA_Nor[jp0+2][i];
	       }
	    const double dv[4] = {d0,d1,d2,d3};
	    for (int q = 0; q < nq; q++)
	    {
	       (*CUDA_LCC).dytemp[(jp0+q)*(CUDA_ma+1) + l] = dv[q];
	       if (Inrel == 1)
		  (*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + dv[q];
	    }
	 }

	 if (threadIdx.x==0)
	    for (int q = 0; q < nq; q++)
	    {
	       (*CUDA_LCC).ytemp[jp0+q] = ymv[q];
	       if (Inrel == 1)
		  lave = lave + ymv[q];
	    }
	 /* save lightcurves */
	 __syncthreads();

/*         if ((*CUDA_LCC).Lastcall == 1) always ==0
			 (*CUDA_LCC).Yout[np] = ymod;*/
      } /* jp, lpoints */
	 if (threadIdx.x==0)
	 {
		  (*CUDA_LCC).np=lnp;
		  (*CUDA_LCC).ave=lave;
	 }
}

