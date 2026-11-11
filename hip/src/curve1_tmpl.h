#pragma once
#ifndef PS_GEO_N
#define PS_GEO_N 26
#ifndef PS_GEO_BATCH
#define PS_GEO_BATCH 16
#endif
#endif
/* Hot per-lightcurve routines (matrix setup, brightness and its derivatives,
   and the alpha/beta accumulation) as they are used by the fused kernels. */
#include <cmath>
#include <hip/hip_runtime.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
#include "curve1_warp.h"


template<int ISTART> __device__ double brightT_impl(freq_context *CUDA_LCC, double cg[],int jp,int Lpoints1,int Inrel,const double *sBl,const double *sDb,double clv,double clsv);
__device__ __forceinline__ double brightT(freq_context *CUDA_LCC, double cg[],int jp,int Lpoints1,int Inrel,const double *sBl,const double *sDb,double clv,double clsv)
{
  return Inrel ? brightT_impl<2>(CUDA_LCC,cg,jp,Lpoints1,Inrel,sBl,sDb,clv,clsv)
               : brightT_impl<1>(CUDA_LCC,cg,jp,Lpoints1,Inrel,sBl,sDb,clv,clsv);
}

template<int WM> __device__ void mrqcof_curve1T(freq_context *CUDA_LCC, double a[],
	      double *alpha, double beta[],int Inrel,int Lpoints)
{
	int l,k,jp, lnp,Lpoints1=Lpoints+1;
   double lave;
   __shared__ double tmave[CUDA_BLOCK_DIM];
   __shared__ double sh_Bl[9], sh_Db[18];
   __shared__ __attribute__((aligned(16))) double sgeo[WM ? 4 : 1][PS_GEO_BATCH*PS_GEO_N];
   __shared__ double swcA[WM ? 4 : 1][32], swcB[WM ? 4 : 1][32];
   __shared__ int sfc[WM ? 4 : 1][32];
   if (threadIdx.x == 0)
   {
      sh_Bl[0]=(*CUDA_LCC).Blmat[1][1]; sh_Bl[1]=(*CUDA_LCC).Blmat[1][2]; sh_Bl[2]=(*CUDA_LCC).Blmat[1][3];
      sh_Bl[3]=(*CUDA_LCC).Blmat[2][1]; sh_Bl[4]=(*CUDA_LCC).Blmat[2][2]; sh_Bl[5]=(*CUDA_LCC).Blmat[2][3];
      sh_Bl[6]=(*CUDA_LCC).Blmat[3][1]; sh_Bl[7]=(*CUDA_LCC).Blmat[3][2]; sh_Bl[8]=(*CUDA_LCC).Blmat[3][3];
      sh_Db[0]=(*CUDA_LCC).Dblm[1][1][1]; sh_Db[1]=(*CUDA_LCC).Dblm[1][1][2]; sh_Db[2]=(*CUDA_LCC).Dblm[1][1][3];
      sh_Db[3]=(*CUDA_LCC).Dblm[1][2][1]; sh_Db[4]=(*CUDA_LCC).Dblm[1][2][2]; sh_Db[5]=(*CUDA_LCC).Dblm[1][2][3];
      sh_Db[6]=(*CUDA_LCC).Dblm[1][3][1]; sh_Db[7]=(*CUDA_LCC).Dblm[1][3][2]; sh_Db[8]=(*CUDA_LCC).Dblm[1][3][3];
      sh_Db[9]=(*CUDA_LCC).Dblm[2][1][1]; sh_Db[10]=(*CUDA_LCC).Dblm[2][1][2]; sh_Db[11]=(*CUDA_LCC).Dblm[2][1][3];
      sh_Db[12]=(*CUDA_LCC).Dblm[2][2][1]; sh_Db[13]=(*CUDA_LCC).Dblm[2][2][2]; sh_Db[14]=(*CUDA_LCC).Dblm[2][2][3];
      sh_Db[15]=(*CUDA_LCC).Dblm[2][3][1]; sh_Db[16]=(*CUDA_LCC).Dblm[2][3][2]; sh_Db[17]=(*CUDA_LCC).Dblm[2][3][3];
   }
   __syncthreads();

   lnp=(*CUDA_LCC).np;
   lave=(*CUDA_LCC).ave;
//precalc thread boundaries
    int brtmph = Lpoints;
	int brtmpl = 1 + threadIdx.x;
//

   const int ncoef_h = CUDA_ma;
   const double cl_h = exp(a[ncoef_h - 1]);
   const double cls_h = a[ncoef_h];
   /* Lockstep point loop: every thread performs the same number of iterations so
      that per-facet __syncthreads() inside brightT is safe (inactive lanes skip
      the work).  Bit-identical to the guarded loop it replaces. */
   if (WM)
     {
       {
         const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
         ps_brightT_warp(CUDA_LCC,a,Inrel,Lpoints,sh_Bl,sh_Db,cl_h,cls_h,wid,lane,
                         sgeo[wid],swcA[wid],swcB[wid],sfc[wid]);
       }
     }
   else
     {
       for (int jp2 = 1 + threadIdx.x; jp2 <= Lpoints; jp2 += CUDA_BLOCK_DIM)
         brightT(CUDA_LCC,a,jp2,Lpoints1,Inrel,sh_Bl,sh_Db,cl_h,cls_h);
     }

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

	  const int dstride = CUDA_ma + 1;
	  for (l=tmpl; l <= tmph; l++)
		{
			double dsum = (*CUDA_LCC).dytemp[1 * dstride + l];
			for (jp = 2; jp <= Lpoints; jp++)
				dsum = dsum + (*CUDA_LCC).dytemp[jp * dstride + l];
			(*CUDA_LCC).dave[l] = dsum;
	  }
		tmave[threadIdx.x] = 0;
	   {
	     int ohi = Lpoints / CUDA_BLOCK_DIM; if (Lpoints % CUDA_BLOCK_DIM) ohi++;
	     int olo = threadIdx.x * ohi; ohi = olo + ohi;
	     if (ohi > Lpoints) ohi = Lpoints; olo++;
	     for (jp = olo; jp <= ohi; jp++) tmave[threadIdx.x] += (*CUDA_LCC).ytemp[jp];
	   }
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


template<int ISTART>
__device__ double brightT_impl(freq_context *CUDA_LCC, double cg[],int jp,int Lpoints1,int Inrel,const double *sBl,const double *sDb,double clv,double clsv)
{
   int ncoef0,ncoef, i, j;
   double dnom, s,Scale,dphp1,dphp2,dphp3;
   double e_1,e_2,e_3,e0_1,e0_2,e0_3,de[4][4],de0[4][4];

   ncoef0 = CUDA_ncoef0;//ncoef - 2 - CUDA_Nphpar;
   ncoef = CUDA_ma;
   const int dstride = ncoef + 1;
   const double cl = clv;   /* Lambert (hoisted) */
   const double cls = clsv; /* Lommel-Seeliger (hoisted) */

  /* matrix from neo */
		/* derivatives */

#define Bl11 sBl[0]
#define Bl12 sBl[1]
#define Bl13 sBl[2]
#define Bl21 sBl[3]
#define Bl22 sBl[4]
#define Bl23 sBl[5]
#define Bl31 sBl[6]
#define Bl32 sBl[7]
#define Bl33 sBl[8]
#define Db111 sDb[0]
#define Db112 sDb[1]
#define Db113 sDb[2]
#define Db121 sDb[3]
#define Db122 sDb[4]
#define Db123 sDb[5]
#define Db131 sDb[6]
#define Db132 sDb[7]
#define Db133 sDb[8]
#define Db211 sDb[9]
#define Db212 sDb[10]
#define Db213 sDb[11]
#define Db221 sDb[12]
#define Db222 sDb[13]
#define Db223 sDb[14]
#define Db231 sDb[15]
#define Db232 sDb[16]
#define Db233 sDb[17]

	/* ---- fused matrix_neo: per-point rotation matrices and phase function ---- */
	const int lnp = (*CUDA_LCC).np + jp;
	const int estr = MAX_N_OBS + 1;
	const double ee_1 = CUDA_ee[0*estr + lnp], ee0_1 = CUDA_ee0[0*estr + lnp];
	const double ee_2 = CUDA_ee[1*estr + lnp], ee0_2 = CUDA_ee0[1*estr + lnp];
	const double ee_3 = CUDA_ee[2*estr + lnp], ee0_3 = CUDA_ee0[2*estr + lnp];
	const double tpoint = CUDA_tim[lnp];

	double ee_dot = ee_1*ee0_1 + ee_2*ee0_2 + ee_3*ee0_3;
	ee_dot = ee_dot > 1.0 ? 1.0 : (ee_dot < -1.0 ? -1.0 : ee_dot);
	const double alpha = acos(ee_dot);
	const double fph = exp(-alpha / cg[CUDA_ncoef0+2]);
	dphp1 = fph;
	dphp2 = cg[CUDA_ncoef0+1] * fph * alpha / (cg[CUDA_ncoef0+2] * cg[CUDA_ncoef0+2]);
	dphp3 = alpha;
	Scale = 1 + cg[CUDA_ncoef0+1] * fph + cg[CUDA_ncoef0+3] * alpha;

	double ff = cg[CUDA_ncoef0] * tpoint + CUDA_Phi_0;
	ff = fmod(ff, 2 * PI);
	double cf, sf;
	sincos(ff, &sf, &cf);
	double tmat, pom, pom0;

	tmat = cf*Bl11 + sf*Bl21 + 0*Bl31; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = cf*Bl12 + sf*Bl22 + 0*Bl32; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = cf*Bl13 + sf*Bl23 + 0*Bl33;
	e_1 = (pom + tmat*ee_3);  e0_1 = (pom0 + tmat*ee0_3);

	tmat = (-sf)*Bl11 + cf*Bl21 + 0*Bl31; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = (-sf)*Bl12 + cf*Bl22 + 0*Bl32; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = (-sf)*Bl13 + cf*Bl23 + 0*Bl33;
	e_2 = (pom + tmat*ee_3);  e0_2 = (pom0 + tmat*ee0_3);

	tmat = 0*Bl11 + 0*Bl21 + 1*Bl31; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = 0*Bl12 + 0*Bl22 + 1*Bl32; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = 0*Bl13 + 0*Bl23 + 1*Bl33;
	e_3 = (pom + tmat*ee_3);  e0_3 = (pom0 + tmat*ee0_3);

	tmat = cf*Db111 + sf*Db121 + 0*Db131; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = cf*Db112 + sf*Db122 + 0*Db132; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = cf*Db113 + sf*Db123 + 0*Db133;
	de[1][1] = (pom + tmat*ee_3);  de0[1][1] = (pom0 + tmat*ee0_3);

	tmat = cf*Db211 + sf*Db221 + 0*Db231; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = cf*Db212 + sf*Db222 + 0*Db232; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = cf*Db213 + sf*Db223 + 0*Db233;
	de[1][2] = (pom + tmat*ee_3);  de0[1][2] = (pom0 + tmat*ee0_3);

	tmat = (-tpoint*sf)*Bl11 + (tpoint*cf)*Bl21 + 0*Bl31; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = (-tpoint*sf)*Bl12 + (tpoint*cf)*Bl22 + 0*Bl32; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = (-tpoint*sf)*Bl13 + (tpoint*cf)*Bl23 + 0*Bl33;
	de[1][3] = (pom + tmat*ee_3);  de0[1][3] = (pom0 + tmat*ee0_3);

	tmat = -sf*Db111 + cf*Db121 + 0*Db131; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = -sf*Db112 + cf*Db122 + 0*Db132; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = -sf*Db113 + cf*Db123 + 0*Db133;
	de[2][1] = (pom + tmat*ee_3);  de0[2][1] = (pom0 + tmat*ee0_3);

	tmat = -sf*Db211 + cf*Db221 + 0*Db231; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = -sf*Db212 + cf*Db222 + 0*Db232; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = -sf*Db213 + cf*Db223 + 0*Db233;
	de[2][2] = (pom + tmat*ee_3);  de0[2][2] = (pom0 + tmat*ee0_3);

	tmat = (-tpoint*cf)*Bl11 + (-tpoint*sf)*Bl21 + 0*Bl31; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = (-tpoint*cf)*Bl12 + (-tpoint*sf)*Bl22 + 0*Bl32; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = (-tpoint*cf)*Bl13 + (-tpoint*sf)*Bl23 + 0*Bl33;
	de[2][3] = (pom + tmat*ee_3);  de0[2][3] = (pom0 + tmat*ee0_3);

	tmat = 0*Db111 + 0*Db121 + 1*Db131; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = 0*Db112 + 0*Db122 + 1*Db132; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = 0*Db113 + 0*Db123 + 1*Db133;
	de[3][1] = (pom + tmat*ee_3);  de0[3][1] = (pom0 + tmat*ee0_3);

	tmat = 0*Db211 + 0*Db221 + 1*Db231; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = 0*Db212 + 0*Db222 + 1*Db232; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = 0*Db213 + 0*Db223 + 1*Db233;
	de[3][2] = (pom + tmat*ee_3);  de0[3][2] = (pom0 + tmat*ee0_3);

	tmat = 0*Bl11 + 0*Bl21 + 0*Bl31; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = 0*Bl12 + 0*Bl22 + 0*Bl32; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = 0*Bl13 + 0*Bl23 + 0*Bl33;
	de[3][3] = (pom + tmat*ee_3);  de0[3][3] = (pom0 + tmat*ee0_3);

   /* Directions (and ders.) in the rotating system */

   //
   /*Integrated brightness (phase coeff. used later) */
   double br,ar,tmp1,tmp2,tmp3,tmp4,tmp5;
   const double fcl = cl, fcls = cls;
   double acc0[25], acc1[25];
   int2 bfr;

	br=0;
	tmp1=0;
	tmp2=0;
	tmp3=0;
	tmp4=0;
	tmp5=0;
	const int iStart = ISTART;
#pragma unroll
	for (int p = 0; p < 25; p++) { acc0[p] = 0.0f; acc1[p] = 0.0f; }
	j=blockIdx.x*(CUDA_Numfac1)+1;
	for (i = 1; i <= CUDA_Numfac; i++,j++)
   {
		  const double fn0 = CUDA_Nor[0][i], fn1 = CUDA_Nor[1][i], fn2 = CUDA_Nor[2][i];
		  const double flmu  = e_1 * fn0 + e_2 * fn1 + e_3 * fn2;
		  const double flmu0 = e0_1 * fn0 + e0_2 * fn1 + e0_3 * fn2;
		  if((flmu > TINY) && (flmu0 > TINY))
		  {
			const double fdnom = flmu + flmu0;
			const double finv = 1.0 / fdnom;
			const double fs = flmu * flmu0 * (fcl + fcls * finv);
			ar = (*CUDA_LCC).Area[i];
			br+= ar * fs;
			//
			/* accumulate d/dg in registers during the facet sweep */
			{
				const double wgt = ar * fs;
				if (iStart == 2)
				{
#pragma unroll
					for (int p = 0; p < 25; p++)
					{
						const int idx = 2 + 2 * p;
						const double2 d2 = *reinterpret_cast<const double2*>(&CUDA_Dsph[i][idx]);
						acc0[p] += wgt * d2.x;
						acc1[p] += wgt * d2.y;
					}
				}
				else
				{
#pragma unroll
					for (int p = 0; p < 25; p++)
					{
						const int idx = 1 + 2 * p;
						const double2 d2 = *reinterpret_cast<const double2*>(&CUDA_Dsph[i][idx]);
						acc0[p] += wgt * d2.x;
						acc1[p] += wgt * d2.y;
					}
				}
			}
			//
			const double flmu0_dnom = flmu0 * finv;
			const double fdsmu = fcls * (flmu0_dnom * flmu0_dnom) + fcl * flmu0;
			const double flmu_dnom = flmu * finv;
			const double fdsmu0 = fcls * (flmu_dnom * flmu_dnom) + fcl * flmu;

			const double fs1  = de[1][1] * fn0 + de[2][1] * fn1 + de[3][1] * fn2;
			const double fs10 = de0[1][1] * fn0 + de0[2][1] * fn1 + de0[3][1] * fn2;
			tmp1 += ar * (fdsmu * fs1 + fdsmu0 * fs10);
			const double fs2  = de[1][2] * fn0 + de[2][2] * fn1 + de[3][2] * fn2;
			const double fs20 = de0[1][2] * fn0 + de0[2][2] * fn1 + de0[3][2] * fn2;
			tmp2 += ar * (fdsmu * fs2 + fdsmu0 * fs20);
			const double fs3  = de[1][3] * fn0 + de[2][3] * fn1 + de[3][3] * fn2;
			const double fs30 = de0[1][3] * fn0 + de0[2][3] * fn1 + de0[3][3] * fn2;
			tmp3 += ar * (fdsmu * fs3 + fdsmu0 * fs30);

			tmp4 += ar * (flmu * flmu0);
			tmp5 += ar * (flmu * flmu0 * finv);
		}
	}
		const int dbase = jp * dstride;
		i = dbase + (ncoef0 - 3 + 1);
	   /* Ders. of brightness w.r.t. rotation parameters */
	   (*CUDA_LCC).dytemp[i]= Scale * tmp1;
	   i++;
	   (*CUDA_LCC).dytemp[i] = Scale * tmp2;
	   i++;
	   (*CUDA_LCC).dytemp[i] = Scale * tmp3;
	   i++;
	   /* Ders. of br. w.r.t. phase function params. */
	   (*CUDA_LCC).dytemp[i] = br * dphp1;
	   i++;
	   (*CUDA_LCC).dytemp[i] = br * dphp2;
	   i++;
	   (*CUDA_LCC).dytemp[i] = br * dphp3;

	   /* Ders. of br. w.r.t. cl, cls */
	   (*CUDA_LCC).dytemp[dbase + (ncoef - 1)] = Scale * tmp4 * cl;
	   (*CUDA_LCC).dytemp[dbase + ncoef] = Scale * tmp5;
	   /* Scaled brightness */
	  (*CUDA_LCC).ytemp[jp]=br * Scale;

	ncoef0-=3;
	int d = dbase + iStart;
	int d1 = d + 1;
   /* Derivatives of brightness w.r.t. g-coeffs (accumulated above) */
#pragma unroll
   for (int p = 0; p < 25; p++)
   {
	   const int idx = iStart + 2 * p;
	   if (idx <= ncoef0)
	   {
		   (*CUDA_LCC).dytemp[d + 2 * p] = Scale * acc0[p];
		   if (idx + 1 <= ncoef0)
			   (*CUDA_LCC).dytemp[d1 + 2 * p] = Scale * acc1[p];
	   }
   }
#undef Bl11
#undef Bl12
#undef Bl13
#undef Bl21
#undef Bl22
#undef Bl23
#undef Bl31
#undef Bl32
#undef Bl33
#undef Db111
#undef Db112
#undef Db113
#undef Db121
#undef Db122
#undef Db123
#undef Db131
#undef Db132
#undef Db133
#undef Db211
#undef Db212
#undef Db213
#undef Db221
#undef Db222
#undef Db223
#undef Db231
#undef Db232
#undef Db233
   return(0);
}


/* ---- original Fast FP32 Curve1 (tail only, PS_FP32_TAIL) ---- */
template<int ISTART> __device__ double brightT_impl_fp(freq_context *CUDA_LCC, double cg[],int jp,int Lpoints1,int Inrel,const double *sBl,const double *sDb,double clv,double clsv);

__device__ __forceinline__ double brightT_fp(freq_context *CUDA_LCC, double cg[],int jp,int Lpoints1,int Inrel,const double *sBl,const double *sDb,double clv,double clsv)
{
  return Inrel ? brightT_impl_fp<2>(CUDA_LCC,cg,jp,Lpoints1,Inrel,sBl,sDb,clv,clsv)
               : brightT_impl_fp<1>(CUDA_LCC,cg,jp,Lpoints1,Inrel,sBl,sDb,clv,clsv);
}

__device__ void mrqcof_curve1T_fp(freq_context *CUDA_LCC, double a[],
	      double *alpha, double beta[],int Inrel,int Lpoints)
{
	int l,k,jp, lnp,Lpoints1=Lpoints+1;
   double lave;
   __shared__ double tmave[CUDA_BLOCK_DIM];
   __shared__ double sh_Bl[9], sh_Db[18];
   if (threadIdx.x == 0)
   {
      sh_Bl[0]=(*CUDA_LCC).Blmat[1][1]; sh_Bl[1]=(*CUDA_LCC).Blmat[1][2]; sh_Bl[2]=(*CUDA_LCC).Blmat[1][3];
      sh_Bl[3]=(*CUDA_LCC).Blmat[2][1]; sh_Bl[4]=(*CUDA_LCC).Blmat[2][2]; sh_Bl[5]=(*CUDA_LCC).Blmat[2][3];
      sh_Bl[6]=(*CUDA_LCC).Blmat[3][1]; sh_Bl[7]=(*CUDA_LCC).Blmat[3][2]; sh_Bl[8]=(*CUDA_LCC).Blmat[3][3];
      sh_Db[0]=(*CUDA_LCC).Dblm[1][1][1]; sh_Db[1]=(*CUDA_LCC).Dblm[1][1][2]; sh_Db[2]=(*CUDA_LCC).Dblm[1][1][3];
      sh_Db[3]=(*CUDA_LCC).Dblm[1][2][1]; sh_Db[4]=(*CUDA_LCC).Dblm[1][2][2]; sh_Db[5]=(*CUDA_LCC).Dblm[1][2][3];
      sh_Db[6]=(*CUDA_LCC).Dblm[1][3][1]; sh_Db[7]=(*CUDA_LCC).Dblm[1][3][2]; sh_Db[8]=(*CUDA_LCC).Dblm[1][3][3];
      sh_Db[9]=(*CUDA_LCC).Dblm[2][1][1]; sh_Db[10]=(*CUDA_LCC).Dblm[2][1][2]; sh_Db[11]=(*CUDA_LCC).Dblm[2][1][3];
      sh_Db[12]=(*CUDA_LCC).Dblm[2][2][1]; sh_Db[13]=(*CUDA_LCC).Dblm[2][2][2]; sh_Db[14]=(*CUDA_LCC).Dblm[2][2][3];
      sh_Db[15]=(*CUDA_LCC).Dblm[2][3][1]; sh_Db[16]=(*CUDA_LCC).Dblm[2][3][2]; sh_Db[17]=(*CUDA_LCC).Dblm[2][3][3];
   }
   __syncthreads();

   lnp=(*CUDA_LCC).np;
   lave=(*CUDA_LCC).ave;
//precalc thread boundaries
    int brtmph = Lpoints;
	int brtmpl = 1 + threadIdx.x;
//

   const int ncoef_h = CUDA_ma;
   const double cl_h = exp(a[ncoef_h - 1]);
   const double cls_h = a[ncoef_h];
   /* Lockstep point loop: every thread performs the same number of iterations so
      that per-facet __syncthreads() inside brightT is safe (inactive lanes skip
      the work).  Bit-identical to the guarded loop it replaces. */
   for (int jp2 = 1 + threadIdx.x; jp2 <= Lpoints; jp2 += CUDA_BLOCK_DIM)
     brightT_fp(CUDA_LCC,a,jp2,Lpoints1,Inrel,sh_Bl,sh_Db,cl_h,cls_h);

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

	  const int dstride = CUDA_ma + 1;
	  for (l=tmpl; l <= tmph; l++)
		{
			double dsum = (*CUDA_LCC).dytemp[1 * dstride + l];
			for (jp = 2; jp <= Lpoints; jp++)
				dsum = dsum + (*CUDA_LCC).dytemp[jp * dstride + l];
			(*CUDA_LCC).dave[l] = dsum;
	  }
		tmave[threadIdx.x] = 0;
	   {
	     int ohi = Lpoints / CUDA_BLOCK_DIM; if (Lpoints % CUDA_BLOCK_DIM) ohi++;
	     int olo = threadIdx.x * ohi; ohi = olo + ohi;
	     if (ohi > Lpoints) ohi = Lpoints; olo++;
	     for (jp = olo; jp <= ohi; jp++) tmave[threadIdx.x] += (*CUDA_LCC).ytemp[jp];
	   }
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



template<int ISTART>
__device__ double brightT_impl_fp(freq_context *CUDA_LCC, double cg[],int jp,int Lpoints1,int Inrel,const double *sBl,const double *sDb,double clv,double clsv)
{
   int ncoef0,ncoef, i, j;
   double dnom, s,Scale,dphp1,dphp2,dphp3;
   float e_1,e_2,e_3,e0_1,e0_2,e0_3,de[4][4],de0[4][4];

   ncoef0 = CUDA_ncoef0;//ncoef - 2 - CUDA_Nphpar;
   ncoef = CUDA_ma;
   const int dstride = ncoef + 1;
   const double cl = clv;   /* Lambert (hoisted) */
   const double cls = clsv; /* Lommel-Seeliger (hoisted) */

  /* matrix from neo */
		/* derivatives */

#define Bl11 sBl[0]
#define Bl12 sBl[1]
#define Bl13 sBl[2]
#define Bl21 sBl[3]
#define Bl22 sBl[4]
#define Bl23 sBl[5]
#define Bl31 sBl[6]
#define Bl32 sBl[7]
#define Bl33 sBl[8]
#define Db111 sDb[0]
#define Db112 sDb[1]
#define Db113 sDb[2]
#define Db121 sDb[3]
#define Db122 sDb[4]
#define Db123 sDb[5]
#define Db131 sDb[6]
#define Db132 sDb[7]
#define Db133 sDb[8]
#define Db211 sDb[9]
#define Db212 sDb[10]
#define Db213 sDb[11]
#define Db221 sDb[12]
#define Db222 sDb[13]
#define Db223 sDb[14]
#define Db231 sDb[15]
#define Db232 sDb[16]
#define Db233 sDb[17]

	/* ---- fused matrix_neo: per-point rotation matrices and phase function ---- */
	const int lnp = (*CUDA_LCC).np + jp;
	const int estr = MAX_N_OBS + 1;
	const double ee_1 = CUDA_ee[0*estr + lnp], ee0_1 = CUDA_ee0[0*estr + lnp];
	const double ee_2 = CUDA_ee[1*estr + lnp], ee0_2 = CUDA_ee0[1*estr + lnp];
	const double ee_3 = CUDA_ee[2*estr + lnp], ee0_3 = CUDA_ee0[2*estr + lnp];
	const double tpoint = CUDA_tim[lnp];

	double ee_dot = ee_1*ee0_1 + ee_2*ee0_2 + ee_3*ee0_3;
	ee_dot = ee_dot > 1.0 ? 1.0 : (ee_dot < -1.0 ? -1.0 : ee_dot);
	const double alpha = acos(ee_dot);
	const double fph = exp(-alpha / cg[CUDA_ncoef0+2]);
	dphp1 = fph;
	dphp2 = cg[CUDA_ncoef0+1] * fph * alpha / (cg[CUDA_ncoef0+2] * cg[CUDA_ncoef0+2]);
	dphp3 = alpha;
	Scale = 1 + cg[CUDA_ncoef0+1] * fph + cg[CUDA_ncoef0+3] * alpha;

	const double ff = cg[CUDA_ncoef0] * tpoint + CUDA_Phi_0;
	double cf, sf;
	sincos(ff, &sf, &cf);
	double tmat, pom, pom0;

	tmat = cf*Bl11 + sf*Bl21 + 0*Bl31; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = cf*Bl12 + sf*Bl22 + 0*Bl32; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = cf*Bl13 + sf*Bl23 + 0*Bl33;
	e_1 = (float)(pom + tmat*ee_3);  e0_1 = (float)(pom0 + tmat*ee0_3);

	tmat = (-sf)*Bl11 + cf*Bl21 + 0*Bl31; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = (-sf)*Bl12 + cf*Bl22 + 0*Bl32; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = (-sf)*Bl13 + cf*Bl23 + 0*Bl33;
	e_2 = (float)(pom + tmat*ee_3);  e0_2 = (float)(pom0 + tmat*ee0_3);

	tmat = 0*Bl11 + 0*Bl21 + 1*Bl31; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = 0*Bl12 + 0*Bl22 + 1*Bl32; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = 0*Bl13 + 0*Bl23 + 1*Bl33;
	e_3 = (float)(pom + tmat*ee_3);  e0_3 = (float)(pom0 + tmat*ee0_3);

	tmat = cf*Db111 + sf*Db121 + 0*Db131; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = cf*Db112 + sf*Db122 + 0*Db132; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = cf*Db113 + sf*Db123 + 0*Db133;
	de[1][1] = (float)(pom + tmat*ee_3);  de0[1][1] = (float)(pom0 + tmat*ee0_3);

	tmat = cf*Db211 + sf*Db221 + 0*Db231; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = cf*Db212 + sf*Db222 + 0*Db232; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = cf*Db213 + sf*Db223 + 0*Db233;
	de[1][2] = (float)(pom + tmat*ee_3);  de0[1][2] = (float)(pom0 + tmat*ee0_3);

	tmat = (-tpoint*sf)*Bl11 + (tpoint*cf)*Bl21 + 0*Bl31; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = (-tpoint*sf)*Bl12 + (tpoint*cf)*Bl22 + 0*Bl32; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = (-tpoint*sf)*Bl13 + (tpoint*cf)*Bl23 + 0*Bl33;
	de[1][3] = (float)(pom + tmat*ee_3);  de0[1][3] = (float)(pom0 + tmat*ee0_3);

	tmat = -sf*Db111 + cf*Db121 + 0*Db131; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = -sf*Db112 + cf*Db122 + 0*Db132; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = -sf*Db113 + cf*Db123 + 0*Db133;
	de[2][1] = (float)(pom + tmat*ee_3);  de0[2][1] = (float)(pom0 + tmat*ee0_3);

	tmat = -sf*Db211 + cf*Db221 + 0*Db231; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = -sf*Db212 + cf*Db222 + 0*Db232; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = -sf*Db213 + cf*Db223 + 0*Db233;
	de[2][2] = (float)(pom + tmat*ee_3);  de0[2][2] = (float)(pom0 + tmat*ee0_3);

	tmat = (-tpoint*cf)*Bl11 + (-tpoint*sf)*Bl21 + 0*Bl31; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = (-tpoint*cf)*Bl12 + (-tpoint*sf)*Bl22 + 0*Bl32; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = (-tpoint*cf)*Bl13 + (-tpoint*sf)*Bl23 + 0*Bl33;
	de[2][3] = (float)(pom + tmat*ee_3);  de0[2][3] = (float)(pom0 + tmat*ee0_3);

	tmat = 0*Db111 + 0*Db121 + 1*Db131; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = 0*Db112 + 0*Db122 + 1*Db132; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = 0*Db113 + 0*Db123 + 1*Db133;
	de[3][1] = (float)(pom + tmat*ee_3);  de0[3][1] = (float)(pom0 + tmat*ee0_3);

	tmat = 0*Db211 + 0*Db221 + 1*Db231; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = 0*Db212 + 0*Db222 + 1*Db232; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = 0*Db213 + 0*Db223 + 1*Db233;
	de[3][2] = (float)(pom + tmat*ee_3);  de0[3][2] = (float)(pom0 + tmat*ee0_3);

	tmat = 0*Bl11 + 0*Bl21 + 0*Bl31; pom = tmat*ee_1; pom0 = tmat*ee0_1;
	tmat = 0*Bl12 + 0*Bl22 + 0*Bl32; pom += tmat*ee_2; pom0 += tmat*ee0_2;
	tmat = 0*Bl13 + 0*Bl23 + 0*Bl33;
	de[3][3] = (float)(pom + tmat*ee_3);  de0[3][3] = (float)(pom0 + tmat*ee0_3);

   /* Directions (and ders.) in the rotating system */

   //
   /*Integrated brightness (phase coeff. used later) */
   float br,ar,tmp1,tmp2,tmp3,tmp4,tmp5;
   const float fcl = (float)cl, fcls = (float)cls;
   float acc0[25], acc1[25];
   int2 bfr;

	br=0;
	tmp1=0;
	tmp2=0;
	tmp3=0;
	tmp4=0;
	tmp5=0;
	const int iStart = ISTART;
#pragma unroll
	for (int p = 0; p < 25; p++) { acc0[p] = 0.0f; acc1[p] = 0.0f; }
	j=blockIdx.x*(CUDA_Numfac1)+1;
	for (i = 1; i <= CUDA_Numfac; i++,j++)
   {
		  const float fn0 = CUDA_Norf[i][0], fn1 = CUDA_Norf[i][1], fn2 = CUDA_Norf[i][2];
		  const float flmu  = (float)e_1 * fn0 + (float)e_2 * fn1 + (float)e_3 * fn2;
		  const float flmu0 = (float)e0_1 * fn0 + (float)e0_2 * fn1 + (float)e0_3 * fn2;
		  if((flmu > (float)TINY) && (flmu0 > (float)TINY))
		  {
			const float fdnom = flmu + flmu0;
			const float finv = __fdividef(1.0f, fdnom);
			const float fs = flmu * flmu0 * (fcl + fcls * finv);
			ar = (float)(*CUDA_LCC).Area[i];   /* same cast curv() used for Areaf */
			br+= ar * fs;
			//
			/* accumulate d/dg in registers during the facet sweep */
			{
				const float wgt = ar * fs;
				if (iStart == 2)
				{
#pragma unroll
					for (int p = 0; p < 25; p++)
					{
						const int idx = 2 + 2 * p;
						const float2 v = *reinterpret_cast<const float2*>(&CUDA_DsphF[i][idx]);
						acc0[p] += wgt * v.x;
						acc1[p] += wgt * v.y;
					}
				}
				else
				{
#pragma unroll
					for (int p = 0; p < 25; p++)
					{
						const int idx = 1 + 2 * p;
						acc0[p] += wgt * CUDA_DsphF[i][idx];
						acc1[p] += wgt * CUDA_DsphF[i][idx + 1];
					}
				}
			}
			//
			const float flmu0_dnom = flmu0 * finv;
			const float fdsmu = fcls * (flmu0_dnom * flmu0_dnom) + fcl * flmu0;
			const float flmu_dnom = flmu * finv;
			const float fdsmu0 = fcls * (flmu_dnom * flmu_dnom) + fcl * flmu;

			const float fs1  = (float)de[1][1] * fn0 + (float)de[2][1] * fn1 + (float)de[3][1] * fn2;
			const float fs10 = (float)de0[1][1] * fn0 + (float)de0[2][1] * fn1 + (float)de0[3][1] * fn2;
			tmp1 += ar * (fdsmu * fs1 + fdsmu0 * fs10);
			const float fs2  = (float)de[1][2] * fn0 + (float)de[2][2] * fn1 + (float)de[3][2] * fn2;
			const float fs20 = (float)de0[1][2] * fn0 + (float)de0[2][2] * fn1 + (float)de0[3][2] * fn2;
			tmp2 += ar * (fdsmu * fs2 + fdsmu0 * fs20);
			const float fs3  = (float)de[1][3] * fn0 + (float)de[2][3] * fn1 + (float)de[3][3] * fn2;
			const float fs30 = (float)de0[1][3] * fn0 + (float)de0[2][3] * fn1 + (float)de0[3][3] * fn2;
			tmp3 += ar * (fdsmu * fs3 + fdsmu0 * fs30);

			tmp4 += ar * (flmu * flmu0);
			tmp5 += ar * (flmu * flmu0 * finv);
		}
	}
		const int dbase = jp * dstride;
		i = dbase + (ncoef0 - 3 + 1);
	   /* Ders. of brightness w.r.t. rotation parameters */
	   (*CUDA_LCC).dytemp[i]= Scale * tmp1;
	   i++;
	   (*CUDA_LCC).dytemp[i] = Scale * tmp2;
	   i++;
	   (*CUDA_LCC).dytemp[i] = Scale * tmp3;
	   i++;
	   /* Ders. of br. w.r.t. phase function params. */
	   (*CUDA_LCC).dytemp[i] = br * dphp1;
	   i++;
	   (*CUDA_LCC).dytemp[i] = br * dphp2;
	   i++;
	   (*CUDA_LCC).dytemp[i] = br * dphp3;

	   /* Ders. of br. w.r.t. cl, cls */
	   (*CUDA_LCC).dytemp[dbase + (ncoef - 1)] = Scale * tmp4 * cl;
	   (*CUDA_LCC).dytemp[dbase + ncoef] = Scale * tmp5;
	   /* Scaled brightness */
	  (*CUDA_LCC).ytemp[jp]=br * Scale;

	ncoef0-=3;
	int d = dbase + iStart;
	int d1 = d + 1;
   /* Derivatives of brightness w.r.t. g-coeffs (accumulated above) */
#pragma unroll
   for (int p = 0; p < 25; p++)
   {
	   const int idx = iStart + 2 * p;
	   if (idx <= ncoef0)
	   {
		   (*CUDA_LCC).dytemp[d + 2 * p] = Scale * (double)acc0[p];
		   if (idx + 1 <= ncoef0)
			   (*CUDA_LCC).dytemp[d1 + 2 * p] = Scale * (double)acc1[p];
	   }
   }
#undef Bl11
#undef Bl12
#undef Bl13
#undef Bl21
#undef Bl22
#undef Bl23
#undef Bl31
#undef Bl32
#undef Bl33
#undef Db111
#undef Db112
#undef Db113
#undef Db121
#undef Db122
#undef Db123
#undef Db131
#undef Db132
#undef Db133
#undef Db211
#undef Db212
#undef Db213
#undef Db221
#undef Db222
#undef Db223
#undef Db231
#undef Db232
#undef Db233
   return(0);
}
