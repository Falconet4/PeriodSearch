#pragma once
/* Warp-cooperative Curve1T. The warp-cooperative formulation follows
   IanSteveC's PeriodSearch (mrqcof_curve1_opt): lane = facet, shared
   compaction, column-owned coefficient accumulation, butterfly reduction.
   Selected at runtime by PS_WARP=1. */


__device__ __forceinline__ void ps_geo_point(freq_context *CUDA_LCC, double cg[], int jp,
                                             const double *Bl, const double *Db, double *g)
{
   const int ncoef0 = CUDA_ncoef0;
   const int lnp = (*CUDA_LCC).np + jp;
   const int estr = MAX_N_OBS + 1;
   const double ee_1 = CUDA_ee[0*estr + lnp], ee0_1 = CUDA_ee0[0*estr + lnp];
   const double ee_2 = CUDA_ee[1*estr + lnp], ee0_2 = CUDA_ee0[1*estr + lnp];
   const double ee_3 = CUDA_ee[2*estr + lnp], ee0_3 = CUDA_ee0[2*estr + lnp];
   const double tpoint = CUDA_tim[lnp];

   double ee_dot = ee_1*ee0_1 + ee_2*ee0_2 + ee_3*ee0_3;
   ee_dot = ee_dot > 1.0 ? 1.0 : (ee_dot < -1.0 ? -1.0 : ee_dot);
   const double alpha = acos(ee_dot);
   const double fph = exp(-alpha / cg[ncoef0+2]);
   g[23] = fph;
   g[24] = cg[ncoef0+1] * fph * alpha / (cg[ncoef0+2] * cg[ncoef0+2]);
   g[25] = alpha;
   g[22] = 1 + cg[ncoef0+1] * fph + cg[ncoef0+3] * alpha;

   double ff = cg[ncoef0] * tpoint + CUDA_Phi_0;
   ff = fmod(ff, 2 * PI);
   double cf, sf;
   sincos(ff, &sf, &cf);
   double tmat, pom, pom0;

   tmat = cf*Bl[0] + sf*Bl[3] + 0*Bl[6]; pom = tmat*ee_1; pom0 = tmat*ee0_1;
   tmat = cf*Bl[1] + sf*Bl[4] + 0*Bl[7]; pom += tmat*ee_2; pom0 += tmat*ee0_2;
   tmat = cf*Bl[2] + sf*Bl[5] + 0*Bl[8];
   g[16] = pom + tmat*ee_3;  g[19] = pom0 + tmat*ee0_3;

   tmat = (-sf)*Bl[0] + cf*Bl[3] + 0*Bl[6]; pom = tmat*ee_1; pom0 = tmat*ee0_1;
   tmat = (-sf)*Bl[1] + cf*Bl[4] + 0*Bl[7]; pom += tmat*ee_2; pom0 += tmat*ee0_2;
   tmat = (-sf)*Bl[2] + cf*Bl[5] + 0*Bl[8];
   g[17] = pom + tmat*ee_3;  g[20] = pom0 + tmat*ee0_3;

   tmat = 0*Bl[0] + 0*Bl[3] + 1*Bl[6]; pom = tmat*ee_1; pom0 = tmat*ee0_1;
   tmat = 0*Bl[1] + 0*Bl[4] + 1*Bl[7]; pom += tmat*ee_2; pom0 += tmat*ee0_2;
   tmat = 0*Bl[2] + 0*Bl[5] + 1*Bl[8];
   g[18] = pom + tmat*ee_3;  g[21] = pom0 + tmat*ee0_3;

   tmat = cf*Db[0] + sf*Db[3] + 0*Db[6]; pom = tmat*ee_1; pom0 = tmat*ee0_1;
   tmat = cf*Db[1] + sf*Db[4] + 0*Db[7]; pom += tmat*ee_2; pom0 += tmat*ee0_2;
   tmat = cf*Db[2] + sf*Db[5] + 0*Db[8];
   g[0] = pom + tmat*ee_3;  g[3] = pom0 + tmat*ee0_3;

   tmat = cf*Db[9] + sf*Db[12] + 0*Db[15]; pom = tmat*ee_1; pom0 = tmat*ee0_1;
   tmat = cf*Db[10] + sf*Db[13] + 0*Db[16]; pom += tmat*ee_2; pom0 += tmat*ee0_2;
   tmat = cf*Db[11] + sf*Db[14] + 0*Db[17];
   g[6] = pom + tmat*ee_3;  g[9] = pom0 + tmat*ee0_3;

   tmat = (-tpoint*sf)*Bl[0] + (tpoint*cf)*Bl[3] + 0*Bl[6]; pom = tmat*ee_1; pom0 = tmat*ee0_1;
   tmat = (-tpoint*sf)*Bl[1] + (tpoint*cf)*Bl[4] + 0*Bl[7]; pom += tmat*ee_2; pom0 += tmat*ee0_2;
   tmat = (-tpoint*sf)*Bl[2] + (tpoint*cf)*Bl[5] + 0*Bl[8];
   g[12] = pom + tmat*ee_3;  g[14] = pom0 + tmat*ee0_3;

   tmat = -sf*Db[0] + cf*Db[3] + 0*Db[6]; pom = tmat*ee_1; pom0 = tmat*ee0_1;
   tmat = -sf*Db[1] + cf*Db[4] + 0*Db[7]; pom += tmat*ee_2; pom0 += tmat*ee0_2;
   tmat = -sf*Db[2] + cf*Db[5] + 0*Db[8];
   g[1] = pom + tmat*ee_3;  g[4] = pom0 + tmat*ee0_3;

   tmat = -sf*Db[9] + cf*Db[12] + 0*Db[15]; pom = tmat*ee_1; pom0 = tmat*ee0_1;
   tmat = -sf*Db[10] + cf*Db[13] + 0*Db[16]; pom += tmat*ee_2; pom0 += tmat*ee0_2;
   tmat = -sf*Db[11] + cf*Db[14] + 0*Db[17];
   g[7] = pom + tmat*ee_3;  g[10] = pom0 + tmat*ee0_3;

   tmat = (-tpoint*cf)*Bl[0] + (-tpoint*sf)*Bl[3] + 0*Bl[6]; pom = tmat*ee_1; pom0 = tmat*ee0_1;
   tmat = (-tpoint*cf)*Bl[1] + (-tpoint*sf)*Bl[4] + 0*Bl[7]; pom += tmat*ee_2; pom0 += tmat*ee0_2;
   tmat = (-tpoint*cf)*Bl[2] + (-tpoint*sf)*Bl[5] + 0*Bl[8];
   g[13] = pom + tmat*ee_3;  g[15] = pom0 + tmat*ee0_3;

   tmat = 0*Db[0] + 0*Db[3] + 1*Db[6]; pom = tmat*ee_1; pom0 = tmat*ee0_1;
   tmat = 0*Db[1] + 0*Db[4] + 1*Db[7]; pom += tmat*ee_2; pom0 += tmat*ee0_2;
   tmat = 0*Db[2] + 0*Db[5] + 1*Db[8];
   g[2] = pom + tmat*ee_3;  g[5] = pom0 + tmat*ee0_3;

   tmat = 0*Db[9] + 0*Db[12] + 1*Db[15]; pom = tmat*ee_1; pom0 = tmat*ee0_1;
   tmat = 0*Db[10] + 0*Db[13] + 1*Db[16]; pom += tmat*ee_2; pom0 += tmat*ee0_2;
   tmat = 0*Db[11] + 0*Db[14] + 1*Db[17];
   g[8] = pom + tmat*ee_3;  g[11] = pom0 + tmat*ee0_3;
}

/* Contribution of one point to one facet; returns the weight (0 == facet not
   visible from this point, which is exactly a zero contribution). */
__device__ __forceinline__ double ps_facet_pt(const double * __restrict__ pg, double n0, double n1, double n2,
                                              double ar, double cl, double cls,
                                              double &t1, double &t2, double &t3,
                                              double &t4, double &t5)
{
   const double * __restrict__ pa = (const double *)__builtin_assume_aligned(pg, 16);
   const double lmu  = pa[16]*n0 + pa[17]*n1 + pa[18]*n2;
   const double lmu0 = pa[19]*n0 + pa[20]*n1 + pa[21]*n2;
   if (!((lmu > TINY) && (lmu0 > TINY))) return 0.0;

   const double dnom = lmu + lmu0;
   const double r    = __drcp_rn(dnom);
   const double s    = lmu * lmu0 * (cl + cls * r);
   const double lmu0_dnom = lmu0 * r;
   const double lmu_dnom  = lmu * r;
   const double dsmu  = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
   const double dsmu0 = cls * (lmu_dnom * lmu_dnom) + cl * lmu;

   const double sum1  = n0*pa[0] + n1*pa[1] + n2*pa[2];
   const double sum10 = n0*pa[3] + n1*pa[4] + n2*pa[5];
   const double sum2  = n0*pa[6] + n1*pa[7] + n2*pa[8];
   const double sum20 = n0*pa[9] + n1*pa[10] + n2*pa[11];
   const double sum3  = n0*pa[12] + n1*pa[13];
   const double sum30 = n0*pa[14] + n1*pa[15];

   t1 += ar * (dsmu * sum1 + dsmu0 * sum10);
   t2 += ar * (dsmu * sum2 + dsmu0 * sum20);
   t3 += ar * (dsmu * sum3 + dsmu0 * sum30);
   t4 += ar * (lmu * lmu0);
   t5 += ar * (lmu * lmu0 * r);
   return ar * s;
}

__device__ void ps_brightT_warp(freq_context *CUDA_LCC, double cg[], int Inrel, int Lpoints,
                                const double *sBl, const double *sDb, double cl, double cls,
                                const int wid, const int lane,
                                double *sgeo, double *swcA, double *swcB, int *sfc)
{
   const int nf = CUDA_Numfac;
   const int ncoef = CUDA_ma;
   const int dstride = ncoef + 1;
   const int iStart = Inrel ? 2 : 1;
   const int nn = CUDA_ncoef0 - 3;
   const int nrot = CUDA_ncoef0 - 2;

   /* Spread a curve's points evenly over the warps of the block instead of giving
      each warp a fixed PS_GEO_BATCH of them.  Under the fixed scheme a curve shorter
      than PS_GEO_BATCH leaves every warp but the first idle, and one shorter than
      4*PS_GEO_BATCH uses only half the block.  Measured on a single workunit with
      identical data, the same points grouped into curves of <= 16 cost 66 % more
      wall clock than the same points in longer curves.  Only the point-to-warp
      assignment changes; every point's arithmetic is untouched. */
   /* Give each warp an equal share of the curve.  The old split rounded up, so the
      remainder all landed on the last warp and the block ran one partial round for
      every point count that is not a multiple of nwarps*chunk.  Here the shares
      differ by at most one point, and each warp walks its own contiguous range. */
   const int nwarps = (int)(blockDim.x >> 5);
   const int wbase  = Lpoints / nwarps;
   const int wrem   = Lpoints % nwarps;
   const int wcount = wbase + ((wid < wrem) ? 1 : 0);
   const int wstart = 1 + wid*wbase + ((wid < wrem) ? wid : wrem);
   #pragma unroll 1
   for (int jp0 = wstart; jp0 < wstart + wcount; jp0 += PS_GEO_BATCH)
   {
      int nb = wstart + wcount - jp0;
      if (nb > PS_GEO_BATCH) nb = PS_GEO_BATCH;

      if (lane < nb)
         ps_geo_point(CUDA_LCC, cg, jp0 + lane, sBl, sDb, sgeo + lane*PS_GEO_N);
      __syncwarp();

      /* A/B variant: one point per pass instead of the two-point pairing.
         Halves the work per wavefront (the property the counters attribute to
         Ahorek's build) and leaves every point's sums and their order alone,
         so the output is bit-identical. */
      for (int jj = 0; jj < nb; jj += 1)
      {
         const int haveB = 0;
         const double *pgA = sgeo + jj*PS_GEO_N;
         const double *pgB = sgeo + (haveB ? jj + 1 : jj)*PS_GEO_N;
         const int jpA = jp0 + jj;
         const int jpB = jpA + 1;

         double brA=0, t1A=0, t2A=0, t3A=0, t4A=0, t5A=0;
         double brB=0, t1B=0, t2B=0, t3B=0, t4B=0, t5B=0;
         double a0A=0, a1A=0, a0B=0, a1B=0;

         for (int f0 = 0; f0 < nf; f0 += 32)
         {
            const int i = f0 + lane + 1;
            double dbrA = 0.0, dbrB = 0.0;

            if (i <= nf)
            {
               const double n0 = CUDA_Nor[0][i];
               const double n1 = CUDA_Nor[1][i];
               const double n2 = CUDA_Nor[2][i];
               const double ar = (*CUDA_LCC).Area[i];

               dbrA = ps_facet_pt(pgA, n0, n1, n2, ar, cl, cls, t1A, t2A, t3A, t4A, t5A);
               if (dbrA != 0.0) brA += dbrA;
               if (haveB)
               {
                  dbrB = ps_facet_pt(pgB, n0, n1, n2, ar, cl, cls, t1B, t2B, t3B, t4B, t5B);
                  if (dbrB != 0.0) brB += dbrB;
               }
            }

            const unsigned long long vis = __ballot_sync(0xffffffffffffffffULL, (dbrA != 0.0) || (dbrB != 0.0));
            const int cnt = (int)__popcll(vis);
            if (dbrA != 0.0 || dbrB != 0.0)
            {
               const int pos = (int)__popcll(vis & ((1ULL << lane) - 1ULL));
               swcA[pos] = dbrA;
               swcB[pos] = dbrB;
               sfc[pos]  = i;
            }
            __syncwarp();

            if (lane < 32)
            {
               const int c = iStart + 2*lane;
               #pragma unroll 4
               for (int j = 0; j < cnt; j++)
               {
                  const double * __restrict__ row = CUDA_Dsph[sfc[j]];
                  const double2 v = *reinterpret_cast<const double2*>(&row[c]);
                  const double wA = swcA[j];
                  const double wB = swcB[j];
                  a0A += wA * v.x;  a1A += wA * v.y;
                  a0B += wB * v.x;  a1B += wB * v.y;
               }
            }
            __syncwarp();
         }

         #pragma unroll
         for (int off = 16; off > 0; off >>= 1)
         {
            brA += __shfl_xor_sync(0xffffffffffffffffULL, brA, off);
            t1A += __shfl_xor_sync(0xffffffffffffffffULL, t1A, off);
            t2A += __shfl_xor_sync(0xffffffffffffffffULL, t2A, off);
            t3A += __shfl_xor_sync(0xffffffffffffffffULL, t3A, off);
            t4A += __shfl_xor_sync(0xffffffffffffffffULL, t4A, off);
            t5A += __shfl_xor_sync(0xffffffffffffffffULL, t5A, off);
            brB += __shfl_xor_sync(0xffffffffffffffffULL, brB, off);
            t1B += __shfl_xor_sync(0xffffffffffffffffULL, t1B, off);
            t2B += __shfl_xor_sync(0xffffffffffffffffULL, t2B, off);
            t3B += __shfl_xor_sync(0xffffffffffffffffULL, t3B, off);
            t4B += __shfl_xor_sync(0xffffffffffffffffULL, t4B, off);
            t5B += __shfl_xor_sync(0xffffffffffffffffULL, t5B, off);
         }

         if (lane == 0)
         {
            const int dA = jpA * dstride;
            int iw = dA + nrot;
            (*CUDA_LCC).dytemp[iw]     = pgA[22] * t1A;
            (*CUDA_LCC).dytemp[iw + 1] = pgA[22] * t2A;
            (*CUDA_LCC).dytemp[iw + 2] = pgA[22] * t3A;
            (*CUDA_LCC).dytemp[iw + 3] = brA * pgA[23];
            (*CUDA_LCC).dytemp[iw + 4] = brA * pgA[24];
            (*CUDA_LCC).dytemp[iw + 5] = brA * pgA[25];
            (*CUDA_LCC).dytemp[dA + ncoef - 1] = pgA[22] * t4A * cl;
            (*CUDA_LCC).dytemp[dA + ncoef]     = pgA[22] * t5A;
            (*CUDA_LCC).ytemp[jpA] = brA * pgA[22];
         }
         if (lane < 32)
         {
            const int c = iStart + 2*lane;
            const int dA = jpA * dstride;
            if (c <= nn)
            {
               (*CUDA_LCC).dytemp[dA + c] = pgA[22] * a0A;
               if (c + 1 <= nn)
                  (*CUDA_LCC).dytemp[dA + c + 1] = pgA[22] * a1A;
            }
         }
         if (lane == 0 && haveB)
         {
            const int dB = jpB * dstride;
            int iw = dB + nrot;
            (*CUDA_LCC).dytemp[iw]     = pgB[22] * t1B;
            (*CUDA_LCC).dytemp[iw + 1] = pgB[22] * t2B;
            (*CUDA_LCC).dytemp[iw + 2] = pgB[22] * t3B;
            (*CUDA_LCC).dytemp[iw + 3] = brB * pgB[23];
            (*CUDA_LCC).dytemp[iw + 4] = brB * pgB[24];
            (*CUDA_LCC).dytemp[iw + 5] = brB * pgB[25];
            (*CUDA_LCC).dytemp[dB + ncoef - 1] = pgB[22] * t4B * cl;
            (*CUDA_LCC).dytemp[dB + ncoef]     = pgB[22] * t5B;
            (*CUDA_LCC).ytemp[jpB] = brB * pgB[22];
         }
         if (lane < 32 && haveB)
         {
            const int c = iStart + 2*lane;
            const int dB = jpB * dstride;
            if (c <= nn)
            {
               (*CUDA_LCC).dytemp[dB + c] = pgB[22] * a0B;
               if (c + 1 <= nn)
                  (*CUDA_LCC).dytemp[dB + c + 1] = pgB[22] * a1B;
            }
         }
         __syncwarp();
      }
      __syncwarp();
   }
}
