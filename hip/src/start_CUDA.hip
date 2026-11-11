#ifndef NVML_NO_UNVERSIONED_FUNC_DEFS
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#endif


#include <hip/hip_runtime.h>
#include <cstdio>
#include "mfile.h"
#include "globals.h"
#include "globals_CUDA.h"
#include "start_CUDA.h"
#include "declarations_CUDA.h"
#include "boinc_api.h"
#include "Start.cuh"
//#include "cuda_runtime.h"
#include <hip/hip_runtime_api.h>
//#include <cuda_occupancy.h>
#include <hip/hip_texture_types.h>

#ifdef __GNUC__
#include <time.h>
#endif

//global to all freq
__constant__ int /*CUDA_n,*/CUDA_Ncoef, CUDA_Nphpar, CUDA_Numfac, CUDA_Numfac1, CUDA_Dg_block;
__constant__ int CUDA_ia[MAX_N_PAR + 1];
__constant__ int CUDA_ma, CUDA_mfit, CUDA_mfit1, CUDA_lastone, CUDA_lastma, CUDA_ncoef0;
__device__ double CUDA_cg_first[MAX_N_PAR + 1];
__device__ double CUDA_beta_pole[N_POLES + 1];
__device__ double CUDA_lambda_pole[N_POLES + 1];
__device__ double CUDA_par[4];
//__device__ __constant__ double CUDA_cl, CUDA_Alamda_start, CUDA_Alamda_incr;
__device__ double CUDA_cl, CUDA_Alamda_start, CUDA_Alamda_incr;
__device__ int CUDA_n_iter_max, CUDA_n_iter_min, CUDA_ndata;
__device__ double CUDA_iter_diff_max;
__device__ double CUDA_Nor[3][MAX_N_FAC + 1];
__constant__ float CUDA_Norf[MAX_N_FAC + 1][3];
__constant__ double CUDA_conw_r;
__constant__ int CUDA_Lmax, CUDA_Mmax;
__device__ double CUDA_Fc[MAX_LM + 1][MAX_N_FAC + 1];
__device__ double CUDA_Fs[MAX_LM + 1][MAX_N_FAC + 1];
__device__ double CUDA_Pleg[MAX_LM + 1][MAX_LM + 1][MAX_N_FAC + 1];
__device__ double CUDA_Darea[MAX_N_FAC + 1];
__device__ double CUDA_Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
__device__ float CUDA_DsphF[MAX_N_FAC + 1][DSPHF_STRIDE];
__device__ double* CUDA_ee/*[MAX_N_OBS+1][3]*/;
__device__ double* CUDA_ee0/*[MAX_N_OBS+1][3]*/;
__device__ double CUDA_tim[MAX_N_OBS + 1];
//__device__ double CUDA_brightness[MAX_N_OBS+1];
//__device__ double CUDA_sig[MAX_N_OBS+1];
//__device__ double *CUDA_Weight/*[MAX_N_OBS+1]*/;
__constant__ double CUDA_Phi_0;
__device__ int CUDA_End;
__device__ int CUDA_Is_Precalc;

//__device__ int icol;
//__device__ double pivinv;
//__shared__ int sh_icol[CUDA_BLOCK_DIM];
//__shared__ int sh_irow[CUDA_BLOCK_DIM];
//__shared__ double sh_big[CUDA_BLOCK_DIM];

__device__ const int2 * texWeight;
__device__ const int2 * texbrightness;
__device__ const int2 * texsig;

//global to one thread
__device__ freq_context* CUDA_CC;
__device__ freq_result* CUDA_FR;

__device__ const int2 * texArea;
__device__ const int2 * texDg;

int CUDA_grid_dim;
double* pee, * pee0, * pWeight;
bool nvml_enabled = false;
//bool if_freq_measured = false;

//void GetPeakClock(const int cudadev)
//{
//	unsigned int currentSmClock;
//	unsigned int currentMemoryClock;
//	const unsigned int devId = cudadev;
//	nvmlDevice_t nvmlDevice;
//	nvmlDeviceGetHandleByIndex(devId, &nvmlDevice);
//	nvmlDeviceGetClock(nvmlDevice, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &currentSmClock);
//	nvmlDeviceGetClock(nvmlDevice, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &currentMemoryClock);
//	currentMemoryClock /= 2;
//	cudaDeviceProp deviceProp;
//	cudaGetDeviceProperties(&deviceProp, cudadev);
//	const auto deviceClock = deviceProp.clockRate / 1000;
//	const auto memoryClock = deviceProp.memoryClockRate / 1000 /2;
//	fprintf(stderr, "CUDA Device SM clock [base|current]: %u MHz | %u MHz\n", deviceClock, currentSmClock);
//	fprintf(stderr, "CUDA Device Memory clock [base|current]: %u MHz | %u MHz\n", memoryClock, currentMemoryClock);
//
//	if_freq_measured = true;
//}

// NOTE: https://boinc.berkeley.edu/trac/wiki/CudaApps
bool SetCUDABlockingSync(const int device) {
	hipDevice_t  hcuDevice;
	hipCtx_t hcuContext;

	hipError_t status = hipInit(0);
	if (status != hipSuccess)
		return false;

	status = hipDeviceGet(&hcuDevice, device);
	if (status != hipSuccess)
		return false;

	status = hipCtxCreate(&hcuContext, 0x4, hcuDevice);
	if (status != hipSuccess)
		return false;

	return true;
}

int CUDAPrepare(int cudadev, double* beta_pole, double* lambda_pole, double* par, double cl, double Alamda_start, double Alamda_incr,
	double ee[][3], double ee0[][3], double* tim, double Phi_0, int checkex, int ndata)
{
	//init gpu
	auto initResult = SetCUDABlockingSync(cudadev);
	if (!initResult)
	{
		fprintf(stderr, "CUDA: Error while initialising CUDA\n");
		exit(999);
	}

	hipSetDevice(cudadev);
	// TODO: Check if this is obsolete when calling SetCUDABlockingSync()
	hipSetDeviceFlags(hipDeviceScheduleBlockingSync);
	// TODO: Check if this will help to free some CPU core utilization
	//cudaSetDeviceFlags(cudaDeviceScheduleYield);

	/* LDS-fit guard.  The Curve1/Curve2 tiles and the Gauss-Jordan solver are
	   sized at compile time; if this device offers less shared memory than the
	   kernels were built for, fail with a clear message instead of computing
	   garbage.  PS_LDS_MAX is a test hook for exercising the failure path. */
	{
		int maxsh = 0;
		hipDeviceGetAttribute(&maxsh, hipDeviceAttributeMaxSharedMemoryPerBlock, cudadev);
		{ const char* e = getenv("PS_LDS_MAX"); if (e) maxsh = atoi(e); }

		static const void* const ks[] = {
			(const void*)CudaCalculateIter1Mrqmin1End,
			(const void*)CudaCalculateIter1Mrqmin2End,
			(const void*)CudaCalculateIter1Mrqcof1Curve1Tw,
			(const void*)CudaCalculateIter1Mrqcof2Curve1Tw,
			(const void*)CudaCalculateIter1Mrqcof1Curve1T,
			(const void*)CudaCalculateIter1Mrqcof2Curve1T,
			(const void*)CudaCalculateIter1Mrqcof1Curve2,
			(const void*)CudaCalculateIter1Mrqcof2Curve2 };

		for (unsigned i = 0; i < sizeof(ks) / sizeof(ks[0]); i++)
		{
			hipFuncAttributes fa;
			if (hipFuncGetAttributes(&fa, ks[i]) == hipSuccess)
			{
				size_t need = (size_t)fa.sharedSizeBytes + (size_t)fa.localSizeBytes;
				if (need > (size_t)maxsh)
				{
					fprintf(stderr, "Error: kernel %u needs %zu B of local memory (+ ~4 KB scratch) "
					                "but the device offers %d B\n", i, need, maxsh);
					exit(999);
				}
			}
		}
	}

	nvml_enabled = false;  /* NVML is NVIDIA-only; no equivalent needed */

	//determine gridDim
	hipDeviceProp_t deviceProp;

	hipGetDeviceProperties(&deviceProp, cudadev);
	if (!checkex)
	{
		auto cudaVersion = HIP_VERSION;
		auto totalGlobalMemory = deviceProp.totalGlobalMem / 1048576;
		auto sharedMemorySm = deviceProp.sharedMemPerMultiprocessor;
		auto sharedMemoryBlock = deviceProp.sharedMemPerBlock;


		/*auto peakClk = 1;
		cudaDeviceGetAttribute(&peakClk, cudaDevAttrClockRate, cudadev);
		auto devicePeakClock = peakClk / 1024;*/

		fprintf(stderr, "HIP runtime version: %d\n", cudaVersion);
		fprintf(stderr, "HIP device number: %d\n", cudadev);
		fprintf(stderr, "HIP device: %s %lluMB \n", deviceProp.name, totalGlobalMemory);
		fprintf(stderr, "GPU architecture: %s\n", deviceProp.gcnArchName);
		//fprintf(stderr, "Device peak clock: %d MHz\n", devicePeakClock);
		fprintf(stderr, "Shared memory per Block | per SM: %llu | %llu\n", sharedMemoryBlock, sharedMemorySm);
		fprintf(stderr, "Multiprocessors: %d\n", deviceProp.multiProcessorCount);

	}


	//int cudaBlockDim = CUDA_BLOCK_DIM;
	// NOTE: See this https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities , Table 15.
	// NOTE: Also this https://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid
	// NOTE: NB - Always set MaxUsedRegisters to 32 in order to achieve 100% SM occupancy (project's Configuration properties -> CUDA C/C++ -> Device)


	// Maximum number of resident thread blocks per multiprocessor
	auto smxBlock = 256;  /* AMD: no SMX sizing; 256 is the app default */

	CUDA_grid_dim = deviceProp.multiProcessorCount * smxBlock;

	if (!checkex)
	{
		fprintf(stderr, "Resident blocks per multiprocessor: %d\n", smxBlock);
		fprintf(stderr, "Grid dim: %d = %d*%d\n", CUDA_grid_dim, deviceProp.multiProcessorCount, smxBlock);
		fprintf(stderr, "Block dim: %d\n", CUDA_BLOCK_DIM);
	}

	hipError_t res;

	//Global parameters
	res = hipMemcpyToSymbol(HIP_SYMBOL(CUDA_beta_pole), beta_pole, sizeof(double) * (N_POLES + 1));
	res = hipMemcpyToSymbol(HIP_SYMBOL(CUDA_lambda_pole), lambda_pole, sizeof(double) * (N_POLES + 1));
	res = hipMemcpyToSymbol(HIP_SYMBOL(CUDA_par), par, sizeof(double) * 4);
	res = hipMemcpyToSymbol(HIP_SYMBOL(CUDA_cl), &cl, sizeof(cl));
	res = hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Alamda_start), &Alamda_start, sizeof(Alamda_start));
	res = hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Alamda_incr), &Alamda_incr, sizeof(Alamda_incr));
	res = hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Mmax), &m_max, sizeof(m_max));
	res = hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Lmax), &l_max, sizeof(l_max));
	res = hipMemcpyToSymbol(HIP_SYMBOL(CUDA_tim), tim, sizeof(double) * (MAX_N_OBS + 1));
	res = hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Phi_0), &Phi_0, sizeof(Phi_0));

	res = hipMalloc(&pWeight, (ndata + 3 + 1) * sizeof(double));
	res = hipMemcpy(pWeight, weight, (ndata + 3 + 1) * sizeof(double), hipMemcpyHostToDevice);
	res = hipMemcpyToSymbol(HIP_SYMBOL(texWeight), &pWeight, sizeof(pWeight));

	res = hipMalloc(&pee, (MAX_N_OBS + 1) * 3 * sizeof(double));
	{
		double* soa = (double*)malloc((size_t)(MAX_N_OBS + 1) * 3 * sizeof(double));
		for (int ii = 0; ii <= ndata; ii++)
			for (int cc = 0; cc < 3; cc++) soa[cc * (MAX_N_OBS + 1) + ii] = ee[ii][cc];
		res = hipMemcpy(pee, soa, (MAX_N_OBS + 1) * 3 * sizeof(double), hipMemcpyHostToDevice);
		free(soa);
	}
	res = hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ee), &pee, sizeof(void*));

	res = hipMalloc(&pee0, (MAX_N_OBS + 1) * 3 * sizeof(double));
	{
		double* soa = (double*)malloc((size_t)(MAX_N_OBS + 1) * 3 * sizeof(double));
		for (int ii = 0; ii <= ndata; ii++)
			for (int cc = 0; cc < 3; cc++) soa[cc * (MAX_N_OBS + 1) + ii] = ee0[ii][cc];
		res = hipMemcpy(pee0, soa, (MAX_N_OBS + 1) * 3 * sizeof(double), hipMemcpyHostToDevice);
		free(soa);
	}
	res = hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ee0), &pee0, sizeof(void*));

	if (res == hipSuccess)
	{
		return 1;
	}

	else return 0;
}

void CUDAUnprepare(void)
{
	hipFree(pee);
	hipFree(pee0);
	hipFree(pWeight);
}

int CUDAPrecalc(int cudadev, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double* conw_r,
	int ndata, int* ia, int* ia_par, int* new_conw, double* cg_first, double* sig, int Numfac, double* brightness)
{
	//int* endPtr;
	int max_test_periods, iC, theEnd;
	double sum_dark_facet, ave_dark_facet;
	int i, n, m;
	int n_iter_max;
	double iter_diff_max;
	freq_result* res;
	void* pcc, * pfr, * pbrightness, * psig;

	// NOTE: max_test_periods dictates the CUDA_Grid_dim_precalc value which is actual Threads-per-Block
	/*	Cuda Compute profiler gives the following advice for almost every kernel launched:
		"Threads are executed in groups of 32 threads called warps. This kernel launch is configured to execute 16 threads per block.
		Consequently, some threads in a warp are masked off and those hardware resources are unused. Try changing the number of threads per block to be a multiple of 32 threads.
		Between 128 and 256 threads per block is a good initial range for experimentation. Use smaller thread blocks rather than one large thread block per multiprocessor
		if latency affects performance. This is particularly beneficial to kernels that frequently call __syncthreads().*/

	max_test_periods = 10; //10;
	sum_dark_facet = 0.0;
	ave_dark_facet = 0.0;

	//#ifdef _DEBUG
	//	int n_max = (int)((freq_start - freq_end) / freq_step) + 1;
	//	if (n_max < max_test_periods)
	//	{
	//		max_test_periods = n_max;
	//		fprintf(stderr, "n_max(%d) < max_test_periods (%d)\n", n_max, max_test_periods);
	//	}
	//	else
	//	{
	//		fprintf(stderr, "n_max(%d) > max_test_periods (%d)\n", n_max, max_test_periods);
	//	}
	//
	//	fprintf(stderr, "freq_start (%.3f) - freq_end (%.3f) / freq_step (%.3f) = n_max (%d)\n", freq_start, freq_end, freq_step, n_max);
	//#endif

	for (i = 1; i <= n_ph_par; i++)
	{
		ia[n_coef + 3 + i] = ia_par[i];
	}

	n_iter_max = 0;
	iter_diff_max = -1;
	if (stop_condition > 1)
	{
		n_iter_max = (int)stop_condition;
		iter_diff_max = 0;
		n_iter_min = 0; /* to not overwrite the n_iter_max value */
	}
	if (stop_condition < 1)
	{
		n_iter_max = MAX_N_ITER; /* to avoid neverending loop */
		iter_diff_max = stop_condition;
	}

	hipError_t err;
	int isPrecalc = 1;
	/*int i_col, sh_icol_local[CUDA_BLOCK_DIM], sh_irow_local[CUDA_BLOCK_DIM];
	double piv_inv, sh_big_local[CUDA_BLOCK_DIM];*/

	//here move data to device
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Ncoef), &n_coef, sizeof(n_coef));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Nphpar), &n_ph_par, sizeof(n_ph_par));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Numfac), &Numfac, sizeof(Numfac));
	m = Numfac + 1;
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Numfac1), &m, sizeof(m));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ia), ia, sizeof(int) * (MAX_N_PAR + 1));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_cg_first), cg_first, sizeof(double) * (MAX_N_PAR + 1));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_n_iter_max), &n_iter_max, sizeof(n_iter_max));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_n_iter_min), &n_iter_min, sizeof(n_iter_min));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ndata), &ndata, sizeof(ndata));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_iter_diff_max), &iter_diff_max, sizeof(iter_diff_max));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_conw_r), conw_r, sizeof(conw_r));
	{
		static double normalSoA[3][MAX_N_FAC + 1];
		for (int nf_i = 0; nf_i <= MAX_N_FAC; nf_i++)
			for (int nf_k = 0; nf_k < 3; nf_k++)
				normalSoA[nf_k][nf_i] = normal[nf_i][nf_k];
		hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Nor), normalSoA, sizeof(normalSoA));
	}
	{
		static float normalf[(MAX_N_FAC + 1) * 3];
		for (int nf_i = 0; nf_i <= MAX_N_FAC; nf_i++)
			for (int nf_k = 0; nf_k < 3; nf_k++)
				normalf[nf_i * 3 + nf_k] = (float)normal[nf_i][nf_k];
		hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Norf), normalf, sizeof(normalf));
	}
	{
		static double fcSoA[MAX_LM + 1][MAX_N_FAC + 1];
		static double fsSoA[MAX_LM + 1][MAX_N_FAC + 1];
		for (int i = 0; i <= MAX_N_FAC; i++)
			for (int m = 0; m <= MAX_LM; m++) { fcSoA[m][i] = f_c[i][m]; fsSoA[m][i] = f_s[i][m]; }
		hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Fc), fcSoA, sizeof(fcSoA));
		hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Fs), fsSoA, sizeof(fsSoA));
	}
	{
		static double plegSoA[MAX_LM + 1][MAX_LM + 1][MAX_N_FAC + 1];
		for (int i = 0; i <= MAX_N_FAC; i++)
			for (int l = 0; l <= MAX_LM; l++)
				for (int m = 0; m <= MAX_LM; m++)
					plegSoA[l][m][i] = pleg[i][l][m];
		hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Pleg), plegSoA, sizeof(plegSoA));
	}
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Darea), d_area, sizeof(double) * (MAX_N_FAC + 1));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Dsph), d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1));
	{
		static float dsphf[(MAX_N_FAC + 1) * DSPHF_STRIDE];
		for (int i = 0; i <= MAX_N_FAC; i++)
			for (int k = 0; k <= MAX_N_PAR; k++)
				dsphf[i * DSPHF_STRIDE + k] = (float)d_sphere[i][k];
		hipMemcpyToSymbol(HIP_SYMBOL(CUDA_DsphF), dsphf, sizeof(dsphf));
	}
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Is_Precalc), &isPrecalc, sizeof isPrecalc, 0, hipMemcpyHostToDevice);
	/*cudaMemcpyToSymbol(icol, &i_col, sizeof(i_col));
	cudaMemcpyToSymbol(pivinv, &piv_inv, sizeof(piv_inv));
	cudaMemcpyToSymbol(sh_icol, sh_icol_local, sizeof(int) * CUDA_BLOCK_DIM);
	cudaMemcpyToSymbol(sh_irow, sh_irow_local, sizeof(int) * CUDA_BLOCK_DIM);
	cudaMemcpyToSymbol(sh_big, sh_big_local, sizeof(double) * CUDA_BLOCK_DIM);*/

	err = hipMalloc(&pbrightness, (ndata + 1) * sizeof(double));
	err = hipMemcpy(pbrightness, brightness, (ndata + 1) * sizeof(double), hipMemcpyHostToDevice);
	err = hipMemcpyToSymbol(HIP_SYMBOL(texbrightness), &pbrightness, sizeof(pbrightness));

	err = hipMalloc(&psig, (ndata + 1) * sizeof(double));
	err = hipMemcpy(psig, sig, (ndata + 1) * sizeof(double), hipMemcpyHostToDevice);
	err = hipMemcpyToSymbol(HIP_SYMBOL(texsig), &psig, sizeof(psig));
	if (err) printf("Error: %s\n", hipGetErrorString(err));

	/* number of fitted parameters */
	int lmfit = 0, llastma = 0, llastone = 1, ma = n_coef + 5 + n_ph_par;
	for (m = 1; m <= ma; m++)
	{
		if (ia[m])
		{
			lmfit++;
			llastma = m;
		}
	}
	llastone = 1;
	for (m = 2; m <= llastma; m++) //ia[1] is skipped because ia[1]=0 is acceptable inside mrqcof
	{
		if (!ia[m]) break;
		llastone = m;
	}
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ma), &ma, sizeof(ma));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_mfit), &lmfit, sizeof(lmfit));
	m = lmfit + 1;
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_mfit1), &m, sizeof(m));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_lastma), &llastma, sizeof(llastma));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_lastone), &llastone, sizeof(llastone));
	m = ma - 2 - n_ph_par;
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ncoef0), &m, sizeof(m));

	int CUDA_Grid_dim_precalc = max_test_periods * N_POLES;

	{
		size_t freeB = 0, totalB = 0;
		(void)hipMemGetInfo(&freeB, &totalB);
		const size_t needB = (size_t)CUDA_Grid_dim_precalc * (sizeof(freq_context) + sizeof(freq_result)
			+ 3 * (size_t)(Numfac + 1) * sizeof(double)
			+ 2 * (size_t)(lmfit + 1) * (lmfit + 1) * sizeof(double)
			+ (size_t)(max_l_points + 1) * (size_t)(ma + 1) * sizeof(double)
			+ (size_t)(max_l_points + 1) * sizeof(double));
		if (needB > freeB - freeB / 10)
			fprintf(stderr, "period_search: WARNING need %.1f MB, only %.1f MB free\n",
			        (double)needB / 1048576.0, (double)freeB / 1048576.0);
	}

	err = hipMalloc(&pcc, CUDA_Grid_dim_precalc * sizeof(freq_context));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_CC), &pcc, sizeof(pcc));
	err = hipMalloc(&pfr, CUDA_Grid_dim_precalc * sizeof(freq_result));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_FR), &pfr, sizeof(pfr));

	m = (Numfac + 1) * (n_coef + 1);
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Dg_block), &m, sizeof(m));

	double* pa, * pal, * pco, * pdytemp, * pytemp, * pgf;
	float* paf;

	err = hipMalloc(&pa, CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double));
	err = hipMemcpyToSymbol(HIP_SYMBOL(texArea), &pa, sizeof(pa));
	err = hipMalloc(&pal, CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	err = hipMalloc(&pco, CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	err = hipMalloc(&pdytemp, CUDA_Grid_dim_precalc * (max_l_points + 1) * (ma + 1) * sizeof(double));
	err = hipMalloc(&pytemp, CUDA_Grid_dim_precalc * (max_l_points + 1) * sizeof(double));
	err = hipMalloc(&pgf, CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double));
	err = hipMalloc(&paf, CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(float));
	if (err != hipSuccess) { fprintf(stderr, "period_search: device allocation failed: %s\n", hipGetErrorString(err)); exit(1); }

	{
		void** hctx = (void**)malloc((size_t)CUDA_Grid_dim_precalc * 7 * sizeof(void*));
		for (m = 0; m < CUDA_Grid_dim_precalc; m++)
		{
			void** q = hctx + (size_t)m * 7;
			q[0] = &pa[m * (Numfac + 1)];
			q[1] = &pgf[m * (Numfac + 1)];
			q[2] = &pal[m * (lmfit + 1) * (lmfit + 1)];
			q[3] = &pco[m * (lmfit + 1) * (lmfit + 1)];
			q[4] = &pdytemp[m * (max_l_points + 1) * (ma + 1)];
			q[5] = &pytemp[m * (max_l_points + 1)];
			q[6] = &paf[m * (Numfac + 1)];
		}
		err = hipMemcpy2D((void*)pcc, sizeof(freq_context), hctx, 7 * sizeof(void*),
		                  7 * sizeof(void*), CUDA_Grid_dim_precalc, hipMemcpyHostToDevice);
		free(hctx);
	}

	res = NULL;
	err = hipHostMalloc((void**)&res, CUDA_Grid_dim_precalc * sizeof(freq_result), hipHostMallocDefault);
	if (err != hipSuccess || res == NULL) res = (freq_result*)malloc(CUDA_Grid_dim_precalc * sizeof(freq_result));

	for (n = 1; n <= max_test_periods; n += CUDA_Grid_dim_precalc)
	{
		CudaCalculatePreparePacked <<<CUDA_Grid_dim_precalc, 1 >>> (max_test_periods, freq_start, freq_step);
		//err = cudaThreadSynchronize();
		err = hipDeviceSynchronize();

		{
			//zero global End signal
			theEnd = 0;
			hipMemcpyToSymbol(HIP_SYMBOL(CUDA_End), &theEnd, sizeof(theEnd), 0, hipMemcpyHostToDevice);
			//cudaGetSymbolAddress((void**)&endPtr, CUDA_End);
			//
			CudaCalculatePreparePolePacked <<<CUDA_Grid_dim_precalc, 1 >>> (max_test_periods);
			//
#ifdef _DEBUG
			printf(".");
#endif
			auto count = 0;
					bool firstIter = true;
	while (!theEnd)
			{
				count++;
				CudaCalculateIter1Begin <<<CUDA_Grid_dim_precalc, 1 >>> ();
				//mrqcof
								if (firstIter)
				{
CudaCalculateIter1Mrqcof1Start <<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >>> ();
				for (iC = 1; iC < l_curves; iC++)
				{
					LaunchCurve1Kernel(1, CUDA_Grid_dim_precalc, in_rel[iC], l_points[iC], 0);
					CudaCalculateIter1Mrqcof1Curve2 <<<CUDA_Grid_dim_precalc, CURVE2_BLOCK_DIM >>> (in_rel[iC], l_points[iC]);
				}
				CudaCalculateIter1Mrqcof1Curve1Last <<<CUDA_Grid_dim_precalc, CURVE1LAST_BLOCK_DIM >>> (in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof1Curve2 <<<CUDA_Grid_dim_precalc, CURVE2_BLOCK_DIM >>> (in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof1End <<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >>> ();
				}
				firstIter = false;
				//mrqcof
				CudaCalculateIter1Mrqmin1End <<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >>> ();
				//mrqcof
				CudaCalculateIter1Mrqcof2Start <<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >>> ();
				for (iC = 1; iC < l_curves; iC++)
				{
					LaunchCurve1Kernel(2, CUDA_Grid_dim_precalc, in_rel[iC], l_points[iC], 0);
					CudaCalculateIter1Mrqcof2Curve2 <<<CUDA_Grid_dim_precalc, CURVE2_BLOCK_DIM >>> (in_rel[iC], l_points[iC]);
				}
				CudaCalculateIter1Mrqcof2Curve1Last <<<CUDA_Grid_dim_precalc, CURVE1LAST_BLOCK_DIM >>> (in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof2Curve2 <<<CUDA_Grid_dim_precalc, CURVE2_BLOCK_DIM >>> (in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof2End <<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >>> ();
				//mrqcof
				CudaCalculateIter1Mrqmin2End <<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >>> ();
				CudaCalculateIter2 <<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >>> ();
				//err=cudaThreadSynchronize(); memcpy is synchro itself
				err = hipMemcpyFromSymbol(&theEnd, HIP_SYMBOL(CUDA_End), sizeof theEnd, 0, hipMemcpyDeviceToHost);
				theEnd = theEnd == CUDA_Grid_dim_precalc;

				//break;//debug
			}
			CudaCalculateFinishPole <<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >>> ();
			//err = cudaThreadSynchronize();
			err = hipDeviceSynchronize();
			//			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*CUDA_Grid_dim_precalc);
			//			err=cudaMemcpyFromSymbol(&resc,CUDA_CC,sizeof(freq_context)*CUDA_Grid_dim_precalc);
						//break; //debug
		}
		printf("\n");

		CudaCalculateFinish <<<CUDA_Grid_dim_precalc, 1 >>> ();
		//err=cudaThreadSynchronize(); memcpy is synchro itself

		//read results here
		err = hipMemcpy(res, pfr, sizeof(freq_result) * CUDA_Grid_dim_precalc, hipMemcpyDeviceToHost);

		/* The unpacked precalc reuses the last pole's FR slots, which hold the
		   best-dev pole across all poles for each period (strict <, earlier
		   pole wins ties). Reproduce that reduction before summing. */
		for (int p = 0; p < max_test_periods; p++)
		{
			double bd = 1e40, bk = 0;
			int have = 0;
			for (int pole = 0; pole < N_POLES; pole++)
			{
				const freq_result& r = res[pole * max_test_periods + p];
				const double cand = r.dev_best;
				if (cand == cand && ((bd != bd) || cand < bd))
				{
					bd = cand; bk = r.dark_best; have = 1;
				}
			}
			if (have) sum_dark_facet = sum_dark_facet + bk;
		}
	} /* period loop */

	isPrecalc = 0;
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Is_Precalc), &isPrecalc, sizeof(isPrecalc), 0, hipMemcpyHostToDevice);

	hipFree(pa);
	hipFree(paf);
	hipFree(pgf);
	hipFree(pal);
	hipFree(pco);
	hipFree(pdytemp);
	hipFree(pytemp);
	hipFree(pcc);
	hipFree(pfr);
	hipFree(pbrightness);
	hipFree(psig);

	hipHostFree(res);

	ave_dark_facet = sum_dark_facet / max_test_periods;

	if (ave_dark_facet < 1.0)
		*new_conw = 1; /* new correct conwexity weight */
	if (ave_dark_facet >= 1.0)
		*conw_r = *conw_r * 2; /* still not good */

	return 1;
}

int CUDAStart(int cudadev, int n_start_from, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double conw_r,
	int ndata, int* ia, int* ia_par, double* cg_first, MFILE& mf, double escl, double* sig, int Numfac, double* brightness)
{
	int retval, i, n, m, iC, n_max = (int)((freq_start - freq_end) / freq_step) + 1;
	/* only the frequencies that will actually be launched need per-block storage */
	const int allocBlocks = (n_max - n_start_from + 1 < CUDA_grid_dim) ? (n_max - n_start_from + 1) : CUDA_grid_dim;
	int n_iter_max, theEnd, LinesWritten;
	double iter_diff_max;
	freq_result* res;
	void* pcc, * pfr, * pbrightness, * psig;
	char buf[256];

	for (i = 1; i <= n_ph_par; i++)
	{
		ia[n_coef + 3 + i] = ia_par[i];
	}

	n_iter_max = 0;
	iter_diff_max = -1;
	if (stop_condition > 1)
	{
		n_iter_max = (int)stop_condition;
		iter_diff_max = 0;
		n_iter_min = 0; /* to not overwrite the n_iter_max value */
	}
	if (stop_condition < 1)
	{
		n_iter_max = MAX_N_ITER; /* to avoid neverending loop */
		iter_diff_max = stop_condition;
	}

	hipError_t err;

	/*int i_col, sh_icol_local[CUDA_BLOCK_DIM], sh_irow_local[CUDA_BLOCK_DIM];
	double piv_inv, sh_big_local[CUDA_BLOCK_DIM];*/

	//here move data to device
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Ncoef), &n_coef, sizeof(n_coef));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Nphpar), &n_ph_par, sizeof(n_ph_par));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Numfac), &Numfac, sizeof(Numfac));
	m = Numfac + 1;
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Numfac1), &m, sizeof(m));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ia), ia, sizeof(int) * (MAX_N_PAR + 1));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_cg_first), cg_first, sizeof(double) * (MAX_N_PAR + 1));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_n_iter_max), &n_iter_max, sizeof(n_iter_max));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_n_iter_min), &n_iter_min, sizeof(n_iter_min));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ndata), &ndata, sizeof(ndata));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_iter_diff_max), &iter_diff_max, sizeof(iter_diff_max));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_conw_r), &conw_r, sizeof(conw_r));
	{
		static double normalSoA[3][MAX_N_FAC + 1];
		for (int nf_i = 0; nf_i <= MAX_N_FAC; nf_i++)
			for (int nf_k = 0; nf_k < 3; nf_k++)
				normalSoA[nf_k][nf_i] = normal[nf_i][nf_k];
		hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Nor), normalSoA, sizeof(normalSoA));
	}
	{
		static float normalf[(MAX_N_FAC + 1) * 3];
		for (int nf_i = 0; nf_i <= MAX_N_FAC; nf_i++)
			for (int nf_k = 0; nf_k < 3; nf_k++)
				normalf[nf_i * 3 + nf_k] = (float)normal[nf_i][nf_k];
		hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Norf), normalf, sizeof(normalf));
	}
	{
		static double fcSoA[MAX_LM + 1][MAX_N_FAC + 1];
		static double fsSoA[MAX_LM + 1][MAX_N_FAC + 1];
		for (int i = 0; i <= MAX_N_FAC; i++)
			for (int m = 0; m <= MAX_LM; m++) { fcSoA[m][i] = f_c[i][m]; fsSoA[m][i] = f_s[i][m]; }
		hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Fc), fcSoA, sizeof(fcSoA));
		hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Fs), fsSoA, sizeof(fsSoA));
	}
	{
		static double plegSoA[MAX_LM + 1][MAX_LM + 1][MAX_N_FAC + 1];
		for (int i = 0; i <= MAX_N_FAC; i++)
			for (int l = 0; l <= MAX_LM; l++)
				for (int m = 0; m <= MAX_LM; m++)
					plegSoA[l][m][i] = pleg[i][l][m];
		hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Pleg), plegSoA, sizeof(plegSoA));
	}
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Darea), d_area, sizeof(double) * (MAX_N_FAC + 1));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Dsph), d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1));
	{
		static float dsphf[(MAX_N_FAC + 1) * DSPHF_STRIDE];
		for (int i = 0; i <= MAX_N_FAC; i++)
			for (int k = 0; k <= MAX_N_PAR; k++)
				dsphf[i * DSPHF_STRIDE + k] = (float)d_sphere[i][k];
		hipMemcpyToSymbol(HIP_SYMBOL(CUDA_DsphF), dsphf, sizeof(dsphf));
	}
	/*cudaMemcpyToSymbol(icol, &i_col, sizeof(i_col));
	cudaMemcpyToSymbol(pivinv, &piv_inv, sizeof(piv_inv));
	cudaMemcpyToSymbol(sh_icol, sh_icol_local, sizeof(int) * CUDA_BLOCK_DIM);
	cudaMemcpyToSymbol(sh_irow, sh_irow_local, sizeof(int) * CUDA_BLOCK_DIM);
	cudaMemcpyToSymbol(sh_big, sh_big_local, sizeof(double) * CUDA_BLOCK_DIM);*/


	err = hipMalloc(&pbrightness, (ndata + 1) * sizeof(double));
	err = hipMemcpy(pbrightness, brightness, (ndata + 1) * sizeof(double), hipMemcpyHostToDevice);
	err = hipMemcpyToSymbol(HIP_SYMBOL(texbrightness), &pbrightness, sizeof(pbrightness));

	err = hipMalloc(&psig, (ndata + 1) * sizeof(double));
	err = hipMemcpy(psig, sig, (ndata + 1) * sizeof(double), hipMemcpyHostToDevice);
	err = hipMemcpyToSymbol(HIP_SYMBOL(texsig), &psig, sizeof(psig));
	if (err) printf("Error: %s", hipGetErrorString(err));

	/* number of fitted parameters */
	int lmfit = 0, llastma = 0, llastone = 1, ma = n_coef + 5 + n_ph_par;
	for (m = 1; m <= ma; m++)
	{
		if (ia[m])
		{
			lmfit++;
			llastma = m;
		}
	}
	llastone = 1;
	for (m = 2; m <= llastma; m++) //ia[1] is skipped because ia[1]=0 is acceptable inside mrqcof
	{
		if (!ia[m]) break;
		llastone = m;
	}
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ma), &ma, sizeof(ma));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_mfit), &lmfit, sizeof(lmfit));
	m = lmfit + 1;
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_mfit1), &m, sizeof(m));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_lastma), &llastma, sizeof(llastma));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_lastone), &llastone, sizeof(llastone));
	m = ma - 2 - n_ph_par;
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ncoef0), &m, sizeof(m));

	{
		size_t freeB = 0, totalB = 0;
		(void)hipMemGetInfo(&freeB, &totalB);
		const size_t needB = (size_t)allocBlocks * (sizeof(freq_context) + sizeof(freq_result)
			+ 3 * (size_t)(Numfac + 1) * sizeof(double)
			+ 2 * (size_t)(lmfit + 1) * (lmfit + 1) * sizeof(double)
			+ (size_t)(max_l_points + 1) * (size_t)(ma + 1) * sizeof(double)
			+ (size_t)(max_l_points + 1) * sizeof(double));
		if (needB > freeB - freeB / 10)
			fprintf(stderr, "period_search: WARNING need %.1f MB, only %.1f MB free\n",
			        (double)needB / 1048576.0, (double)freeB / 1048576.0);
	}

	err = hipMalloc(&pcc, allocBlocks * sizeof(freq_context));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_CC), &pcc, sizeof(pcc));
	err = hipMalloc(&pfr, allocBlocks * sizeof(freq_result));
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_FR), &pfr, sizeof(pfr));

	m = (Numfac + 1) * (n_coef + 1);
	hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Dg_block), &m, sizeof(m));

	double* pa, * pal, * pco, * pdytemp, * pytemp, * pgf;
	float* paf;

	err = hipMalloc(&pa, allocBlocks * (Numfac + 1) * sizeof(double));
	err = hipMemcpyToSymbol(HIP_SYMBOL(texArea), &pa, sizeof(pa));
	err = hipMalloc(&pal, allocBlocks * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	err = hipMalloc(&pco, allocBlocks * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	err = hipMalloc(&pdytemp, allocBlocks * (max_l_points + 1) * (ma + 1) * sizeof(double));
	err = hipMalloc(&pytemp, allocBlocks * (max_l_points + 1) * sizeof(double));
	err = hipMalloc(&pgf, allocBlocks * (Numfac + 1) * sizeof(double));
	err = hipMalloc(&paf, allocBlocks * (Numfac + 1) * sizeof(float));
	if (err != hipSuccess) { fprintf(stderr, "period_search: device allocation failed: %s\n", hipGetErrorString(err)); exit(1); }

	{
		void** hctx = (void**)malloc((size_t)allocBlocks * 7 * sizeof(void*));
		for (m = 0; m < allocBlocks; m++)
		{
			void** q = hctx + (size_t)m * 7;
			q[0] = &pa[m * (Numfac + 1)];
			q[1] = &pgf[m * (Numfac + 1)];
			q[2] = &pal[m * (lmfit + 1) * (lmfit + 1)];
			q[3] = &pco[m * (lmfit + 1) * (lmfit + 1)];
			q[4] = &pdytemp[m * (max_l_points + 1) * (ma + 1)];
			q[5] = &pytemp[m * (max_l_points + 1)];
			q[6] = &paf[m * (Numfac + 1)];
		}
		err = hipMemcpy2D((void*)pcc, sizeof(freq_context), hctx, 7 * sizeof(void*),
		                  7 * sizeof(void*), allocBlocks, hipMemcpyHostToDevice);
		free(hctx);
	}

	res = NULL;
	err = hipHostMalloc((void**)&res, allocBlocks * sizeof(freq_result), hipHostMallocDefault);
	if (err != hipSuccess || res == NULL) res = (freq_result*)malloc(allocBlocks * sizeof(freq_result));

	//int firstreport = 0;//beta debug
	auto oldFractionDone = 0.0001;

	for (n = n_start_from; n <= n_max; n += CUDA_grid_dim)
	{
		/* only launch blocks that correspond to real trial frequencies */
		const int grid = (n_max - n + 1 < CUDA_grid_dim) ? (n_max - n + 1) : CUDA_grid_dim;
		/* fractional progress at the END of this batch; 'n' is the first
		   frequency in it, so n/n_max would stay near zero when n_max is
		   smaller than the batch size (single-batch workunits). */
		const double total_freqs = (double)(n_max - n_start_from + 1);
		auto fractionDone = (double)(n - n_start_from + grid) / total_freqs;
		//boinc_fraction_done(fractionDone);

//#if _DEBUG
//		float fraction = fractionDone * 100;
//		std::time_t t = std::time(nullptr);   // get time now
//		std::tm* now = std::localtime(&t);
//
//		printf("%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction);
//		fprintf(stderr, "%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction);
//#endif

		CudaCalculatePrepare <<<grid, 1 >>> (n, n_max, freq_start, freq_step);
		//err = cudaThreadSynchronize();
		err = hipDeviceSynchronize();

		for (m = 1; m <= N_POLES; m++)
		{
			auto mid = (double(fractionDone) - double(oldFractionDone));
			auto inner = (double(mid) / double(N_POLES) * (double(m - 1)));
			//printf("mid: %.4f, inner: %.4f\n", mid, inner);
			auto fractionDone2 = oldFractionDone + inner;
			boinc_fraction_done(fractionDone2);

#ifdef _DEBUG
			float fraction2 = fractionDone2 * 100;
			//float fraction = fractionDone * 100;
			std::time_t t = std::time(nullptr);   // get time now
			std::tm* now = std::localtime(&t);

			printf("%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction2);
			fprintf(stderr, "%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction2);
#endif
			//zero global End signal
			theEnd = 0;
			hipMemcpyToSymbol(HIP_SYMBOL(CUDA_End), &theEnd, sizeof(theEnd));
			//
			CudaCalculatePreparePole <<<grid, 1 >>> (m);
			//
					bool firstIter = true;
	int count = 0;
	while (!theEnd)
			{
				count++;
				CudaCalculateIter1Begin <<<grid, 1 >>> ();
				//mrqcof
								if (firstIter)
				{
CudaCalculateIter1Mrqcof1Start <<<grid, CUDA_BLOCK_DIM >>> ();
				for (iC = 1; iC < l_curves; iC++)
				{
					LaunchCurve1Kernel(1, grid, in_rel[iC], l_points[iC], count);
					CudaCalculateIter1Mrqcof1Curve2 <<<grid, CURVE2_BLOCK_DIM >>> (in_rel[iC], l_points[iC]);
				}
				CudaCalculateIter1Mrqcof1Curve1Last <<<grid, CURVE1LAST_BLOCK_DIM >>> (in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof1Curve2 <<<grid, CURVE2_BLOCK_DIM >>> (in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof1End <<<grid, CUDA_BLOCK_DIM >>> ();
				}
				firstIter = false;
				//mrqcof
				CudaCalculateIter1Mrqmin1End <<<grid, CUDA_BLOCK_DIM >>> ();

				/*if (!if_freq_measured && nvml_enabled && n == n_start_from && m == N_POLES)
				{
					GetPeakClock(cudadev);
				}*/

				//mrqcof
				CudaCalculateIter1Mrqcof2Start <<<grid, CUDA_BLOCK_DIM >>> ();
				for (iC = 1; iC < l_curves; iC++)
				{
					LaunchCurve1Kernel(2, grid, in_rel[iC], l_points[iC], count);
					CudaCalculateIter1Mrqcof2Curve2 <<<grid, CURVE2_BLOCK_DIM >>> (in_rel[iC], l_points[iC]);
				}
				CudaCalculateIter1Mrqcof2Curve1Last <<<grid, CURVE1LAST_BLOCK_DIM >>> (in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof2Curve2 <<<grid, CURVE2_BLOCK_DIM >>> (in_rel[l_curves], l_points[l_curves]);
				CudaCalculateIter1Mrqcof2End <<<grid, CUDA_BLOCK_DIM >>> ();
				//mrqcof
				CudaCalculateIter1Mrqmin2End <<<grid, CUDA_BLOCK_DIM >>> ();
				CudaCalculateIter2 <<<grid, CUDA_BLOCK_DIM >>> ();
				/* Poll the completion counter every fifth iteration: the copy is a
				   blocking host round-trip that drains the queue, and all contexts
				   run a fixed iteration count, so at most four extra iterations are
				   executed and every kernel returns immediately in them. */
				if ((count % 5) == 0)
				{
					err = hipMemcpyFromSymbol(&theEnd, HIP_SYMBOL(CUDA_End), sizeof theEnd, 0, hipMemcpyDeviceToHost);
					theEnd = theEnd == grid;
				}

				//break;//debug
			}
			CudaCalculateFinishPole <<<grid, CUDA_BLOCK_DIM >>> ();
			//err = cudaThreadSynchronize();
			err = hipDeviceSynchronize();
			//			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*grid);
			//			err=cudaMemcpyFromSymbol(&resc,CUDA_CC,sizeof(freq_context)*grid);
						//break; //debug
		}

		CudaCalculateFinish <<<grid, 1 >>> ();
		//err=cudaThreadSynchronize(); memcpy is synchro itself

		//read results here
		err = hipMemcpy(res, pfr, sizeof(freq_result) * grid, hipMemcpyDeviceToHost);

		oldFractionDone = fractionDone;
		LinesWritten = 0;
		for (m = 1; m <= grid; m++)
		{
			if (res[m - 1].isReported == 1)
			{
				LinesWritten++;
				/* output file */
				if (n == 1 && m == 1)
				{
					mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * res[m - 1].per_best, res[m - 1].dev_best, res[m - 1].dev_best * res[m - 1].dev_best * (ndata - 3), conw_r * escl * escl, round(res[m - 1].la_best), round(res[m - 1].be_best));
				}
				else
				{
					mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * res[m - 1].per_best, res[m - 1].dev_best, res[m - 1].dev_best * res[m - 1].dev_best * (ndata - 3), res[m - 1].dark_best, round(res[m - 1].la_best), round(res[m - 1].be_best));
				}
			}
		}
		if (boinc_time_to_checkpoint() || boinc_is_standalone())
		{
			retval = DoCheckpoint(mf, (n - 1) + LinesWritten, 1, conw_r); //zero lines
			if (retval) { fprintf(stderr, "%s APP: period_search checkpoint failed %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval); exit(retval); }
			boinc_checkpoint_completed();
		}

		//		break;//debug
	} /* period loop */

	hipFree(pa);
	hipFree(paf);
	hipFree(pgf);
	hipFree(pal);
	hipFree(pco);
	hipFree(pdytemp);
	hipFree(pytemp);
	hipFree(pcc);
	hipFree(pfr);
	hipFree(pbrightness);
	hipFree(psig);

	hipHostFree(res);

	//nvmlShutdown();

	return 1;
}
