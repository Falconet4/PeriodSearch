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

/* A4: report the exact failing HIP call instead of collapsing many results
   into one status that is tested only after the last call. */
static void ps_hip_ok(hipError_t e, const char* what)
{
	if (e != hipSuccess)
	{
		fprintf(stderr, "period_search: %s failed: %s\n", what, hipGetErrorString(e));
		fflush(stderr);
		exit(1);
	}
}

/* C1: the spin guard must allow the configured iteration count, which for
   stop_condition > 1 is n_iter_max (not just MAX_N_ITER). */
static long long ps_max_spin(int n_iter_max, int n_iter_min)
{
	int m = MAX_N_ITER;
	if (n_iter_max > m) m = n_iter_max;
	if (n_iter_min > m) m = n_iter_min;
	return (long long)m + 16;
}

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
int CUDA_warp_size = 32;
double* pee, * pee0, * pWeight;
bool nvml_enabled = false;

/* A9: the wave size the device code was actually compiled for.  warpSize is a
   device-side constant; on RDNA it is 32 by default but can be 64 in an
   explicit wave64 build, which deviceProp.warpSize does not reveal. */
__device__ int CUDA_WarpSizeCompiled;
__global__ void CudaRecordWarpSize(void)
{
	CUDA_WarpSizeCompiled = (int)warpSize;
}
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
	/* A9: use the wave size the kernels were compiled for, not the device's
	   native/default wave size.  The two differ for explicit wave64 RDNA
	   builds, where deviceProp.warpSize still reports 32. */
	CudaRecordWarpSize <<<1, 1 >>> ();
	ps_hip_ok(hipGetLastError(), "CudaRecordWarpSize launch");
	ps_hip_ok(hipMemcpyFromSymbol(&CUDA_warp_size, HIP_SYMBOL(CUDA_WarpSizeCompiled), sizeof(int), 0, hipMemcpyDeviceToHost), "CUDA_WarpSizeCompiled");
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

	//Global parameters
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_beta_pole), beta_pole, sizeof(double) * (N_POLES + 1)), "CUDA_beta_pole");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_lambda_pole), lambda_pole, sizeof(double) * (N_POLES + 1)), "CUDA_lambda_pole");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_par), par, sizeof(double) * 4), "CUDA_par");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_cl), &cl, sizeof(cl)), "CUDA_cl");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Alamda_start), &Alamda_start, sizeof(Alamda_start)), "CUDA_Alamda_start");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Alamda_incr), &Alamda_incr, sizeof(Alamda_incr)), "CUDA_Alamda_incr");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Mmax), &m_max, sizeof(m_max)), "CUDA_Mmax");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Lmax), &l_max, sizeof(l_max)), "CUDA_Lmax");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_tim), tim, sizeof(double) * (MAX_N_OBS + 1)), "CUDA_tim");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Phi_0), &Phi_0, sizeof(Phi_0)), "CUDA_Phi_0");

	ps_hip_ok(hipMalloc(&pWeight, (ndata + 3 + 1) * sizeof(double)), "hipMalloc(pWeight)");
	ps_hip_ok(hipMemcpy(pWeight, weight, (ndata + 3 + 1) * sizeof(double), hipMemcpyHostToDevice), "weight copy");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(texWeight), &pWeight, sizeof(pWeight)), "texWeight");

	ps_hip_ok(hipMalloc(&pee, (MAX_N_OBS + 1) * 3 * sizeof(double)), "hipMalloc(pee)");
	{
		double* soa = (double*)malloc((size_t)(MAX_N_OBS + 1) * 3 * sizeof(double));
		if (soa == NULL) { fprintf(stderr, "period_search: cannot allocate ee staging buffer\n"); fflush(stderr); exit(1); }
		for (int ii = 0; ii <= ndata; ii++)
			for (int cc = 0; cc < 3; cc++) soa[cc * (MAX_N_OBS + 1) + ii] = ee[ii][cc];
		ps_hip_ok(hipMemcpy(pee, soa, (MAX_N_OBS + 1) * 3 * sizeof(double), hipMemcpyHostToDevice), "ee copy");
		free(soa);
	}
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ee), &pee, sizeof(void*)), "CUDA_ee");

	ps_hip_ok(hipMalloc(&pee0, (MAX_N_OBS + 1) * 3 * sizeof(double)), "hipMalloc(pee0)");
	{
		double* soa = (double*)malloc((size_t)(MAX_N_OBS + 1) * 3 * sizeof(double));
		if (soa == NULL) { fprintf(stderr, "period_search: cannot allocate ee0 staging buffer\n"); fflush(stderr); exit(1); }
		for (int ii = 0; ii <= ndata; ii++)
			for (int cc = 0; cc < 3; cc++) soa[cc * (MAX_N_OBS + 1) + ii] = ee0[ii][cc];
		ps_hip_ok(hipMemcpy(pee0, soa, (MAX_N_OBS + 1) * 3 * sizeof(double), hipMemcpyHostToDevice), "ee0 copy");
		free(soa);
	}
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ee0), &pee0, sizeof(void*)), "CUDA_ee0");

	return 1;
}

void CUDAUnprepare(void)
{
	hipFree(pee);
	hipFree(pee0);
	hipFree(pWeight);
	/* A16: do not leave freed/dangling file-scope state behind if the process
	   is ever reused for another device or workunit. */
	pee = pee0 = pWeight = nullptr;
	CUDA_grid_dim = 0;
	CUDA_warp_size = 32;
	nvml_enabled = false;
}

/* A11: the host-side device setup that CUDAPrecalc and CUDAStart used to carry
   twice, verbatim.  The only real differences are the number of blocks, the
   precalc flag and conw_r (a pointer in CUDAPrecalc, a value in CUDAStart). */
struct ps_device_buffers
{
	void* pcc;
	void* pfr;
	void* pbrightness;
	void* psig;
	double* pa;
	double* pal;
	double* pco;
	double* pdytemp;
	double* pytemp;
	double* pgf;
	freq_result* res;
	bool resPinned;
	ps_device_buffers()
		: pcc(nullptr), pfr(nullptr), pbrightness(nullptr), psig(nullptr),
		  pa(nullptr), pal(nullptr), pco(nullptr), pdytemp(nullptr), pytemp(nullptr),
		  pgf(nullptr), res(nullptr), resPinned(false) {}
};

static void ps_device_setup(int blocks, int isPrecalc, double conw_r,
                            int* ia, int* ia_par, const double* cg_first, double* sig,
                            double* brightness, int Numfac,
                            int n_iter_max, int n_iter_min, int ndata,
                            double iter_diff_max, ps_device_buffers& b)
{
	int m;
	int isp = isPrecalc;

	/* A11 follow-up: this loop was duplicated in both callers. */
	for (int i = 1; i <= n_ph_par; i++)
		ia[n_coef + 3 + i] = ia_par[i];

	//here move data to device
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Ncoef), &n_coef, sizeof(n_coef)), "CUDA_Ncoef");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Nphpar), &n_ph_par, sizeof(n_ph_par)), "CUDA_Nphpar");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Numfac), &Numfac, sizeof(Numfac)), "CUDA_Numfac");
	m = Numfac + 1;
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Numfac1), &m, sizeof(m)), "CUDA_Numfac1");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ia), ia, sizeof(int) * (MAX_N_PAR + 1)), "CUDA_ia");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_cg_first), cg_first, sizeof(double) * (MAX_N_PAR + 1)), "CUDA_cg_first");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_n_iter_max), &n_iter_max, sizeof(n_iter_max)), "CUDA_n_iter_max");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_n_iter_min), &n_iter_min, sizeof(n_iter_min)), "CUDA_n_iter_min");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ndata), &ndata, sizeof(ndata)), "CUDA_ndata");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_iter_diff_max), &iter_diff_max, sizeof(iter_diff_max)), "CUDA_iter_diff_max");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_conw_r), &conw_r, sizeof(conw_r)), "CUDA_conw_r");
	{
		static double normalSoA[3][MAX_N_FAC + 1];
		for (int nf_i = 0; nf_i <= MAX_N_FAC; nf_i++)
			for (int nf_k = 0; nf_k < 3; nf_k++)
				normalSoA[nf_k][nf_i] = normal[nf_i][nf_k];
		ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Nor), normalSoA, sizeof(normalSoA)), "CUDA_Nor");
	}
	{
		static float normalf[(MAX_N_FAC + 1) * 3];
		for (int nf_i = 0; nf_i <= MAX_N_FAC; nf_i++)
			for (int nf_k = 0; nf_k < 3; nf_k++)
				normalf[nf_i * 3 + nf_k] = (float)normal[nf_i][nf_k];
		ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Norf), normalf, sizeof(normalf)), "CUDA_Norf");
	}
	{
		static double fcSoA[MAX_LM + 1][MAX_N_FAC + 1];
		static double fsSoA[MAX_LM + 1][MAX_N_FAC + 1];
		for (int i = 0; i <= MAX_N_FAC; i++)
			for (int m = 0; m <= MAX_LM; m++) { fcSoA[m][i] = f_c[i][m]; fsSoA[m][i] = f_s[i][m]; }
		ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Fc), fcSoA, sizeof(fcSoA)), "CUDA_Fc");
		ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Fs), fsSoA, sizeof(fsSoA)), "CUDA_Fs");
	}
	{
		static double plegSoA[MAX_LM + 1][MAX_LM + 1][MAX_N_FAC + 1];
		for (int i = 0; i <= MAX_N_FAC; i++)
			for (int l = 0; l <= MAX_LM; l++)
				for (int m = 0; m <= MAX_LM; m++)
					plegSoA[l][m][i] = pleg[i][l][m];
		ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Pleg), plegSoA, sizeof(plegSoA)), "CUDA_Pleg");
	}
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Darea), d_area, sizeof(double) * (MAX_N_FAC + 1)), "CUDA_Darea");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Dsph), d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1)), "CUDA_Dsph");
	{
		static float dsphf[(MAX_N_FAC + 1) * DSPHF_STRIDE];
		for (int i = 0; i <= MAX_N_FAC; i++)
			for (int k = 0; k <= MAX_N_PAR; k++)
				dsphf[i * DSPHF_STRIDE + k] = (float)d_sphere[i][k];
		ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_DsphF), dsphf, sizeof(dsphf)), "CUDA_DsphF");
	}
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Is_Precalc), &isp, sizeof isp, 0, hipMemcpyHostToDevice), "CUDA_Is_Precalc");

	ps_hip_ok(hipMalloc(&b.pbrightness, (ndata + 1) * sizeof(double)), "hipMalloc(pbrightness)");
	ps_hip_ok(hipMemcpy(b.pbrightness, brightness, (ndata + 1) * sizeof(double), hipMemcpyHostToDevice), "brightness copy");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(texbrightness), &b.pbrightness, sizeof(b.pbrightness)), "texbrightness");

	ps_hip_ok(hipMalloc(&b.psig, (ndata + 1) * sizeof(double)), "hipMalloc(psig)");
	ps_hip_ok(hipMemcpy(b.psig, sig, (ndata + 1) * sizeof(double), hipMemcpyHostToDevice), "sig copy");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(texsig), &b.psig, sizeof(b.psig)), "texsig");

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
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ma), &ma, sizeof(ma)), "CUDA_ma");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_mfit), &lmfit, sizeof(lmfit)), "CUDA_mfit");
	m = lmfit + 1;
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_mfit1), &m, sizeof(m)), "CUDA_mfit1");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_lastma), &llastma, sizeof(llastma)), "CUDA_lastma");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_lastone), &llastone, sizeof(llastone)), "CUDA_lastone");
	m = ma - 2 - n_ph_par;
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_ncoef0), &m, sizeof(m)), "CUDA_ncoef0");

	{
		size_t freeB = 0, totalB = 0;
		(void)hipMemGetInfo(&freeB, &totalB);
		const size_t needB = (size_t)blocks * (sizeof(freq_context) + sizeof(freq_result)
			+ 3 * (size_t)(Numfac + 1) * sizeof(double)
			+ 2 * (size_t)(lmfit + 1) * (lmfit + 1) * sizeof(double)
			+ (size_t)(max_l_points + 1) * (size_t)(ma + 1) * sizeof(double)
			+ (size_t)(max_l_points + 1) * sizeof(double));
		if (needB > freeB - freeB / 10)
			fprintf(stderr, "period_search: WARNING need %.1f MB, only %.1f MB free\n",
			        (double)needB / 1048576.0, (double)freeB / 1048576.0);
	}

	ps_hip_ok(hipMalloc(&b.pcc, blocks * sizeof(freq_context)), "hipMalloc(pcc)");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_CC), &b.pcc, sizeof(b.pcc)), "CUDA_CC");
	ps_hip_ok(hipMalloc(&b.pfr, blocks * sizeof(freq_result)), "hipMalloc(pfr)");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_FR), &b.pfr, sizeof(b.pfr)), "CUDA_FR");

	m = (Numfac + 1) * (n_coef + 1);
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Dg_block), &m, sizeof(m)), "CUDA_Dg_block");

	ps_hip_ok(hipMalloc(&b.pa, blocks * (Numfac + 1) * sizeof(double)), "hipMalloc(pa)");
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(texArea), &b.pa, sizeof(b.pa)), "texArea");
	ps_hip_ok(hipMalloc(&b.pal, blocks * (lmfit + 1) * (lmfit + 1) * sizeof(double)), "hipMalloc(pal)");
	ps_hip_ok(hipMalloc(&b.pco, blocks * (lmfit + 1) * (lmfit + 1) * sizeof(double)), "hipMalloc(pco)");
	ps_hip_ok(hipMalloc(&b.pdytemp, blocks * (max_l_points + 1) * (ma + 1) * sizeof(double)), "hipMalloc(pdytemp)");
	ps_hip_ok(hipMalloc(&b.pytemp, blocks * (max_l_points + 1) * sizeof(double)), "hipMalloc(pytemp)");
	ps_hip_ok(hipMalloc(&b.pgf, blocks * (Numfac + 1) * sizeof(double)), "hipMalloc(pgf)");

	{
		/* A12/B1: freq_context starts with exactly six pointer members then
		   double cg[MAX_N_PAR+1] at offset 48.  Copy six pointers, not seven. */
		void** hctx = (void**)malloc((size_t)blocks * 6 * sizeof(void*));
		if (hctx == NULL) { fprintf(stderr, "period_search: cannot allocate struct staging array\n"); fflush(stderr); exit(1); }
		for (m = 0; m < blocks; m++)
		{
			void** q = hctx + (size_t)m * 6;
			q[0] = &b.pa[m * (Numfac + 1)];
			q[1] = &b.pgf[m * (Numfac + 1)];
			q[2] = &b.pal[m * (lmfit + 1) * (lmfit + 1)];
			q[3] = &b.pco[m * (lmfit + 1) * (lmfit + 1)];
			q[4] = &b.pdytemp[m * (max_l_points + 1) * (ma + 1)];
			q[5] = &b.pytemp[m * (max_l_points + 1)];
		}
		ps_hip_ok(hipMemcpy2D((void*)b.pcc, sizeof(freq_context), hctx, 6 * sizeof(void*),
		                      6 * sizeof(void*), blocks, hipMemcpyHostToDevice), "hipMemcpy2D(freq_context)");
		free(hctx);
	}

	b.res = NULL;
	b.resPinned = false;
	hipError_t resErr = hipHostMalloc((void**)&b.res, blocks * sizeof(freq_result), hipHostMallocDefault);
	if (resErr == hipSuccess && b.res != NULL)
		b.resPinned = true;
	else
	{
		b.res = (freq_result*)malloc(blocks * sizeof(freq_result));
		if (b.res == NULL) { fprintf(stderr, "period_search: cannot allocate result buffer\n"); fflush(stderr); exit(1); }
	}
}

static void ps_device_teardown(ps_device_buffers& b)
{
	hipFree(b.pa);
	hipFree(b.pgf);
	hipFree(b.pal);
	hipFree(b.pco);
	hipFree(b.pdytemp);
	hipFree(b.pytemp);
	hipFree(b.pcc);
	hipFree(b.pfr);
	hipFree(b.pbrightness);
	hipFree(b.psig);
	if (b.resPinned) hipHostFree(b.res); else free(b.res);
	b = ps_device_buffers();
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

	n_iter_max = 0;
	iter_diff_max = -1;
	if (stop_condition >= 1)
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
	int CUDA_Grid_dim_precalc = max_test_periods * N_POLES;
	ps_device_buffers dev;
	ps_device_setup(CUDA_Grid_dim_precalc, isPrecalc, *conw_r, ia, ia_par, cg_first, sig, brightness, Numfac,
	                n_iter_max, n_iter_min, ndata, iter_diff_max, dev);
	freq_result* res = dev.res;
	void* pfr = dev.pfr;
	/*cudaMemcpyToSymbol(icol, &i_col, sizeof(i_col));
	cudaMemcpyToSymbol(pivinv, &piv_inv, sizeof(piv_inv));
	cudaMemcpyToSymbol(sh_icol, sh_icol_local, sizeof(int) * CUDA_BLOCK_DIM);
	cudaMemcpyToSymbol(sh_irow, sh_irow_local, sizeof(int) * CUDA_BLOCK_DIM);
	cudaMemcpyToSymbol(sh_big, sh_big_local, sizeof(double) * CUDA_BLOCK_DIM);*/


	for (n = 1; n <= max_test_periods; n += CUDA_Grid_dim_precalc)
	{
		CudaCalculatePreparePacked <<<CUDA_Grid_dim_precalc, 1 >>> (max_test_periods, freq_start, freq_step);
		//err = cudaThreadSynchronize();
		ps_hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize");

		{
			//zero global End signal
			theEnd = 0;
			ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_End), &theEnd, sizeof(theEnd), 0, hipMemcpyHostToDevice), "CUDA_End");
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
				/* B6: a failed kernel launch never raises CUDA_End.  Abort
				   instead of spinning forever.  The bound allows the configured
				   minimum iteration count plus polling slack. */
				if ((long long)count > ps_max_spin(n_iter_max, n_iter_min))
				{
					fprintf(stderr, "period_search: precalc did not terminate after %d iterations (kernel launch failure?)\n", count - 1);
					fflush(stderr);
					exit(1);
				}
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
				if (err != hipSuccess)
				{
					fprintf(stderr, "period_search: precalc completion read failed: %s\n", hipGetErrorString(err));
					fflush(stderr);
					exit(1);
				}
				theEnd = theEnd == CUDA_Grid_dim_precalc;

				//break;//debug
			}
			CudaCalculateFinishPole <<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >>> ();
			//err = cudaThreadSynchronize();
			ps_hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize");
			//			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*CUDA_Grid_dim_precalc);
			//			err=cudaMemcpyFromSymbol(&resc,CUDA_CC,sizeof(freq_context)*CUDA_Grid_dim_precalc);
						//break; //debug
		}
		printf("\n");

		CudaCalculateFinish <<<CUDA_Grid_dim_precalc, 1 >>> ();
		//err=cudaThreadSynchronize(); memcpy is synchro itself

		//read results here
		ps_hip_ok(hipMemcpy(res, pfr, sizeof(freq_result) * CUDA_Grid_dim_precalc, hipMemcpyDeviceToHost), "result copy");

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
	ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_Is_Precalc), &isPrecalc, sizeof(isPrecalc), 0, hipMemcpyHostToDevice), "CUDA_Is_Precalc");

	ps_device_teardown(dev);

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
	/* H2: a corrupt/stale checkpoint can put n_start_from outside the sweep.
	   Clamp it, and finish cleanly if there is nothing left to compute, so the
	   block count cannot go negative or wrap. */
	if (n_start_from < 1) n_start_from = 1;
	if (n_start_from > n_max) return 1;
	/* only the frequencies that will actually be launched need per-block storage */
	const int allocBlocks = (n_max - n_start_from + 1 < CUDA_grid_dim) ? (n_max - n_start_from + 1) : CUDA_grid_dim;
	int n_iter_max, theEnd, LinesWritten;
	double iter_diff_max;
	char buf[256];

	n_iter_max = 0;
	iter_diff_max = -1;
	if (stop_condition >= 1)
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
	ps_device_buffers dev;
	ps_device_setup(allocBlocks, 0, conw_r, ia, ia_par, cg_first, sig, brightness, Numfac,
	                n_iter_max, n_iter_min, ndata, iter_diff_max, dev);
	freq_result* res = dev.res;
	void* pfr = dev.pfr;
	/*cudaMemcpyToSymbol(icol, &i_col, sizeof(i_col));
	cudaMemcpyToSymbol(pivinv, &piv_inv, sizeof(piv_inv));
	cudaMemcpyToSymbol(sh_icol, sh_icol_local, sizeof(int) * CUDA_BLOCK_DIM);
	cudaMemcpyToSymbol(sh_irow, sh_irow_local, sizeof(int) * CUDA_BLOCK_DIM);
	cudaMemcpyToSymbol(sh_big, sh_big_local, sizeof(double) * CUDA_BLOCK_DIM);*/



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
		ps_hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize");

		for (m = 1; m <= N_POLES; m++)
		{
			auto mid = (double(fractionDone) - double(oldFractionDone));
			auto inner = (double(mid) / double(N_POLES) * (double(m)));
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
			ps_hip_ok(hipMemcpyToSymbol(HIP_SYMBOL(CUDA_End), &theEnd, sizeof(theEnd)), "CUDA_End");
			//
			CudaCalculatePreparePole <<<grid, 1 >>> (m);
			//
					bool firstIter = true;
	int count = 0;
	while (!theEnd)
			{
				count++;
				/* B6: bound the spin; see CUDAPrecalc. */
				if ((long long)count > ps_max_spin(n_iter_max, n_iter_min))
				{
					fprintf(stderr, "period_search: solve did not terminate after %d iterations (kernel launch failure?)\n", count - 1);
					fflush(stderr);
					exit(1);
				}
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
					if (err != hipSuccess)
					{
						fprintf(stderr, "period_search: solve completion read failed: %s\n", hipGetErrorString(err));
						fflush(stderr);
						exit(1);
					}
					theEnd = theEnd == grid;
				}

				//break;//debug
			}
			CudaCalculateFinishPole <<<grid, CUDA_BLOCK_DIM >>> ();
			//err = cudaThreadSynchronize();
			ps_hip_ok(hipDeviceSynchronize(), "hipDeviceSynchronize");
			//			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*grid);
			//			err=cudaMemcpyFromSymbol(&resc,CUDA_CC,sizeof(freq_context)*grid);
						//break; //debug
		}

		CudaCalculateFinish <<<grid, 1 >>> ();
		//err=cudaThreadSynchronize(); memcpy is synchro itself

		//read results here
		ps_hip_ok(hipMemcpy(res, pfr, sizeof(freq_result) * grid, hipMemcpyDeviceToHost), "result copy");

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

	ps_device_teardown(dev);

	//nvmlShutdown();

	return 1;
}
