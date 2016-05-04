#include "testgpu.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math_functions.h>

// Important Variables
float *hostSwarm1, *hostVeloc1, *hostPBest1, *hostGBest1, *hostSvals1, *hostPvals1, *hostGBval1;
float *hostSwarm2, *hostVeloc2, *hostPBest2, *hostGBest2, *hostSvals2, *hostPvals2, *hostGBval2;
float *GBest1;
float *devSwarm1, *devVeloc1, *devPBest1, *devGBest1, *devSvals1, *devPvals1, *devGBval1;
float *devSwarm2, *devVeloc2, *devPBest2, *devGBest2, *devSvals2, *devPvals2, *devGBval2;

// Global function
__global__ void cudaInitparticles(float *swarm, float *vel, const int loopIdx, unsigned long seed);
__global__ void cudaEvalObjFunc(float *fvals, float *swarm, float *fixed, const int loopIdx);
__global__ void cudaUpdateSwarm(float *swarm, float *vel, float *pBests, float *gBest, const int loopIdx, unsigned long seed);
__global__ void cudaUpdatePBest(float *swarm, float *pBests, float *fvals, float *fpvals, const int loopIdx, const int maximize, const int initial);
__global__ void cudaUpdateGBest(float *pBests, float *gBest, float *fpvals, float *fgval, const int loopIdx, const int maximize, const int initial);

// Device function
__device__ float obj(float x, float y);

//
__device__ float obj(float x, float y) 
{
	float res = (x - 1.0)*(x - 1.0) - y*y;
  return res;
}

//
__global__ void cudaInitparticles(float *swarm, float *vel, const int loopIdx, unsigned long seed) 
{
	int I = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned int seed = i;
	curandState state;
	curand_init(seed, I, 0, &state);
	
	__shared__ float tmp_swarm[512]; 
	__shared__ float tmp_vel[512];
	
	int LENGTH;
	float U, L; 
	switch (loopIdx) {
		case 0:
			LENGTH = nSwarm1; U = Ux; L = Lx; 
		break;
		case 1:
			LENGTH = nSwarm1 * nSwarm2; U = Uy; L = Ly; 
		break;
	}
	if (i < LENGTH) {
		tmp_swarm[I] = curand_uniform(&state) * (U - L) + L;
		swarm[i] = tmp_swarm[I];
		//swarm[i] = curand_uniform(&state) * (U - L) + L; 
		//if (i == 0) printf("Loop: %d, S_0: %2.2f\n", loopIdx, swarm[i]);
		tmp_vel[I] = curand_uniform(&state);
		vel[i] = tmp_vel[I];
		//vel[i] = curand_uniform(&state); 
		//if (i == 0) printf("Loop: %d, V_0: %2.2f\n", loopIdx, vel[i]);
	}	
}

//
__global__ void cudaUpdateSwarm(float *swarm, float *vel, float *pBests, float *gBest, const int loopIdx, unsigned long seed)
{
	int I = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int indexTmp, LENGTH;
	float L, U;
	switch (loopIdx) {
		case 0:
			LENGTH = nSwarm1; 
			indexTmp = (int)(i/nSwarm1);
			L = Lx; U = Ux;
		break;
		case 1:
			LENGTH = nSwarm1 * nSwarm2; 
			indexTmp = (int)(i/nSwarm2);
			L = Ly; U = Uy;
		break;
	}
	
	//unsigned int seed = i;
	curandState state;
	curand_init(seed, I, 0, &state);
	
	__shared__ float tmp_swarm[512]; 
	__shared__ float tmp_vel[512];
	__shared__ float tmp_pBests[512];
	
	float r1 = curand_uniform(&state);
	float r2 = curand_uniform(&state);
	
	if (i < LENGTH) {
		//printf("%d: S_iA: %2.2f\n", loopIdx, swarm[i]);
		//if (i == 0) printf("Loop: %d, V_0A: %2.2f\n", loopIdx, vel[i]);
		//printf("%d: pBests: %2.2f\n", loopIdx, pBests[i]);
		//printf("%d: Gindex: %d\n", loopIdx, indexTmp);
		//printf("%d: gBest: %2.2f\n", loopIdx, gBest[indexTmp]);
		tmp_swarm[I] = swarm[i];
		tmp_vel[I] = vel[i];
		tmp_pBests[I] = pBests[i];
		
		tmp_vel[I] = omg * tmp_vel[I] + c1 * r1 * (tmp_pBests[I] - tmp_swarm[I]) + c2 * r2 * (gBest[indexTmp] - tmp_swarm[I]);
		if (tmp_vel[I] > vmax) {
			tmp_vel[I] = vmax;
		}
		if (tmp_vel[I] < -1.0*vmax) {
			tmp_vel[I] = -1.0*vmax;
		}
		tmp_swarm[I] += tmp_vel[I];
		if (tmp_swarm[I] > U) tmp_swarm[I] = U;
		if (tmp_swarm[I] < L) tmp_swarm[I] = L;
		
		swarm[i] = tmp_swarm[I];
		vel[i] = tmp_vel[I];
		/*
		vel[i] = omg * vel[i] + c1 * r1 * (pBests[i] - swarm[i]) + c2 * r2 * (gBest[indexTmp] - swarm[i]);
		if (vel[i] > vmax) {
			vel[i] = vmax;
		}
		if (vel[i] < -1.0*vmax) {
			vel[i] = -1.0*vmax;
		}
		swarm[i] += vel[i];
		if (swarm[i] > U) swarm[i] = U;
		if (swarm[i] < L) swarm[i] = L;
		*/
		//if (i == 0) printf("Loop: %d, V_0B: %2.2f\n", loopIdx, vel[i]);
		//printf("%d: S_i: %2.2f\n", loopIdx, swarm[i]);
	}
}

//
__global__ void cudaUpdatePBest(float *swarm, float *pBests, float *fvals, float *fpvals, const int loopIdx, const int maximize, const int initial) 
{
	int I = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int LENGTH;
	switch (loopIdx) {
		case 0:
			LENGTH = nSwarm1; 
		break;
		case 1:
			LENGTH = nSwarm1 * nSwarm2; 
		break;
	}
	
	__shared__ float tmp_swarm[512]; 
	__shared__ float tmp_pBests[512];
	__shared__ float tmp_fvals[512];
	__shared__ float tmp_fpvals[512];
	
	if (i < LENGTH) {
		tmp_swarm[I] = swarm[i];
		tmp_pBests[I] = pBests[i];
		tmp_fvals[I] = fvals[i];
		tmp_fpvals[I] = fpvals[i];
		
		if (initial == 1) {
			tmp_fpvals[I] = tmp_fvals[I];
			tmp_pBests[I] = tmp_swarm[I];
		} else {
			if (maximize == 1) {
				if (tmp_fvals[I] > tmp_fpvals[I]) {
					tmp_fpvals[I] = tmp_fvals[I];
					tmp_pBests[I] = tmp_swarm[I];
				}
			} else {
				if (tmp_fvals[I] < tmp_fpvals[I]) {
					tmp_fpvals[I] = tmp_fvals[I];
					tmp_pBests[I] = tmp_swarm[I];
				}
			}
		}
		pBests[i] = tmp_pBests[I];
		fpvals[i] = tmp_fpvals[I];
		/*if (initial == 1) {
			fpvals[i] = fvals[i];
			pBests[i] = swarm[i];
		} else {
			if (maximize == 1) {
				if (fvals[i] > fpvals[i]) {
					fpvals[i] = fvals[i];
					pBests[i] = swarm[i];
				}
			} else {
				if (fvals[i] < fpvals[i]) {
					fpvals[i] = fvals[i];
					pBests[i] = swarm[i];
				}
			}
		}*/
	}
}

//
__global__ void cudaUpdateGBest(float *pBests, float *gBest, float *fpvals, float *fgval, const int loopIdx, const int maximize, const int initial)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int lenGBest, lenPBest;
	
	switch (loopIdx) {
		case 0:
			lenGBest = 1; lenPBest = nSwarm1;
		break;
		case 1: 
			lenGBest = nSwarm1; lenPBest = nSwarm2;
		break;
	}
	
	if (i < lenGBest) {
		
		float bestValInEachSwarm = fpvals[i*lenPBest];
		int bestLocInEachSwarm = 0;
		int k;
		if (maximize == 1) {
			for (k = 1; k < lenPBest; k++) {
				if (fpvals[i*lenPBest + k] > bestValInEachSwarm) {
					bestValInEachSwarm = fpvals[i*lenPBest + k];	bestLocInEachSwarm = k;
				}
			}
		} else {
			for (k = 1; k < lenPBest; k++) {
				if (fpvals[i*lenPBest + k] < bestValInEachSwarm) {
					bestValInEachSwarm = fpvals[i*lenPBest + k];	bestLocInEachSwarm = k;
				}
			}
		}
		
		//if (loopIdx == 1) printf("Swarm %d, Gbest: %2.2f \n", i, gBest[i]);
		
		if (initial == 1) {
			fgval[i] = fpvals[i*lenPBest + bestLocInEachSwarm];
			gBest[i] = pBests[i*lenPBest + bestLocInEachSwarm];
		} else {		
			if (maximize == 1) {
				if (fpvals[i*lenPBest + bestLocInEachSwarm] > fgval[i]) {
					fgval[i] = fpvals[i*lenPBest + bestLocInEachSwarm];
					gBest[i] = pBests[i*lenPBest + bestLocInEachSwarm];
				}
			}	else {
				if (fpvals[i*lenPBest + bestLocInEachSwarm] < fgval[i]) {
					fgval[i] = fpvals[i*lenPBest + bestLocInEachSwarm];
					gBest[i] = pBests[i*lenPBest + bestLocInEachSwarm];
				}
			}
		}
		//if (loopIdx == 0) printf("Gbest: %2.2f \n", gBest[i]);
	}
}

__global__ void cudaEvalObjFunc(float *fvals, float *swarm, float *fixed, const int loopIdx)
{
	int I = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int LENGTH;
	int indexTmp;
	
	__shared__ float tmp_swarm[512]; 
	__shared__ float tmp_fvals[512];
		
	switch (loopIdx) {
		case 0:
			LENGTH = nSwarm1; 
			if (i < LENGTH) {
				fvals[i] = fixed[i];				
			}
		break;
		case 1:
			LENGTH = nSwarm1 * nSwarm2; 
			if (i < LENGTH) {
				indexTmp = (int)(i/nSwarm2);
				tmp_swarm[I] = swarm[i];
				tmp_fvals[I] = fvals[i];
				tmp_fvals[I] = obj(fixed[indexTmp], tmp_swarm[I]);
				swarm[i] = tmp_swarm[I];
				fvals[i] = tmp_fvals[I];				
				//fvals[i] = obj(fixed[indexTmp], swarm[i]);
			}			
		break;
	}
}

//
void cudaHPSO()
{
	//
	int threadsNum = 512;
	int blocksNum0 = (int)((nSwarm1 + threadsNum - 1)/threadsNum); 
	int blocksNum1 = (int)(((nSwarm1 * nSwarm2) + threadsNum - 1)/threadsNum); 
	//
	int t1, t2;
	// Start Hierarchical PSO with GPU
	cudaInitparticles<<<blocksNum0, threadsNum>>>(devSwarm1, devVeloc1, 0, unsigned(time(NULL)));
	// --------------------------------------------------------------------------------------------------- //
	cudaInitparticles<<<blocksNum1, threadsNum>>>(devSwarm2, devVeloc2, 1, unsigned(time(NULL))); 
	cudaEvalObjFunc<<<blocksNum1, threadsNum>>>(devSvals2, devSwarm2, devSwarm1, 1); 
	cudaUpdatePBest<<<blocksNum1, threadsNum>>>(devSwarm2, devPBest2, devSvals2, devPvals2, 1, 1, 1);
	cudaUpdateGBest<<<blocksNum0, threadsNum>>>(devPBest2, devGBest2, devPvals2, devGBval2, 1, 1, 1);
	for (t2 = 0; t2 < nIter2; t2++) {
		//printf("2: iteration %d\n", t2);
		cudaUpdateSwarm<<<blocksNum1, threadsNum>>>(devSwarm2, devVeloc2, devPBest2, devGBest2, 1, unsigned(time(NULL)));
		cudaEvalObjFunc<<<blocksNum1, threadsNum>>>(devSvals2, devSwarm2, devSwarm1, 1); 
		cudaUpdatePBest<<<blocksNum1, threadsNum>>>(devSwarm2, devPBest2, devSvals2, devPvals2, 1, 1, 0);
		cudaUpdateGBest<<<blocksNum0, threadsNum>>>(devPBest2, devGBest2, devPvals2, devGBval2, 1, 1, 0);
	}
	// --------------------------------------------------------------------------------------------------- //
	cudaEvalObjFunc<<<blocksNum0, threadsNum>>>(devSvals1, devSwarm1, devGBval2, 0); 
	cudaUpdatePBest<<<blocksNum0, threadsNum>>>(devSwarm1, devPBest1, devSvals1, devPvals1, 0, 0, 1);
	cudaUpdateGBest<<<1, 1>>>(devPBest1, devGBest1, devPvals1, devGBval1, 0, 0, 1);
	for (t1 = 0; t1 < nIter1; t1++) {
		cudaUpdateSwarm<<<blocksNum0, threadsNum>>>(devSwarm1, devVeloc1, devPBest1, devGBest1, 0, unsigned(time(NULL)));
		// --------------------------------------------------------------------------------------------------- //
		cudaInitparticles<<<blocksNum1, threadsNum>>>(devSwarm2, devVeloc2, 1, unsigned(time(NULL))); 
		cudaEvalObjFunc<<<blocksNum1, threadsNum>>>(devSvals2, devSwarm2, devSwarm1, 1); 
		cudaUpdatePBest<<<blocksNum1, threadsNum>>>(devSwarm2, devPBest2, devSvals2, devPvals2, 1, 1, 1);
		cudaUpdateGBest<<<blocksNum0, threadsNum>>>(devPBest2, devGBest2, devPvals2, devGBval2, 1, 1, 1);
		for (t2 = 0; t2 < nIter2; t2++) {
			//printf("2: iteration %d\n", t2);
			cudaUpdateSwarm<<<blocksNum1, threadsNum>>>(devSwarm2, devVeloc2, devPBest2, devGBest2, 1, unsigned(time(NULL)));
			cudaEvalObjFunc<<<blocksNum1, threadsNum>>>(devSvals2, devSwarm2, devSwarm1, 1); 
			cudaUpdatePBest<<<blocksNum1, threadsNum>>>(devSwarm2, devPBest2, devSvals2, devPvals2, 1, 1, 0);
			cudaUpdateGBest<<<blocksNum0, threadsNum>>>(devPBest2, devGBest2, devPvals2, devGBval2, 1, 1, 0);
		}
		// --------------------------------------------------------------------------------------------------- //
		cudaEvalObjFunc<<<blocksNum0, threadsNum>>>(devSvals1, devSwarm1, devGBval2, 0); 
		cudaUpdatePBest<<<blocksNum0, threadsNum>>>(devSwarm1, devPBest1, devSvals1, devPvals1, 0, 0, 0);
		cudaUpdateGBest<<<1, 1>>>(devPBest1, devGBest1, devPvals1, devGBval1, 0, 0, 0);
	}
	// HPSO done
}

//
void Free_Memory() {
	// cleanup
	if (hostSwarm1) free(hostSwarm1);
	if (hostVeloc1) free(hostVeloc1);
	if (hostSvals1) free(hostSvals1);
	if (hostPvals1) free(hostPvals1);
	if (hostGBval1) free(hostGBval1);
	if (hostPBest1) free(hostPBest1);
	if (hostGBest1) free(hostGBest1);
	//
	if (hostSwarm2) free(hostSwarm2);
	if (hostVeloc2) free(hostVeloc2);
	if (hostSvals2) free(hostSvals2);
	if (hostPvals2) free(hostPvals2);
	if (hostGBval2) free(hostGBval2);
	if (hostPBest2) free(hostPBest2);
	if (hostGBest2) free(hostGBest2);
	
	cudaError_t fError;	
	if (GBest1) fError = cudaFreeHost(GBest1); if (fError != 0) printf("CUDA error (free hostGBest1) = %s\n", cudaGetErrorString(fError));
	
  if (devSwarm1) fError = cudaFree(devSwarm1); if (fError != 0) printf("CUDA error (free devSwarm1) = %s\n", cudaGetErrorString(fError));
  if (devVeloc1) fError = cudaFree(devVeloc1); if (fError != 0) printf("CUDA error (free devVeloc1) = %s\n", cudaGetErrorString(fError));
  if (devSvals1) fError = cudaFree(devSvals1); if (fError != 0) printf("CUDA error (free devSvals1) = %s\n", cudaGetErrorString(fError));
  if (devPvals1) fError = cudaFree(devPvals1); if (fError != 0) printf("CUDA error (free devPvals1) = %s\n", cudaGetErrorString(fError));
  if (devGBval1) fError = cudaFree(devGBval1); if (fError != 0) printf("CUDA error (free devGBval1) = %s\n", cudaGetErrorString(fError));
  if (devPBest1) fError = cudaFree(devPBest1); if (fError != 0) printf("CUDA error (free devPBest1) = %s\n", cudaGetErrorString(fError));
  if (devGBest1) fError = cudaFree(devGBest1); if (fError != 0) printf("CUDA error (free devGBest1) = %s\n", cudaGetErrorString(fError));
	//
  if (devSwarm2) fError = cudaFree(devSwarm2); if (fError != 0) printf("CUDA error (free devSwarm2) = %s\n", cudaGetErrorString(fError));
  if (devVeloc2) fError = cudaFree(devVeloc2); if (fError != 0) printf("CUDA error (free devVeloc2) = %s\n", cudaGetErrorString(fError));
  if (devSvals2) fError = cudaFree(devSvals2); if (fError != 0) printf("CUDA error (free devSvals2) = %s\n", cudaGetErrorString(fError));
  if (devPvals2) fError = cudaFree(devPvals2); if (fError != 0) printf("CUDA error (free devPvals2) = %s\n", cudaGetErrorString(fError));
  if (devGBval2) fError = cudaFree(devGBval2); if (fError != 0) printf("CUDA error (free devGBval2) = %s\n", cudaGetErrorString(fError));
  if (devPBest2) fError = cudaFree(devPBest2); if (fError != 0) printf("CUDA error (free devPBest2) = %s\n", cudaGetErrorString(fError));
  if (devGBest2) fError = cudaFree(devGBest2); if (fError != 0) printf("CUDA error (free devGBest2) = %s\n", cudaGetErrorString(fError));
}

//
void Allocate_Memory() {
	
	size_t alignment = 32;
	posix_memalign((void**)&hostSwarm1, alignment, sizeof(float) * nSwarm1);
	posix_memalign((void**)&hostVeloc1, alignment, sizeof(float) * nSwarm1);
	posix_memalign((void**)&hostSvals1, alignment, sizeof(float) * nSwarm1);
	posix_memalign((void**)&hostPvals1, alignment, sizeof(float) * nSwarm1);
	posix_memalign((void**)&hostGBval1, alignment, sizeof(float) * 1);
	posix_memalign((void**)&hostPBest1, alignment, sizeof(float) * nSwarm1);
	posix_memalign((void**)&hostGBest1, alignment, sizeof(float) * 1);
	
	posix_memalign((void**)&hostSwarm2, alignment, sizeof(float) * nSwarm2);
	posix_memalign((void**)&hostVeloc2, alignment, sizeof(float) * nSwarm2);
	posix_memalign((void**)&hostSvals2, alignment, sizeof(float) * nSwarm2);
	posix_memalign((void**)&hostPvals2, alignment, sizeof(float) * nSwarm2);
	posix_memalign((void**)&hostGBval2, alignment, sizeof(float) * 1);
	posix_memalign((void**)&hostPBest2, alignment, sizeof(float) * nSwarm2);
	posix_memalign((void**)&hostGBest2, alignment, sizeof(float) * 1);
	
	cudaError_t Error;
	
  Error = cudaMallocHost((void**)&GBest1, sizeof(float) * 1); 
	if (Error != 0) printf("CUDA error (malloc hostGBest1) = %s\n", cudaGetErrorString(Error));
	
  Error = cudaMalloc((void**)&devSwarm1, sizeof(float) * nSwarm1); 
	if (Error != 0) printf("CUDA error (malloc devSwarm1) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devVeloc1, sizeof(float) * nSwarm1); 
	if (Error != 0) printf("CUDA error (malloc devVeloc1) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devSvals1, sizeof(float) * nSwarm1); 
	if (Error != 0) printf("CUDA error (malloc devSvals1) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devPvals1, sizeof(float) * nSwarm1); 
	if (Error != 0) printf("CUDA error (malloc devPvals1) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devGBval1, sizeof(float) * 1);			 
	if (Error != 0) printf("CUDA error (malloc devGBval1) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devPBest1, sizeof(float) * nSwarm1); 
	if (Error != 0) printf("CUDA error (malloc devPBest1) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devGBest1, sizeof(float) * 1);
	if (Error != 0) printf("CUDA error (malloc devGBest1) = %s\n", cudaGetErrorString(Error));
	//
  Error = cudaMalloc((void**)&devSwarm2, sizeof(float) * nSwarm1 * nSwarm2);
	if (Error != 0) printf("CUDA error (malloc devSwarm2) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devVeloc2, sizeof(float) * nSwarm1 * nSwarm2);
	if (Error != 0) printf("CUDA error (malloc devVeloc2) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devSvals2, sizeof(float) * nSwarm1 * nSwarm2);
	if (Error != 0) printf("CUDA error (malloc devSvals2) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devPvals2, sizeof(float) * nSwarm1 * nSwarm2);
	if (Error != 0) printf("CUDA error (malloc devPvals2) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devGBval2, sizeof(float) * nSwarm1);
	if (Error != 0) printf("CUDA error (malloc devGBval2) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devPBest2, sizeof(float) * nSwarm1 * nSwarm2);
	if (Error != 0) printf("CUDA error (malloc devPBest2) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devGBest2, sizeof(float) * nSwarm1);
	if (Error != 0) printf("CUDA error (malloc devGBest2) = %s\n", cudaGetErrorString(Error));
}

//
void Get_From_Device() {
	cudaError_t Error;
	Error = cudaMemcpy(GBest1, devGBest1, sizeof(float) * 1, cudaMemcpyDeviceToHost);
	if (Error != 0) printf("CUDA error (memcpy devGBest1 -> GBest1) = %s\n", cudaGetErrorString(Error));
}