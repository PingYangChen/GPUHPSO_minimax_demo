#include "testgpu.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math_functions.h>

// Important Variables
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
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned int seed = i;
	curandState state;
	curand_init(seed, i, 0, &state);
	
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
		swarm[i] = curand_uniform(&state) * (U - L) + L; 
		//printf("S_i: %2.2f\n", swarm[i]);
		vel[i] = curand_uniform(&state); 
		//printf("V_i: %2.2f\n", vel[i]);
	}	
}

//
__global__ void cudaUpdateSwarm(float *swarm, float *vel, float *pBests, float *gBest, const int loopIdx, unsigned long seed)
{
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
	curand_init(seed, i, 0, &state);
	
	float r1 = curand_uniform(&state);
	float r2 = curand_uniform(&state);
	
	if (i < LENGTH) {
		//printf("%d: S_iA: %2.2f\n", loopIdx, swarm[i]);
		//printf("%d: V_iA: %2.2f\n", loopIdx, vel[i]);
		//printf("%d: pBests: %2.2f\n", loopIdx, pBests[i]);
		//printf("%d: Gindex: %d\n", loopIdx, indexTmp);
		//printf("%d: gBest: %2.2f\n", loopIdx, gBest[indexTmp]);
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
		//printf("%d: V_iB: %2.2f\n", loopIdx, vel[i]);
		//printf("%d: S_i: %2.2f\n", loopIdx, swarm[i]);
	}
}

//
__global__ void cudaUpdatePBest(float *swarm, float *pBests, float *fvals, float *fpvals, const int loopIdx, const int maximize, const int initial) {
	
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
	
	if (i < LENGTH) {
		if (initial == 1) {
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
		}
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
		if (loopIdx == 0) printf("Gbest: %2.2f \n", gBest[i]);
	}
}

__global__ void cudaEvalObjFunc(float *fvals, float *swarm, float *fixed, const int loopIdx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int LENGTH;
	int indexTmp;
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
				fvals[i] = obj(fixed[indexTmp], swarm[i]);
			}			
		break;
	}
}

//
void cudaHPSO()
{
	//
	Allocate_Memory();
	//
	int threadsNum = 32;
	int blocksNum0 = nSwarm1/threadsNum;
	int blocksNum1 = (nSwarm1 * nSwarm2)/threadsNum;
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
		cudaUpdatePBest<<<blocksNum1, threadsNum>>>(devSwarm2, devPBest2, devSvals2, devPvals2, 1, 1, 1);
		cudaUpdateGBest<<<blocksNum0, threadsNum>>>(devPBest2, devGBest2, devPvals2, devGBval2, 1, 1, 0);
	}
	// --------------------------------------------------------------------------------------------------- //
	cudaEvalObjFunc<<<blocksNum0, threadsNum>>>(devSvals1, devSwarm1, devGBval2, 0); 
	cudaUpdatePBest<<<blocksNum0, threadsNum>>>(devSwarm1, devPBest1, devSvals1, devPvals1, 0, 0, 1);
	cudaUpdateGBest<<<blocksNum0, threadsNum>>>(devPBest1, devGBest1, devPvals1, devGBval1, 0, 0, 1);
	for (t1 = 0; t1 < nIter1; t1++) {
		cudaUpdateSwarm<<<blocksNum0, threadsNum>>>(devSwarm1, devVeloc1, devPBest1, devGBest1, 0, unsigned(time(NULL)));
		// --------------------------------------------------------------------------------------------------- //
		cudaInitparticles<<<blocksNum1, threadsNum>>>(devSwarm2, devVeloc2, 1, unsigned(time(NULL))); 
		cudaEvalObjFunc<<<blocksNum1, threadsNum>>>(devSvals2, devSwarm2, devSwarm1, 1); 
		cudaUpdatePBest<<<blocksNum1, threadsNum>>>(devSwarm2, devPBest2, devSvals2, devPvals2, 1, 1, 1);
		cudaUpdateGBest<<<blocksNum0, threadsNum>>>(devPBest2, devGBest2, devPvals2, devGBval2, 1, 1, 0);
		for (t2 = 0; t2 < nIter2; t2++) {
			//printf("2: iteration %d\n", t2);
			cudaUpdateSwarm<<<blocksNum1, threadsNum>>>(devSwarm2, devVeloc2, devPBest2, devGBest2, 1, unsigned(time(NULL)));
			cudaEvalObjFunc<<<blocksNum1, threadsNum>>>(devSvals2, devSwarm2, devSwarm1, 1); 
			cudaUpdatePBest<<<blocksNum1, threadsNum>>>(devSwarm2, devPBest2, devSvals2, devPvals2, 1, 1, 1);
			cudaUpdateGBest<<<blocksNum0, threadsNum>>>(devPBest2, devGBest2, devPvals2, devGBval2, 1, 1, 1);
		}
		// --------------------------------------------------------------------------------------------------- //
		cudaEvalObjFunc<<<blocksNum0, threadsNum>>>(devSvals1, devSwarm1, devGBval2, 0); 
		cudaUpdatePBest<<<blocksNum0, threadsNum>>>(devSwarm1, devPBest1, devSvals1, devPvals1, 0, 0, 1);
		cudaUpdateGBest<<<blocksNum0, threadsNum>>>(devPBest1, devGBest1, devPvals1, devGBval1, 0, 0, 1);
	}
	
	Get_From_Device();
	printf("The best location is at x = %2.2f\n", GBest1[0]);
	//
	Free_Memory();
}

//
void Free_Memory() {
	// cleanup
	if (GBest1) free(GBest1);
	cudaError_t fError;
  if (devSwarm1) fError = cudaFree(devSwarm1); printf("CUDA error (free devSwarm1) = %s\n", cudaGetErrorString(fError));
  if (devVeloc1) fError = cudaFree(devVeloc1); printf("CUDA error (free devVeloc1) = %s\n", cudaGetErrorString(fError));
  if (devSvals1) fError = cudaFree(devSvals1); printf("CUDA error (free devSvals1) = %s\n", cudaGetErrorString(fError));
  if (devPvals1) fError = cudaFree(devPvals1); printf("CUDA error (free devPvals1) = %s\n", cudaGetErrorString(fError));
  if (devGBval1) fError = cudaFree(devGBval1); printf("CUDA error (free devGBval1) = %s\n", cudaGetErrorString(fError));
  if (devPBest1) fError = cudaFree(devPBest1); printf("CUDA error (free devPBest1) = %s\n", cudaGetErrorString(fError));
  if (devGBest1) fError = cudaFree(devGBest1); printf("CUDA error (free devGBest1) = %s\n", cudaGetErrorString(fError));
	//
  if (devSwarm2) fError = cudaFree(devSwarm2); printf("CUDA error (free devSwarm2) = %s\n", cudaGetErrorString(fError));
  if (devVeloc2) fError = cudaFree(devVeloc2); printf("CUDA error (free devVeloc2) = %s\n", cudaGetErrorString(fError));
  if (devSvals2) fError = cudaFree(devSvals2); printf("CUDA error (free devSvals2) = %s\n", cudaGetErrorString(fError));
  if (devPvals2) fError = cudaFree(devPvals2); printf("CUDA error (free devPvals2) = %s\n", cudaGetErrorString(fError));
  if (devGBval2) fError = cudaFree(devGBval2); printf("CUDA error (free devGBval2) = %s\n", cudaGetErrorString(fError));
  if (devPBest2) fError = cudaFree(devPBest2); printf("CUDA error (free devPBest2) = %s\n", cudaGetErrorString(fError));
  if (devGBest2) fError = cudaFree(devGBest2); printf("CUDA error (free devGBest2) = %s\n", cudaGetErrorString(fError));
}

//
void Allocate_Memory() {
	
	GBest1  = (float*)malloc(sizeof(float) * 1);
	cudaError_t Error;
  Error = cudaMalloc((void**)&devSwarm1, sizeof(float) * nSwarm1); 
	printf("CUDA error (malloc devSwarm1) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devVeloc1, sizeof(float) * nSwarm1); 
	printf("CUDA error (malloc devVeloc1) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devSvals1, sizeof(float) * nSwarm1); 
	printf("CUDA error (malloc devSvals1) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devPvals1, sizeof(float) * nSwarm1); 
	printf("CUDA error (malloc devPvals1) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devGBval1, sizeof(float) * 1);			 
	printf("CUDA error (malloc devGBval1) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devPBest1, sizeof(float) * nSwarm1); 
	printf("CUDA error (malloc devPBest1) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devGBest1, sizeof(float) * 1);			 
	printf("CUDA error (malloc devGBest1) = %s\n", cudaGetErrorString(Error));
	//
  Error = cudaMalloc((void**)&devSwarm2, sizeof(float) * nSwarm1 * nSwarm2);
	printf("CUDA error (malloc devSwarm2) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devVeloc2, sizeof(float) * nSwarm1 * nSwarm2);
	printf("CUDA error (malloc devVeloc2) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devSvals2, sizeof(float) * nSwarm1 * nSwarm2);
	printf("CUDA error (malloc devSvals2) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devPvals2, sizeof(float) * nSwarm1 * nSwarm2);
	printf("CUDA error (malloc devPvals2) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devGBval2, sizeof(float) * nSwarm1);
	printf("CUDA error (malloc devGBval2) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devPBest2, sizeof(float) * nSwarm1 * nSwarm2);
	printf("CUDA error (malloc devPBest2) = %s\n", cudaGetErrorString(Error));
  Error = cudaMalloc((void**)&devGBest2, sizeof(float) * nSwarm1);
	printf("CUDA error (malloc devGBest2) = %s\n", cudaGetErrorString(Error));
}

//
void Get_From_Device() {
	cudaError_t Error;
	Error = cudaMemcpy(GBest1, devGBest1, sizeof(float) * 1, cudaMemcpyDeviceToHost);
	printf("CUDA error (memcpy devGBest1 -> GBest1) = %s\n", cudaGetErrorString(Error));
}