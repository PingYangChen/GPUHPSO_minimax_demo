#include "testgpu.h"

extern float *GBest1;
extern float *hostSwarm1, *hostVeloc1, *hostPBest1, *hostGBest1, *hostSvals1, *hostPvals1, *hostGBval1;
extern float *hostSwarm2, *hostVeloc2, *hostPBest2, *hostGBest2, *hostSvals2, *hostPvals2, *hostGBval2;

void Initparticles(float *swarm, float *vel, const int loopIdx);
void EvalObjFunc(float *fvals, float *swarm, const float fixed, const int loopIdx);
void UpdateSwarm(float *swarm, float *vel, float *pBests, float *gBest, const int loopIdx);
void UpdatePBest(float *swarm, float *pBests, float *fvals, float *fpvals, const int loopIdx, const int maximize, const int initial);
void UpdateGBest(float *pBests, float *gBest, float *fpvals, float *fgval, const int loopIdx, const int maximize, const int initial);
float hostObjFunc(float x, float y);
void cpuHPSO(float *swarm, float *vel, float *fvals, const float fixed, float *pBests, float *gBest, float *fpvals, float *fgval, const int loopIdx, const int maximize, const int nIter);

int main() 
{
	printf("------------------------------------------- \n");
	printf("Loop 0: nSwarm = %d; nIteration: %d\n", nSwarm1, nIter1);
	printf("Loop 1: nSwarm = %d; nIteration: %d\n", nSwarm2, nIter2);
	
	Allocate_Memory();
	//
	struct timeval start, end;   
	double cputime, gputime, gputime1, gputime2;
		// Set timer
	gettimeofday(&start,NULL); 	
	cudaHPSO();
	Get_From_Device();
	gettimeofday(&end,NULL);	// Stop stopwatch and compute time
	printf("The best location is at x = %2.2f\n", GBest1[0]);
	gputime = ((end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec)/1000000.0);
	printf("GPU Time = %f sec.\n", gputime);
	//printf("------------------------------------------- \n");
	//	
	gettimeofday(&start,NULL); 	
	cpuHPSO(hostSwarm1, hostVeloc1, hostSvals1, 0.0, hostPBest1, hostGBest1, hostPvals1, hostGBval1, 0, 0, nIter1);
	gettimeofday(&end,NULL);	// Stop stopwatch and compute time
	printf("------------------------------------------- \n");
	printf("CPU: The best location is at x = %2.2f\n", hostGBest1[0]);
	cputime = ((end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec)/1000000.0);
	printf("CPU Time = %f sec.\n", cputime);
	printf("------------------------------------------- \n");

	Free_Memory();
  return 0;
	
}

float hostObjFunc(float x, float y) 
{
	float res = (x - 1.0)*(x - 1.0) - y*y;
	return res;
}

//
void Initparticles(float *swarm, float *vel, const int loopIdx) 
{
	int i;	
	int LENGTH;
	float U, L; 
	switch (loopIdx) {
		case 0:
			LENGTH = nSwarm1; U = Ux; L = Lx; 
		break;
		case 1:
			LENGTH = nSwarm2; U = Uy; L = Ly; 
		break;
	}
	for (i = 0; i < LENGTH; i++) {
		swarm[i] = ((float)rand()/(float)RAND_MAX) * (U - L) + L; 
		vel[i] = ((float)rand()/(float)RAND_MAX); 
		//if (i == 0) printf("Loop: %d, S_%d: %2.2f\n", loopIdx, i, swarm[i]);
	}	
}

//
void UpdateSwarm(float *swarm, float *vel, float *pBests, float *gBest, const int loopIdx)
{
	int i;
	int indexTmp, LENGTH;
	float L, U;
	switch (loopIdx) {
		case 0:
			LENGTH = nSwarm1; 
			L = Lx; U = Ux;
		break;
		case 1:
			LENGTH = nSwarm2; 
			L = Ly; U = Uy;
		break;
	}
	
	float r1;
	float r2;
	
	for (i = 0; i < LENGTH; i++) {
		r1 = (float)rand()/(float)RAND_MAX;
		r2 = (float)rand()/(float)RAND_MAX;
		//if (i == 0) printf("Loop: %d, V_0A: %2.2f\n", loopIdx, vel[i]);
		vel[i] = omg * vel[i] + c1 * r1 * (pBests[i] - swarm[i]) + c2 * r2 * (gBest[0] - swarm[i]);
		if (vel[i] > vmax) {
			vel[i] = vmax;
		}
		if (vel[i] < -1.0*vmax) {
			vel[i] = -1.0*vmax;
		}
		swarm[i] += vel[i];
		if (swarm[i] > U) swarm[i] = U;
		if (swarm[i] < L) swarm[i] = L;
		//if (i == 0) printf("Loop: %d, V_0B: %2.2f\n", loopIdx, vel[i]);
	}
}

//
void UpdatePBest(float *swarm, float *pBests, float *fvals, float *fpvals, const int loopIdx, const int maximize, const int initial) {
	
	int i;
	int LENGTH;
	switch (loopIdx) {
		case 0:
			LENGTH = nSwarm1; 
		break;
		case 1:
			LENGTH = nSwarm2; 
		break;
	}
	
	for (i = 0; i < LENGTH; i++) {
		//if (i == 0) printf("Loop: %d, fvals0: %2.2f \n", loopIdx, fvals[i]);
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
		//if (i == 0) printf("Loop: %d, pbest0: %2.2f \n", loopIdx, pBests[i]);
	}
}

//
void UpdateGBest(float *pBests, float *gBest, float *fpvals, float *fgval, const int loopIdx, const int maximize, const int initial)
{
	
	int lenPBest;
	switch (loopIdx) {
		case 0:
			lenPBest = nSwarm1;
		break;
		case 1: 
			lenPBest = nSwarm2;
		break;
	}
		
	float bestValInEachSwarm = fpvals[0];
	int bestLocInEachSwarm = 0;
	int k;
	if (maximize == 1) {
		for (k = 1; k < lenPBest; k++) {
			if (fpvals[k] > bestValInEachSwarm) {
				bestValInEachSwarm = fpvals[k];	bestLocInEachSwarm = k;
			}
		}
	} else {
		for (k = 1; k < lenPBest; k++) {
			if (fpvals[k] < bestValInEachSwarm) {
				bestValInEachSwarm = fpvals[k];	bestLocInEachSwarm = k;
			}
		}
	}
		
	if (initial == 1) {
		fgval[0] = fpvals[bestLocInEachSwarm];
		gBest[0] = pBests[bestLocInEachSwarm];
	} else {		
		if (maximize == 1) {
			if (fpvals[bestLocInEachSwarm] > fgval[0]) {
				fgval[0] = fpvals[bestLocInEachSwarm];
				gBest[0] = pBests[bestLocInEachSwarm];
			}
		}	else {
			if (fpvals[bestLocInEachSwarm] < fgval[0]) {
				fgval[0] = fpvals[bestLocInEachSwarm];
				gBest[0] = pBests[bestLocInEachSwarm];
			}
		}
	}
	//printf("Loop: %d, Gbest: %2.2f \n", loopIdx, gBest[0]);
}

void EvalObjFunc(float *fvals, float *swarm, const float fixed, const int loopIdx)
{
	int i;
	int LENGTH;
	int indexTmp;
	switch (loopIdx) {
		case 0:
			LENGTH = nSwarm1; 
			for (i = 0; i < LENGTH; i++) {
				cpuHPSO(hostSwarm2, hostVeloc2, hostSvals2, swarm[i], hostPBest2, hostGBest2, hostPvals2, hostGBval2, 1, 1, nIter2);
				fvals[i] = hostGBval2[0];
				//if (i == 0) printf("Loop: %d, F: %2.2f\n", loopIdx, fvals[i]);
			}
		break;
		case 1:
			LENGTH = nSwarm2; 
			for (i = 0; i < LENGTH; i++) {
				fvals[i] = hostObjFunc(fixed, swarm[i]);
				//if (i == 0) printf("Loop: %d, %d, F: %2.2f\n", loopIdx, i, fvals[i]);
			}			
		break;
	}
}

//
void cpuHPSO(float *swarm, float *vel, float *fvals, const float fixed, float *pBests, float *gBest, float *fpvals, float *fgval, const int loopIdx, const int maximize, const int nIter) 
{
	int t;
	Initparticles(swarm, vel, loopIdx);
	EvalObjFunc(fvals, swarm, fixed, loopIdx); 
	UpdatePBest(swarm, pBests, fvals, fpvals, loopIdx, maximize, 1);
	UpdateGBest(pBests, gBest, fpvals, fgval, loopIdx, maximize, 1);
	for (t = 0; t < nIter; t++) {
		UpdateSwarm (swarm, hostVeloc1, pBests, gBest, loopIdx);
		EvalObjFunc(fvals, swarm, fixed, loopIdx); 
		UpdatePBest(swarm, pBests, fvals, fpvals, loopIdx, maximize, 0);
		UpdateGBest(pBests, gBest, fpvals, fgval, loopIdx, maximize, 0);
	}
}

