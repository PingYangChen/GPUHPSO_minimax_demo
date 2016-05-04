#include "testgpu.h"

extern double *GBest1;
extern double *hostSwarm1, *hostVeloc1, *hostPBest1, *hostGBest1, *hostSvals1, *hostPvals1, *hostGBval1;
extern double *hostSwarm2, *hostVeloc2, *hostPBest2, *hostGBest2, *hostSvals2, *hostPvals2, *hostGBval2;

void Initparticles(double *swarm, double *vel, const int loopIdx);
void EvalObjFunc(double *fvals, double *swarm, const double fixed, const int loopIdx);
void UpdateSwarm(double *swarm, double *vel, double *pBests, double *gBest, const int loopIdx);
void UpdatePBest(double *swarm, double *pBests, double *fvals, double *fpvals, const int loopIdx, const int maximize, const int initial);
void UpdateGBest(double *pBests, double *gBest, double *fpvals, double *fgval, const int loopIdx, const int maximize, const int initial);
double hostObjFunc(double x, double y);
void cpuHPSO(double *swarm, double *vel, double *fvals, const double fixed, double *pBests, double *gBest, double *fpvals, double *fgval, const int loopIdx, const int maximize, const int nIter);

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
	printf("GPU: The best location is at x = %2.3f\n", GBest1[0]);
	gputime = ((end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec)/1000000.0);
	printf("GPU Time = %f sec.\n", gputime);
	//printf("------------------------------------------- \n");
	//	
	gettimeofday(&start,NULL); 	
	cpuHPSO(hostSwarm1, hostVeloc1, hostSvals1, 0.0, hostPBest1, hostGBest1, hostPvals1, hostGBval1, 0, 0, nIter1);
	gettimeofday(&end,NULL);	// Stop stopwatch and compute time
	printf("------------------------------------------- \n");
	printf("CPU: The best location is at x = %2.3f\n", hostGBest1[0]);
	cputime = ((end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec)/1000000.0);
	printf("CPU Time = %f sec.\n", cputime);
	printf("------------------------------------------- \n");

	Free_Memory();
  return 0;
	
}

double hostObjFunc(double x, double y) 
{
	double res = (x - 1.0)*(x - 1.0) - y*y;
	return res;
}

//
void Initparticles(double *swarm, double *vel, const int loopIdx) 
{
	long int i;	
	long int LENGTH;
	double U, L; 
	switch (loopIdx) {
		case 0:
			LENGTH = nSwarm1; U = Ux; L = Lx; 
		break;
		case 1:
			LENGTH = nSwarm2; U = Uy; L = Ly; 
		break;
	}
	for (i = 0; i < LENGTH; i++) {
		swarm[i] = ((double)rand()/(double)RAND_MAX) * (U - L) + L; 
		vel[i] = ((double)rand()/(double)RAND_MAX); 
		//if (i == 0) printf("Loop: %d, S_%d: %2.2f\n", loopIdx, i, swarm[i]);
	}	
}

//
void UpdateSwarm(double *swarm, double *vel, double *pBests, double *gBest, const int loopIdx)
{
	long int i;
	long int indexTmp, LENGTH;
	double L, U;
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
	
	double r1;
	double r2;
	
	for (i = 0; i < LENGTH; i++) {
		r1 = (double)rand()/(double)RAND_MAX;
		r2 = (double)rand()/(double)RAND_MAX;
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
void UpdatePBest(double *swarm, double *pBests, double *fvals, double *fpvals, const int loopIdx, const int maximize, const int initial) {
	
	long int i;
	long int LENGTH;
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
void UpdateGBest(double *pBests, double *gBest, double *fpvals, double *fgval, const int loopIdx, const int maximize, const int initial)
{
	
	long int lenPBest;
	switch (loopIdx) {
		case 0:
			lenPBest = nSwarm1;
		break;
		case 1: 
			lenPBest = nSwarm2;
		break;
	}
		
	double bestValInEachSwarm = fpvals[0];
	long int bestLocInEachSwarm = 0;
	long int k;
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
	
	//if (loopIdx == 0) printf("Current Gbest Loc: %g ; ", gBest[0]);
	//if (loopIdx == 0) printf("Current Gbest Val: %g \n", fgval[0]);	
	//if (loopIdx == 0) printf("Best Pbest Loc: %g ; ", pBests[bestLocInEachSwarm]);
	//if (loopIdx == 0) printf("Best Pbest Val: %g \n", fpvals[bestLocInEachSwarm]);
		
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
	//if (loopIdx == 0) printf("Updated Gbest Loc: %g ; ", gBest[0]);
	//if (loopIdx == 0) printf("Updated Gbest Val: %g \n", fgval[0]);	
}

void EvalObjFunc(double *fvals, double *swarm, const double fixed, const int loopIdx)
{
	long int i;
	long int LENGTH;
	long int indexTmp;
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
void cpuHPSO(double *swarm, double *vel, double *fvals, const double fixed, double *pBests, double *gBest, double *fpvals, double *fgval, const int loopIdx, const int maximize, const int nIter) 
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

