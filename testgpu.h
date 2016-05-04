#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <immintrin.h>
#include <sys/time.h>

#define nSwarm1 256*4 // should be a multiple of 32
#define nSwarm2 256*4 // should be a power of 2

#define nIter1 50
#define nIter2 50

#define Ux 5.0
#define Lx -5.0

#define Uy 5.0
#define Ly -5.0

#define c1 2.0
#define c2 2.0
#define vmax 3.0
#define omg 0.9

//Declare functions
void Allocate_Memory();
void cudaHPSO();
void Get_From_Device();
void Free_Memory();
void Call_GPUFunction();