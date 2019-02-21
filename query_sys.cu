#include <cuda.h>
#include <stdio.h>

void printDevProp ( cudaDeviceProp devp )
{
	printf ( " No . of multiprocessors : % d \n ",devp.multiProcessorCount ) ; // 24
	printf ( " Size of warp % d \n " , devp.warpSize ) ; // 32
	return ;
}

int main ()
{
	int devCount ;
	cudaGetDeviceCount(& devCount);
	for (int i = 0; i < devCount;++i)
	{
		cudaDeviceProp devp ;
		cudaGetDeviceProperties(&devp ,i);
		printDevProp(devp) ;
	}
	return 0;
}