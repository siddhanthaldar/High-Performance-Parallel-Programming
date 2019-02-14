#include <iostream>
#include <omp.h>
#include <fstream>
#include <list>

using namespace std;

int prime_is(int i)
{
	int x;
	for(x=2;x<=i/2;x++)
	{
		if(i%x) continue;
		else return 0;
	}
	return 1;
}


int main()
{
	int i;
	double wtime;

	FILE *f;
	f = freopen("16cores.txt", "a+", stdout);

	wtime = omp_get_wtime();
	#pragma omp parallel for num_threads(16) shared(i)//,prime, time) 

	for(i=2; i<=131072;i++)
	{			
		if(prime_is(i))
		{
			printf("Prime: %d\t Time: %f\n",i,omp_get_wtime() - wtime);
		}		
	}

	float core_time = omp_get_wtime() - wtime;
	cout<<"Total core time : "<<core_time<<endl;

	fclose(f);
	return 0;
}