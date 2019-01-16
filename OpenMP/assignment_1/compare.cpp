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

	ofstream f;
	f.open("performance.txt");
	// Serial
	int count = 0;
	wtime = omp_get_wtime();
	for(i=2; i<=131072;i++)
	{			
		if(prime_is(i))
		{
			count++;	
		}
	}

	cout<<count<<endl;	
	float cpu_time = omp_get_wtime() - wtime;
	cout<<"Total time without parallelization : "<<cpu_time<<endl;


	// 4 cores
	count = 0;
	wtime = omp_get_wtime();
	#pragma omp parallel for num_threads(4) shared(i) 
	for(i=2; i<=131072;i++)
	{			
		if(prime_is(i))
		{
			count++;	
		}
	}

	cout<<count<<endl;
	float four_core_time = omp_get_wtime() - wtime;
	cout<<"Total four core time : "<<four_core_time<<endl;

	// 16 cores
	count=0;
	wtime = omp_get_wtime();
	#pragma omp parallel for num_threads(16) shared(i) 
	for(i=2; i<=131072;i++)
	{			
		if(prime_is(i))
		{
			count++;	
		}
	}
	
	cout<<count<<endl;
	float sixteen_core_time = omp_get_wtime() - wtime;
	cout<<"Total sixteen core time : "<<sixteen_core_time<<endl;

	f<<"Serial time = "<<cpu_time<<endl;
	f<<"Total time for four cores = "<<four_core_time<<"\t\t Percentage Improvement = "<<(cpu_time - four_core_time)/cpu_time*100.0<<endl;
	f<<"Total time for sixteen cores = "<<sixteen_core_time<<"\t\t Percentage Improvement = "<<(cpu_time - sixteen_core_time)/cpu_time*100.0<<endl;

	f.close();
	return 0;
}