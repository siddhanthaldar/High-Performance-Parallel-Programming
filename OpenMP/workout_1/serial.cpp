#include <iostream>
#include <omp.h>
#include <fstream>

using namespace std;

int prime(int i)
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

	int count = 0;	
	ofstream f;
	f.open ("serial.txt");

	wtime = omp_get_wtime();
	{		
		for(i=2; i<=131072;i++)
		{			
			if(prime(i))
			{
				count++;	
				f << "Prime : "<<i<<"\tTime : "<<omp_get_wtime() - wtime<<endl;
			}		
		}
	}	
		
	cout<<count<<endl;
	float cpu_time = omp_get_wtime() - wtime;
	cout<<"Total time without parallelization : "<<cpu_time<<endl;
	f<<"Total time = "<<cpu_time<<endl;
	f.close();
	return 0;
}