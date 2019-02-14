#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
	int num_var,i,j,k;
	float m;
	cout<<"Give number of variables:";
	cin>>num_var;

	float *var = new float[(num_var+1)*num_var];
	float *sol = new float[num_var];

	for(i=0; i<num_var;i++)
		sol[i] = 0;


	for(i=0; i<(num_var+1)*num_var;i++)
			cin>>var[i];
	
	for(i=0; i<num_var-1;i++)
	{
		#pragma omp parallel shared(num_var,i,var) private(j,k)
		{
			#pragma omp for //collapse(2) 
				for(j=i+1; j<num_var;j++)		
					for(k=num_var; k>=0;k--)
					{
						var[j*(num_var+1)+k] -= var[i*(num_var+1)+k]*var[j*(num_var+1)+i]/var[i*(num_var+1)+i];
					}	
		}	
		#pragma omp barrier
		
	}		
		

	cout<<"Matrix : "<<endl;
	for(int i=0; i<(num_var+1)*num_var;i++)
	{
		if(i%(num_var+1)==0) cout<<endl;
		cout<<var[i]<<"	";
	}	
	cout<<endl;

	cout<<"Answer :"<<endl;
	for(i=num_var-1; i>=0;i--)
	{
		for(j=num_var-1;j>i;j--)
			sol[i] += sol[j]*var[i*(num_var+1)+j];
		sol[i] = (var[i*(num_var+1)+num_var] - sol[i])/var[i*(num_var+1)+i];
	}

	for(i=0; i<num_var;i++)
		cout<<sol[i]<<"	";
	cout<<endl;

	return 0;
}