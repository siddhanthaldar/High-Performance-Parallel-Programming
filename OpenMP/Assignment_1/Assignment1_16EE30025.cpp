#include <iostream>
#include <fstream>
#include <omp.h>
#include <cmath>
#include <stdlib.h>
#include <string.h>
#include <bits/stdc++.h>

using namespace std;

// Initialize variables
const int num_data = 6400000;
const int str_len = 15;
const char characters[] = {'A', 'C', 'T', 'G'};
int thread_num;

// Function to generate data
double gen_data(char **data)
{	
	cout<<"Generating data"<<endl;
	double wtime = omp_get_wtime();
	#pragma omp parallel num_threads(thread_num)
	{
		#pragma omp for collapse(2)
		for(int i=0; i<num_data; i++)
			for(int j=0; j<str_len; j++)
				data[i][j] = characters[rand()%4];	
	}
	wtime = omp_get_wtime() - wtime;
	cout<<"Time taken to generate data = "<<wtime<<endl;
	return wtime;
}

// Function to search for matching strings in the database
double match(char **data, char* str)
{
	cout<<endl<<"Search for matching strings"<<endl;
	int i,j,count=0,flag,x;
	vector <int> idx;
	double wtime, time[10], res[2];
	for(x=0; x<10;x++)
	{
		wtime = omp_get_wtime();
		#pragma omp parallel num_threads(thread_num)
		{
			#pragma omp for schedule(static) private(i,j,flag)
				for(i=0; i<num_data;i++)
				{
					flag = 0;
					for(j=0; j<str_len;j++)
					{
						if(data[i][j] != str[j])
						{
							flag = 1; break;
						}
					}			
					if(!flag) idx.push_back(i);
				}
		}
		time[x] = omp_get_wtime()- wtime;
	}
	
	// Calculate average time
	wtime = 0;
	for(x=0; x<10; x++)
		wtime += time[x];
	wtime /= 10.0;

	// Find number of strings matched and their indices
	cout<<"Number of strings matched = "<<idx.size()/10.0<<endl;
	if(idx.size() > 0)
	{
		cout<<"Indices :"<<endl;
		for(i=0; i<idx.size()/10.0;i++) 
			cout<<idx[i]<<"	";
	}
	cout<<endl<<"Time taken to match string = "<<wtime<<endl;
	

	// return {(double)idx.size()/10.0, wtime};
	return wtime;
}


// Function to calculate percentage redundancy 
double redundancy(char **data)
{
	int i,j,count,x;
	long sum;
	long val = (int)(10*num_data);
	int *elements = new int [val];
	double wtime, redundancy[10];

	cout<<endl<<"Calculating percentage redundancy"<<endl;
	for(x=0; x<10; x++)
	{
		count = 0;
		for(i=0;i<val;i++) elements[i] = 0;
		wtime = omp_get_wtime();
		#pragma omp parallel num_threads(thread_num)
		{
			#pragma omp for private(sum,j) 
			for(i=0; i<num_data;i++)
			{
				sum=0;
				for(j=0;j<str_len;j++)
				{
					if(data[i][j] == 'A') sum += 3*pow(4,j);
					else if(data[i][j] == 'C') sum += 2*pow(4,j);
					else if(data[i][j] == 'T') sum += 1*pow(4,j);
					else sum += 0;
				}
				sum = sum%val;

				#pragma omp critical
				{
					if(elements[sum] == 1) count++;
					else elements[sum] = 1;	
				}
				
			}	

		}
		wtime = omp_get_wtime()- wtime;
		redundancy[x] = 100.0*(count)/num_data;
	}

	cout<<"Time taken to calculate redundancy = "<<wtime<<endl;	
	
	// Calculate average redundancy
	double red = 0;
	for(x=0; x<10; x++)
		red += redundancy[x];
	red /= 10.0;
	cout<<"Average Percentage redundancy = "<<red<<endl;

	// return {wtime. red};
	return red;
	
}

int main()
{
	int i,j;

	// Read searching pattern and number of threads
	char *str = new char [str_len];

	cout<<"Give number of threads : ";
	cin>>thread_num;
	cout<<"Give searching pattern :";
	cin>>str;
	
	// Read data from database
	char **data = new char*[num_data];
	for(i=0; i<num_data;i++)
		data[i] = new char [str_len];

	double gen_time = gen_data(data);

	// Find matching strings
	double match_dat = match(data, str);

	// Calculate redundacy
	double red_dat = redundancy(data);

	ofstream f;
	f.open("Report_3_thread.txt");
	f<<"Number of threads = "<<thread_num<<endl<<endl;
	f<<"Time taken to generate data = "<<gen_time<<endl;
	f<<endl<<"Search for matching strings"<<endl;
	f<<"Time taken to match string = "<<match_dat<<endl;
	f<<endl<<"Calculating percentage redundancy"<<endl;
	f<<"Average Percentage redundancy = "<<red_dat<<endl;
	f.close();

	return 0; 

}