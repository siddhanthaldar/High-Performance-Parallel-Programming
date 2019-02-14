#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include "bmp.h"
#include <fstream>

using namespace std;

char file[] = "../img/foggy_3.bmp";

int num_levels = 255;
int image_width = 500;//9984;
int image_height = 500;//5616;

int image_size = image_height*image_width;


int main()
{
	// int image_size = 64*64;
	// int num_levels = 255;
	int i;

	unsigned char* img = read_bmp(file);
	unsigned char* output = new unsigned char[image_size];

	int *hist = new int[num_levels];
	float *tf = new float[num_levels];

	#pragma omp parallel for 
	for(i=0; i<num_levels;i++)
		hist[i] = 0;
	#pragma omp parallel for shared(hist)
		for(i=0;i<image_size;i++)
		{
			#pragma omp critical
				hist[img[i]]++;
		}	
	#pragma omp parallel for schedule(static,1) shared(tf, hist)
	for(i=0; i<num_levels;i++)
	{
		#pragma omp critical
			tf[i] = (hist[i]*1.0/image_size)*(num_levels-1);
	}

	for(i=1; i<num_levels;i++)
		tf[i] += tf[i-1];

	#pragma omp parallel for
	for(i=0;i<image_size;i++)
		output[i] = (int)tf[img[i]];

	ofstream f;
	f.open("input_histogram_3.txt");
	for(i=0; i<num_levels; i++)
	{
		f<<i<<" : "<< hist[i]<<endl;
	}
	f.close();

	f.open("output_histogram_3.txt");
	for(i=0; i<num_levels; i++)
	{
		f<<i<<" : "<< tf[i]<<endl;
	}
	f.close();

	write_bmp(output,image_width, image_height);

	return 0;
}