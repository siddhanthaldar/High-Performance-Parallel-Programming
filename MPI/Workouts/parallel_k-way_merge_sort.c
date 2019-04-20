// Code for k-way merge sort parallelized using MPI

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>

void merge(int *a, int *b, int l, int m, int r) {
	int h, i, j, k;
	h = l;
	i = l;
	j = m + 1;
	
	while((h <= m) && (j <= r)) {
		if(a[h] <= a[j]) {	
			b[i] = a[h];
			h++;	
		}	
		else {			
			b[i] = a[j];
			j++;	
		}			
		i++;	
	}
			
	while(h<=m)
	{
		b[i] = a[h];
		h++;i++;
	}
	while(j<=r)
	{
		b[i] = a[j];
		i++;j++;
	}

	for(k = l; k <= r; k++) {		
		a[k] = b[k];		
	}
}

void mergeSort(int *a, int *b, int l, int r,int k) {
	// int m;
	// int m1,m2;
	int *m = (int*)malloc((k+1)*sizeof(int));
	int i;
	if(l < r) {
		//2 way
		// m = (l + r)/2;
				
		// mergeSort(a, b, l, m);
		// mergeSort(a, b, (m + 1), r);
		// merge(a, b, l, m, r);	

		// //3 way
		// m1 = l + (r-l)/3;
		// m2 = m1 + (r-l)/3;
		// mergeSort(a, b, l, m1);
		// mergeSort(a, b, (m1 + 1), m2);
		// mergeSort(a, b, (m2+1), r);
		// merge(a, b, l, m1, m2);
		// merge(a, b, l, m2, r);	
	
		// k way
		m[0] = l-1;
		m[1] = m[0]+1+(r-l)/k;
		for(i=2;i<k;i++)
			m[i] = m[i-1] + (r-l)/k;
		m[k] = r;

		for(i=0;i<k;i++)
			mergeSort(a,b,m[i]+1,m[i+1],k);

		for(i=1;i<k;i++)
			merge(a,b,l,m[i],m[i+1]);  

	}	
}

int main(int argc, char **argv)
{
	// Generate array to be sorted
	int n = atoi(argv[1]);
	int k = atoi(argv[2]);
	
	srand(time(NULL));
	
	// MPI code for sorting
	int world_rank, world_size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int num_elements_per_proc = n/world_size;

	int i;
	int *arr=NULL;
	if(world_rank==0)
	{
		printf("Unsorted Array : \n");
		arr = (int *)malloc(n*sizeof(int));
		for(i=0;i<n;i++)
		{
			arr[i] = rand()%n;
			printf("%d ", arr[i]);
		}
		printf("\n\n");
	}

	int *sub_arr = (int *)malloc(num_elements_per_proc*sizeof(int));
	MPI_Scatter(arr, num_elements_per_proc, MPI_INT, sub_arr,num_elements_per_proc,MPI_INT,0,MPI_COMM_WORLD);

	int *dummy_arr = (int *)malloc(num_elements_per_proc*sizeof(int));
	mergeSort(sub_arr,dummy_arr,0,num_elements_per_proc-1,k);

	int *sub_sorted = NULL;
	if(world_rank ==0)
		sub_sorted = (int *)malloc(n*sizeof(int));

	MPI_Gather(sub_arr, num_elements_per_proc, MPI_INT, sub_sorted, num_elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

	if(world_rank==0)
	{
		int *dummy_fin = (int*)malloc(n*sizeof(int));
		mergeSort(sub_sorted,dummy_fin,0,n-1,k);

		printf("Sorted Array :\n");
		for(i=0; i<n;i++)
			printf("%d  ",sub_sorted[i]);
		printf("\n");

		free(sub_sorted);
		free(dummy_fin);
	}

	free(arr);
	free(sub_arr);
	free(dummy_arr);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}