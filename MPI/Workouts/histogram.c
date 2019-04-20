// Code for calculating histogram of an image using MPI functions
// MPI_Scatter and MPI_Gather/MPI_Allgather
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>


int *create_rand_nums(int num_elements, int level) {
  int *rand_nums = (int *)malloc(sizeof(int) * num_elements);
  assert(rand_nums != NULL);
  int i;
  for (i = 0; i < num_elements; i++) {
    rand_nums[i] = rand()%level;
  }
  return rand_nums;
}

void* compute_hist(int *array, int* hist, int num_elements) {
  int i;
  for (i = 0; i < num_elements; i++) {
    hist[array[i]]+=1;
  }
}

int main(int argc, char** argv) {
  int num_elements = 100;
  int level = 5;
  int num_elements_per_proc;

  srand(time(NULL));

  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  num_elements_per_proc = num_elements/world_size;

  int *image = NULL;
  if (world_rank == 0) {
    image = (int *)malloc(sizeof(int) * num_elements);
    image = create_rand_nums(num_elements,level);
    printf("\nImage:\n");
    int i;
    for(i=0;i<num_elements;i++)
      printf("%d  ",image[i]);
    printf("\n");

  }

  int *sub_image = (int*)malloc(sizeof(int) * num_elements_per_proc);
  assert(sub_image != NULL);

  MPI_Scatter(image, num_elements_per_proc, MPI_INT, sub_image,
              num_elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

  int* sub_hist = (int*)calloc(level,sizeof(int)); 
  compute_hist(sub_image,sub_hist, num_elements_per_proc);

  int *sub_hists = NULL;
  // Using MPI_Gather
  // if (world_rank == 0) {
  //   sub_hists = (int *)malloc(sizeof(int) * world_size*level);
  //   assert(sub_hists != NULL);
  // }
  // MPI_Gather(sub_hist, level, MPI_INT, sub_hists, level, MPI_INT, 0, MPI_COMM_WORLD);
  
  // Using MPI_Allgather
  sub_hists = (int *)malloc(sizeof(int) * world_size*level);
  assert(sub_hists != NULL);
  MPI_Allgather(sub_hist, level, MPI_INT, sub_hists, level, MPI_INT, MPI_COMM_WORLD);


  if (world_rank == 0) {
    int* act_hist = (int*)calloc(level,sizeof(int));
    int i,j;

    for(i=0;i<level;i++)
      for(j=0;j<world_size;j++)
        act_hist[i] += sub_hists[j*world_size+i];
    printf("\nHistogram:\n");
    for(i=0;i<level;i++)
      printf("%d  ",act_hist[i]);
    printf("\n");

  }

  // Clean up
  if (world_rank == 0) {
    free(image);
    free(sub_hists);
  }
  free(sub_hist);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}