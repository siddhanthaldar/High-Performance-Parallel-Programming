#include <mpi.h>
#include <stdio.h>
#include <stdio.h>

// Using send and receive
void my_bcast(void* data, int count, MPI_Datatype datatype, int root,
              MPI_Comm communicator) {
  int world_rank;
  MPI_Comm_rank(communicator, &world_rank);
  int world_size;
  MPI_Comm_size(communicator, &world_size);

  if (world_rank == root) {
    // If we are the root process, send our data to everyone
    int i;
    for (i = 0; i < world_size; i++) {
      if (i != world_rank) {
        MPI_Send(data, count, datatype, i, 0, communicator);
      }
    }
  } else {
    // If we are a receiver process, receive the data from the root
    MPI_Recv(data, count, datatype, root, 0, communicator,
             MPI_STATUS_IGNORE);
  }
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Time my_bcast
    int num_elements = 1;
    int data = 5;
    // Synchronize before starting timing
    MPI_Barrier(MPI_COMM_WORLD);
    // float total_my_bcast_time -= MPI_Wtime();
    my_bcast(&data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);
    // Synchronize again before obtaining final time
    MPI_Barrier(MPI_COMM_WORLD);
    // total_my_bcast_time += MPI_Wtime();
    // print("my_bcast time = %f\n",total_my_bcast_time);

    // Time MPI_Bcast
    MPI_Barrier(MPI_COMM_WORLD);
    // float total_mpi_bcast_time -= MPI_Wtime();
    MPI_Bcast(&data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    // total_mpi_bcast_time += MPI_Wtime();
    // print("MPI_bcast time = %f\n\n",total_mpi_bcast_time);


    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}