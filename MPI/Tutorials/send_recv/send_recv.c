#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int number;
    if (world_rank == 0) {
        number = -1;
        MPI_Send(&number, 1, MPI_INT, 3, 2, MPI_COMM_WORLD);
    } else if (world_rank == 3) {
        MPI_Recv(&number, 1, MPI_INT, 0, 2, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        printf("Process 3 received number %d from process 0\n",
               number);
    }
    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}