// Code to find number of numbers received by a node using MPI_Status

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int MAX_NUMBERS = 100;
    int numbers[MAX_NUMBERS];
    int number_amount;
    if (world_rank == 0) {
        // Pick a random amount of integers to send to process one
        srand(time(NULL));
        number_amount = (rand() / (float)RAND_MAX) * MAX_NUMBERS;

        // Send the amount of integers to process one
        MPI_Send(numbers, number_amount, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("0 sent %d numbers to 1\n", number_amount);
    } else if (world_rank == 1) {
        MPI_Status status;
        // Receive at most MAX_NUMBERS from process zero
        MPI_Recv(numbers, MAX_NUMBERS, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 &status);

        // After receiving the message, check the status to determine
        // how many numbers were actually received
        MPI_Get_count(&status, MPI_INT, &number_amount);

        // Print off the amount of numbers, and also print additional
        // information in the status object
        printf("1 received %d numbers from 0. Message source = %d, "
               "tag = %d\n",
               number_amount, status.MPI_SOURCE, status.MPI_TAG);
    }
    
    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}