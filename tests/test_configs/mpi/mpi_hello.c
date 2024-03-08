#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>



#if 1

int main(int argc, char** argv) {
    sleep(1);
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    char filename[64];
    sprintf(filename, "mpi_hello.%d.log", getpid());
    FILE *log = fopen(filename, "w");

    fprintf(log, "Hello world from processor %s, rank %d out of %d processors\n",
            processor_name, world_rank, world_size);
    fflush(log);

    // unlink(filename);
    fclose(log);

    // Finalize the MPI environment.
    MPI_Finalize();
}

#else

int main(int argc, char** argv) {
    sleep(1);
    char filename[64];
    sprintf(filename, "mpi_hello.log");
    FILE *log = fopen(filename, "w");

    printf("HELLO.\n");

    fprintf(log, "Hello world from processor");
    fflush(log);

    // unlink(filename);
    fclose(log);

}

#endif