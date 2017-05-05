#include <mpi.h>
#include <stdlib.h>
#include "mpi_utils.h"

void *MPI_Isend_float_wrapper(float *buff, size_t sz, int dst, int tag) {
    MPI_Request req;
    MPI_Isend(buff, sz, MPI_FLOAT, dst, tag, MPI_COMM_WORLD, &req);

    return (void *) req;
}

int MPI_Recv_float_wrapper(float *buff, size_t sz, int src, int tag) {
    MPI_Status stat;
    return MPI_Recv(buff, sz, MPI_FLOAT, src, tag, MPI_COMM_WORLD, &stat);
}

void MPI_Waitall_wrapper(void *req[], int cnt) {

    MPI_Waitall(cnt, (MPI_Request *) req, MPI_STATUSES_IGNORE);

}