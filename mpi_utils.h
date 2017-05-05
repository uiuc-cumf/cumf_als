#ifndef MPI_UTILS_H_
#define MPI_UTILS_H_

void *MPI_Isend_float_wrapper(float *buff, size_t sz, int dst, int tag);
int MPI_Recv_float_wrapper(float *buff, size_t sz, int src, int tag);

void MPI_Waitall_wrapper(void *req[], int cnt);

#endif
