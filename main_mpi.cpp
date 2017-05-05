/*
 * main.cpp
 *
 *  Created on: Feb 10, 2015
 *      Author: Wei Tan (wtan@us.ibm.com)
 *  Test als.cu using netflix or yahoo data
 *  Alternating Least Square for Matrix Factorization on CUDA 7.0+
 *  Code optimized for F = 100, and on cc 3.5, 3.7 platforms. Also tested in cc 5.2
 */
#include "cuda_wrapper.h"
#include<stdlib.h>
#include<stdio.h>
#include <string>
#include <mpi.h>

#define DEVICEID 0
#define ITERS 10

int main(int argc, char **argv) {

	MPI_Init(&argc, &argv);

	int nProcs, myRank;

	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

	//parse input parameters
	if(argc != 10){
		if (myRank == 0) {
			printf("Usage: give M, N, F, NNZ, NNZ_TEST, lambda, X_BATCH, THETA_BATCH and DATA_DIR.\n");
			printf("E.g., for netflix data set, use: \n");
			printf("./main 17770 480189 100 99072112 1408395 0.048 1 3 ./data/netflix/ \n");
			printf("E.g., for movielens 10M data set, use: \n");
			printf("./main 71567 65133 100 9000048 1000006 0.05 1 1 ./data/ml10M/ \n");
			printf("E.g., for yahooMusic data set, use: \n");
			printf("./main 1000990 624961 100 252800275 4003960 1.4 6 3 ./data/yahoo/ \n");
		}
		return 0;
	}
	
	int ret = main_wrapper(argc, argv, myRank, nProcs);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return ret;

}
