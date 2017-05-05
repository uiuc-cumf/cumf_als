/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * als.cu
 *
 *  Created on: Feb 10, 2015
 *      Author: Wei Tan (wtan@us.ibm.com)
 *  Alternating Least Square for Matrix Factorization on CUDA 7.0+
 *  Code optimized for F = 100, and on cc 3.5, 3.7 platforms. Also tested in cc 5.2
 */
//do not use fp16 by default  
//#define CUMF_USE_HALF
//#define SURPASS_NAN
#define USE_CG
//if cojugate gradient solver generates results in FP16 
//#define CUMF_TT_FP16
//#define CUMF_XX_FP16
#define CG_ITER 6
//#define CUMF_SAVE_MODEL
#include "als.h"
#include "device_utilities.h"
#include "cg.h"
#include "host_utilities.h"
#include <fstream>
#include <assert.h>
#include <cuda_fp16.h>
#include <omp.h>
#ifdef CUMF_USE_HALF
#define SCAN_BATCH 24
#else
#define SCAN_BATCH 28
#endif
#include <iostream>
using namespace std;

void saveDeviceFloatArrayToFile(string fileName, int size, float* d_array){
    float* h_array;
    cudacall(cudaMallocHost( (void** ) &h_array, size * sizeof(h_array[0])) );
    cudacall(cudaMemcpy(h_array, d_array, size * sizeof(h_array[0]),cudaMemcpyDeviceToHost));
    FILE * outfile = fopen(fileName.c_str(), "wb");
    fwrite(h_array, sizeof(float), size, outfile);
    fclose(outfile);
    cudaFreeHost(h_array);
}
int updateX(const int batch_size, const int batch_offset, float * ythetaT, float * tt, float * XT,
        cublasHandle_t handle, const int m, const int n, const int f, const int nnz,
        float** devPtrTTHost, float **devPtrYthetaTHost){
    #ifdef DEBUG
    float elapsed;
    struct timeval tv0, tv1, tv2;
    gettimeofday(&tv0, NULL);
    printf("*******Batch LU factorization of tt.\n");
    #endif
    //pointers needed by batch op
    float **devPtrTT = 0;
    int *INFO;
    for (int k = 0; k < batch_size; k++) {
        devPtrTTHost[k] = &tt[k * f * f];
    }
    cudacall(cudaMalloc((void** ) &devPtrTT, batch_size * sizeof(*devPtrTT)));
    cudacall(cudaMemcpy(devPtrTT, devPtrTTHost, batch_size * sizeof(*devPtrTT),cudaMemcpyHostToDevice));
    //cudacall( cudaMalloc(&P, f * batch_size * sizeof(int)) );
    cudacall( cudaMalloc(&INFO, batch_size * sizeof(int) ));
    cublascall(cublasSgetrfBatched(handle, f, devPtrTT, f, NULL, INFO, batch_size));

    cudaThreadSynchronize();
    #ifdef DEBUG
    gettimeofday(&tv1, NULL);
    elapsed = (tv1.tv_sec - tv0.tv_sec)
            + (tv1.tv_usec - tv0.tv_usec) / 1000000.0;
    printf("\t %f seconds. \n", elapsed);

    printf("*******solve: tt * XT = ythetaT use cublas, with LU decomposition.\n");
    #endif

    float **devPtrYthetaT = 0;

    for (int k = 0; k < batch_size; k++) {
        devPtrYthetaTHost[k] = &ythetaT[batch_offset * f + k * f];
    }
    cudacall(cudaMalloc((void** ) &devPtrYthetaT, batch_size * sizeof(*devPtrYthetaT)));
    cudacall(cudaMemcpy(devPtrYthetaT, devPtrYthetaTHost, batch_size * sizeof(*devPtrYthetaT), cudaMemcpyHostToDevice));

    int * info2 = (int *) malloc(sizeof(int));
    cublascall( cublasSgetrsBatched(handle, CUBLAS_OP_N, f, 1,
            (const float ** ) devPtrTT, f, NULL, devPtrYthetaT, f, info2, batch_size) );

    cudaThreadSynchronize();
    cudaError_t cudaStat1 = cudaGetLastError();
    if (cudaStat1 != cudaSuccess) {
        fprintf(stderr,"Failed to launch cublasSgetrsBatched (error code: %s)!\n", cudaGetErrorString(cudaStat1));
        exit(EXIT_FAILURE);
    }

    cudacall( cudaMemcpy(&XT[batch_offset * f], &ythetaT[batch_offset * f],
            batch_size * f * sizeof(float), cudaMemcpyDeviceToDevice) );
    #ifdef DEBUG
    gettimeofday(&tv2, NULL);
    elapsed = (tv2.tv_sec - tv1.tv_sec)
            + (tv2.tv_usec - tv1.tv_usec) / 1000000.0;
    printf("\t %f seconds. \n", elapsed);
    #endif

    cudacall(cudaFree(devPtrTT));
    //cudacall(cudaFree(P));
    cudacall(cudaFree(INFO));
    cudacall(cudaFree(devPtrYthetaT));
    return 0;
}

int updateTheta(const int batch_size, const int batch_offset, float * xx,
          float * yTXT, float * thetaT,
        cublasHandle_t handle,
         const int m, const int n, const int f, const int nnz,
         float ** devPtrXXHost, float **devPtrYTXTHost ){

    #ifdef DEBUG
    float elapsed;
    struct timeval tv0, tv1, tv2;
    gettimeofday(&tv0, NULL);
    printf("*******LU factorize xx.\n");
    #endif
    float **devPtrXX = 0;

    for (int k = 0; k < batch_size; k++) {
        devPtrXXHost[k] = &xx[k * f * f];
    }
    cudacall(cudaMalloc((void** ) &devPtrXX, batch_size * sizeof(*devPtrXX)));
    cudacall(cudaMemcpy(devPtrXX, devPtrXXHost, batch_size * sizeof(*devPtrXX), cudaMemcpyHostToDevice));
    int *INFO;
    //cudacall(cudaMalloc(&P, f * batch_size * sizeof(int)));
    cudacall(cudaMalloc(&INFO, batch_size * sizeof(int)));
    cublascall(cublasSgetrfBatched(handle, f, devPtrXX, f, NULL, INFO, batch_size));
    cudaThreadSynchronize();
    #ifdef DEBUG
    gettimeofday(&tv1, NULL);
    elapsed = (tv1.tv_sec - tv0.tv_sec)
            + (tv1.tv_usec - tv0.tv_usec) / 1000000.0;
    printf("\t %f seconds. \n", elapsed);

    printf("******* solve xx * thetaT = yTXT with CUDA 7.\n");
    #endif
    float **devPtrYTXT = 0;

    for (int k = 0; k < batch_size; k++) {
        devPtrYTXTHost[k] = &yTXT[batch_offset * f + k * f];
    }

    cudacall(cudaMalloc((void** ) &devPtrYTXT, batch_size * sizeof(*devPtrYTXT)));
    cudacall(cudaMemcpy(devPtrYTXT, devPtrYTXTHost, batch_size * sizeof(*devPtrYTXT),cudaMemcpyHostToDevice));

    int * info2 = (int *) malloc(sizeof(int));
    cublascall( cublasSgetrsBatched(handle, CUBLAS_OP_N, f, 1,
            (const float ** ) devPtrXX, f, NULL, devPtrYTXT, f, info2, batch_size) );
    cudaThreadSynchronize();
    cudaError_t cudaStat1 = cudaGetLastError();
    if (cudaStat1 != cudaSuccess) {
        fprintf(stderr,"Failed to launch cublasSgetrsBatched (error code: %s)!\n", cudaGetErrorString(cudaStat1));
        exit(EXIT_FAILURE);
    }

    cudacall( cudaMemcpy( &thetaT[batch_offset * f], &yTXT[batch_offset * f],
                            batch_size * f * sizeof(float), cudaMemcpyDeviceToDevice) );
    #ifdef DEBUG
    gettimeofday(&tv2, NULL);
    elapsed = (tv2.tv_sec - tv1.tv_sec)
            + (tv2.tv_usec - tv1.tv_usec) / 1000000.0;
    printf("\t %f seconds. \n", elapsed);
    #endif

    cudaFree(devPtrXX);
    cudaFree(INFO);
    free(info2);
    cudaFree(devPtrYTXT);
    return 0;
}

__global__ void RMSE(const float * csrVal, const int* cooRowIndex,
        const int* csrColIndex, const float * __restrict__ thetaT, const float * __restrict__ XT, float * error, const int nnz,
        const int error_size, const int f) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < nnz) {
        int row = cooRowIndex[i];
        int col = csrColIndex[i];
        float e = csrVal[i];
        //if(i%1000000==0) printf("row: %d, col: %d, csrVal[%d]: %f.\n", row, col, i, e);
        for (int k = 0; k < f; k++) {
            e -= __ldg(&thetaT[f * col + k]) * __ldg(&XT[f * row + k]);
        }
        atomicAdd(&error[i%error_size], e*e);
        //if(i%1000000==0) printf("error[%d]: %f.\n", i, e);
    }
}

__inline__ __device__
float warpRedSum(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__inline__ __device__
float blockRedSum(float val) {

    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpRedSum(val);     // Each warp performs partial reduction

    if (lane==0) {
        shared[wid] = val; // Write reduced value to shared memory
    }

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;

    if (wid == 0) {
        val = warpRedSum(val); //Final reduce within first warp
    }

    return val;
}

__global__ void RMSE_reduction(const float * csrVal, const int* cooRowIndex,
        const int* csrColIndex, const float * __restrict__ thetaT, const float * __restrict__ XT, float * error, const int nnz, const int f) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float e = 0.0;
    if (i < nnz) {
        int row = cooRowIndex[i];
        int col = csrColIndex[i];
        e = csrVal[i];
        //if(i%1000000==0) printf("row: %d, col: %d, csrVal[%d]: %f.\n", row, col, i, e);
        for (int k = 0; k < f; k++) {
            e -= __ldg(&thetaT[f * col + k]) * __ldg(&XT[f * row + k]);
        }

        e = e * e;
    }

    float sum = blockRedSum(e);
    if (threadIdx.x == 0) {
        error[blockIdx.x] = sum;
    }

}

__global__ void RMSE_reduction2(const int offset, const float * csrVal, const int* cooRowIndex,
        const int* csrColIndex, const float * __restrict__ thetaT, const float * __restrict__ XT, float * error,
        const int nnz, const int f) {

    __shared__ float sdata[256];

    int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
    float e = 0.0;
    if (i < nnz) {
        int row = cooRowIndex[i];
        int col = csrColIndex[i];
        e = csrVal[i];
        //if(i%1000000==0) printf("row: %d, col: %d, csrVal[%d]: %f.\n", row, col, i, e);
        for (int k = 0; k < f; k++) {
            e -= __ldg(&thetaT[f * col + k]) * __ldg(&XT[f * row + k]);
        }

        e = e * e;
    }

    sdata[threadIdx.x] = e;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        error[blockIdx.x] = sdata[0];
    }

}

__global__ void reduction(float *in, float *out, int N) {

  float sum = 0.0;
  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = blockRedSum(sum);
  if (threadIdx.x == 0)
    out[blockIdx.x] = sum;

}


__global__ void
__launch_bounds__(64)
get_hermitian100(const int batch_offset, float2* tt,
        const int* csrRowIndex, const int* csrColIndex, const float lambda, const int m, const int F,
        const float2* __restrict__ thetaT) {
    extern __shared__ float2 thetaTemp[];
    int row = blockIdx.x + batch_offset;
    if (row < m) {
        //this block needs to handle end - start thetaT columns
        int start = csrRowIndex[row];
        int end = csrRowIndex[row + 1];
        //slide through [start, end] by window size SCAN_BATCH
        int iterations = (end - start - 1)/SCAN_BATCH + 1;
        float temp0= 0, temp1= 0, temp2= 0, temp3= 0, temp4= 0, temp5= 0, temp6= 0, temp7= 0, temp8= 0, temp9 = 0;
        float temp10= 0, temp11= 0, temp12= 0, temp13= 0, temp14= 0, temp15= 0, temp16= 0, temp17= 0, temp18= 0, temp19 = 0;
        float temp20= 0, temp21= 0, temp22= 0, temp23= 0, temp24= 0, temp25= 0, temp26= 0, temp27= 0, temp28= 0, temp29 = 0;
        float temp30= 0, temp31= 0, temp32= 0, temp33= 0, temp34= 0, temp35= 0, temp36= 0, temp37= 0, temp38= 0, temp39 = 0;
        float temp40= 0, temp41= 0, temp42= 0, temp43= 0, temp44= 0, temp45= 0, temp46= 0, temp47= 0, temp48= 0, temp49 = 0;
        float temp50= 0, temp51= 0, temp52= 0, temp53= 0, temp54= 0, temp55= 0, temp56= 0, temp57= 0, temp58= 0, temp59 = 0;
        float temp60= 0, temp61= 0, temp62= 0, temp63= 0, temp64= 0, temp65= 0, temp66= 0, temp67= 0, temp68= 0, temp69 = 0;
        float temp70= 0, temp71= 0, temp72= 0, temp73= 0, temp74= 0, temp75= 0, temp76= 0, temp77= 0, temp78= 0, temp79 = 0;
        float temp80= 0, temp81= 0, temp82= 0, temp83= 0, temp84= 0, temp85= 0, temp86= 0, temp87= 0, temp88= 0, temp89 = 0;
        float temp90= 0, temp91= 0, temp92= 0, temp93= 0, temp94= 0, temp95= 0, temp96= 0, temp97= 0, temp98= 0, temp99 = 0;

        int tile_x = 0;
        int tile_y = 0;

        int tile = F/10;
        for ( int i = 0; i < 10; i++){
            int end = ((20-i)*(i+1))/2;
            if(threadIdx.x < end){
                tile_x = i * tile;
                tile_y = (10 + threadIdx.x - end) * tile;
                break;
            }
        }
        //iteration: copy gmem-->smem; aggregate smem-->register
        for (int iter = 0; iter < iterations; iter ++){
            //copy texture --> smem, and sync
            /*
            This is the fastest implementation
            thetaT is NOT coalesced loaded but cached by L1 and L2
            faster than coalesced version (see the next paragraph commented out) 
            because it concurrently load multiple thetaT columns
            two threads per theta column, e.g., threads 0 & 1 for theta[0], threads 2 & 3 for theta[1]
            require: blockDim.x (64) >= 2*SCAN_BATCH
            */
///* 
            if(threadIdx.x < 2*SCAN_BATCH){
                int anchor = start + iter*SCAN_BATCH + threadIdx.x/2;
                if(anchor < end){
                    int col = csrColIndex[anchor];
                    //IMPORTANT: for loop has constant and identical start and end
                    for (int k = 0; k < 50; k += 2)
                        //thetaTemp[threadIdx.x*F/4 + k/2] =__ldg(&thetaT[ F/2 * col + threadIdx.x%2*F/4 + k/2]);
                        thetaTemp[threadIdx.x*F/4 + k/2] = thetaT[ F/2 * col + threadIdx.x%2*F/4 + k/2];
                }
            }
//*/            

/*          
                //coalesced load thetaT, has to load column by column, less concurrency, worse performance
                int anchor = start + iter*SCAN_BATCH + threadIdx.x%32;
                int col_local;
                if(anchor < end && threadIdx.x%32 < SCAN_BATCH)
                    col_local = csrColIndex[anchor];
                int stop = (end - start - iter*SCAN_BATCH < SCAN_BATCH)? end - start - iter*SCAN_BATCH: SCAN_BATCH;
                for (int k = 0; k < stop; k++){
                    //deal with col_local in lane[k]
                    int col = __shfl(col_local, k);
                    //if(blockIdx.x==0 && threadIdx.x==0)
                    //  printf("iter=%d,k=%d,col=%d,stop=%d,anchor=%d\n", iter,k, col, stop, anchor);
                    //this type of for is bad in performance
                    //for(int i = threadIdx.x; i < F; i += 64)
                    if(threadIdx.x<F/2)
                        thetaTemp[k*F/2 + threadIdx.x] = __ldg(&thetaT[ F/2 * col + threadIdx.x]);
                }
*/
            __syncthreads();
///*
            //tile: 10*10
            if(threadIdx.x < 55){
                if(iter < iterations - 1){
                    for(int k = 0; k < SCAN_BATCH; k++)
                        accumulate_in_registers();
                }
                else{
                    for(int k = 0; k < end - start - iter*SCAN_BATCH; k++)
                        accumulate_in_registers();
                }
                
            }
//*/            
        }
        //end of iteration in copying from smem and aggregating in register
        __syncthreads();
        #ifdef DEBUG
        //if(threadIdx.x==0)
        //  printf("***temp 0~9: %f %f %f %f %f %f %f %f %f %f\n", temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9);
        #endif
        if(threadIdx.x < 55 ){
            //weighted-lambda regularization
            if(tile_x == tile_y){
                float temp = (end - start) * lambda;
                temp0 += temp;
                temp11 += temp;
                temp22 += temp;
                temp33 += temp;
                temp44 += temp;
                temp55 += temp;
                temp66 += temp;
                temp77 += temp;
                temp88 += temp;
                temp99 += temp;
            }
            //copy output to gmem
            int index = blockIdx.x*F*F/2;
            //fill_lower_half_from_registers();
            fill_lower_half_from_registers_float2();
            //symmetric
            if(tile_x!=tile_y){
                //fill_upper_half_from_registers();
                fill_upper_half_from_registers_float2();
            }
        }
    }
}

/*a generic kernel to get the hermitian matrices
 * as the left-hand side of the equations, to update X in ALS
 *examplary F = 100, T = 10
 */
__global__ void
get_hermitianT10(const int batch_offset, float* tt,
        const int* csrRowIndex, const int* csrColIndex, const float lambda, const int m, const int F,
        const float* __restrict__ thetaT) {
    extern __shared__ float2 thetaTemp [];
    int row = blockIdx.x + batch_offset;
    if (row < m) {
        //this block needs to handle end - start thetaT columns
        int start = csrRowIndex[row];
        int end = csrRowIndex[row + 1];
        //slide through [start, end] by window size SCAN_BATCH
        int iterations = (end - start - 1)/SCAN_BATCH + 1;
        float temp0= 0, temp1= 0, temp2= 0, temp3= 0, temp4= 0, temp5= 0, temp6= 0, temp7= 0, temp8= 0, temp9 = 0;
        float temp10= 0, temp11= 0, temp12= 0, temp13= 0, temp14= 0, temp15= 0, temp16= 0, temp17= 0, temp18= 0, temp19 = 0;
        float temp20= 0, temp21= 0, temp22= 0, temp23= 0, temp24= 0, temp25= 0, temp26= 0, temp27= 0, temp28= 0, temp29 = 0;
        float temp30= 0, temp31= 0, temp32= 0, temp33= 0, temp34= 0, temp35= 0, temp36= 0, temp37= 0, temp38= 0, temp39 = 0;
        float temp40= 0, temp41= 0, temp42= 0, temp43= 0, temp44= 0, temp45= 0, temp46= 0, temp47= 0, temp48= 0, temp49 = 0;
        float temp50= 0, temp51= 0, temp52= 0, temp53= 0, temp54= 0, temp55= 0, temp56= 0, temp57= 0, temp58= 0, temp59 = 0;
        float temp60= 0, temp61= 0, temp62= 0, temp63= 0, temp64= 0, temp65= 0, temp66= 0, temp67= 0, temp68= 0, temp69 = 0;
        float temp70= 0, temp71= 0, temp72= 0, temp73= 0, temp74= 0, temp75= 0, temp76= 0, temp77= 0, temp78= 0, temp79 = 0;
        float temp80= 0, temp81= 0, temp82= 0, temp83= 0, temp84= 0, temp85= 0, temp86= 0, temp87= 0, temp88= 0, temp89 = 0;
        float temp90= 0, temp91= 0, temp92= 0, temp93= 0, temp94= 0, temp95= 0, temp96= 0, temp97= 0, temp98= 0, temp99 = 0;

        int N = F/T10; // N = 100/10=10; for F = 100 and T = 10
        int effective_block_size = N*(N+1)/2;
        //get the x and y coordinate
        int tile_x = 0;
        int tile_y = 0;
        for ( int i = 0; i < N; i++ ) {
            int end = ((2*N-i)*(i+1))/2;
            if(threadIdx.x < end){
                tile_x = i * T10;
                tile_y = (N + threadIdx.x - end) * T10;
                break;
            }
        }
        int index = blockIdx.x*F*F;
        //iteration: copy gmem-->smem; aggregate smem-->register
        for (int iter = 0; iter < iterations; iter ++){
            //phase 1 in iteration: gmem --> smem
            
            //REQ: blockDim.x >= F/2
            if(threadIdx.x < F/2){
                for(int k = 0; k< SCAN_BATCH; k++){
                    if(iter*SCAN_BATCH + k < end - start){
                        float2 theta;
                        theta.x = __ldg(&thetaT[F * csrColIndex[start + iter*SCAN_BATCH + k] + 2*threadIdx.x]);
                        theta.y = __ldg(&thetaT[F * csrColIndex[start + iter*SCAN_BATCH + k] + 2*threadIdx.x+1]);
                        thetaTemp[k * F/2 + threadIdx.x] = theta;
                        //this simpler statement is slower.
                        //thetaTemp[k * F/2 + threadIdx.x] = __ldg((float2*)&thetaT[F * csrColIndex[start + iter*SCAN_BATCH + k] + 2*threadIdx.x]);
                    }
                    //not enough theta to copy, set zero
                    else
                        memset(&thetaTemp[k*F/2 + threadIdx.x], 0, 2*sizeof(float));
                }
            }           
            __syncthreads();
            
            //phase 2 in iteration: smem --> register
            if(threadIdx.x < effective_block_size){//this redundant "if" seems improving kernel performance
                for(int k = 0; k < SCAN_BATCH; k++){
                    accumulate_in_registers();
                }
            }
        }
        //end of iteration in copying from smem and aggregating in register
        __syncthreads();

        //phase 3, after iteration: register --> gmem
        if(threadIdx.x < effective_block_size){
            fill_lower_half_from_registers();

            //symmetric
            if(tile_x != tile_y){
                fill_upper_half_from_registers();
            }
            //regularization
            if(tile_x == tile_y){
                for(int k = 0; k < T10; k++)
                    tt[index + (tile_x+k)*(1+F)] += (end - start) * lambda;
            }
        }
    }
}


float doALS(const int* csrRowIndexHostPtr, const int* csrColIndexHostPtr, const float* csrValHostPtr,
        const int* cscRowIndexHostPtr, const int* cscColIndexHostPtr, const float* cscValHostPtr,
        const int* cooRowIndexHostPtr, float* thetaTHost, float* XTHost,
        const int * cooRowIndexTestHostPtr, const int * cooColIndexTestHostPtr, const float * cooValHostTestPtr,
        const int m, const int n, const int f, const long nnz, const long nnz_test, const float lambda,
        const int ITERS, const int X_BATCH, const int THETA_BATCH)
{
    int nDevices;
    cudacall(cudaGetDeviceCount(&nDevices));

    omp_set_num_threads(nDevices);

    cudaStream_t  stream[2][nDevices];
    for (int device = 0; device < nDevices; ++device) {
        cudacall(cudaSetDevice(device));
        cudacall(cudaStreamCreate(&stream[0][device]));
        cudacall(cudaStreamCreate(&stream[1][device]));
    }

    cudacall(cudaSetDevice(0));

    printf("*******parameters: m: %d, n:  %d, f: %d, nnz: %ld \n", m, n, f, nnz);
    //device pointers
    float * csrVal[nDevices];
    float * thetaT[nDevices];
    float * XT[nDevices];
    float * cscVal[nDevices];
    int * cscRowIndex[nDevices];
    int * cscColIndex[nDevices];
    int * csrRowIndex[nDevices];
    int * csrColIndex[nDevices];
    //coo to calculate RMSE
    int * cooRowIndex[nDevices];
    float * cooVal_test[nDevices];
    int * cooRowIndex_test[nDevices];
    int * cooColIndex_test[nDevices];

    float * errors_train[nDevices];
    float * errors_test[nDevices];
    float final_rmse = 0;

    int nnz_device = (nnz - 1) / nDevices + 1;
    int nnz_test_device = (nnz_test - 1) / nDevices + 1;

    int error_size_train = (nnz_device - 1) / 256 + 1;
    int error_size_test = (nnz_test_device - 1) / 256 + 1;

    float *rmse_train_device[nDevices];
    float *rmse_test_device[nDevices];

    printf("*******start allocating memory on GPU...\n");

    printf("*******start copying memory to GPU...\n");

    for (int device = 0; device < nDevices; ++device) {
        cudacall(cudaSetDevice(device));
        //dimension: M*F
        cudacall(cudaMalloc((void** ) &XT[device], f * m * sizeof(XT[0][0])));
        //dimension: F*N
        cudacall(cudaMalloc((void** ) &thetaT[device], f * n * sizeof(thetaT[0][0])));

        cudacall(cudaMalloc((void** ) &cscRowIndex[device], nnz * sizeof(cscRowIndex[0][0])));
        cudacall(cudaMalloc((void** ) &cscColIndex[device], (n+1) * sizeof(cscColIndex[0][0])));

        cudacall(cudaMalloc((void** ) &csrRowIndex[device],(m + 1) * sizeof(csrRowIndex[0][0])));
        cudacall(cudaMalloc((void** ) &csrColIndex[device], nnz * sizeof(csrColIndex[0][0])));

        cudacall(cudaMemcpy(thetaT[device], thetaTHost, (size_t ) (n * f * sizeof(thetaT[0][0])), cudaMemcpyHostToDevice));
        //CG needs XT
        cudacall(cudaMemcpy(XT[device], XTHost, (size_t ) (m * f * sizeof(XT[0][0])), cudaMemcpyHostToDevice));

        cudacall(cudaMemcpy(cscRowIndex[device], cscRowIndexHostPtr, (size_t ) nnz * sizeof(cscRowIndex[0][0]), cudaMemcpyHostToDevice));
        cudacall(cudaMemcpy(cscColIndex[device], cscColIndexHostPtr, (size_t ) (n+1) * sizeof(cscColIndex[0][0]), cudaMemcpyHostToDevice));

        cudacall(cudaMemcpy(csrRowIndex[device], csrRowIndexHostPtr,(size_t ) ((m + 1) * sizeof(csrRowIndex[0][0])), cudaMemcpyHostToDevice));
        cudacall(cudaMemcpy(csrColIndex[device], csrColIndexHostPtr,(size_t ) (nnz * sizeof(csrColIndex[0][0])), cudaMemcpyHostToDevice));

        cudacall(cudaMalloc((void** ) &cooRowIndex[device], nnz * sizeof(cooRowIndex[0][0])));
        cudacall(cudaMemcpy(cooRowIndex[device], cooRowIndexHostPtr,(size_t ) (nnz * sizeof(cooRowIndex[0][0])), cudaMemcpyHostToDevice));

        cudacall(cudaMalloc((void** ) &csrVal[device], nnz * sizeof(csrVal[0][0])));
        cudacall(cudaMemcpy(csrVal[device], csrValHostPtr, (size_t ) (nnz * sizeof(csrVal[0][0])), cudaMemcpyHostToDevice));

        cudacall(cudaMalloc((void** ) &cooRowIndex_test[device], nnz_test * sizeof(cooRowIndex_test[0][0])));
        cudacall(cudaMemcpy(cooRowIndex_test[device], cooRowIndexTestHostPtr,(size_t ) (nnz_test * sizeof(cooRowIndex_test[0][0])), cudaMemcpyHostToDevice));

        cudacall(cudaMalloc((void** ) &cooColIndex_test[device], nnz_test * sizeof(cooColIndex_test[0][0])));
        cudacall(cudaMalloc((void** ) &cooVal_test[device], nnz_test * sizeof(cooVal_test[0][0])));

        cudacall(cudaMemcpy(cooColIndex_test[device], cooColIndexTestHostPtr,(size_t ) (nnz_test * sizeof(cooColIndex_test[0][0])), cudaMemcpyHostToDevice));
        cudacall(cudaMemcpy(cooVal_test[device], cooValHostTestPtr,(size_t ) (nnz_test * sizeof(cooVal_test[0][0])),cudaMemcpyHostToDevice));

        cudacall(cudaMalloc((void** ) &errors_train[device], error_size_train * sizeof(errors_train[0][0])));
        cudacall(cudaMalloc((void** ) &errors_test[device], error_size_test * sizeof(errors_test[0][0])));

        cudacall(cudaMalloc((void** ) &rmse_train_device[device], sizeof(rmse_train_device[0][0])));
        cudacall(cudaMalloc((void** ) &rmse_test_device[device], sizeof(rmse_test_device[0][0])));

    	cudacall(cudaMalloc((void** ) &cscVal[device], nnz * sizeof(cscVal[0][0])));
    	cudacall(cudaMemcpy(cscVal[device], cscValHostPtr,(size_t ) (nnz * sizeof(cscVal[0][0])),cudaMemcpyHostToDevice));

	    cudacall(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	    //64-bit smem access
	    //http://acceleware.com/blog/maximizing-shared-memory-bandwidth-nvidia-kepler-gpus
	    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    }

    //initialize cublas, cusparse
    cublasHandle_t handle[nDevices];
    cusparseHandle_t cushandle[nDevices];
    cusparseMatDescr_t descr[nDevices];

    #pragma omp parallel
    {
        int device = omp_get_thread_num();

        cudacall(cudaSetDevice(device));

        cublascall(cublasCreate(&handle[device]));
        cusparsecall(cusparseCreate(&cushandle[device]));
        cusparsecall( cusparseCreateMatDescr(&descr[device]));
        cusparseSetMatType(descr[device], CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr[device], CUSPARSE_INDEX_BASE_ZERO);
    }
    using namespace std;
    #ifdef DEBUG
    //variable used to time
    double t0 = 0;
    double t1 = 0;
    #endif

    double t_itr = seconds();

    printf("*******start iterations...\n");
    for(int iter = 0; iter < ITERS ; iter ++){
        #ifdef DEBUG
        printf("---------------------------ALS iteration %d, update X.----------------------------------\n", iter);
        t0 = seconds();
        t1 = seconds();
        #endif

        float * tt[nDevices];
        float * ythetaT[nDevices];
        float * ytheta[nDevices];

        #ifdef DEBUG
        printf("\tgenerate: Y*theta using cusparse.\n");
        #endif

		for (int device = 0; device < nDevices; ++device) {
            cudacall(cudaSetDevice(device));

            cudacall(cudaMalloc((void** ) &tt[device], m / X_BATCH * f * f * sizeof(float)));
            cudacall(cudaMalloc((void** ) &ythetaT[device], f * m * sizeof(ythetaT[0][0])));
            cudacall(cudaMalloc((void** ) &ytheta[device], f * m * sizeof(ytheta[0][0])));

        }

        #pragma omp parallel
        {
            int device = omp_get_thread_num();
            cudacall(cudaSetDevice(device));

            const float alpha = 1.0f;
            const float beta = 0.0f;

            cusparsecall(cusparseScsrmm2(cushandle[device], CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_TRANSPOSE, m, f, n, nnz, &alpha, descr[device], csrVal[device],
                    csrRowIndex[device], csrColIndex[device], thetaT[device], f, &beta, ytheta[device], m) );

            cublascall(cublasSgeam(handle[device], CUBLAS_OP_T, CUBLAS_OP_N, f, m, &alpha,
                    (const float * ) ytheta[device], m, &beta, ythetaT[device], f, ythetaT[device], f));

            cudacall(cudaDeviceSynchronize());
            cudacall(cudaFree(ytheta[device]));

        }
        
        #ifdef DEBUG
        printf("\tgenerate: Y*theta run %f seconds.\n", seconds() - t1);
        #endif

        int block_dim = f/T10*(f/T10+1)/2;
        if (block_dim < f/2) block_dim = f/2;
        for(int batch_id_base = 0; batch_id_base < X_BATCH; batch_id_base += nDevices){

            #pragma omp parallel
            {

                int device = omp_get_thread_num();

                int batch_id = batch_id_base + device;

                if (batch_id < X_BATCH) {

                    cudacall(cudaSetDevice(device));

                    #ifdef DEBUG
                    printf("*******batch %d / %d on device %d*******\n", batch_id, X_BATCH, device);
                    #endif

                    int batch_size = 0;
                    if(batch_id != X_BATCH - 1)
                        batch_size = m/X_BATCH;
                    else
                        batch_size = m - batch_id*(m/X_BATCH);
                    int batch_offset = batch_id * (m/X_BATCH);

                    if(f == 100){
                        get_hermitian100<<<batch_size, 64, SCAN_BATCH * f/2 * sizeof(float2), stream[0][device]>>>
                            (batch_offset, (float2*)tt[device], csrRowIndex[device], csrColIndex[device], lambda, m, f, (float2*)thetaT[device]);
                        //This commented out is the fused kernel
                        //performance not good due to register pressure and low occupancy
                        //alsUpdateFeature100Host
                        //  (batch_offset, csrRowIndex, csrColIndex, lambda, m, f, thetaT, XT, ythetaT, 6);
                    }
                    else {
                        get_hermitianT10<<<batch_size, block_dim, SCAN_BATCH * f/2 * sizeof(float2), stream[0][device]>>>
                            (batch_offset, tt[device], csrRowIndex[device], csrColIndex[device], lambda, m, f, thetaT[device]);
                    }

                    updateXWithCGHostAsync(tt[device], &XT[device][batch_offset*f], &ythetaT[device][batch_offset*f], batch_size, f, CG_ITER, &stream[0][device]);

                }

            }

            #pragma omp parallel
            {
                    int device = omp_get_thread_num();

                    int batch_id = batch_id_base + device;

                    if (batch_id < X_BATCH) {

                        cudacall(cudaSetDevice(device));
                        cudacall(cudaStreamSynchronize(stream[0][device]));

                        int batch_size = 0;
                        if(batch_id != X_BATCH - 1)
                            batch_size = m/X_BATCH;
                        else
                            batch_size = m - batch_id*(m/X_BATCH);
                        int batch_offset = batch_id * (m/X_BATCH);

                        for (int j = 0; j < nDevices; ++j) {
                            if (j != device) {
                                cudacall(cudaMemcpyPeerAsync(&XT[j][batch_offset*f], j, &XT[device][batch_offset*f], device, batch_size * f * sizeof(XT[0][0]), stream[1][device]));
                            }
                        }

                    }

            }
            
        }

        float * xx[nDevices];
        float * yTXT[nDevices];
        float * yTX[nDevices];

        for (int device = 0; device < nDevices; ++device) {
            cudacall(cudaSetDevice(device));

            cudacall(cudaMalloc((void** ) &yTXT[device], f * n * sizeof(yTXT[0][0])));
            cudacall(cudaMalloc((void** ) &yTX[device], n * f * sizeof(yTX[0][0])));
        }

        #ifdef DEBUG
        printf("update X run %f seconds, gridSize: %d, blockSize %d.\n", seconds() - t0, m, f);
        #endif

        #pragma omp parallel
        {
            int device = omp_get_thread_num();

            cudacall(cudaSetDevice(device));
            cudacall(cudaStreamSynchronize(stream[1][device]));

            cudacall(cudaFree(tt[device]));
            cudacall(cudaFree(ythetaT[device]));
        }
///*
        #ifdef DEBUG
        t0 = seconds();
        t1 = seconds();
        printf("---------------------------------- ALS iteration %d, update theta ----------------------------------\n", iter);
        printf("\tgenerate: Y'*X using cusparse.\n");
        #endif

        #pragma omp parallel
        {
            int device = omp_get_thread_num();

            cudacall(cudaSetDevice(device));

	        const float alpha = 1.0f;
	        const float beta = 0.0f;

	        cusparsecall( cusparseScsrmm2(cushandle[device], CUSPARSE_OPERATION_NON_TRANSPOSE,
	                CUSPARSE_OPERATION_TRANSPOSE, n, f, m, nnz, &alpha, descr[device], cscVal[device],
	                cscColIndex[device], cscRowIndex[device], XT[device], f, &beta, yTX[device], n) );
	        cublascall(cublasSgeam(handle[device], CUBLAS_OP_T, CUBLAS_OP_N, f, n, &alpha,
	                (const float * ) yTX[device], n, &beta, yTXT[device], f, yTXT[device], f));

            cudacall(cudaDeviceSynchronize());
            cudacall(cudaFree(yTX[device]));

        }

        #ifdef DEBUG
        printf("\tgenerate: Y'*X run %f seconds.\n", seconds() - t1);
        #endif

        //in batches, when N is huge
        for(int batch_id_base = 0; batch_id_base < THETA_BATCH; batch_id_base += nDevices){

            #pragma omp parallel
            {

                int device = omp_get_thread_num();

                int batch_id = batch_id_base + device;

                if (batch_id < THETA_BATCH) {

                    cudacall(cudaSetDevice(device));

                    #ifdef DEBUG
                    printf("*******batch %d / %d on device %d*******\n", batch_id, THETA_BATCH, device);
                    #endif

                    int batch_size = 0;
                    if(batch_id != THETA_BATCH - 1)
                        batch_size = n/THETA_BATCH;
                    else
                        batch_size = n - batch_id*(n/THETA_BATCH);
                    int batch_offset = batch_id * (n/THETA_BATCH);
                    
                    //cudacall(cudaMemset(xx, 0, f * f * batch_size * sizeof(float)) );

                    cudacall(cudaMalloc((void** ) &xx[device], batch_size * f * f * sizeof(xx[0][0])));

                    if(f == 100){
                        get_hermitian100<<<batch_size, 64, SCAN_BATCH * f/2*sizeof(float2), stream[0][device]>>>
                            (batch_offset, (float2*)xx[device], cscColIndex[device], cscRowIndex[device], lambda, n, f, (float2*)XT[device]);
                    }
                    else {
                        get_hermitianT10<<<batch_size, block_dim, SCAN_BATCH*f*sizeof(float), stream[0][device]>>>
                            (batch_offset, xx[device], cscColIndex[device], cscRowIndex[device], lambda, n, f, XT[device]);
                    }

                    printf("\tCG solver with fp32.\n");
                    updateXWithCGHostAsync(xx[device], &thetaT[device][batch_offset*f], &yTXT[device][batch_offset*f], batch_size, f, CG_ITER, &stream[0][device]);

                }
            }

            #pragma omp parallel
            {

                int device = omp_get_thread_num();

                int batch_id = batch_id_base + device;

                if (batch_id < THETA_BATCH) {

                    cudacall(cudaSetDevice(device));
                    cudacall(cudaStreamSynchronize(stream[0][device]));

                    int batch_size = 0;
                    if(batch_id != THETA_BATCH - 1)
                        batch_size = n/THETA_BATCH;
                    else
                        batch_size = n - batch_id*(n/THETA_BATCH);
                    int batch_offset = batch_id * (n/THETA_BATCH);

                    for (int j = 0; j < nDevices; ++j) {
                        if (j != device) {
                            cudacall(cudaMemcpyPeerAsync(&thetaT[j][batch_offset*f], j, &thetaT[device][batch_offset*f], device, batch_size * f * sizeof(thetaT[0][0]), stream[1][device]));
                        }
                    }

                    cudacall(cudaFree(xx[device]));

                }
            }       
            
        }

        #pragma omp parallel
        {
            int device = omp_get_thread_num();

            cudacall(cudaSetDevice(device));
            cudacall(cudaStreamSynchronize(stream[1][device]));

            cudacall(cudaFree(yTXT[device]));
        }

        #ifdef DEBUG
        printf("update theta run %f seconds, gridSize: %d, blockSize %d.\n",
                seconds() - t0, n, f);
        printf("Calculate RMSE.\n");
        #endif

        printf("Error buffer size = %d\n", error_size_train);

        float rmse_train_host[nDevices];
        float rmse_test_host[nDevices];

        float rmse_train = 0.0;
        float rmse_test = 0.0;

        #pragma omp parallel
        {
            int device = omp_get_thread_num();

            cudacall(cudaSetDevice(device));

            printf("Calculate partial RMSE on device %d\n", device);

            int offset = nnz_device * device;
            int bound = nnz_device * (device + 1);
            if (bound > nnz) {
                bound = nnz;
            }

            RMSE_reduction2<<<error_size_train, 256, 0, stream[0][device]>>>
                    (offset, csrVal[device], cooRowIndex[device], csrColIndex[device], thetaT[device], XT[device], errors_train[device], bound, f);

            reduction<<<1, 1024, 0, stream[0][device]>>>(errors_train[device], rmse_train_device[device], error_size_train);

            cudacall(cudaMemcpyAsync(rmse_train_host + device, rmse_train_device[device], sizeof(rmse_train_host[0]), cudaMemcpyDeviceToHost, stream[0][device]));

            offset = nnz_test_device * device;
            bound = nnz_test_device * (device + 1);
            if (bound > nnz_test) {
                bound = nnz_test;
            }

            RMSE_reduction2<<<error_size_test, 256, 0, stream[1][device]>>>
                    (offset, cooVal_test[device], cooRowIndex_test[device], cooColIndex_test[device], thetaT[device], XT[device], errors_test[device], bound, f);

            reduction<<<1, 1024, 0, stream[1][device]>>>(errors_test[device], rmse_test_device[device], error_size_test);

            cudacall(cudaMemcpyAsync(rmse_test_host + device, rmse_test_device[device], sizeof(rmse_test_host[0]), cudaMemcpyDeviceToHost, stream[1][device]));

        }

        //#pragma omp parallel shared(rmse_train) shared(rmse_test)
        //{
            //int device = omp_get_thread_num();          
		for (int device = 0; device < nDevices; ++device) {

            cudacall(cudaSetDevice(device));

            cudaDeviceSynchronize();

            rmse_train += rmse_train_host[device];
            rmse_test += rmse_test_host[device];
        }

        printf("--------- Train RMSE in iter %d: %f\n", iter, sqrt(rmse_train/nnz));

        final_rmse = sqrt(rmse_test/nnz_test);
        printf("--------- Test RMSE in iter %d: %f\n", iter, final_rmse);

//*/        
    }

    printf("%d iterations takes %lf seconds\n", ITERS, seconds() - t_itr);

    cudacall(cudaSetDevice(0));

    //copy feature vectors back to host
    cudacall(cudaMemcpy(thetaTHost, thetaT[0], (size_t ) (n * f * sizeof(thetaT[0][0])), cudaMemcpyDeviceToHost));
    cudacall(cudaMemcpy(XTHost, XT[0], (size_t ) (m * f * sizeof(XT[0][0])), cudaMemcpyDeviceToHost));

    for (int device = 0; device < nDevices; ++device) {
        cudacall(cudaSetDevice(device));

        cudacall(cudaFree(XT[device]));
        cudacall(cudaFree(thetaT[device]));

        cudacall(cudaFree(cscColIndex[device]));
        cudacall(cudaFree(cscRowIndex[device]));

        cudacall(cudaFree(csrRowIndex[device]));
        cudacall(cudaFree(csrColIndex[device]));

        cudacall(cudaFree(cooRowIndex[device]));
        cudacall(cudaFree(csrVal[device]));

        cudacall(cudaFree(cooRowIndex_test[device]));
        cudacall(cudaFree(cooColIndex_test[device]));
        cudacall(cudaFree(cooVal_test[device]));

        cudacall(cudaFree(errors_train[device]));
        cudacall(cudaFree(errors_test[device]));

        cudacall(cudaFree(rmse_train_device[device]));
        cudacall(cudaFree(rmse_test_device[device]));

        cudacall(cudaFree(cscVal[device]));
    }
    //WARN: do not call cudaDeviceReset inside ALS() 
    //because the caller needs to access XT and thetaT which was in the same context
    //cudacall(cudaDeviceReset());
    return final_rmse;
}
