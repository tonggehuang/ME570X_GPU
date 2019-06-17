#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <string>
#include <fstream>
#include <math.h>

using namespace std;

#define rawR 7
#define rawC 840
#define rawL (rawR*rawC)
#define LENGTH 840
#define BATCH 1
#define LENGTHPAD 1024
#define NRANK 2

static __global__ void cufftComplexScale(cufftComplex *idata, cufftComplex *odata, const int size, float scale)
{
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadID < size)
    {
        odata[threadID].x = idata[threadID].x * scale;
        odata[threadID].y = idata[threadID].y * scale;
    }
}

int main()
{
  int n[NRANK] = {rawR, rawC};
  // create arrays
  float speed2d[rawR][rawC];

  // read raw data 
  ifstream rawData("../data/speedData.txt");

  // generate 2d speed data
  if (!(rawData.is_open())){
    cout<<"faild to read data." << endl;
  } 

  for (int row=0; row<rawR; row++){
    for (int col=0; col<rawC; col++){
      rawData >> speed2d[row][col];
    }
  }

  rawData.close();

  // print array for debug
  // for (int row=0; row<rawR; row++){
  //   for (int col=0; col<10; col++){
  //     cout << speed2d[row][col] << '\t';
  //   }
  //   cout << '\n';
  // }

  // host data pointer
  cufftComplex *CompData2d=(cufftComplex*)malloc(rawC*rawR*sizeof(cufftComplex)); 

  // 2 d

  for (int i=0; i<rawR; i++){
    for (int j=0; j<rawC; j++){
      CompData2d[i*rawC+j].x = speed2d[i][j];
      CompData2d[i*rawC+j].y = 0;
    }
  }

  cufftComplex *d_fftData; // device data pointer
  cudaMalloc((void**)&d_fftData,rawC*rawR*sizeof(cufftComplex));
  cudaMemcpy(d_fftData,CompData2d,rawC*rawR*sizeof(cufftComplex),cudaMemcpyHostToDevice);

  // create the cuda event to count the running time for GPU
  cudaEvent_t start1, stop1;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);

  cufftHandle plan;
  cufftPlanMany(&plan, NRANK, n,
        NULL, 1, 0,
        NULL, 1, 0,
        CUFFT_C2C, BATCH);

  // execute kernel
  

  cufftExecC2C(plan,(cufftComplex*)d_fftData,(cufftComplex*)d_fftData,CUFFT_FORWARD);


  cudaEventRecord(start1);
  cufftExecC2C(plan, (cufftComplex*)d_fftData, (cufftComplex*)d_fftData, CUFFT_INVERSE);
 
  dim3 dimBlock(1024);
  dim3 dimGrid(6); 

  cufftComplexScale <<<dimGrid, dimBlock>>>((cufftComplex*)d_fftData,(cufftComplex*)d_fftData,rawC*rawR,1.0f / (rawC*rawR));
  
  cudaEventRecord(stop1);

  cudaEventSynchronize(stop1);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start1, stop1);
  cout << "GPU FFT time used: "<< milliseconds << " ms\n";



  cudaDeviceSynchronize();

  cudaMemcpy(CompData2d,d_fftData,rawC*rawR*sizeof(cufftComplex)*BATCH,cudaMemcpyDeviceToHost);

  //store result to txt
  ofstream myFile;
  myFile.open("../data/2d_batch_inverse.txt");

  for (int i=0; i<rawR; i++){
    for (int j=0; j<rawC; j++){
      myFile << CompData2d[i*rawC+j].x <<','<< CompData2d[i*rawC+j].y << endl;
    }
  }

  cufftDestroy(plan);
  free(CompData2d);
  cudaFree(d_fftData);

  return 0;
}