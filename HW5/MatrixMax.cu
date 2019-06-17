#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <string>
#include <fstream>
#include <math.h>

using namespace std;

#define cudaCheckError() {																\
	cudaError_t e=cudaGetLastError();													\
	if(e!=cudaSuccess) {																\
		printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));	\
		}																					\
}


inline int GetBlockSize(int b, int maxSize)
{
	if (b <= maxSize)
		return b;
	else
		return maxSize;
}


inline int GetGridSize(int n, int b)
{
	if (n%b == 0)
		return n / b;
	else
		return int(n*1.0 / (b*1.0)) + 1;
}

#define max(a,b)  (((a) > (b)) ? (a) : (b))

__global__ void FindMax4(float *in, float *out, int n)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if (i < n && j < n)
	{
		// Output index
		int index = j * n + i;

		// Input index
		// Set up the input index correctly for the 4 inputs
		//patten 1
		// int index1 = (i * 2) + (j * n * 2 * 2);
		// int index2 = 1 + (i * 2) + (j * n * 2 * 2);
		// int index3 = (i * 2) + (j * n * 2 * 2) + (n * 2);
		// int index4 = 1 + (i * 2) + (j * n * 2 * 2) + (n * 2);

		//patten 2
		int index1 = i + (j * n * 2);
		int index2 = i + n + (j * n * 2);
		int index3 = i + (j * n * 2) + (n * n * 2);
		int index4 = i + n + (j * n * 2) + (n * n * 2);



		// Compute the max of 4 values
		float max1 = max(in[index1], in[index2]);
		float max2 = max(in[index3], in[index4]);
		float max = max(max1, max2);
		out[index] = max;
	}
}

// handle non power of 2
void InitMatrix(float* a, int n, int m, int oldMatrixSize)
{
	srand((int)time(NULL));
	for (int j = 0; j < m; j++)
	{
		for (int i = 0; i < n; i++)
		{
    		if (i>(oldMatrixSize-1)){
    		    a[j*n+i] = 0;
    		}
    		else if (j>(oldMatrixSize-1)){
    		    a[j*n+i] = 0;
    		}
    		else{
    		    a[j*n + i] = float(10.0 * rand() / (RAND_MAX*1.0));
    		}
		}
	}
}
// void InitMatrix(float* a, int n, int m)
// {
// 	srand((int)time(NULL));
// 	for (int j = 0; j < m; j++)
// 		for (int i = 0; i < n; i++)
// 			a[j*n + i] = float(10.0 * rand() / (RAND_MAX*1.0));
	
// 	// Use the code below for debugging if required
// 	//a[j*n + i] = j*n + i;

// }

void PrintMatrix(float* a, int n, int m)
{
	for (int j = 0; j < m; j++)
	{
		for (int i = 0; i < n; i++)
			cout << a[j*n + i] << " ";
		cout << endl;
	}
}
float FindMaxCPU(float* a, int n, int m)
{ 
	float maxVal = 0;
	for (int j = 0; j < m; j++)
		for (int i = 0; i < n; i++)
			maxVal = max(maxVal, a[j*n + i]);
	return maxVal;
}

int main()
{
	//store result to txt
	ofstream myFile;
	myFile.open("../runRes.csv");

	int testMS[7] = {128,256,512,1024,2048,4096,8192};

	for(int t; t<7; t++){
		
		// Set size of the matrix
		int n = testMS[t]+1;
		int mSize = 16;
		// if not power tow
		int oldMatrixSize = n;
		cout<<oldMatrixSize<<endl;
    
	    if ((n&(n-1))==0)
	    {
	    	cout << "power of two - create matrix original" << endl;
	    }
		else
		{
			// float nC = n;
			int newPowerTwo = log2((float)n)+1;
			n = pow(2, newPowerTwo);
			cout << "Non power of two - create matrix with padding zeros" << endl;
		}

		// create the cuda event to count the running time for GPU
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		// Create CPU Array
		float* matrix = new float[n*n];
		InitMatrix(matrix, n, n, oldMatrixSize);
		cout << "Created a " << n << " x " << n << " Matrix." << endl;
		cout << "block size: " << mSize << endl;
		// find max count time cpu
		clock_t c_start = clock();
		float maxVal = FindMaxCPU(matrix, n, n);
		clock_t c_end = clock();
		cout << "Maximum value from CPU computation is : " << maxVal << endl;
		
		long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
	    cout << "CPU time used: " << time_elapsed_ms << " ms\n";

		// Use the following code for print debugging
		// cout << endl;
		// PrintMatrix(matrix, n, n);
		// cout << endl;

		// Allocate GPU Memory
		float* matrix1CUDA;
		float* matrix2CUDA;
		cudaMalloc((void**)&(matrix1CUDA), n*n*sizeof(float));
		cudaMalloc((void**)&(matrix2CUDA), n*n*sizeof(float));
		cudaCheckError();

		// Copy GPU Memory
		cudaMemcpy(matrix1CUDA, matrix, n*n*sizeof(float), cudaMemcpyHostToDevice);
		cudaCheckError();

		// Setup swap of CUDA device pointers
		float* inputCUDA;
		float* outputCUDA;
		inputCUDA = (matrix1CUDA);
		outputCUDA = (matrix2CUDA);

		// start timer 
		cudaEventRecord(start);
		// Run the Kernel
		for (int p = n / 2; p >= 1; p = p / 2)
		{
			dim3 block(GetBlockSize(p, mSize), GetBlockSize(p, mSize), 1);
			dim3 grid(GetGridSize(p, block.x), GetGridSize(p, block.y), 1);
			FindMax4 << < grid, block >> >(inputCUDA, outputCUDA, p);
			cudaCheckError();


			// Use the following code for print debugging
	#ifdef DEBUG
	// 		//float* tempDataIn = new float[2 * p * 2 * p];
	// 		//cudaMemcpy(tempDataIn, inputCUDA, 2 * p * 2 * p * sizeof(float), cudaMemcpyDeviceToHost);
	// 		//cudaCheckError();
	// 		//PrintMatrix(tempDataIn, 2*p, 2*p);
	// 		//cout << endl;
	// 		//delete[] tempDataIn;

			float* tempDataOut = new float[p*p];
			cudaMemcpy(tempDataOut, outputCUDA, p * p * sizeof(float), cudaMemcpyDeviceToHost);
			cudaCheckError();
			// PrintMatrix(tempDataOut, p, p);
			// cout << endl;
			delete[] tempDataOut;
	#endif

			// Swap input output pointers
			float* oldInputCUDA = inputCUDA;
			inputCUDA = outputCUDA;
			outputCUDA = oldInputCUDA;

		}
		// end timer 
		cudaEventRecord(stop);

		float maxValGPU;
		cudaMemcpy(&maxValGPU, inputCUDA, 1 * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Maximum value from GPU computation is : " << maxValGPU << endl;

		// timer time in milliseconds
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		cout << "GPU time used: "<< milliseconds << " ms\n";

		cudaDeviceSynchronize();
		cudaCheckError();

		// Free the Memory
		cudaFree(matrix1CUDA);
		cudaFree(matrix2CUDA);

		
		myFile << oldMatrixSize <<","<< mSize <<"," << mSize <<","<< time_elapsed_ms << "," << milliseconds <<endl;

	#ifdef DEBUG
		cudaCheckError();
	#endif
	}

	return 0;
}
