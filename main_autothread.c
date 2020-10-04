#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>
#include "mmio.h"
#include "mmiohighlevel.h"
#include <omp.h>
#include <math.h>
#include <immintrin.h>
typedef struct
{
	VALUE_TYPE *value;
	int *columnindex;
	int *rowpointer;

} SMatrix;

void scan(int *array, int n)
{
	int nthreads = omp_get_max_threads();
	int *lastitem = (int *)malloc(sizeof(int) * nthreads);
	memset(lastitem, 0, sizeof(int) * nthreads);

	int np = ceil((double)n / (double)nthreads);
#pragma omp parallel for
	for (int tid = 0; tid < nthreads; tid++)
	{
		int start = tid * np;
		start = start > n ? n : start;
		int end = (tid + 1) * np;
		end = end > n ? n : end;
		if (start == end)
		{
			lastitem[tid] = 0;
			continue;
		}
		int old, new;
		old = array[start];
		array[start] = 0;
		for (int i = start + 1; i <= end; i++)
		{
			if (i != end)
			{
				new = array[i];
				array[i] = old + array[i - 1];
				old = new;
			}
			else
			{
				lastitem[tid] = old + array[i - 1];
			}
		}
	}
	int old, new;
	old = lastitem[0];
	lastitem[0] = 0;
	for (int i = 1; i < nthreads; i++)
	{
		new = lastitem[i];
		lastitem[i] = old + lastitem[i - 1];
		old = new;
	}

#pragma omp parallel for
	for (int tid = 0; tid < nthreads; tid++)
	{
		int start = tid * np;
		start = start > n ? n : start;
		int end = (tid + 1) * np;
		end = end > n ? n : end;
		if (start == end)
		{
			continue;
		}
		for (int i = start; i < end; i++)
		{
			array[i] += lastitem[tid];
		}
	}
	//over
	free(lastitem);
}

void Transpose(VALUE_TYPE *A, int m, int n, VALUE_TYPE *AT)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
		{
			AT[j * m + i] = A[i * n + j];
		}
}

void printvec(int *array, int length)
{
	for (int i = 0; i < length; i++)
		printf("%d ", array[i]);
	printf("\n");
}
void DenseToCSR(VALUE_TYPE *C, int m, int n, SMatrix *W)
{
	W->rowpointer = (int *)malloc(sizeof(int) * (m + 1)); //scan 后得到rowpointer
	memset(W->rowpointer, 0, sizeof(int) * (m + 1));
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (C[i * n + j] != 0)
			{
				W->rowpointer[i]++;
			}
		}
	}

	//to scan
	scan(W->rowpointer, (m + 1));
	//printf("scan over\n");
	W->columnindex = (int *)malloc(sizeof(int) * (W->rowpointer[m]));
	W->value = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * W->rowpointer[m]);
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < m; i++)
	{
		int start = W->rowpointer[i];
		int end = W->rowpointer[i + 1];
		for (int j = 0; j < n; j++)
		{
			if (start == end)
				break;
			if (C[i * n + j] != 0)
			{
				W->columnindex[start] = j;
				W->value[start] = C[i * n + j];
				start++;
			}
		}
	}
	//printf("trans over");
}

int main(int argc, char **argv)
{
	struct timeval t1, t2, t3, t4;
	int size1 = 0;
	int size2 = 0;
	int *tc1;
	int *tc2;
	double bias = -0.3000;
	printf("========= threads num : %d ==========\n", omp_get_max_threads());
	int mA;
	int nA;
	int nnzA;
	int isSymmetricA;
	SMatrix A;

	int mB;
	int nB;
	int nnzB;
	int isSymmetricB;
	SMatrix B[120];

	int mC, nC;
	int nnzC_golden = 0;

	// load matrix data from file
	gettimeofday(&t3, NULL);
	char filename1[] = "sparse-images-1024.tsv";
	mmio_info(&mA, &nA, &nnzA, &isSymmetricA, filename1);
	A.value = (VALUE_TYPE *)malloc((nnzA) * sizeof(VALUE_TYPE));
	A.columnindex = (int *)malloc((nnzA) * sizeof(int));
	A.rowpointer = (int *)malloc((mA + 1) * sizeof(int));
	mmio_data(A.rowpointer, A.columnindex, A.value, filename1);
	printf("input matrix A: ( %i, %i ) nnz = %i\n", mA, nA, nnzA);
	VALUE_TYPE *A0 = (VALUE_TYPE *)malloc(mA * nA * sizeof(VALUE_TYPE));
	char neuronfile1[] = "neuron1024/n1024-l";
	char neuronfile2[] = ".tsv";
	char filename3[60];

	for (int k = 0; k < 120; k++)
	{
		char filenum[5];
		int k1 = k + 1;
		snprintf(filenum, sizeof(filenum), "%d", k1);

		strcpy(filename3, neuronfile1);
		strcat(filename3, filenum);
		strcat(filename3, neuronfile2);

		mmio_info(&mB, &nB, &nnzB, &isSymmetricB, filename3);
		B[k].value = (VALUE_TYPE *)malloc((nnzB) * sizeof(VALUE_TYPE));
		B[k].columnindex = (int *)malloc((nnzB) * sizeof(int));
		B[k].rowpointer = (int *)malloc((mB + 1) * sizeof(int));
		mmio_data(B[k].rowpointer, B[k].columnindex, B[k].value, filename3);
	}

	gettimeofday(&t4, NULL);
	double time_load = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
	printf("Weight matrix load time: %f ms \n", time_load);

	mC = mA;
	nC = nB;
	VALUE_TYPE *C0 = (VALUE_TYPE *)malloc((mC * nC) * sizeof(VALUE_TYPE));
	double time_trans;
	gettimeofday(&t3, NULL);
	for (int k = 0; k < 120; k++)
	{
		int k1 = k + 1;
		//memset(C0, 0, sizeof(VALUE_TYPE) * mC * nC);
		int NC = mC * nC;
		/*#pragma omp parallel for
		for (int i = 0; i < NC; i++)
		{
			C0[i] = 0;
		}*/
		__m256 v = _mm256_setzero_ps();
#pragma omp parallel for
		for (int i = 0; i < NC; i += 8)
		{
			_mm256_storeu_ps(&C0[i], v);
		}

		gettimeofday(&t1, NULL);
#pragma omp parallel for schedule(dynamic)
		//csr * csr = Dense(C)
		for (int i = 0; i < mA; i++)
		{
			int Astart = A.rowpointer[i];
			int Aend = A.rowpointer[i + 1];
			for (int j = Astart; j < Aend; j++)
			{
				// ith row , A.columnidex[j] col
				int Bstart = B[k].rowpointer[A.columnindex[j]];
				int Bend = B[k].rowpointer[A.columnindex[j] + 1];
				for (int B_row = Bstart; B_row < Bend; B_row++)
				{
					C0[i * nC + B[k].columnindex[B_row]] += A.value[j] * B[k].value[B_row];
				}
			}
		}
		gettimeofday(&t2, NULL);
		double time_gemm = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

		gettimeofday(&t1, NULL);

		int nthreads = omp_get_max_threads();
		int n = mC * nC;
		int np = ceil((double)n / (double)nthreads);
#pragma omp parallel for
		for (int tid = 0; tid < nthreads; tid++)
		{
			int start = tid * np;
			start = start > n ? n : start;
			int end = (tid + 1) * np;
			end = end > n ? n : end;
			if (start == end)
			{
				continue;
			}
			for (int i = start; i < end; i++)
			{
				C0[i] += bias; //bias = -0.3
				if (C0[i] <= 0)
				{
					C0[i] = 0;
				}
				else if (C0[i] >= 32)
				{
					C0[i] = 32;
				}
			}
		}
		gettimeofday(&t2, NULL);
		double time_biasrelu = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		if (k != 119)
		{
			free(A.rowpointer);
			free(A.columnindex);
			free(A.value);

			gettimeofday(&t1, NULL);
			DenseToCSR(C0, mC, nC, &A);
			gettimeofday(&t2, NULL);
			time_trans = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		}
		else
			memcpy(A0, C0, (mC * nC) * sizeof(VALUE_TYPE));

		printf("k = %d, GEMM time: %4.5f ms, Bias+ReLU time: %4.5f ms,Trans time: %4.5fms\n",
			   k + 1, time_gemm, time_biasrelu, time_trans);
	}

	gettimeofday(&t4, NULL);
	double time_inference = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
	printf("Inference time: %f ms \n", time_inference);

	free(C0);

	// check results

	printf("test\n");
	FILE *fs;
	fs = fopen("sparse-images-1024-1.tsv", "w+");
	for (int i = 0; i < mA; i++)
	{
		int sum = 0;
		for (int j = (i * nA); j < ((i + 1) * nA); j++)
		{
			sum += A0[j];
		}
		if (sum != 0)
		{
			fprintf(fs, "%d\n", i + 1);
		}
	}
	fclose(fs);
	FILE *fp2 = NULL;

	fp2 = fopen("sparse-images-1024-1.tsv", "rb");
	if (fp2 == NULL)
	{
		printf("Error:Open file fail!\n");
	}

	fseek(fp2, 0, SEEK_END);
	size2 = ftell(fp2);
	rewind(fp2);

	tc2 = (int *)malloc(sizeof(int) * size2 / 4);

	int readnum2 = fread(tc2, 4, size2 / 4, fp2);

	fclose(fp2);

	FILE *fp1;

	fp1 = fopen("neuron1024-l120-categories.tsv", "rb");
	if (fp1 == NULL)
	{
		printf("Error:Open file fail!\n");
	}

	fseek(fp1, 0, SEEK_END);
	size1 = ftell(fp1);
	rewind(fp1);

	tc1 = (int *)malloc(sizeof(int) * size1 / 4);

	int readnum1 = fread(tc1, 4, size1 / 4, fp1);

	fclose(fp1);
	int judge = 0;
	for (int i = 0; i < size1 / 4; i++)
	{
		if (tc1[i] - tc2[i] != 0)
		{
			judge++;
		}
	}
	printf("judge:%d\n", judge);
	if (judge == 0)
	{
		printf("CHALLENGE PASSED\n");
	}
	else
	{
		printf("CHALLENGE FAILED\n");
	}

	free(A.rowpointer);
	free(A.columnindex);
	free(A.value);
	free(A0);
	return 0;
}
