#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>
#include "mmio.h"
#include "mmiohighlevel.h"

typedef struct 
{
	VALUE_TYPE *value;
	int *columnindex;
	int *rowpointer;

}SMatrix;

int main(int argc, char ** argv)
{
	struct timeval t1, t2, t3, t4;
	int size1 = 0;
	int size2 = 0;
	int *tc1;
	int *tc2;
	double bias = -0.3000;

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
	
	int mC,nC;
	int nnzC_golden = 0;

    // load matrix data from file
	gettimeofday(&t3, NULL);
	char filename1[]="sparse-images-1024.tsv";
	mmio_info(&mA, &nA, &nnzA, &isSymmetricA, filename1);
	A.value=(VALUE_TYPE*)malloc((nnzA)*sizeof(VALUE_TYPE));
	A.columnindex=(int*)malloc((nnzA)*sizeof(int));
	A.rowpointer=(int*)malloc((mA+1)*sizeof(int));
	mmio_data(A.rowpointer, A.columnindex, A.value, filename1);
	printf("input matrix A: ( %i, %i ) nnz = %i\n", mA, nA, nnzA);
    VALUE_TYPE *A0 = (VALUE_TYPE *)malloc(mA * nA * sizeof(VALUE_TYPE));
    memset(A0, 0, sizeof(VALUE_TYPE) * mA * nA);
    for (int i = 0; i < mA; i++)
    {
        for (int j = A.rowpointer[i]; j < A.rowpointer[i+1]; j++)
        {
            A0[i * nA + A.columnindex[j]] = A.value[j];
        }
    }
	free(A.rowpointer);
	free(A.columnindex);
	free(A.value);

	char neuronfile1[] = "neuron1024/n1024-l";
	char neuronfile2[] = ".tsv";
	char filename3[60];
	
	VALUE_TYPE *B0[120];
	for (int k = 0; k < 120; k++) 
	{	
		char filenum[5];
		int k1=k+1;
		snprintf(filenum,sizeof(filenum),"%d",k1);
		
		strcpy(filename3, neuronfile1);
		strcat(filename3, filenum);
		strcat(filename3, neuronfile2);

		mmio_info(&mB, &nB, &nnzB, &isSymmetricB, filename3);
		B[k].value=(VALUE_TYPE*)malloc((nnzB)*sizeof(VALUE_TYPE));
		B[k].columnindex=(int*)malloc((nnzB)*sizeof(int));
		B[k].rowpointer=(int*)malloc((mB+1)*sizeof(int));
		mmio_data(B[k].rowpointer, B[k].columnindex, B[k].value, filename3);
		
		B0[k] = (VALUE_TYPE *)malloc(mB * nB * sizeof(VALUE_TYPE));
        memset(B0[k], 0, sizeof(VALUE_TYPE) * mB * nB);
		for (int i = 0; i < mB; i++)
        {
            for (int j = B[k].rowpointer[i]; j < B[k].rowpointer[i+1]; j++)
            {
                B0[k][i * nB + B[k].columnindex[j]] = B[k].value[j];
            }
        }

		free(B[k].rowpointer);
		free(B[k].columnindex);
		free(B[k].value);
	}
	gettimeofday(&t4,NULL);
	double time_load = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
	printf("Weight matrix load time: %f ms \n",time_load);

    mC = mA;
	nC = nB;
	VALUE_TYPE *C0 =(VALUE_TYPE *)malloc((mC*nC)*sizeof(VALUE_TYPE));
    
    gettimeofday(&t3, NULL);
	for (int k = 0; k < 120; k++) 
	{
		int k1=k+1;
        memset(C0, 0, sizeof(VALUE_TYPE)*mC*nC);

		gettimeofday(&t1, NULL);
		
        for (int mi = 0; mi < mC; mi++)
        {
            for (int ni = 0; ni < nC; ni++)
            {
                for (int ki = 0; ki < nA; ki++)
                    C0[mi * nB + ni] += A0[mi * nA + ki] * B0[k][ki * nB + ni];
            }
        }
		gettimeofday(&t2,NULL);
        double time_gemm = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		
		gettimeofday(&t1, NULL);
		
		for (int i = 0; i < mC*nC; i++) 
		{
			C0[i] += bias;
			
			if (C0[i] <= 0)
			{   
				C0[i] = 0;
			}
			else if (C0[i] >= 32) 
			{
				C0[i] = 32;
			}
		}
		gettimeofday(&t2,NULL);
        double time_biasrelu = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		printf("k = %d, GEMM time: %4.5f ms, Bias+ReLU time: %4.5f ms\n", 
		       k+1, time_gemm, time_biasrelu);

        free(B0[k]);
		
		memcpy(A0, C0, (mC*nC)*sizeof(VALUE_TYPE));
	}
	
	gettimeofday(&t4,NULL);
	double time_inference = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
	printf("Inference time: %f ms \n",time_inference);
	
	free(C0);
	
    // check results
	printf("test\n");
	FILE* fs;
	fs=fopen("sparse-images-1024-1.tsv","w+");
	for (int i = 0; i <mA; i++)
	{	
		int sum =0;
		for (int j = (i*nA); j < ((i+1)*nA); j++)
		{  
			sum+=A0[j];
		
		}
		if(sum!=0)
		{
			fprintf(fs,"%d\n", i+1);
		}
	}
	fclose(fs);
	FILE* fp2=NULL;

	fp2 = fopen("sparse-images-1024-1.tsv", "rb");
	if (fp2 == NULL)
	{
		printf("Error:Open file fail!\n");
	}

	fseek(fp2, 0, SEEK_END);
	size2 = ftell(fp2);
	rewind(fp2);

	tc2 = (int*)malloc(sizeof(int) * size2/4);

	int readnum2 = fread(tc2, 4, size2/4, fp2);

	fclose(fp2);
	
	FILE* fp1;

	fp1 = fopen("neuron1024-l120-categories.tsv", "rb");
	if (fp1 == NULL)
	{
		printf("Error:Open file fail!\n");
	}

	fseek(fp1, 0, SEEK_END);
	size1 = ftell(fp1);
	rewind(fp1);

	tc1 = (int*)malloc(sizeof(int) * size1/4);

	int readnum1 = fread(tc1, 4, size1/4, fp1);

	fclose(fp1);
	int judge=0;
	for(int i=0;i<size1/4;i++)
	{
		if(tc1[i]-tc2[i] != 0)
		{
			judge++;
		}
	}
	printf("judge:%d\n",judge);
	if (judge == 0) {
		printf("CHALLENGE PASSED\n");
	}
	else
	{
		printf("CHALLENGE FAILED\n");
	}

	free(A0);

	return 0;
}

		
