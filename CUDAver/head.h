#ifndef SETTING_H_
#define SETTING_H_

#include <bits/stdc++.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<curand.h>
#include <cufft.h>
#include <cuda_fp16.h>
#include <random>
#include <sys/timeb.h>   
#include <time.h> 
#define IPNNUM 784
#define HDNNUM 256
//#define H1 
//#define B1
//#define H2
#define OPNNUM 10
#define BATCHSIZE 600

using namespace std;


class epsilon
{

public:
    double *noise1 = NULL;
    double *noise2 = NULL;
    double *noise_b1 = NULL;
    double *noise_b2 = NULL;

    epsilon()
    {
        cudaMalloc((void**)&noise1,sizeof(double)*IPNNUM*HDNNUM);
        cudaMalloc((void**)&noise2,sizeof(double)*HDNNUM*OPNNUM);
        cudaMalloc((void**)&noise_b1,sizeof(double)*HDNNUM);
        cudaMalloc((void**)&noise_b2,sizeof(double)*OPNNUM);

        cudaMemset(noise1,0,sizeof(double)*IPNNUM*HDNNUM);
        cudaMemset(noise2,0,sizeof(double)*OPNNUM*HDNNUM);
        cudaMemset(noise_b1,0,sizeof(double)*HDNNUM);
        cudaMemset(noise_b2,0,sizeof(double)*OPNNUM);

    }
    ~epsilon()
    {
        cudaFree(noise1);
        cudaFree(noise2);
        cudaFree(noise_b1);
        cudaFree(noise_b2);
    }
};





__global__ void nan_chuli(double *d_A,int a);


__global__ void activate(double *d_A,int b,int a);
__global__ void Vector2_Multiply_By_Elements (const double* a, const double* b, int n, int batch,double* out);
__global__ void Vector1_Multiply_By_Elements (const double* a, double* b, int n);
__global__ void dropout(double *d_A,double *d_B,int n);
__global__ void relu(double *d_A,int a);
__global__ void softmax(double *d_A,double *d_B,int a,int b);
__global__ void exp_fun(double *d_A,int a);
void printTensor(double *d_des,long m,long n,long l);
void forward_cuda(double *input,double *W1,double *outh,double *W2,double *outo,double *b1,double *b2,int in,int hid,int out,int batch,cublasHandle_t handle,double *d_t);
void back_cuda(double *Y,double *Y_hat,double *outh,double *W2,double *input,double *W1,int in,int hid,int out,int batch,double rate,cublasHandle_t handle);
double loss_gpu(double *A,double *B,int n,int batch,cublasHandle_t handle);
double NLL_loss(double *Y,double *outo,int out,int batch,double *d_t,cublasHandle_t handle);
void warmup();
__global__ void log_softmax(double *d_A,double *d_B,int a,int b);
__global__ void noise_nul_std(double *d_noise,int num);
__global__ void NLLloos(double *d_A,double sum,int batch);
__global__ void gen_wight(double *d_A,double *d_B,int num);
__global__ void set_0(double *d_A,int num,int i);
__global__ void set_02(double *d_A,double* d_B,int num);
__global__ void averg(double *d_A,int num,double sum);
__global__ void activate(double *d_A,int b,int a);
void forward_cuda_ES(double *input,double *W1,double *outh,double *W2,double *outo,double *target,
                     double *b1,double *b2,int in,int hid,int out,int batch,
                     cublasHandle_t handle,double *d_t);
void forword_ES(double *input,double *W1,double *outh,double *W2,double *outo,
                     double *b1,double *b2,int in,int hid,int out,int batch,
                     cublasHandle_t handle,double *d_t);
#endif /* SETTING_H_ */