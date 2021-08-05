#include "head.h"

__global__ void Vector2_Multiply_By_Elements (const double* a, const double* b, int n,int batch, double* out){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<n)
	{
		out[tid]=a[tid]-b[tid];
		tid+=temp;
	}
	__syncthreads();
}
__global__ void Vector1_Multiply_By_Elements (const double* a, double* b, int n){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<n)
	{
		if(a[tid]<=0)
		{
			b[tid] = 0;
		}
		tid+=temp;
	}
	__syncthreads();
}

__global__ void exp_fun(double *d_A,int a)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<a)
	{
		d_A[tid] =exp(d_A[tid]);
		tid+=temp;
	}
	__syncthreads();
}
__global__ void softmax(double *d_A,double *d_B,int a,int b)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<a*b)
	{
		d_A[tid] = d_A[tid]/d_B[tid/a];
		tid+=temp;
	}
	__syncthreads();
}

__global__ void log_softmax(double *d_A,double *d_B,int a,int b)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<a*b)
	{
		d_A[tid] = log(d_A[tid]/d_B[tid/a]);
		tid+=temp;
	}
	__syncthreads();
}

__global__ void dropout(double *d_A,double *d_B,int n)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(i<n)
	{
		d_A[i] = d_A[i] * d_B[i];
		i+=temp;
	}
	__syncthreads();
}

__global__ void relu(double *d_A,int a)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<a)
	{
		if(d_A[tid]<= 0)
			d_A[tid] = 0;
		tid+=temp;
	}
	__syncthreads();
}

__global__ void nan_chuli(double *d_A,int a)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<a)
	{
		if(d_A[tid]<=1e-10)
			d_A[tid] = 0;
		tid+=temp;
	}
	__syncthreads();
}

__global__ void noise_nul_std(double *d_noise,int num)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<num)
	{
		d_noise[tid] = d_noise[tid] * 0.01;
		tid+=temp;
	}
	__syncthreads();
}
__global__ void NLLloos(double *d_A,double* d_B,int batch)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<batch*batch)
	{	
		long row = tid%batch;
    	long col = tid/batch;
		if(row == col)
		{
			d_B[row] = -d_A[tid];
		}
		tid+=temp;
	}
	__syncthreads();
}
__global__ void gen_wight(double *d_A,double *d_B,int num)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<num)
	{	
		d_B[tid] = 1/(d_A[tid]+0.000000001);
		tid+=temp;
	}
	__syncthreads();
}
__global__ void set_0(double *d_A,int num,int i)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<num)
	{	
		if(tid == i)
		{
			d_A[tid] = 0;
		}
		tid+=temp;
	}
	__syncthreads();
}

__global__ void set_02(double *d_A,double* d_B,int num)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<num)
	{	
		if(d_A[tid] == d_B[tid])
		{
			d_A[tid] = 0;
		}
		tid+=temp;
	}
	__syncthreads();
}
__global__ void averg(double *d_A,int num,double sum)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<num)
	{	
		d_A[tid] = d_A[tid]/sum;
		tid+=temp;
	}
	__syncthreads();
}
__global__ void activate(double *d_A,int b,int a)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<a)
	{
		d_A[tid] = 1/(1+exp(-d_A[tid]+b));
		tid+=temp;
	}
	__syncthreads();
}



void printTensor(double *d_des,long m,long n,long l){
	double *des = new double[m*n*l]();
	cudaMemcpy(des,d_des,sizeof(double)*m*n*l,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for(long k = 0;k<l;k++){
		for(long i = 0;i<n;i++){
			for(long j = 0;j<m;j++){
				cout<<des[k*m*n+i*m+j]<<" ";
			}
			cout<<endl;
		}
		cout<<"~~~~~~~~~~~~~~~~"<<endl;
	}
	delete[] des;des=nullptr;
}
void warmup(){
	double *tmp = new double[9];
	for(unsigned i = 0; i < 9; ++i) {
		tmp[i] = i+1;
	}
	double *d_tmp;
	cudaMalloc((void**)&d_tmp,sizeof(double)*9);
	cudaMemcpy(d_tmp,tmp,sizeof(double)*9,cudaMemcpyHostToDevice);
	relu<<<1,512>>>(d_tmp,3);
	cudaDeviceSynchronize();
	cudaFree(d_tmp);

}
void forward_cuda(double *input,double *W1,double *outh,double *W2,double *outo,double *b1,double *b2,int in,int hid,int out,int batch,cublasHandle_t handle,double *d_t)
{
	//hid 行 in列 W1 ,out行，hid列 W2

	//cout<<"weigh matrix is :"<<endl;printTensor(W1,4,4,1);

	double alpha=1.0, beta=0.0;

	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,hid,batch,in,&alpha,W1,hid,input,in,&beta,outh,hid);
	//printTensor(W1,10,1,1);
	//激活函数
	dim3 blockh((batch*hid+1024-1)/1024,1,1);
	relu<<<blockh,1024>>>(outh,hid*batch);
	//activate<<<blockh,1024>>>(outh,k1,hid*batch);

	cudaDeviceSynchronize();

	dim3 blocko((batch*out+1024-1)/1024,1,1);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,out,batch,hid,&alpha,W2,out,outh,hid,&beta,outo,out);	
	//softmax<<<blocko,1024>>>(outo,k2,out*batch);
	//relu<<<blockh,1024>>>(outo,k2,out*batch);

	//activate<<<blocko,1024>>>(outo,k2,out*batch);
	
	//softmax,先对每一个数取 exp
	/*exp_fun<<<blocko,1024>>>(outo,k2,out*batch);
	double *temp_sum;
	cudaMalloc((void**)&temp_sum,sizeof(double)*1*batch);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,batch,out,&alpha,d_t,1,outo,out,&beta,temp_sum,1);*/
	//softmax<<<blocko,1024>>>(outo,temp_sum,out,batch);
	//计算每一列之和

	//nan_chuli<<<blocko,1024>>>(outo,out*batch);
	//printTensor(outo,10,1,1);
	cudaDeviceSynchronize();
	//cudaFree(temp_sum);

}

void back_cuda(double *Y,double *Y_hat,double *outh,double *W2,double *input,double *W1,int in,int hid,int out,int batch,double rate,cublasHandle_t handle)
{
	double *d_thta3,*d_thta2;
	cudaMalloc((void**)&d_thta3,sizeof(double)*out*batch);
	cudaMalloc((void**)&d_thta2,sizeof(double)*hid*batch);
	//printTensor(Y,10,1,1);
	dim3 block2((batch*out+1024-1)/1024,1,1);
	Vector2_Multiply_By_Elements<<<block2,1024>>>(Y_hat, Y, out*batch, batch,d_thta3);
	double alpha=1.0, beta=0.0;
	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,hid,batch,out,&alpha,W2,out,d_thta3,out,&beta,d_thta2,hid);
	dim3 block1((batch*hid+1024-1)/1024,1,1);
	Vector1_Multiply_By_Elements<<<block1,1024>>>(outh, d_thta2, hid*batch);//RELU反向
	cudaDeviceSynchronize();
	alpha=rate; beta=1.0;

	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,hid,in,batch,&alpha,d_thta2,hid,input,in,&beta,W1,hid);

	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,out,hid,batch,&alpha,d_thta3,out,outh,hid,&beta,W2,out);


	cudaFree(d_thta2);
	cudaFree(d_thta3);
}	
double NLL_loss(double *Y,double *outo,int out,int batch,double *d_t,cublasHandle_t handle)
{	

	double loss = 0;
	double sum_n = 0;  
	double alpha=1.0, beta=0.0;
	double *d_YYhat,*d_Dia;

	dim3 blocko((batch*out+1024-1)/1024,1,1);
	exp_fun<<<blocko,1024>>>(outo,out*batch);
	double *temp_sum;
	cudaMalloc((void**)&temp_sum,sizeof(double)*1*batch);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,batch,out,&alpha,d_t,1,outo,out,&beta,temp_sum,1);
	log_softmax<<<blocko,1024>>>(outo,temp_sum,out,batch);
	//计算每一列之和
	//nan_chuli<<<blocko,1024>>>(outo,out*batch);
	cudaDeviceSynchronize();
	
	
	dim3 blockh((batch*batch+1024-1)/1024,1,1);

	cudaMalloc((void**)&d_YYhat,sizeof(double)*batch*batch);
	cudaMalloc((void**)&d_Dia,sizeof(double)*batch);

	//cout<<"Y_hat is :"<<endl;printTensor(Y_hat,10,1,1);

	// Y与Y_hat相乘，取对角线元素 (batch*10) * (10*batch)
	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,batch,batch,out,&alpha,outo,out,Y,out,&beta,d_YYhat,batch);
	
	//cout<<"YY_hat is :"<<endl;printTensor(d_YYhat,10,1,1);   这里已经是 NAN

	NLLloos<<<blockh,1024>>>(d_YYhat,d_Dia,batch); //对角线取反

	//cout<<"NULLloss :";printTensor(d_Dia,10,1,1);
	double *Dia = new double[batch];
	cudaMemcpy(Dia,d_Dia,sizeof(double)*batch,cudaMemcpyDeviceToHost);
	for(int i = 0;i<batch; i++)
	{
		sum_n += Dia[i];
	}

	loss = sum_n/batch;
	cudaFree(d_YYhat);
	cudaFree(d_Dia);
	delete Dia;
	cudaFree(temp_sum);
	return loss;
}


void forword_ES(double *input,double *W1,double *outh,double *W2,double *outo,
                     double *b1,double *b2,int in,int hid,int out,int batch,
                     cublasHandle_t handle,double *d_t)
{
	double alpha=1.0, beta=0.0,beta1 = 1.0;

	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,hid,batch,in,&alpha,W1,in,input,in,&beta,outh,hid);
	for(int i = 0;i<batch;i++)
	{
		cublasDgeam(handle,CUBLAS_OP_N,CUBLAS_OP_N,hid,1,&alpha,outh+i*hid,hid,&beta1,b1,hid,outh+i*hid,hid);
	}
	

	//激活函数
	dim3 blockh((batch*hid+1024-1)/1024,1,1);
	relu<<<blockh,1024>>>(outh,hid*batch);
	//activate<<<blockh,1024>>>(outh,k1,hid*batch);
	cudaDeviceSynchronize();
	
	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,out,batch,hid,&alpha,W2,hid,outh,hid,&beta,outo,out);
	for(int i = 0;i<batch;i++)
	{
		cublasDgeam(handle,CUBLAS_OP_N,CUBLAS_OP_N,out,1,&alpha,outo+i*out,out,&beta1,b2,out,outo+i*out,out);
	}
	
	//printTensor(outo,10,1,1);  // 结果输出
	//relu<<<blockh,1024>>>(outo,k2,out*batch);
	//activate<<<blocko,1024>>>(outo,k2,out*batch);
	//softmax<<<blocko,1024>>>(outo,k2,out*batch);
	//log_softmax,先对每一个数取 exp

}


void forward_cuda_ES(double *input,double *W1,double *outh,double *W2,double *outo,double *target,
                     double *b1,double *b2,int in,int hid,int out,int batch,
                     cublasHandle_t handle,double *d_t)
{

	bool if_mirror = true;
	int num_directions = 40; //若if_mirror为true，这里是20，否则40
	//printTensor(W1,3,3,1);
	//printTensor(W2,3,3,1);
	
	epsilon *ep = new epsilon[num_directions*2];
	// 1、 产生random noise;
	//W1 size: in*hid; 激活函数：hid； W2 hid*out; softmax: out（只给W加noise）
	//一共产生2个 noise
	double *noise1,*noise2,*noise_b1,*noise_b2;
	cudaMalloc((void**)&noise1,sizeof(double)*in*hid);
	cudaMalloc((void**)&noise_b1,sizeof(double)*hid);
	cudaMalloc((void**)&noise2,sizeof(double)*out*hid);
	cudaMalloc((void**)&noise_b2,sizeof(double)*out);

	cudaMemset(noise1,0,sizeof(double)*in*hid);
	cudaMemset(noise2,0, sizeof(double)*out*hid);
	cudaMemset(noise_b1,0,sizeof(double)*hid);
	cudaMemset(noise_b2,0 , sizeof(double)*out);

	double *loss = new double[num_directions*2]; 
	//double *epsilons[num_directions*2];

	dim3 block1((in*hid+1024-1)/1024,1,1);
    dim3 block2((out*hid+1024-1)/1024,1,1);
    dim3 block11((hid+1024-1)/1024,1,1);
    dim3 block22((out+1024-1)/1024,1,1);
    double alpha=1.0, beta=1.0, beta1 = -1.0;
    double alpha2;


    // cout<<"start -----"<<endl;
    for(int i = 0;i<num_directions;i++)
    {	
    	struct timeb timeSeed;
		ftime(&timeSeed);

    	curandGenerator_t gen;
	    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	    curandSetPseudoRandomGeneratorSeed(gen, timeSeed.time * 1000 + timeSeed.millitm);
	    curandGenerateNormalDouble(gen, noise1, in*hid, 0, 1);

	    curandSetPseudoRandomGeneratorSeed(gen, timeSeed.time * 1000 + timeSeed.millitm);
	    curandGenerateNormalDouble(gen, noise2, hid*out, 0, 1);

	    curandSetPseudoRandomGeneratorSeed(gen, timeSeed.time * 1000 + timeSeed.millitm);
	    curandGenerateNormalDouble(gen, noise_b1, hid, 0, 1);

	    curandSetPseudoRandomGeneratorSeed(gen, timeSeed.time * 1000 + timeSeed.millitm);
	    curandGenerateNormalDouble(gen, noise_b2, out, 0, 1);


	    //noise 的每个元素 *0.01
	    
	    noise_nul_std<<<block1,1024>>>(noise1, hid*in);
	    cudaDeviceSynchronize();
	    noise_nul_std<<<block2,1024>>>(noise2, hid*out);
	    cudaDeviceSynchronize();
	    noise_nul_std<<<block11,1024>>>(noise_b1, hid);
	    cudaDeviceSynchronize();
	    noise_nul_std<<<block22,1024>>>(noise_b2, out);
	    cudaDeviceSynchronize();

	    //cudaMalloc((void**)&epsilons[2*i],sizeof(double)*in*hid);
	    cublasDcopy(handle,in*hid,noise1,1,ep[2*i].noise1,1);
	  	//cudaMalloc((void**)&epsilons[2*i+1],sizeof(double)*out*hid);
	    cublasDcopy(handle,out*hid,noise2,1,ep[2*i].noise2,1);
	    cublasDcopy(handle,hid,noise_b1,1,ep[2*i].noise_b1,1);
	    cublasDcopy(handle,out,noise_b2,1,ep[2*i].noise_b2,1);

	    
	    //2. W1 W2 与noise1 noise2 相加，
	    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,in,hid,
	                          &alpha,W1,in,
	                          &beta,noise1,in,
	                          W1,in);
	    
	    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,hid,out,
	                          &alpha,W2,hid,
	                          &beta,noise2,hid,
	                          W2,hid);
	    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,hid,1,
	                          &alpha,b1,hid,
	                          &beta,noise_b1,hid,
	                          b1,hid);
	    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,out,1,
	                          &alpha,b2,out,
	                          &beta,noise_b2,out,
	                          b2,out);
	  
	    //3. 进行前向。记下 损失
	    forword_ES(input,W1,outh,W2,outo,b1,b2,in,hid,out,batch,handle,d_t);
	    // 求得损失 loss
	    loss[2*i] = NLL_loss(target,outo,out,batch,d_t,handle);
	    
	    //4. remove noise
	    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,in,hid,
	                          &alpha,W1,in,
	                          &beta1,noise1,in,
	                          W1,in);
	    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,hid,out,
	                          &alpha,W2,hid,
	                          &beta1,noise2,hid,
	                          W2,hid);
	    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,hid,1,
	                          &alpha,b1,hid,
	                          &beta1,noise_b1,hid,
	                          b1,hid);
	    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,out,1,
	                          &alpha,b2,out,
	                          &beta1,noise_b2,out,
	                          b2,out);
	    //若需要镜像，if_mirror 为true
	   
	    if(if_mirror)
	    {
	    	cublasDaxpy(handle,in*hid,&beta1,noise1,1,ep[2*i+1].noise1,1);
	    	cublasDaxpy(handle,out*hid,&beta1,noise2,1,ep[2*i+1].noise2,1);
	    	cublasDaxpy(handle,hid,&beta1,noise_b1,1,ep[2*i+1].noise_b1,1);
	    	cublasDaxpy(handle,out,&beta1,noise_b2,1,ep[2*i+1].noise_b2,1);

		    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,in,hid,
		                          &alpha,W1,in,
		                          &beta1,noise1,in,
		                          W1,in);
		    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,hid,out,
		                          &alpha,W2,hid,
		                          &beta1,noise2,hid,
		                          W2,hid);
		    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,hid,1,
	                          &alpha,b1,hid,
	                          &beta1,noise_b1,hid,
	                          b1,hid);
		    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,out,1,
		                          &alpha,b2,out,
		                          &beta1,noise_b2,out,
		                          b2,out);

		    forword_ES(input,W1,outh,W2,outo,b1,b2,in,hid,out,batch,handle,d_t);

		   
		    // 求得损失 loss
		    loss[2*i+1] = NLL_loss(target,outo,out,batch,d_t,handle);
		    
		    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,in,hid,
	                          &alpha,W1,in,
	                          &beta,noise1,in,
	                          W1,in);
		    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,hid,out,
		                          &alpha,W2,hid,
		                          &beta,noise2,hid,
		                          W2,hid);
		    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,hid,1,
	                          &alpha,b1,hid,
	                          &beta,noise_b1,hid,
	                          b1,hid);
		    cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,out,1,
		                          &alpha,b2,out,
		                          &beta,noise_b2,out,
		                          b2,out);	  
	    }

    }
    // double *loss_gpu;
    // cudaMalloc((void**)&loss_gpu,sizeof(double)*num_directions*2);
    // cudaMemcpy(loss_gpu,loss,sizeof(double)*num_directions*2,cudaMemcpyHostToDevice);

    //printTensor(loss_gpu,10,1,1);
    ///////////////////////////////////////////////////////////////////////////////
    //求梯度，并更新W1，W2
    //////////////////////////////////////////////////////////////////////////////
	double elite_rate = 0.2;
	double lr = 0.8;
	int elite_num = max(int(elite_rate * num_directions*2), 1);
	//double *weight = new double[num_directions*2];
	vector<double> weight(num_directions*2);
	//cout<<"mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"<<endl;
	for(int i = 0;i < num_directions*2; i++)
	{
		weight[i] = 1/(loss[i] + 0.000000001);
		//cout<<weight[i]<<" ";
	}
	//cout<<endl;
	// cout<<"mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"<<endl;
	// printTensor(W1,10,1,1);
	// printTensor(W2,10,1,1);
	// printTensor(b1,10,1,1);
	// printTensor(b2,10,1,1);
	// char ch=getchar();

    // cudaMalloc((void**)&d_weight,sizeof(double)*num_directions*2);
    // cudaMemset(d_weight,0,sizeof(double)*num_directions*2);

    // dim3 block3((num_directions*2+1024-1)/1024,1,1);
    // gen_wight<<<block3,1024>>>(loss_gpu,d_weight,num_directions*2);
    // cudaDeviceSynchronize();

    //在d_weight中找到最大的前 elite_num 个，其余为0
    //int *result = new int[elite_num];
    vector<int> result(elite_num);
    vector<double> weight_copy = weight;
    // double *d_weight_copy;
    // cudaMalloc((void**)&d_weight_copy,sizeof(double)*num_directions*2);
    // cudaMemset(d_weight_copy,0,num_directions*2);

    // cublasDcopy(handle,num_directions*2,d_weight,1,d_weight_copy,1);
    for(int i = 0;i<elite_num;i++)
    {
    	// cublasIdamax(handle,num_directions*2,d_weight_copy, 1,&result[i]);
    	// result[i] -=1;
    	
    	// set_0<<<block3,1024>>>(d_weight_copy,num_directions*2,result[i]);
    	// cudaDeviceSynchronize();
    	vector<double>::iterator biggest = std::max_element(weight_copy.begin(), weight_copy.end());
    	result[i]=std::distance(weight_copy.begin(), biggest);
    	weight_copy[result[i]] = 0;

    }
    // set_02<<<block3,1024>>>(d_weight,d_weight_copy,num_directions*2); //d_weight 和 d_weight_copy对比，元素相等的位置set0
    // cudaDeviceSynchronize();
    //delete weight_copy;
    for(int i = 0; i<num_directions * 2; i++)
    {
    	if(find(result.begin(),result.end(),i) == result.end())
    		weight[i] = 0;
    }


    //printTensor(d_weight,8,5,1);

    //  求sum和平均
    // double *weigh = new double[num_directions*2];
     double sum = 0;
    // cudaMemcpy(weigh,d_weight,sizeof(double)*num_directions*2,cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // for(int i = 0;i<num_directions*2;i++){
    // 	sum += weigh[i];
    // }
    for(int i = 0;i<num_directions*2;i++){
    	sum += weight[i];
    }
    for(int i = 0;i<num_directions*2;i++){
    	 weight[i] = weight[i]/sum;
    }
    //averg<<<block3,1024>>>(d_weight,num_directions*2,sum);

    //cudaMemcpy(weigh,d_weight,sizeof(double)*num_directions*2,cudaMemcpyDeviceToHost);
    //printTensor(d_weight,8,5,1);
    // 求梯度
    double *grad_w1,*grad_w2,*grad_b1,*grad_b2;
    cudaMalloc((void**)&grad_w1,sizeof(double)*in*hid);
    cudaMalloc((void**)&grad_w2,sizeof(double)*out*hid);
    cudaMalloc((void**)&grad_b1,sizeof(double)*hid);
    cudaMalloc((void**)&grad_b2,sizeof(double)*out);

    cudaMemset(grad_w1, 0, sizeof(double)*in * hid);
    cudaMemset(grad_w2, 0, sizeof(double)*out * hid);
    cudaMemset(grad_b1, 0, sizeof(double)* hid);
    cudaMemset(grad_b2, 0, sizeof(double)* out);


    for(int i=0;i<elite_num;i++)
    {
    	alpha2 = weight[result[i]];
    	//cout<< "alpha2 :" <<alpha2;
    	cublasDaxpy(handle,in*hid,&alpha2,ep[result[i]].noise1,1,grad_w1,1);
    	cublasDaxpy(handle,out*hid,&alpha2,ep[result[i]].noise2,1,grad_w2,1);
    	cublasDaxpy(handle,hid,&alpha2,ep[result[i]].noise_b1,1,grad_b1,1);
    	cublasDaxpy(handle,out,&alpha2,ep[result[i]].noise_b2,1,grad_b2,1);
    }

 //    cout<<"mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"<<endl;
	// printTensor(grad_w1,10,1,1);
	// printTensor(grad_w2,10,1,1);
	// printTensor(grad_b1,10,1,1);
	// printTensor(grad_b2,10,1,1);
	// char ch=getchar();

    // 权重更新, W = W - lr*grad
    //cout<<"bbbbbbbbbbbbbbbbbbbW1 is :"<<endl;printTensor(W1,10,1,1);
     cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,in,hid,
	                          &alpha,W1,in,
	                          &lr,grad_w1,in,
	                          W1,in);
     cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,hid,out,
	                          &alpha,W2,hid,
	                          &lr,grad_w2,hid,
	                          W2,hid);
     cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,hid,1,
	                          &alpha,b1,hid,
	                          &lr,grad_b1,hid,
	                          b1,hid);
     cublasDgeam(handle,CUBLAS_OP_N, CUBLAS_OP_N,out,1,
	                          &alpha,b2,out,
	                          &lr,grad_b2,out,
	                          b2,out);
     //cout<<"aaaaaaaaaaaaaaaaaaaW1 is :"<<endl;printTensor(W1,10,1,1);
    // cout<<"W2 is :"<<endl; printTensor(W2,10,1,1);

     cudaFree(noise1);
     cudaFree(noise2);
     cudaFree(noise_b1);
     cudaFree(noise_b2);
     //cudaFree(loss_gpu);
     //cudaFree(d_weight);
     //cudaFree(d_weight_copy);
     cudaFree(grad_w1);
     cudaFree(grad_w2);
     cudaFree(grad_b1);
     cudaFree(grad_b2);

     delete[] ep;
}
