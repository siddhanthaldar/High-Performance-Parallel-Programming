
__global__ void op1(float *mat, int mat_size)
{
	int i=blockIdx.y*blockDim.y+threadIdx.y;
	int j=blockIdx.x*blockDim.x+threadIdx.x;
    float temp;
 	if((i<mat_size) && (j<mat_size) && (j<mat_size-1)){
 		if(j%2==0){
 			temp = mat[i*mat_size+j];
 			mat[i*mat_size+j] = mat[i*mat_size+j+1];
 			mat[i*mat_size+j+1] = temp;
 		}
 	}  
 	__syncthreads(); 
}

__global__ void op2(float *mat, int mat_size)
{
	int i=blockIdx.y*blockDim.y+threadIdx.y;
	int j=blockIdx.x*blockDim.x+threadIdx.x;
    float temp;
 	if((i<mat_size) && (j<i)){
		temp = mat[i*mat_size+j];
		mat[i*mat_size+j] = mat[j*mat_size+i];
		mat[j*mat_size+i] = temp;
 	}  
 	__syncthreads(); 
}
