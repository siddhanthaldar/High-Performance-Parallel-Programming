// #include "headers.h"

// #define dimension TILE_SIZE+MAX_MASK_WIDTH-1

__global__ void 
conv(float *N, float *M, float *P, int mask_width, int width, int TILE_SIZE)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	// int dimension = TILE_SIZE + mask_width - 1;

	__shared__ float N_ds[6][6];

	int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
	int halo_index_right = (blockIdx.x+1)*blockDim.x + threadIdx.x ;
	int halo_index_up = (blockIdx.y - 1)*blockDim.y + threadIdx.y;
	int halo_index_down = (blockIdx.y+1)*blockDim.y+threadIdx.y ;

	int n = mask_width/2;

	// Outer corners
	if(threadIdx.x >= blockDim.x-n && threadIdx.y >= blockDim.y-n)
		N_ds[threadIdx.y-(blockDim.y-n)][threadIdx.x-(blockDim.x-n)] = (halo_index_left<0 || halo_index_up < 0)?0:N[halo_index_up*width + halo_index_left];
	if(threadIdx.y >= blockDim.y-n && threadIdx.x<n)
		N_ds[threadIdx.y-(blockDim.y-n)][n+blockDim.x+threadIdx.x] = (halo_index_right>=width || halo_index_up<0)?0:N[halo_index_up*width + halo_index_right];
	if(threadIdx.y<n && threadIdx.x>=blockDim.x-n)
		N_ds[n+blockDim.y+threadIdx.y][threadIdx.x+(blockDim.x-n)] = (halo_index_left<0 || halo_index_down>=width)?0:N[halo_index_down*width + halo_index_left];
	if(threadIdx.x<n && threadIdx.y<n)
		N_ds[n+blockDim.y+threadIdx.y][n+blockDim.x+threadIdx.x] = (halo_index_right>=width || halo_index_down>=width)?0:N[halo_index_down*width+halo_index_right];

	//Tile elements
	N_ds[n+threadIdx.y][n+threadIdx.x] = N[(blockIdx.y*blockDim.y+threadIdx.y)*width + blockIdx.x*blockDim.x+threadIdx.x];

	if(threadIdx.y >= blockDim.y-n)
		N_ds[threadIdx.y-(blockDim.y-n)][n+threadIdx.x] = (halo_index_up<0)?0:N[halo_index_up*width +threadIdx.x];
	if(threadIdx.x >= blockDim.x-n)
		N_ds[n+threadIdx.y][threadIdx.x-(blockDim.x-n)] = (halo_index_left<0)?0:N[(n+threadIdx.y)*width + halo_index_left];
	if(threadIdx.x<n)
		N_ds[n+threadIdx.y][n+blockDim.x+threadIdx.x] = (halo_index_right>=width)?0:N[(n+threadIdx.x)*width + halo_index_right];
	if(threadIdx.y<n)
		N_ds[n+blockDim.y+threadIdx.y][n+threadIdx.x] = (halo_index_down>-width)?0:N[halo_index_down*width+threadIdx.x];

	__syncthreads();

	float Pvalue = 0;
	for(int a=0; a<mask_width;a++)
		for(int b=0; b<mask_width; b++)
			Pvalue += N_ds[threadIdx.y+a][threadIdx.x+b]*M[a*mask_width+b];

	P[j*width + i] = Pvalue;

}

