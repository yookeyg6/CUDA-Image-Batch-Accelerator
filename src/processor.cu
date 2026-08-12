#include "cuda_processor.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
namespace {
__global__ void Gray(const unsigned char* in,unsigned char* g,int w,int h){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=w||y>=h)return; int p=y*w+x,r=3*p;
 g[p]=(unsigned char)(0.299f*in[r]+0.587f*in[r+1]+0.114f*in[r+2]);
}
__global__ void Sobel(const unsigned char* g,unsigned char* out,int w,int h){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=w||y>=h)return; int o=3*(y*w+x);
 if(x==0||y==0||x==w-1||y==h-1){out[o]=out[o+1]=out[o+2]=0;return;}
 int gx=-g[(y-1)*w+x-1]-2*g[y*w+x-1]-g[(y+1)*w+x-1]+g[(y-1)*w+x+1]+2*g[y*w+x+1]+g[(y+1)*w+x+1];
 int gy=-g[(y-1)*w+x-1]-2*g[(y-1)*w+x]-g[(y-1)*w+x+1]+g[(y+1)*w+x-1]+2*g[(y+1)*w+x]+g[(y+1)*w+x+1];
 unsigned char v=(unsigned char)min(255,abs(gx)+abs(gy)); out[o]=out[o+1]=out[o+2]=v;
}
void Check(cudaError_t e,const char* m){if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",m,cudaGetErrorString(e));std::exit(1);}}
}
void ProcessImage(const unsigned char* hi,unsigned char* ho,int w,int h,int threads,float* ms){
 unsigned char *di,*dg,*do_; size_t rgb=(size_t)w*h*3,gray=(size_t)w*h;
 Check(cudaMalloc(&di,rgb),"malloc input");Check(cudaMalloc(&dg,gray),"malloc gray");Check(cudaMalloc(&do_,rgb),"malloc output");
 Check(cudaMemcpy(di,hi,rgb,cudaMemcpyHostToDevice),"copy input");
 int s=threads>1024?32:16; dim3 b(s,s),grid((w+s-1)/s,(h+s-1)/s); cudaEvent_t a,z;
 Check(cudaEventCreate(&a),"event");Check(cudaEventCreate(&z),"event");Check(cudaEventRecord(a),"record");
 Gray<<<grid,b>>>(di,dg,w,h);Check(cudaGetLastError(),"gray");Sobel<<<grid,b>>>(dg,do_,w,h);Check(cudaGetLastError(),"sobel");
 Check(cudaEventRecord(z),"record");Check(cudaEventSynchronize(z),"sync");Check(cudaEventElapsedTime(ms,a,z),"time");
 Check(cudaMemcpy(ho,do_,rgb,cudaMemcpyDeviceToHost),"copy output");
 cudaEventDestroy(a);cudaEventDestroy(z);cudaFree(di);cudaFree(dg);cudaFree(do_);
}
