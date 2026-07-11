// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

extern "C" void minidwarf_solve(const void* const*, void* const*, const long*, int);

int main(int argc, char** argv){
  const char* in=argv[1]; const char* out=argv[2]; const char* tj=argv[3];
  int n_in=atoi(argv[4]), n_out=atoi(argv[5]), reps=atoi(argv[6]), n_dims=atoi(argv[7]);
  int p=8;
  std::vector<long> dims; for(int i=0;i<n_dims;i++) dims.push_back(atol(argv[p++]));
  std::vector<long> in_cnt; for(int i=0;i<n_in;i++) in_cnt.push_back(atol(argv[p++]));
  std::vector<long> out_cnt; for(int i=0;i<n_out;i++) out_cnt.push_back(atol(argv[p++]));

  FILE* f=fopen(in,"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",in); exit(1);}
  std::vector<void*> din(n_in), dout(n_out);
  for(int i=0;i<n_in;i++){ long b=in_cnt[i]*4; std::vector<char> h(b); fread(h.data(),1,b,f);
    CUDA_CHECK(cudaMalloc(&din[i], b)); CUDA_CHECK(cudaMemcpy(din[i],h.data(),b,cudaMemcpyHostToDevice)); }
  fclose(f);
  for(int i=0;i<n_out;i++) CUDA_CHECK(cudaMalloc(&dout[i], out_cnt[i]*4));

  minidwarf_solve((const void* const*)din.data(), (void* const*)dout.data(), dims.data(), n_dims);
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());

  cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
  std::vector<float> times(reps);
  for(int r=0;r<reps;r++){ CUDA_CHECK(cudaEventRecord(s));
    minidwarf_solve((const void* const*)din.data(), (void* const*)dout.data(), dims.data(), n_dims);
    CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(&times[r], s, e)); }
  for(size_t i=0;i+1<times.size();i++) for(size_t j=i+1;j<times.size();j++)
    if(times[j]<times[i]) std::swap(times[i],times[j]);
  float med = times[reps/2];

  FILE* fo=fopen(out,"wb"); if(!fo){ fprintf(stderr,"cannot open %s\n",out); exit(1);}
  for(int i=0;i<n_out;i++){ long b=out_cnt[i]*4; std::vector<char> h(b);
    CUDA_CHECK(cudaMemcpy(h.data(),dout[i],b,cudaMemcpyDeviceToHost)); fwrite(h.data(),1,b,fo); }
  fclose(fo);
  FILE* ft=fopen(tj,"w"); fprintf(ft,"{\"median_ms\": %f}\n", med); fclose(ft);
  return 0;
}
