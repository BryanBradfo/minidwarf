// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

extern "C" void minidwarf_solve(const float* const*, float* const*, const long*, int);

static long prod(const std::vector<long>& d){ long p=1; for(long x:d) p*=x; return p; }

int main(int argc, char** argv){
  const char* in=argv[1]; const char* out=argv[2]; const char* tj=argv[3];
  int n_in=atoi(argv[4]), n_out=atoi(argv[5]), reps=atoi(argv[6]);
  std::vector<long> dims; for(int i=7;i<argc;i++) dims.push_back(atol(argv[i]));
  long n = prod(dims);

  FILE* f=fopen(in,"rb");
  std::vector<float*> din(n_in), dout(n_out);
  for(int i=0;i<n_in;i++){ std::vector<float> h(n); fread(h.data(),4,n,f);
    cudaMalloc(&din[i], n*4); cudaMemcpy(din[i],h.data(),n*4,cudaMemcpyHostToDevice);}
  fclose(f);
  for(int i=0;i<n_out;i++) cudaMalloc(&dout[i], n*4);

  minidwarf_solve(din.data(), dout.data(), dims.data(), (int)dims.size()); // warmup
  cudaDeviceSynchronize();

  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  std::vector<float> times(reps);
  for(int r=0;r<reps;r++){ cudaEventRecord(s);
    minidwarf_solve(din.data(), dout.data(), dims.data(), (int)dims.size());
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&times[r], s, e); }
  for(size_t i=0;i+1<times.size();i++) for(size_t j=i+1;j<times.size();j++)
    if(times[j]<times[i]) std::swap(times[i],times[j]);
  float med = times[reps/2];

  FILE* fo=fopen(out,"wb");
  for(int i=0;i<n_out;i++){ std::vector<float> h(n);
    cudaMemcpy(h.data(),dout[i],n*4,cudaMemcpyDeviceToHost); fwrite(h.data(),4,n,fo);}
  fclose(fo);
  FILE* ft=fopen(tj,"w"); fprintf(ft,"{\"median_ms\": %f}\n", med); fclose(ft);
  return 0;
}
