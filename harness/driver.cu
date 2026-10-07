// SPDX-License-Identifier: Apache-2.0
// MiniDwarf v3 driver. Timing protocol (see docs/superpowers/specs/2026-10-07-hardened-harness-design.md):
// D input data sets rotate through the SAME device buffers; before every call (untimed) the set is
// re-uploaded, outputs are poisoned with NaN and L2 is flushed; the timed region ends with a
// device-wide sync so work on any stream is counted.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

extern "C" void minidwarf_solve(const void* const*, void* const*, const long*, int);

int main(int argc, char** argv){
  if(argc<11){ fprintf(stderr,"usage: in out timing.json n_in n_out n_sets reps warmup n_dims dims... in_counts... out_counts...\n"); return 2; }
  const char* in=argv[1]; const char* out=argv[2]; const char* tj=argv[3];
  int n_in=atoi(argv[4]), n_out=atoi(argv[5]), n_sets=atoi(argv[6]), reps=atoi(argv[7]), warmup=atoi(argv[8]), n_dims=atoi(argv[9]);
  if(n_sets<1 || reps<n_sets){ fprintf(stderr,"need reps >= n_sets >= 1\n"); return 2; }
  int p=10;
  std::vector<long> dims; for(int i=0;i<n_dims;i++) dims.push_back(atol(argv[p++]));
  std::vector<long> in_cnt; for(int i=0;i<n_in;i++) in_cnt.push_back(atol(argv[p++]));
  std::vector<long> out_cnt; for(int i=0;i<n_out;i++) out_cnt.push_back(atol(argv[p++]));

  std::vector<std::vector<char>> hin((size_t)n_sets*n_in);
  FILE* f=fopen(in,"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",in); return 1; }
  for(int s=0;s<n_sets;s++) for(int i=0;i<n_in;i++){
    std::vector<char>& h=hin[(size_t)s*n_in+i]; h.resize(in_cnt[i]*4);
    if(fread(h.data(),1,h.size(),f)!=h.size()){ fprintf(stderr,"short input file\n"); return 1; } }
  fclose(f);

  std::vector<void*> din(n_in), dout(n_out);
  for(int i=0;i<n_in;i++) CUDA_CHECK(cudaMalloc(&din[i], in_cnt[i]*4));
  for(int i=0;i<n_out;i++) CUDA_CHECK(cudaMalloc(&dout[i], out_cnt[i]*4));
  int l2=0; CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
  size_t flush_bytes=2*(size_t)(l2>0 ? l2 : (64<<20)); void* scratch; CUDA_CHECK(cudaMalloc(&scratch, flush_bytes));

  auto prep=[&](int s, int r){
    for(int i=0;i<n_in;i++) CUDA_CHECK(cudaMemcpy(din[i], hin[(size_t)s*n_in+i].data(), in_cnt[i]*4, cudaMemcpyHostToDevice));
    for(int i=0;i<n_out;i++) CUDA_CHECK(cudaMemset(dout[i], 0xFF, out_cnt[i]*4));  // NaN poison
    CUDA_CHECK(cudaMemset(scratch, r & 0xFF, flush_bytes));                          // L2 flush
    CUDA_CHECK(cudaDeviceSynchronize()); };
  auto call=[&](){ minidwarf_solve((const void* const*)din.data(), (void* const*)dout.data(), dims.data(), n_dims); };

  prep(0, 0); call(); CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
  for(int w=0; w<warmup; w++){ prep(w % n_sets, w + 1); call(); CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError()); }

  cudaEvent_t s0,e0; CUDA_CHECK(cudaEventCreate(&s0)); CUDA_CHECK(cudaEventCreate(&e0));
  std::vector<float> times(reps);
  std::vector<std::vector<char>> hout((size_t)n_sets*n_out);
  for(int r=0;r<reps;r++){
    int s=r % n_sets; prep(s, r + warmup + 1);
    CUDA_CHECK(cudaEventRecord(s0)); call(); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0)); CUDA_CHECK(cudaEventSynchronize(e0)); CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&times[r], s0, e0));
    if(r >= reps - n_sets) for(int j=0;j<n_out;j++){
      std::vector<char>& h=hout[(size_t)s*n_out+j]; h.resize(out_cnt[j]*4);
      CUDA_CHECK(cudaMemcpy(h.data(), dout[j], h.size(), cudaMemcpyDeviceToHost)); } }

  std::vector<float> sorted_t(times); std::sort(sorted_t.begin(), sorted_t.end());
  FILE* fo=fopen(out,"wb"); if(!fo){ fprintf(stderr,"cannot open %s\n",out); return 1; }
  for(size_t k=0;k<hout.size();k++) fwrite(hout[k].data(),1,hout[k].size(),fo);
  fclose(fo);
  FILE* ft=fopen(tj,"w"); if(!ft){ fprintf(stderr,"cannot open %s\n",tj); return 1; }
  fprintf(ft,"{\"median_ms\": %f, \"p25_ms\": %f, \"p75_ms\": %f, \"times_ms\": [",
          sorted_t[reps/2], sorted_t[reps/4], sorted_t[(3*reps)/4]);
  for(int r=0;r<reps;r++) fprintf(ft,"%s%f", r ? ", " : "", times[r]);
  fprintf(ft,"]}\n"); fclose(ft);
  return 0;
}
