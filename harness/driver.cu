// SPDX-License-Identifier: Apache-2.0
// MiniDwarf v3 driver. Timing protocol (see docs/superpowers/specs/2026-10-07-hardened-harness-design.md):
// D input data sets rotate (random order) through the SAME device buffers; before every call (untimed) the set
// is re-uploaded, outputs are poisoned with NaN and L2 is flushed; the timed region ends with a device-wide sync.
// EVERY call (first, warm-up, timed) is verified against the expected outputs outside the timed region.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

extern "C" void minidwarf_solve(const void* const*, void* const*, const long*, int);

static bool wr(const void* p, size_t n, FILE* f){ return n==0 || fwrite(p,1,n,f)==n; }

int main(int argc, char** argv){
  const char* usage="usage: in out exp|- timing.json n_in n_out n_sets reps warmup rtol atol n_dims dims... in_counts... out_counts...\n";
  if(argc<13){ fputs(usage,stderr); return 2; }
  const char* in=argv[1]; const char* out=argv[2]; const char* exp=argv[3]; const char* tj=argv[4];
  int n_in=atoi(argv[5]), n_out=atoi(argv[6]), n_sets=atoi(argv[7]), reps=atoi(argv[8]), warmup=atoi(argv[9]);
  double rtol=atof(argv[10]), atl=atof(argv[11]); int n_dims=atoi(argv[12]);
  if(n_dims<0||n_in<0||n_out<0||argc<13+n_dims+n_in+n_out){ fputs(usage,stderr); return 2; }
  if(n_sets<1 || reps<n_sets){ fprintf(stderr,"need reps >= n_sets >= 1\n"); return 2; }
  int p=13;
  std::vector<long> dims; for(int i=0;i<n_dims;i++) dims.push_back(atol(argv[p++]));
  std::vector<long> in_cnt; for(int i=0;i<n_in;i++) in_cnt.push_back(atol(argv[p++]));
  std::vector<long> out_cnt; for(int i=0;i<n_out;i++) out_cnt.push_back(atol(argv[p++]));

  std::vector<std::vector<char>> hin((size_t)n_sets*n_in);
  FILE* f=fopen(in,"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",in); return 1; }
  for(int s=0;s<n_sets;s++) for(int i=0;i<n_in;i++){
    std::vector<char>& h=hin[(size_t)s*n_in+i]; h.resize(in_cnt[i]*4);
    if(fread(h.data(),1,h.size(),f)!=h.size()){ fprintf(stderr,"short input file\n"); return 1; } }
  fclose(f);
  bool check = strcmp(exp,"-")!=0;
  std::vector<std::vector<char>> hexp;
  if(check){
    hexp.resize((size_t)n_sets*n_out);
    FILE* fe=fopen(exp,"rb"); if(!fe){ fprintf(stderr,"cannot open %s\n",exp); return 1; }
    for(int s=0;s<n_sets;s++) for(int j=0;j<n_out;j++){
      std::vector<char>& h=hexp[(size_t)s*n_out+j]; h.resize(out_cnt[j]*4);
      if(fread(h.data(),1,h.size(),fe)!=h.size()){ fprintf(stderr,"short expected file\n"); return 1; } }
    fclose(fe);
  }

  // master copies: the driver's own memcpys/poisoning only ever use these; the candidate gets fresh copies per call
  std::vector<void*> din(n_in), dout(n_out);
  for(int i=0;i<n_in;i++) CUDA_CHECK(cudaMalloc(&din[i], in_cnt[i]*4));
  for(int i=0;i<n_out;i++) CUDA_CHECK(cudaMalloc(&dout[i], out_cnt[i]*4));
  std::vector<void*> arg_in(n_in), arg_out(n_out); std::vector<long> arg_dims(n_dims>0?n_dims:1);
  int l2=0; CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
  size_t flush_bytes=2*(size_t)(l2>0 ? l2 : (64<<20)); void* scratch; CUDA_CHECK(cudaMalloc(&scratch, flush_bytes));

  auto prep=[&](int s, int r){
    CUDA_CHECK(cudaSetDevice(0));
    for(int i=0;i<n_in;i++) CUDA_CHECK(cudaMemcpy(din[i], hin[(size_t)s*n_in+i].data(), in_cnt[i]*4, cudaMemcpyHostToDevice));
    for(int i=0;i<n_out;i++) CUDA_CHECK(cudaMemset(dout[i], 0xFF, out_cnt[i]*4));  // NaN poison
    CUDA_CHECK(cudaMemset(scratch, r & 0xFF, flush_bytes));                          // L2 flush
    CUDA_CHECK(cudaDeviceSynchronize()); };
  auto call=[&](){
    for(int i=0;i<n_in;i++) arg_in[i]=din[i];
    for(int i=0;i<n_out;i++) arg_out[i]=dout[i];
    for(int i=0;i<n_dims;i++) arg_dims[i]=dims[i];
    minidwarf_solve((const void* const*)arg_in.data(), (void* const*)arg_out.data(), arg_dims.data(), n_dims); };
  std::vector<std::vector<char>> hout((size_t)n_sets*n_out);
  long n_bad=0;
  auto verify=[&](int s){  // after sync; copies outputs of this call (kept as the set's latest) and compares
    int d=-1; CUDA_CHECK(cudaGetDevice(&d)); if(d!=0){ fprintf(stderr,"candidate changed the current device to %d\n",d); exit(1); }
    CUDA_CHECK(cudaGetLastError()); bool bad=false;
    for(int j=0;j<n_out;j++){
      std::vector<char>& h=hout[(size_t)s*n_out+j]; h.resize(out_cnt[j]*4);
      CUDA_CHECK(cudaMemcpy(h.data(), dout[j], h.size(), cudaMemcpyDeviceToHost));
      if(check){ const float* a=(const float*)h.data(); const float* e=(const float*)hexp[(size_t)s*n_out+j].data();
        for(long k=0;k<out_cnt[j] && !bad;k++){ double x=a[k], y=e[k];
          if(!(fabs(x-y) <= atl + rtol*fabs(y))) bad=true; } } }
    if(bad) n_bad++; };
  auto sync=[&](){ CUDA_CHECK(cudaDeviceSynchronize()); };

  std::mt19937 rng{std::random_device{}()};
  std::uniform_int_distribution<int> pick(0, n_sets-1);
  { int s=pick(rng); prep(s,0); call(); sync(); verify(s); }
  for(int w=0; w<warmup; w++){ int s=pick(rng); prep(s, w+1); call(); sync(); verify(s); }

  std::vector<int> seq(reps); for(int i=0;i<reps;i++) seq[i]=i % n_sets;
  std::shuffle(seq.begin(), seq.end(), rng);
  cudaEvent_t s0,e0; CUDA_CHECK(cudaEventCreate(&s0)); CUDA_CHECK(cudaEventCreate(&e0));
  std::vector<float> times(reps);
  for(int r=0;r<reps;r++){
    int s=seq[r]; prep(s, r + warmup + 1);
    CUDA_CHECK(cudaEventRecord(s0)); call(); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0)); CUDA_CHECK(cudaEventSynchronize(e0));
    CUDA_CHECK(cudaEventElapsedTime(&times[r], s0, e0));
    verify(s); }

  std::vector<float> sorted_t(times); std::sort(sorted_t.begin(), sorted_t.end());
  FILE* fo=fopen(out,"wb"); if(!fo){ fprintf(stderr,"cannot open %s\n",out); return 1; }
  for(size_t k=0;k<hout.size();k++) if(!wr(hout[k].data(),hout[k].size(),fo)){ fprintf(stderr,"short write to %s\n",out); return 1; }
  if(fclose(fo)!=0){ fprintf(stderr,"short write to %s\n",out); return 1; }
  FILE* ft=fopen(tj,"w"); if(!ft){ fprintf(stderr,"cannot open %s\n",tj); return 1; }
  bool ok = fprintf(ft,"{\"median_ms\": %.9g, \"p25_ms\": %.9g, \"p75_ms\": %.9g, \"n_bad_calls\": ",
          (double)sorted_t[reps/2], (double)sorted_t[reps/4], (double)sorted_t[(3*reps)/4])>0;
  ok = ok && (check ? fprintf(ft,"%ld",n_bad) : fprintf(ft,"null"))>0;
  ok = ok && fprintf(ft,", \"times_ms\": [")>0;
  for(int r=0;r<reps && ok;r++) ok = fprintf(ft,"%s%.9g", r ? ", " : "", (double)times[r])>0;
  ok = ok && fprintf(ft,"], \"timed_sets\": [")>0;
  for(int r=0;r<reps && ok;r++) ok = fprintf(ft,"%s%d", r ? ", " : "", seq[r])>0;
  ok = ok && fprintf(ft,"]}\n")>0;
  if(fclose(ft)!=0 || !ok){ fprintf(stderr,"short write to %s\n",tj); return 1; }
  return 0;
}
