#include <cassert>
#include <ATen/cpu/Utils.h>
#if !defined(__s390x__ ) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <string>
#include <iostream>
#include <ATen/Parallel.h>

struct measure {
  measure(std::string);
  void update(double);
  void show();
  ~measure();
  int parallel_num;
  double time[300];
  double num[300];
  double final_time = 0;
  double final_num = 0;
  std::string pre;
};

measure::measure(std::string s){
  parallel_num = at::get_num_threads();
  pre = s;
  for (int i = 0; i < 300; i ++) {
    time[i] = 0;
    num[i] = 0;
  }
}

void measure::update(double t) {
  int tid = at::get_thread_num();
  time[tid] += t;
  num[tid] +=1;
}

void measure::show() {
  for (int i = 0; i < parallel_num; i ++) {
    final_time += time[i];
    final_num += num[i];
  }
  std::cout << pre << " time ms: " <<  (final_time / 1000 / 1000) << std::endl;
  std::cout << pre << " num: " <<  final_num << std::endl;
}

measure::~measure() {
  show();
}


measure& get_bf16_measure() {
  static measure bf16_measure = measure("bf16 upconversion"); 
  return bf16_measure;
}

measure& get_fp16_measure() {
  static measure fp16_measure = measure("fp16 upconversion"); 
  return fp16_measure;
}

void bf16_measure_update(double t) {
  get_bf16_measure().update(t);
}
void fp16_measure_update(double t) {
  get_fp16_measure().update(t);
}

namespace at::cpu {
bool is_avx2_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return cpuinfo_initialize() && cpuinfo_has_x86_avx2();
#else
  return false;
#endif
}

bool is_avx512_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512vl() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq();
#else
  return false;
#endif
}

bool is_avx512_vnni_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512vnni();
#else
  return false;
#endif
}

bool is_avx512_bf16_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512bf16();
#else
  return false;
#endif
}

bool is_amx_tile_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return cpuinfo_initialize() && cpuinfo_has_x86_amx_tile();
#else
  return false;
#endif
}

bool is_amx_fp16_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return is_amx_tile_supported() && cpuinfo_has_x86_amx_fp16();
#else
  return false;
#endif
}

bool init_amx() {
  if (!is_amx_tile_supported()) {
    return false;
  }

#if defined(__linux__) && !defined(__ANDROID__) && defined(__x86_64__)
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

  unsigned long bitmask = 0;
  // Request permission to use AMX instructions
  long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (rc) {
      return false;
  }
  // Check if the system supports AMX instructions
  rc = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
  if (rc) {
      return false;
  }
  if (bitmask & XFEATURE_MASK_XTILE) {
      return true;
  }
  return false;
#else
  return true;
#endif
}

bool is_arm_sve_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return cpuinfo_initialize() && cpuinfo_has_arm_sve();
#else
  return false;
#endif
}

static uint32_t get_cache_size(int level) {
#if !defined(__s390x__) && !defined(__powerpc__)
  if (!cpuinfo_initialize()) {
    return 0;
  }
  const struct cpuinfo_processor* processors = cpuinfo_get_processors();
  if (!processors) {
    return 0;
  }
  const struct cpuinfo_cache* cache = nullptr;
  switch (level) {
    case 1:
      cache = processors[0].cache.l1d;
      break;
    case 2:
      cache = processors[0].cache.l2;
      break;
    default:
      assert(false && "Unsupported cache level");
  }

  if (!cache) {
    return 0;
  }
  return cache->size;
#else
  return 0;
#endif
}

uint32_t L1d_cache_size() {
  return get_cache_size(1);
}

uint32_t L2_cache_size() {
  return get_cache_size(2);
}

} // namespace at::cpu
