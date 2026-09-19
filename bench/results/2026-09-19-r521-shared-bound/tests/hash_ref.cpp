// Verbatim copies of the served autotune key functions, compiled on the CPU by test_cpu.py to
// cross-check the Python port in autotune_geometry.py:
//   roundup_pow2, gemm_autotune_hash, mgemm_autotune_hash : src/exllamav3/exllamav3_ext/quant/exl3_gemm.cu:40-108
//   salt_hash (COOP_AUTOTUNE_VERSION = 4)                  : src/exllamav3/exllamav3_ext/quant/coop_autotune.cu:27,69-74
// Usage: hash_ref gemm|mgemm size_m size_k size_n K c_fp32 device cc max_num_sms cb [bszm_in bszm_out]
// Prints the salted key (the value stored in coop_autotune_v1.bin) in hex.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ---- begin verbatim exl3_gemm.cu:40-108 ----
uint64_t roundup_pow2(uint64_t x)
{
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	x |= x >> 32;
    return x + 1;
}

uint64_t gemm_autotune_hash
(
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int device,
    int cc,
    int max_num_sms,
    int cb
)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&] (uint64_t v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) MIN(roundup_pow2(size_m), 16));
    mix((uint64_t) size_k);
    mix((uint64_t) size_n);
    mix((uint64_t) K);
    mix(c_fp32 ? 1ull : 0ull);
    mix((uint64_t) device);
    mix((uint64_t) cc);
    mix((uint64_t) max_num_sms);
    mix((uint64_t) cb);
    return h;
}

uint64_t mgemm_autotune_hash
(
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int device,
    int cc,
    int max_num_sms,
    int cb,
    int bszm_in,
    int bszm_out
)
{
    uint64_t h = gemm_autotune_hash(size_m, size_k, size_n, K, c_fp32, device, cc, max_num_sms, cb);
    auto mix = [&] (uint64_t v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) MIN(bszm_in, 24));
    mix((uint64_t) MIN(bszm_out, 24));
    return h;
}
// ---- end verbatim ----

constexpr uint64_t COOP_AUTOTUNE_VERSION = 4;
uint64_t salt_hash(uint64_t hash)
{
    hash ^= COOP_AUTOTUNE_VERSION;
    hash *= 1099511628211ull;
    return hash;
}

int main(int argc, char** argv)
{
    if (argc < 11) { fprintf(stderr, "usage: hash_ref gemm|mgemm m k n K c_fp32 device cc sms cb [bszm_in bszm_out]\n"); return 2; }
    bool mg = !strcmp(argv[1], "mgemm");
    int a[12] = {};
    for (int i = 2; i < argc && i < 14; ++i) a[i - 2] = atoi(argv[i]);
    uint64_t h = mg
        ? mgemm_autotune_hash(a[0], a[1], a[2], a[3], a[4] != 0, a[5], a[6], a[7], a[8], a[9], a[10])
        : gemm_autotune_hash(a[0], a[1], a[2], a[3], a[4] != 0, a[5], a[6], a[7], a[8]);
    printf("%016llx\n", (unsigned long long) salt_hash(h));
    return 0;
}
