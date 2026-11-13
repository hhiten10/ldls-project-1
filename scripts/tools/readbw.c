// Read-only memory bandwidth (complements STREAM, whose kernels all write).
// LLM decode is dominated by READING weights once per token, so this is the most relevant ceiling.
// AVX2 sum over a 1.6 GB buffer (>> 8 MB L3), OpenMP static split, best of NTIMES passes.
// build: gcc -O3 -march=native -fopenmp readbw.c -o readbw
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_FLOATS (400UL * 1000 * 1000)   /* 1.6 GB */
#define NTIMES 10

int main(void) {
    float* a = aligned_alloc(64, N_FLOATS * sizeof(float));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N_FLOATS; i++) a[i] = (float)(i & 7);   // first-touch by the reading threads
    double best = 1e30; volatile float sink = 0;
    for (int k = 0; k < NTIMES; k++) {
        double t0 = omp_get_wtime(); float tot = 0;
        #pragma omp parallel reduction(+:tot)
        {
            __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
            #pragma omp for schedule(static)
            for (size_t i = 0; i < N_FLOATS; i += 32) {
                s0 = _mm256_add_ps(s0, _mm256_load_ps(a + i));
                s1 = _mm256_add_ps(s1, _mm256_load_ps(a + i + 8));
                s2 = _mm256_add_ps(s2, _mm256_load_ps(a + i + 16));
                s3 = _mm256_add_ps(s3, _mm256_load_ps(a + i + 24));
            }
            float b[8]; _mm256_storeu_ps(b, _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
            for (int j = 0; j < 8; j++) tot += b[j];
        }
        double dt = omp_get_wtime() - t0; sink += tot;
        if (k > 0 && dt < best) best = dt;   // skip first pass
    }
    printf("ReadOnly: %.1f MB/s\n", N_FLOATS * sizeof(float) / best / 1e6);
    free(a); return 0;
}
