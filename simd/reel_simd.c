/* AVX2 kernels for Reel: the integer DCT (4x4, 8x8, 16x16, forward and
 * inverse), SAD and the 8x8 Hadamard SATD. They compute exactly what the Rust
 * scalar code computes (same arithmetic, same rounding, same clipping); the
 * Rust tests compare them on random input. */
#include <immintrin.h>
#include <stdint.h>

/* ---- transforms ------------------------------------------------------ */

/* out[k][c] = (sum_j m[k][j] * in[j][c] + round) >> shift, for an n x n
 * block with n a multiple of 8 (columns in vectors of 8) */
static void mul_cols8(int n, const int32_t *m, const int32_t *in, int32_t *out, int shift, int clip16) {
    const __m256i rnd = _mm256_set1_epi32(1 << (shift - 1));
    const __m256i lo = _mm256_set1_epi32(-32768), hi = _mm256_set1_epi32(32767);
    for (int k = 0; k < n; k++) {
        for (int c = 0; c < n; c += 8) {
            __m256i acc = _mm256_setzero_si256();
            for (int j = 0; j < n; j++) {
                __m256i row = _mm256_loadu_si256((const __m256i *)(in + j * n + c));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(_mm256_set1_epi32(m[k * n + j]), row));
            }
            acc = _mm256_srai_epi32(_mm256_add_epi32(acc, rnd), shift);
            if (clip16) acc = _mm256_min_epi32(_mm256_max_epi32(acc, lo), hi);
            _mm256_storeu_si256((__m256i *)(out + k * n + c), acc);
        }
    }
}

static void transpose(int n, const int32_t *in, int32_t *out) {
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) out[c * n + r] = in[r * n + c];
}

static void mul_cols4(const int32_t *m, const int32_t *in, int32_t *out, int shift, int clip16) {
    const __m128i rnd = _mm_set1_epi32(1 << (shift - 1));
    const __m128i lo = _mm_set1_epi32(-32768), hi = _mm_set1_epi32(32767);
    for (int k = 0; k < 4; k++) {
        __m128i acc = _mm_setzero_si128();
        for (int j = 0; j < 4; j++) {
            __m128i row = _mm_loadu_si128((const __m128i *)(in + j * 4));
            acc = _mm_add_epi32(acc, _mm_mullo_epi32(_mm_set1_epi32(m[k * 4 + j]), row));
        }
        acc = _mm_srai_epi32(_mm_add_epi32(acc, rnd), shift);
        if (clip16) acc = _mm_min_epi32(_mm_max_epi32(acc, lo), hi);
        _mm_storeu_si128((__m128i *)(out + k * 4), acc);
    }
}

static void mul_cols(int n, const int32_t *m, const int32_t *in, int32_t *out, int shift, int clip16) {
    if (n == 4) mul_cols4(m, in, out, shift, clip16);
    else mul_cols8(n, m, in, out, shift, clip16);
}

static int log2n(int n) { return n == 4 ? 2 : n == 8 ? 3 : 4; }

/* forward: Y = ((M X) >> s1) M^T >> s2, with s1 = log2 n - 1, s2 = log2 n + 6.
 * `m` is the n x n basis (rows = frequencies). */
void reel_forward_avx2(int n, const int32_t *m, const int32_t *x, int32_t *y) {
    int32_t t[256], tt[256], u[256];
    mul_cols(n, m, x, t, log2n(n) - 1, 0);     /* t = M X (columns) */
    transpose(n, t, tt);                        /* tt = t^T */
    mul_cols(n, m, tt, u, log2n(n) + 6, 0);    /* u = M t^T = (t M^T)^T */
    transpose(n, u, y);
}

/* inverse: X = clip16((M^T Y) >> 7) M >> 12 */
void reel_inverse_avx2(int n, const int32_t *m, const int32_t *y, int32_t *x) {
    int32_t mt[256], t[256], tt[256], u[256];
    transpose(n, m, mt);
    mul_cols(n, mt, y, t, 7, 1);               /* t = M^T Y */
    transpose(n, t, tt);
    mul_cols(n, mt, tt, u, 12, 0);             /* u = M^T t^T = (t M)^T */
    transpose(n, u, x);
}

/* ---- SAD ----------------------------------------------------------- */

uint32_t reel_sad_avx2(const uint8_t *a, intptr_t sa, const uint8_t *b, intptr_t sb, int w, int h) {
    __m128i acc = _mm_setzero_si128();
    if (w == 16) {
        for (int r = 0; r < h; r++) {
            __m128i va = _mm_loadu_si128((const __m128i *)(a + r * sa));
            __m128i vb = _mm_loadu_si128((const __m128i *)(b + r * sb));
            acc = _mm_add_epi64(acc, _mm_sad_epu8(va, vb));
        }
    } else {
        for (int r = 0; r < h; r++) {
            __m128i va = _mm_loadl_epi64((const __m128i *)(a + r * sa));
            __m128i vb = _mm_loadl_epi64((const __m128i *)(b + r * sb));
            acc = _mm_add_epi64(acc, _mm_sad_epu8(va, vb));
        }
    }
    return (uint32_t)(_mm_cvtsi128_si64(acc) + _mm_extract_epi64(acc, 1));
}

/* ---- SATD (8x8 Hadamard) ------------------------------------------- */

static void butterflies(__m256i *v) {
    for (int h = 1; h < 8; h *= 2)
        for (int i = 0; i < 8; i += 2 * h)
            for (int j = i; j < i + h; j++) {
                __m256i a = v[j], b = v[j + h];
                v[j] = _mm256_add_epi32(a, b);
                v[j + h] = _mm256_sub_epi32(a, b);
            }
}

uint32_t reel_satd8_avx2(const int32_t *d) {
    __m256i v[8];
    int32_t t[64], tt[64];
    for (int r = 0; r < 8; r++) v[r] = _mm256_loadu_si256((const __m256i *)(d + r * 8));
    butterflies(v); /* vertical: along the rows' index */
    for (int r = 0; r < 8; r++) _mm256_storeu_si256((__m256i *)(t + r * 8), v[r]);
    transpose(8, t, tt);
    for (int r = 0; r < 8; r++) v[r] = _mm256_loadu_si256((const __m256i *)(tt + r * 8));
    butterflies(v); /* horizontal */
    __m256i acc = _mm256_setzero_si256();
    for (int r = 0; r < 8; r++) acc = _mm256_add_epi32(acc, _mm256_abs_epi32(v[r]));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return ((uint32_t)_mm_cvtsi128_si32(s) + 2) >> 2;
}
