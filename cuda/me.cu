// Exhaustive integer motion search on the GPU: one thread block per 16x16
// macroblock, every candidate vector in [-range, range]^2 evaluated in
// parallel. It minimizes exactly what the CPU reference (me::full_search)
// minimizes: SAD + (lam * (|dx| + |dy|)) >> 4, ties broken by scan order, so
// both return identical vectors.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define MAXR 32
#define WIN (16 + 2 * MAXR)

__global__ void search(const uint8_t *src, const uint8_t *ref, int stride, int origin, int mbw, int range, unsigned lam,
                       unsigned long long *best16, unsigned long long *best8) {
    __shared__ uint8_t s_src[16 * 16];
    __shared__ uint8_t s_ref[WIN * WIN];
    __shared__ unsigned long long s_best[5];
    const int mb = blockIdx.x;
    const int x0 = (mb % mbw) * 16, y0 = (mb / mbw) * 16;
    const int win = 16 + 2 * range;
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        s_src[i] = src[origin + (y0 + i / 16) * stride + x0 + i % 16];
    for (int i = threadIdx.x; i < win * win; i += blockDim.x) {
        int wx = i % win, wy = i / win;
        s_ref[wy * WIN + wx] = ref[origin + (y0 - range + wy) * stride + x0 - range + wx];
    }
    if (threadIdx.x < 5) s_best[threadIdx.x] = ~0ull;
    __syncthreads();
    const int side = 2 * range + 1;
    unsigned long long b16 = ~0ull, b8[4] = {~0ull, ~0ull, ~0ull, ~0ull};
    for (int idx = threadIdx.x; idx < side * side; idx += blockDim.x) {
        const int dy = idx / side - range, dx = idx % side - range;
        unsigned s8[4] = {0, 0, 0, 0};
        for (int r = 0; r < 16; r++) {
            const uint8_t *a = s_src + r * 16;
            const uint8_t *b = s_ref + (r + range + dy) * WIN + range + dx;
            unsigned left = 0, right = 0;
            for (int c = 0; c < 8; c++) left += abs((int)a[c] - (int)b[c]);
            for (int c = 8; c < 16; c++) right += abs((int)a[c] - (int)b[c]);
            s8[(r >= 8) * 2] += left;
            s8[(r >= 8) * 2 + 1] += right;
        }
        const unsigned mvc = (lam * (unsigned)(abs(dx) + abs(dy))) >> 4;
        const unsigned long long k16 = ((unsigned long long)(s8[0] + s8[1] + s8[2] + s8[3] + mvc) << 24) | (unsigned)idx;
        if (k16 < b16) b16 = k16;
        for (int q = 0; q < 4; q++) {
            const unsigned long long k = ((unsigned long long)(s8[q] + mvc) << 24) | (unsigned)idx;
            if (k < b8[q]) b8[q] = k;
        }
    }
    atomicMin(&s_best[0], b16);
    for (int q = 0; q < 4; q++) atomicMin(&s_best[1 + q], b8[q]);
    __syncthreads();
    if (threadIdx.x == 0) {
        best16[mb] = s_best[0];
        for (int q = 0; q < 4; q++) best8[mb * 4 + q] = s_best[1 + q];
    }
}

static uint8_t *d_src = nullptr, *d_ref = nullptr;
static unsigned long long *d_b16 = nullptr, *d_b8 = nullptr;
static size_t cap_plane = 0, cap_mbs = 0;

static int fail(cudaError_t e, char *err, int errlen) {
    snprintf(err, errlen, "%s", cudaGetErrorString(e));
    return 1;
}

#define CK(x)                                    \
    do {                                         \
        cudaError_t e_ = (x);                    \
        if (e_ != cudaSuccess) return fail(e_, err, errlen); \
    } while (0)

// src/ref: whole padded planes (`bytes` long); `origin` is the offset of pixel
// (0, 0). out: per macroblock 11 ints: mv16 x, y, cost16, then 4 x (x, y).
extern "C" int reel_cuda_full_search(const uint8_t *src, const uint8_t *ref, size_t bytes, int stride, int origin, int w, int h,
                                     int range, unsigned lam, int *out, char *err, int errlen) {
    if (range > MAXR || range < 0) {
        snprintf(err, errlen, "range %d is above the kernel limit %d", range, MAXR);
        return 1;
    }
    const int mbw = w / 16, mbs = mbw * (h / 16);
    if (bytes > cap_plane) {
        cudaFree(d_src);
        cudaFree(d_ref);
        CK(cudaMalloc(&d_src, bytes));
        CK(cudaMalloc(&d_ref, bytes));
        cap_plane = bytes;
    }
    if ((size_t)mbs > cap_mbs) {
        cudaFree(d_b16);
        cudaFree(d_b8);
        CK(cudaMalloc(&d_b16, mbs * sizeof(unsigned long long)));
        CK(cudaMalloc(&d_b8, mbs * 4 * sizeof(unsigned long long)));
        cap_mbs = mbs;
    }
    CK(cudaMemcpy(d_src, src, bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ref, ref, bytes, cudaMemcpyHostToDevice));
    search<<<mbs, 256>>>(d_src, d_ref, stride, origin, mbw, range, lam, d_b16, d_b8);
    CK(cudaGetLastError());
    unsigned long long *b16 = new unsigned long long[mbs], *b8 = new unsigned long long[mbs * 4];
    cudaError_t e1 = cudaMemcpy(b16, d_b16, mbs * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaError_t e2 = cudaMemcpy(b8, d_b8, mbs * 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        delete[] b16;
        delete[] b8;
        return fail(e1 != cudaSuccess ? e1 : e2, err, errlen);
    }
    const int side = 2 * range + 1;
    for (int m = 0; m < mbs; m++) {
        int idx = (int)(b16[m] & 0xFFFFFF);
        out[m * 11 + 0] = idx % side - range;
        out[m * 11 + 1] = idx / side - range;
        out[m * 11 + 2] = (int)(b16[m] >> 24);
        for (int q = 0; q < 4; q++) {
            idx = (int)(b8[m * 4 + q] & 0xFFFFFF);
            out[m * 11 + 3 + q * 2] = idx % side - range;
            out[m * 11 + 4 + q * 2] = idx / side - range;
        }
    }
    delete[] b16;
    delete[] b8;
    return 0;
}
