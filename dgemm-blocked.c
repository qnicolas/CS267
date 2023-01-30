#include <immintrin.h> //AVX intrinsics for SIMD operations
#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 40
#endif

#ifndef MICROKERNEL_SIZE
#define MICROKERNEL_SIZE 4
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))


static void micro_kernel(int lda,double* A, double* B, double* C) {
    // Declare
    __m256d Ar0,Ar1,Ar2,Ar3;
    __m256d Br0,Br1,Br2,Br3;
    __m256d Cr0,Cr1,Cr2,Cr3;
    
    Ar0 = _mm256_loadu_pd(A + 0 * lda);
    Ar1 = _mm256_loadu_pd(A + 1 * lda);
    Ar2 = _mm256_loadu_pd(A + 2 * lda);
    Ar3 = _mm256_loadu_pd(A + 3 * lda);
    Cr0 = _mm256_loadu_pd(C + 0 * lda);
    Cr1 = _mm256_loadu_pd(C + 1 * lda);
    Cr2 = _mm256_loadu_pd(C + 2 * lda);
    Cr3 = _mm256_loadu_pd(C + 3 * lda);

    //// First pass
    // Column 0 of C
    Br0 = _mm256_set1_pd(B[0 + 0 * lda]);
    Cr0 = _mm256_fmadd_pd(Ar0,Br0,Cr0);
    // Column 1 of C
    Br1 = _mm256_set1_pd(B[1 + 1 * lda]);
    Cr1 = _mm256_fmadd_pd(Ar1,Br1,Cr1);
    // Column 2 of C
    Br2 = _mm256_set1_pd(B[2 + 2 * lda]);
    Cr2 = _mm256_fmadd_pd(Ar2,Br2,Cr2);
    // Column 3 of C
    Br3 = _mm256_set1_pd(B[3 + 3 * lda]);
    Cr3 = _mm256_fmadd_pd(Ar3,Br3,Cr3);

    //// Second pass
    // Column 0 of C
    Br3 = _mm256_set1_pd(B[3 + 0 * lda]);
    Cr0 = _mm256_fmadd_pd(Ar3,Br3,Cr0);
    // Column 1 of C
    Br0 = _mm256_set1_pd(B[0 + 1 * lda]);
    Cr1 = _mm256_fmadd_pd(Ar0,Br0,Cr1);
    // Column 2 of C
    Br1 = _mm256_set1_pd(B[1 + 2 * lda]);
    Cr2 = _mm256_fmadd_pd(Ar1,Br1,Cr2);
    // Column 3 of C
    Br2 = _mm256_set1_pd(B[2 + 3 * lda]);
    Cr3 = _mm256_fmadd_pd(Ar2,Br2,Cr3);

    //// Third pass
    // Column 0 of C
    Br2 = _mm256_set1_pd(B[2 + 0 * lda]);
    Cr0 = _mm256_fmadd_pd(Ar2,Br2,Cr0);
    // Column 1 of C
    Br3 = _mm256_set1_pd(B[3 + 1 * lda]);
    Cr1 = _mm256_fmadd_pd(Ar3,Br3,Cr1);
    // Column 2 of C
    Br0 = _mm256_set1_pd(B[0 + 2 * lda]);
    Cr2 = _mm256_fmadd_pd(Ar0,Br0,Cr2);
    // Column 3 of C
    Br1 = _mm256_set1_pd(B[1 + 3 * lda]);
    Cr3 = _mm256_fmadd_pd(Ar1,Br1,Cr3);

    //// Fourth pass
    // Column 0 of C
    Br1 = _mm256_set1_pd(B[1 + 0 * lda]);
    Cr0 = _mm256_fmadd_pd(Ar1,Br1,Cr0);
    // Column 1 of C
    Br2 = _mm256_set1_pd(B[2 + 1 * lda]);
    Cr1 = _mm256_fmadd_pd(Ar2,Br2,Cr1);
    // Column 2 of C
    Br3 = _mm256_set1_pd(B[3 + 2 * lda]);
    Cr2 = _mm256_fmadd_pd(Ar3,Br3,Cr2);
    // Column 3 of C
    Br0 = _mm256_set1_pd(B[0 + 3 * lda]);
    Cr3 = _mm256_fmadd_pd(Ar0,Br0,Cr3);

    //// Store
    _mm256_storeu_pd(C + 0 * lda, Cr0);
    _mm256_storeu_pd(C + 1 * lda, Cr1);
    _mm256_storeu_pd(C + 2 * lda, Cr2);
    _mm256_storeu_pd(C + 3 * lda, Cr3);
}

static void do_dummyblock(int lda, int M1, int N1, int K1, double* A, double* B, double* C) {
    for (int j = 0; j < N1; ++j) {
        for (int k = 0; k < K1; ++k) {
            for (int i = 0; i < M1; ++i) {
                double cij = C[i + j * lda];
                cij += A[i + k * lda] * B[k + j * lda];
                C[i + j * lda] = cij;
            }
        }
    }
}



/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    for (int j = 0; j < N; j+=MICROKERNEL_SIZE) {
        for (int k = 0; k < K; k+=MICROKERNEL_SIZE) {
            for (int i = 0; i < M; i+=MICROKERNEL_SIZE) {
                /*if ((N-j < MICROKERNEL_SIZE) || (K-k < MICROKERNEL_SIZE) || (M-i < MICROKERNEL_SIZE)){
                    int M1 = min(MICROKERNEL_SIZE, M - i);
                    int N1 = min(MICROKERNEL_SIZE, N - j);
                    int K1 = min(MICROKERNEL_SIZE, K - k);
                    // Perform individual block dgemm
                    do_dummyblock(lda, M1, N1, K1, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                }
                else{*/
                    micro_kernel(lda, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                //}
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
    // For each block-row of A
        // For each block-column of B
            // Accumulate block dgemms into block of C
                // Correct block dimensions if block "goes off edge of" the matrix

void square_dgemm(int lda, double* A, double* B, double* C) {
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}



