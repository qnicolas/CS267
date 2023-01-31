#include <immintrin.h> //AVX intrinsics for SIMD operations
#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 48
#endif

#ifndef MICROKERNEL_ISIZE
#define MICROKERNEL_ISIZE 8
#define MICROKERNEL_KSIZE 8
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

static void micro_kernel_8by8(int lda,double* A, double* B, double* C) {
    // Declare
    __m256d Ar00,Ar01,Ar02,Ar03,Ar04,Ar05,Ar06,Ar07,Ar10,Ar11,Ar12,Ar13,Ar14,Ar15,Ar16,Ar17;
    __m256d Br0,Br1,Br2,Br3,Br4,Br5,Br6,Br7;
    __m256d Cr00,Cr01,Cr02,Cr03,Cr04,Cr05,Cr06,Cr07,Cr10,Cr11,Cr12,Cr13,Cr14,Cr15,Cr16,Cr17;
    Ar00 = _mm256_loadu_pd(A + 0 + 0 * lda);
    Ar10 = _mm256_loadu_pd(A + 4 + 0 * lda);
    Ar01 = _mm256_loadu_pd(A + 0 + 1 * lda);
    Ar11 = _mm256_loadu_pd(A + 4 + 1 * lda);
    Ar02 = _mm256_loadu_pd(A + 0 + 2 * lda);
    Ar12 = _mm256_loadu_pd(A + 4 + 2 * lda);
    Ar03 = _mm256_loadu_pd(A + 0 + 3 * lda);
    Ar13 = _mm256_loadu_pd(A + 4 + 3 * lda);
    Ar04 = _mm256_loadu_pd(A + 0 + 4 * lda);
    Ar14 = _mm256_loadu_pd(A + 4 + 4 * lda);
    Ar05 = _mm256_loadu_pd(A + 0 + 5 * lda);
    Ar15 = _mm256_loadu_pd(A + 4 + 5 * lda);
    Ar06 = _mm256_loadu_pd(A + 0 + 6 * lda);
    Ar16 = _mm256_loadu_pd(A + 4 + 6 * lda);
    Ar07 = _mm256_loadu_pd(A + 0 + 7 * lda);
    Ar17 = _mm256_loadu_pd(A + 4 + 7 * lda);

    Cr00 = _mm256_loadu_pd(C + 0 + 0 * lda);
    Cr10 = _mm256_loadu_pd(C + 4 + 0 * lda);
    Cr01 = _mm256_loadu_pd(C + 0 + 1 * lda);
    Cr11 = _mm256_loadu_pd(C + 4 + 1 * lda);
    Cr02 = _mm256_loadu_pd(C + 0 + 2 * lda);
    Cr12 = _mm256_loadu_pd(C + 4 + 2 * lda);
    Cr03 = _mm256_loadu_pd(C + 0 + 3 * lda);
    Cr13 = _mm256_loadu_pd(C + 4 + 3 * lda);
    Cr04 = _mm256_loadu_pd(C + 0 + 4 * lda);
    Cr14 = _mm256_loadu_pd(C + 4 + 4 * lda);
    Cr05 = _mm256_loadu_pd(C + 0 + 5 * lda);
    Cr15 = _mm256_loadu_pd(C + 4 + 5 * lda);
    Cr06 = _mm256_loadu_pd(C + 0 + 6 * lda);
    Cr16 = _mm256_loadu_pd(C + 4 + 6 * lda);
    Cr07 = _mm256_loadu_pd(C + 0 + 7 * lda);
    Cr17 = _mm256_loadu_pd(C + 4 + 7 * lda);

    //// First pass
    Br0 = _mm256_set1_pd(B[0 + 0 * lda]);
    Cr00 = _mm256_fmadd_pd(Ar00,Br0,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar10,Br0,Cr10);
    Br1 = _mm256_set1_pd(B[1 + 1 * lda]);
    Cr01 = _mm256_fmadd_pd(Ar01,Br1,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar11,Br1,Cr11);
    Br2 = _mm256_set1_pd(B[2 + 2 * lda]);
    Cr02 = _mm256_fmadd_pd(Ar02,Br2,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar12,Br2,Cr12);
    Br3 = _mm256_set1_pd(B[3 + 3 * lda]);
    Cr03 = _mm256_fmadd_pd(Ar03,Br3,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar13,Br3,Cr13);
    Br4 = _mm256_set1_pd(B[4 + 4 * lda]);
    Cr04 = _mm256_fmadd_pd(Ar04,Br4,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar14,Br4,Cr14);
    Br5 = _mm256_set1_pd(B[5 + 5 * lda]);
    Cr05 = _mm256_fmadd_pd(Ar05,Br5,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar15,Br5,Cr15);
    Br6 = _mm256_set1_pd(B[6 + 6 * lda]);
    Cr06 = _mm256_fmadd_pd(Ar06,Br6,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar16,Br6,Cr16);
    Br7 = _mm256_set1_pd(B[7 + 7 * lda]);
    Cr07 = _mm256_fmadd_pd(Ar07,Br7,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar17,Br7,Cr17);

    //// Second pass
    Br7 = _mm256_set1_pd(B[7 + 0 * lda]);
    Cr00 = _mm256_fmadd_pd(Ar07,Br7,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar17,Br7,Cr10);
    Br0 = _mm256_set1_pd(B[0 + 1 * lda]);
    Cr01 = _mm256_fmadd_pd(Ar00,Br0,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar10,Br0,Cr11);
    Br1 = _mm256_set1_pd(B[1 + 2 * lda]);
    Cr02 = _mm256_fmadd_pd(Ar01,Br1,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar11,Br1,Cr12);
    Br2 = _mm256_set1_pd(B[2 + 3 * lda]);
    Cr03 = _mm256_fmadd_pd(Ar02,Br2,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar12,Br2,Cr13);
    Br3 = _mm256_set1_pd(B[3 + 4 * lda]);
    Cr04 = _mm256_fmadd_pd(Ar03,Br3,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar13,Br3,Cr14);
    Br4 = _mm256_set1_pd(B[4 + 5 * lda]);
    Cr05 = _mm256_fmadd_pd(Ar04,Br4,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar14,Br4,Cr15);
    Br5 = _mm256_set1_pd(B[5 + 6 * lda]);
    Cr06 = _mm256_fmadd_pd(Ar05,Br5,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar15,Br5,Cr16);
    Br6 = _mm256_set1_pd(B[6 + 7 * lda]);
    Cr07 = _mm256_fmadd_pd(Ar06,Br6,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar16,Br6,Cr17);

    //// Third pass
    Br6 = _mm256_set1_pd(B[6 + 0 * lda]);
    Cr00 = _mm256_fmadd_pd(Ar06,Br6,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar16,Br6,Cr10);
    Br7 = _mm256_set1_pd(B[7 + 1 * lda]);
    Cr01 = _mm256_fmadd_pd(Ar07,Br7,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar17,Br7,Cr11);
    Br0 = _mm256_set1_pd(B[0 + 2 * lda]);
    Cr02 = _mm256_fmadd_pd(Ar00,Br0,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar10,Br0,Cr12);
    Br1 = _mm256_set1_pd(B[1 + 3 * lda]);
    Cr03 = _mm256_fmadd_pd(Ar01,Br1,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar11,Br1,Cr13);
    Br2 = _mm256_set1_pd(B[2 + 4 * lda]);
    Cr04 = _mm256_fmadd_pd(Ar02,Br2,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar12,Br2,Cr14);
    Br3 = _mm256_set1_pd(B[3 + 5 * lda]);
    Cr05 = _mm256_fmadd_pd(Ar03,Br3,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar13,Br3,Cr15);
    Br4 = _mm256_set1_pd(B[4 + 6 * lda]);
    Cr06 = _mm256_fmadd_pd(Ar04,Br4,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar14,Br4,Cr16);
    Br5 = _mm256_set1_pd(B[5 + 7 * lda]);
    Cr07 = _mm256_fmadd_pd(Ar05,Br5,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar15,Br5,Cr17);

    //// Fourth pass
    Br5 = _mm256_set1_pd(B[5 + 0 * lda]);
    Cr00 = _mm256_fmadd_pd(Ar05,Br5,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar15,Br5,Cr10);
    Br6 = _mm256_set1_pd(B[6 + 1 * lda]);
    Cr01 = _mm256_fmadd_pd(Ar06,Br6,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar16,Br6,Cr11);
    Br7 = _mm256_set1_pd(B[7 + 2 * lda]);
    Cr02 = _mm256_fmadd_pd(Ar07,Br7,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar17,Br7,Cr12);
    Br0 = _mm256_set1_pd(B[0 + 3 * lda]);
    Cr03 = _mm256_fmadd_pd(Ar00,Br0,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar10,Br0,Cr13);
    Br1 = _mm256_set1_pd(B[1 + 4 * lda]);
    Cr04 = _mm256_fmadd_pd(Ar01,Br1,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar11,Br1,Cr14);
    Br2 = _mm256_set1_pd(B[2 + 5 * lda]);
    Cr05 = _mm256_fmadd_pd(Ar02,Br2,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar12,Br2,Cr15);
    Br3 = _mm256_set1_pd(B[3 + 6 * lda]);
    Cr06 = _mm256_fmadd_pd(Ar03,Br3,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar13,Br3,Cr16);
    Br4 = _mm256_set1_pd(B[4 + 7 * lda]);
    Cr07 = _mm256_fmadd_pd(Ar04,Br4,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar14,Br4,Cr17);

    //// Fifth pass
    Br4 = _mm256_set1_pd(B[4 + 0 * lda]);
    Cr00 = _mm256_fmadd_pd(Ar04,Br4,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar14,Br4,Cr10);
    Br5 = _mm256_set1_pd(B[5 + 1 * lda]);
    Cr01 = _mm256_fmadd_pd(Ar05,Br5,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar15,Br5,Cr11);
    Br6 = _mm256_set1_pd(B[6 + 2 * lda]);
    Cr02 = _mm256_fmadd_pd(Ar06,Br6,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar16,Br6,Cr12);
    Br7 = _mm256_set1_pd(B[7 + 3 * lda]);
    Cr03 = _mm256_fmadd_pd(Ar07,Br7,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar17,Br7,Cr13);
    Br0 = _mm256_set1_pd(B[0 + 4 * lda]);
    Cr04 = _mm256_fmadd_pd(Ar00,Br0,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar10,Br0,Cr14);
    Br1 = _mm256_set1_pd(B[1 + 5 * lda]);
    Cr05 = _mm256_fmadd_pd(Ar01,Br1,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar11,Br1,Cr15);
    Br2 = _mm256_set1_pd(B[2 + 6 * lda]);
    Cr06 = _mm256_fmadd_pd(Ar02,Br2,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar12,Br2,Cr16);
    Br3 = _mm256_set1_pd(B[3 + 7 * lda]);
    Cr07 = _mm256_fmadd_pd(Ar03,Br3,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar13,Br3,Cr17);

    //// Sixth pass
    Br3 = _mm256_set1_pd(B[3 + 0 * lda]);
    Cr00 = _mm256_fmadd_pd(Ar03,Br3,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar13,Br3,Cr10);
    Br4 = _mm256_set1_pd(B[4 + 1 * lda]);
    Cr01 = _mm256_fmadd_pd(Ar04,Br4,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar14,Br4,Cr11);
    Br5 = _mm256_set1_pd(B[5 + 2 * lda]);
    Cr02 = _mm256_fmadd_pd(Ar05,Br5,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar15,Br5,Cr12);
    Br6 = _mm256_set1_pd(B[6 + 3 * lda]);
    Cr03 = _mm256_fmadd_pd(Ar06,Br6,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar16,Br6,Cr13);
    Br7 = _mm256_set1_pd(B[7 + 4 * lda]);
    Cr04 = _mm256_fmadd_pd(Ar07,Br7,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar17,Br7,Cr14);
    Br0 = _mm256_set1_pd(B[0 + 5 * lda]);
    Cr05 = _mm256_fmadd_pd(Ar00,Br0,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar10,Br0,Cr15);
    Br1 = _mm256_set1_pd(B[1 + 6 * lda]);
    Cr06 = _mm256_fmadd_pd(Ar01,Br1,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar11,Br1,Cr16);
    Br2 = _mm256_set1_pd(B[2 + 7 * lda]);
    Cr07 = _mm256_fmadd_pd(Ar02,Br2,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar12,Br2,Cr17);

    //// Seventh pass
    Br2 = _mm256_set1_pd(B[2 + 0 * lda]);
    Cr00 = _mm256_fmadd_pd(Ar02,Br2,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar12,Br2,Cr10);
    Br3 = _mm256_set1_pd(B[3 + 1 * lda]);
    Cr01 = _mm256_fmadd_pd(Ar03,Br3,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar13,Br3,Cr11);
    Br4 = _mm256_set1_pd(B[4 + 2 * lda]);
    Cr02 = _mm256_fmadd_pd(Ar04,Br4,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar14,Br4,Cr12);
    Br5 = _mm256_set1_pd(B[5 + 3 * lda]);
    Cr03 = _mm256_fmadd_pd(Ar05,Br5,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar15,Br5,Cr13);
    Br6 = _mm256_set1_pd(B[6 + 4 * lda]);
    Cr04 = _mm256_fmadd_pd(Ar06,Br6,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar16,Br6,Cr14);
    Br7 = _mm256_set1_pd(B[7 + 5 * lda]);
    Cr05 = _mm256_fmadd_pd(Ar07,Br7,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar17,Br7,Cr15);
    Br0 = _mm256_set1_pd(B[0 + 6 * lda]);
    Cr06 = _mm256_fmadd_pd(Ar00,Br0,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar10,Br0,Cr16);
    Br1 = _mm256_set1_pd(B[1 + 7 * lda]);
    Cr07 = _mm256_fmadd_pd(Ar01,Br1,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar11,Br1,Cr17);

    //// Eighth pass
    Br1 = _mm256_set1_pd(B[1 + 0 * lda]);
    Cr00 = _mm256_fmadd_pd(Ar01,Br1,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar11,Br1,Cr10);
    Br2 = _mm256_set1_pd(B[2 + 1 * lda]);
    Cr01 = _mm256_fmadd_pd(Ar02,Br2,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar12,Br2,Cr11);
    Br3 = _mm256_set1_pd(B[3 + 2 * lda]);
    Cr02 = _mm256_fmadd_pd(Ar03,Br3,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar13,Br3,Cr12);
    Br4 = _mm256_set1_pd(B[4 + 3 * lda]);
    Cr03 = _mm256_fmadd_pd(Ar04,Br4,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar14,Br4,Cr13);
    Br5 = _mm256_set1_pd(B[5 + 4 * lda]);
    Cr04 = _mm256_fmadd_pd(Ar05,Br5,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar15,Br5,Cr14);
    Br6 = _mm256_set1_pd(B[6 + 5 * lda]);
    Cr05 = _mm256_fmadd_pd(Ar06,Br6,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar16,Br6,Cr15);
    Br7 = _mm256_set1_pd(B[7 + 6 * lda]);
    Cr06 = _mm256_fmadd_pd(Ar07,Br7,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar17,Br7,Cr16);
    Br0 = _mm256_set1_pd(B[0 + 7 * lda]);
    Cr07 = _mm256_fmadd_pd(Ar00,Br0,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar10,Br0,Cr17);

    //// Store
    _mm256_storeu_pd(C + 0 + 0 * lda, Cr00);
    _mm256_storeu_pd(C + 4 + 0 * lda, Cr10);
    _mm256_storeu_pd(C + 0 + 1 * lda, Cr01);
    _mm256_storeu_pd(C + 4 + 1 * lda, Cr11);
    _mm256_storeu_pd(C + 0 + 2 * lda, Cr02);
    _mm256_storeu_pd(C + 4 + 2 * lda, Cr12);
    _mm256_storeu_pd(C + 0 + 3 * lda, Cr03);
    _mm256_storeu_pd(C + 4 + 3 * lda, Cr13);
    _mm256_storeu_pd(C + 0 + 4 * lda, Cr04);
    _mm256_storeu_pd(C + 4 + 4 * lda, Cr14);
    _mm256_storeu_pd(C + 0 + 5 * lda, Cr05);
    _mm256_storeu_pd(C + 4 + 5 * lda, Cr15);
    _mm256_storeu_pd(C + 0 + 6 * lda, Cr06);
    _mm256_storeu_pd(C + 4 + 6 * lda, Cr16);
    _mm256_storeu_pd(C + 0 + 7 * lda, Cr07);
    _mm256_storeu_pd(C + 4 + 7 * lda, Cr17);    
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
    for (int j = 0; j < N; j+=MICROKERNEL_ISIZE) {
        for (int k = 0; k < K; k+=MICROKERNEL_KSIZE) {
            for (int i = 0; i < M; i+=MICROKERNEL_ISIZE) {
                /*if ((N-j < MICROKERNEL_SIZE) || (K-k < MICROKERNEL_SIZE) || (M-i < MICROKERNEL_SIZE)){
                    int M1 = min(MICROKERNEL_SIZE, M - i);
                    int N1 = min(MICROKERNEL_SIZE, N - j);
                    int K1 = min(MICROKERNEL_SIZE, K - k);
                    // Perform individual block dgemm
                    do_dummyblock(lda, M1, N1, K1, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                }
                else{*/
                    micro_kernel_8by8(lda, A + i + k * lda, B + k + j * lda, C + i + j * lda);
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



