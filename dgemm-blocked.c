#include <immintrin.h> //AVX intrinsics for SIMD operations
#include <malloc.h>
#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef L1_BLOCK_SIZE
#define L1_BLOCK_SIZE 128
#endif

#ifndef L2_BLOCK_SIZE
#define L2_BLOCK_SIZE 512
#endif

#ifndef MICROKERNEL_ISIZE
#define MICROKERNEL_ISIZE 8
#define MICROKERNEL_KSIZE 8
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

static void micro_kernel_8by8(double* A, double* B, double* C) {
    // Declare
    __m256d Ar00,Ar01,Ar02,Ar03,Ar04,Ar05,Ar06,Ar07,Ar10,Ar11,Ar12,Ar13,Ar14,Ar15,Ar16,Ar17;
    __m256d Br0,Br1,Br2,Br3,Br4,Br5,Br6,Br7;
    __m256d Cr00,Cr01,Cr02,Cr03,Cr04,Cr05,Cr06,Cr07,Cr10,Cr11,Cr12,Cr13,Cr14,Cr15,Cr16,Cr17;
    Ar00 = _mm256_load_pd(A + 0 + 0 * MICROKERNEL_ISIZE);
    Ar10 = _mm256_load_pd(A + 4 + 0 * MICROKERNEL_ISIZE);
    Ar01 = _mm256_load_pd(A + 0 + 1 * MICROKERNEL_ISIZE);
    Ar11 = _mm256_load_pd(A + 4 + 1 * MICROKERNEL_ISIZE);
    Ar02 = _mm256_load_pd(A + 0 + 2 * MICROKERNEL_ISIZE);
    Ar12 = _mm256_load_pd(A + 4 + 2 * MICROKERNEL_ISIZE);
    Ar03 = _mm256_load_pd(A + 0 + 3 * MICROKERNEL_ISIZE);
    Ar13 = _mm256_load_pd(A + 4 + 3 * MICROKERNEL_ISIZE);
    Ar04 = _mm256_load_pd(A + 0 + 4 * MICROKERNEL_ISIZE);
    Ar14 = _mm256_load_pd(A + 4 + 4 * MICROKERNEL_ISIZE);
    Ar05 = _mm256_load_pd(A + 0 + 5 * MICROKERNEL_ISIZE);
    Ar15 = _mm256_load_pd(A + 4 + 5 * MICROKERNEL_ISIZE);
    Ar06 = _mm256_load_pd(A + 0 + 6 * MICROKERNEL_ISIZE);
    Ar16 = _mm256_load_pd(A + 4 + 6 * MICROKERNEL_ISIZE);
    Ar07 = _mm256_load_pd(A + 0 + 7 * MICROKERNEL_ISIZE);
    Ar17 = _mm256_load_pd(A + 4 + 7 * MICROKERNEL_ISIZE);

    Cr00 = _mm256_load_pd(C + 0 + 0 * MICROKERNEL_ISIZE);
    Cr10 = _mm256_load_pd(C + 4 + 0 * MICROKERNEL_ISIZE);
    Cr01 = _mm256_load_pd(C + 0 + 1 * MICROKERNEL_ISIZE);
    Cr11 = _mm256_load_pd(C + 4 + 1 * MICROKERNEL_ISIZE);
    Cr02 = _mm256_load_pd(C + 0 + 2 * MICROKERNEL_ISIZE);
    Cr12 = _mm256_load_pd(C + 4 + 2 * MICROKERNEL_ISIZE);
    Cr03 = _mm256_load_pd(C + 0 + 3 * MICROKERNEL_ISIZE);
    Cr13 = _mm256_load_pd(C + 4 + 3 * MICROKERNEL_ISIZE);
    Cr04 = _mm256_load_pd(C + 0 + 4 * MICROKERNEL_ISIZE);
    Cr14 = _mm256_load_pd(C + 4 + 4 * MICROKERNEL_ISIZE);
    Cr05 = _mm256_load_pd(C + 0 + 5 * MICROKERNEL_ISIZE);
    Cr15 = _mm256_load_pd(C + 4 + 5 * MICROKERNEL_ISIZE);
    Cr06 = _mm256_load_pd(C + 0 + 6 * MICROKERNEL_ISIZE);
    Cr16 = _mm256_load_pd(C + 4 + 6 * MICROKERNEL_ISIZE);
    Cr07 = _mm256_load_pd(C + 0 + 7 * MICROKERNEL_ISIZE);
    Cr17 = _mm256_load_pd(C + 4 + 7 * MICROKERNEL_ISIZE);

    //// First pass
    Br0 = _mm256_set1_pd(B[0 + 0 * MICROKERNEL_KSIZE]);
    Cr00 = _mm256_fmadd_pd(Ar00,Br0,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar10,Br0,Cr10);
    Br1 = _mm256_set1_pd(B[1 + 1 * MICROKERNEL_KSIZE]);
    Cr01 = _mm256_fmadd_pd(Ar01,Br1,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar11,Br1,Cr11);
    Br2 = _mm256_set1_pd(B[2 + 2 * MICROKERNEL_KSIZE]);
    Cr02 = _mm256_fmadd_pd(Ar02,Br2,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar12,Br2,Cr12);
    Br3 = _mm256_set1_pd(B[3 + 3 * MICROKERNEL_KSIZE]);
    Cr03 = _mm256_fmadd_pd(Ar03,Br3,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar13,Br3,Cr13);
    Br4 = _mm256_set1_pd(B[4 + 4 * MICROKERNEL_KSIZE]);
    Cr04 = _mm256_fmadd_pd(Ar04,Br4,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar14,Br4,Cr14);
    Br5 = _mm256_set1_pd(B[5 + 5 * MICROKERNEL_KSIZE]);
    Cr05 = _mm256_fmadd_pd(Ar05,Br5,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar15,Br5,Cr15);
    Br6 = _mm256_set1_pd(B[6 + 6 * MICROKERNEL_KSIZE]);
    Cr06 = _mm256_fmadd_pd(Ar06,Br6,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar16,Br6,Cr16);
    Br7 = _mm256_set1_pd(B[7 + 7 * MICROKERNEL_KSIZE]);
    Cr07 = _mm256_fmadd_pd(Ar07,Br7,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar17,Br7,Cr17);

    //// Second pass
    Br7 = _mm256_set1_pd(B[7 + 0 * MICROKERNEL_KSIZE]);
    Cr00 = _mm256_fmadd_pd(Ar07,Br7,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar17,Br7,Cr10);
    Br0 = _mm256_set1_pd(B[0 + 1 * MICROKERNEL_KSIZE]);
    Cr01 = _mm256_fmadd_pd(Ar00,Br0,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar10,Br0,Cr11);
    Br1 = _mm256_set1_pd(B[1 + 2 * MICROKERNEL_KSIZE]);
    Cr02 = _mm256_fmadd_pd(Ar01,Br1,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar11,Br1,Cr12);
    Br2 = _mm256_set1_pd(B[2 + 3 * MICROKERNEL_KSIZE]);
    Cr03 = _mm256_fmadd_pd(Ar02,Br2,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar12,Br2,Cr13);
    Br3 = _mm256_set1_pd(B[3 + 4 * MICROKERNEL_KSIZE]);
    Cr04 = _mm256_fmadd_pd(Ar03,Br3,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar13,Br3,Cr14);
    Br4 = _mm256_set1_pd(B[4 + 5 * MICROKERNEL_KSIZE]);
    Cr05 = _mm256_fmadd_pd(Ar04,Br4,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar14,Br4,Cr15);
    Br5 = _mm256_set1_pd(B[5 + 6 * MICROKERNEL_KSIZE]);
    Cr06 = _mm256_fmadd_pd(Ar05,Br5,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar15,Br5,Cr16);
    Br6 = _mm256_set1_pd(B[6 + 7 * MICROKERNEL_KSIZE]);
    Cr07 = _mm256_fmadd_pd(Ar06,Br6,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar16,Br6,Cr17);

    //// Third pass
    Br6 = _mm256_set1_pd(B[6 + 0 * MICROKERNEL_KSIZE]);
    Cr00 = _mm256_fmadd_pd(Ar06,Br6,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar16,Br6,Cr10);
    Br7 = _mm256_set1_pd(B[7 + 1 * MICROKERNEL_KSIZE]);
    Cr01 = _mm256_fmadd_pd(Ar07,Br7,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar17,Br7,Cr11);
    Br0 = _mm256_set1_pd(B[0 + 2 * MICROKERNEL_KSIZE]);
    Cr02 = _mm256_fmadd_pd(Ar00,Br0,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar10,Br0,Cr12);
    Br1 = _mm256_set1_pd(B[1 + 3 * MICROKERNEL_KSIZE]);
    Cr03 = _mm256_fmadd_pd(Ar01,Br1,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar11,Br1,Cr13);
    Br2 = _mm256_set1_pd(B[2 + 4 * MICROKERNEL_KSIZE]);
    Cr04 = _mm256_fmadd_pd(Ar02,Br2,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar12,Br2,Cr14);
    Br3 = _mm256_set1_pd(B[3 + 5 * MICROKERNEL_KSIZE]);
    Cr05 = _mm256_fmadd_pd(Ar03,Br3,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar13,Br3,Cr15);
    Br4 = _mm256_set1_pd(B[4 + 6 * MICROKERNEL_KSIZE]);
    Cr06 = _mm256_fmadd_pd(Ar04,Br4,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar14,Br4,Cr16);
    Br5 = _mm256_set1_pd(B[5 + 7 * MICROKERNEL_KSIZE]);
    Cr07 = _mm256_fmadd_pd(Ar05,Br5,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar15,Br5,Cr17);

    //// Fourth pass
    Br5 = _mm256_set1_pd(B[5 + 0 * MICROKERNEL_KSIZE]);
    Cr00 = _mm256_fmadd_pd(Ar05,Br5,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar15,Br5,Cr10);
    Br6 = _mm256_set1_pd(B[6 + 1 * MICROKERNEL_KSIZE]);
    Cr01 = _mm256_fmadd_pd(Ar06,Br6,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar16,Br6,Cr11);
    Br7 = _mm256_set1_pd(B[7 + 2 * MICROKERNEL_KSIZE]);
    Cr02 = _mm256_fmadd_pd(Ar07,Br7,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar17,Br7,Cr12);
    Br0 = _mm256_set1_pd(B[0 + 3 * MICROKERNEL_KSIZE]);
    Cr03 = _mm256_fmadd_pd(Ar00,Br0,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar10,Br0,Cr13);
    Br1 = _mm256_set1_pd(B[1 + 4 * MICROKERNEL_KSIZE]);
    Cr04 = _mm256_fmadd_pd(Ar01,Br1,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar11,Br1,Cr14);
    Br2 = _mm256_set1_pd(B[2 + 5 * MICROKERNEL_KSIZE]);
    Cr05 = _mm256_fmadd_pd(Ar02,Br2,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar12,Br2,Cr15);
    Br3 = _mm256_set1_pd(B[3 + 6 * MICROKERNEL_KSIZE]);
    Cr06 = _mm256_fmadd_pd(Ar03,Br3,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar13,Br3,Cr16);
    Br4 = _mm256_set1_pd(B[4 + 7 * MICROKERNEL_KSIZE]);
    Cr07 = _mm256_fmadd_pd(Ar04,Br4,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar14,Br4,Cr17);

    //// Fifth pass
    Br4 = _mm256_set1_pd(B[4 + 0 * MICROKERNEL_KSIZE]);
    Cr00 = _mm256_fmadd_pd(Ar04,Br4,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar14,Br4,Cr10);
    Br5 = _mm256_set1_pd(B[5 + 1 * MICROKERNEL_KSIZE]);
    Cr01 = _mm256_fmadd_pd(Ar05,Br5,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar15,Br5,Cr11);
    Br6 = _mm256_set1_pd(B[6 + 2 * MICROKERNEL_KSIZE]);
    Cr02 = _mm256_fmadd_pd(Ar06,Br6,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar16,Br6,Cr12);
    Br7 = _mm256_set1_pd(B[7 + 3 * MICROKERNEL_KSIZE]);
    Cr03 = _mm256_fmadd_pd(Ar07,Br7,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar17,Br7,Cr13);
    Br0 = _mm256_set1_pd(B[0 + 4 * MICROKERNEL_KSIZE]);
    Cr04 = _mm256_fmadd_pd(Ar00,Br0,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar10,Br0,Cr14);
    Br1 = _mm256_set1_pd(B[1 + 5 * MICROKERNEL_KSIZE]);
    Cr05 = _mm256_fmadd_pd(Ar01,Br1,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar11,Br1,Cr15);
    Br2 = _mm256_set1_pd(B[2 + 6 * MICROKERNEL_KSIZE]);
    Cr06 = _mm256_fmadd_pd(Ar02,Br2,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar12,Br2,Cr16);
    Br3 = _mm256_set1_pd(B[3 + 7 * MICROKERNEL_KSIZE]);
    Cr07 = _mm256_fmadd_pd(Ar03,Br3,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar13,Br3,Cr17);

    //// Sixth pass
    Br3 = _mm256_set1_pd(B[3 + 0 * MICROKERNEL_KSIZE]);
    Cr00 = _mm256_fmadd_pd(Ar03,Br3,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar13,Br3,Cr10);
    Br4 = _mm256_set1_pd(B[4 + 1 * MICROKERNEL_KSIZE]);
    Cr01 = _mm256_fmadd_pd(Ar04,Br4,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar14,Br4,Cr11);
    Br5 = _mm256_set1_pd(B[5 + 2 * MICROKERNEL_KSIZE]);
    Cr02 = _mm256_fmadd_pd(Ar05,Br5,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar15,Br5,Cr12);
    Br6 = _mm256_set1_pd(B[6 + 3 * MICROKERNEL_KSIZE]);
    Cr03 = _mm256_fmadd_pd(Ar06,Br6,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar16,Br6,Cr13);
    Br7 = _mm256_set1_pd(B[7 + 4 * MICROKERNEL_KSIZE]);
    Cr04 = _mm256_fmadd_pd(Ar07,Br7,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar17,Br7,Cr14);
    Br0 = _mm256_set1_pd(B[0 + 5 * MICROKERNEL_KSIZE]);
    Cr05 = _mm256_fmadd_pd(Ar00,Br0,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar10,Br0,Cr15);
    Br1 = _mm256_set1_pd(B[1 + 6 * MICROKERNEL_KSIZE]);
    Cr06 = _mm256_fmadd_pd(Ar01,Br1,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar11,Br1,Cr16);
    Br2 = _mm256_set1_pd(B[2 + 7 * MICROKERNEL_KSIZE]);
    Cr07 = _mm256_fmadd_pd(Ar02,Br2,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar12,Br2,Cr17);

    //// Seventh pass
    Br2 = _mm256_set1_pd(B[2 + 0 * MICROKERNEL_KSIZE]);
    Cr00 = _mm256_fmadd_pd(Ar02,Br2,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar12,Br2,Cr10);
    Br3 = _mm256_set1_pd(B[3 + 1 * MICROKERNEL_KSIZE]);
    Cr01 = _mm256_fmadd_pd(Ar03,Br3,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar13,Br3,Cr11);
    Br4 = _mm256_set1_pd(B[4 + 2 * MICROKERNEL_KSIZE]);
    Cr02 = _mm256_fmadd_pd(Ar04,Br4,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar14,Br4,Cr12);
    Br5 = _mm256_set1_pd(B[5 + 3 * MICROKERNEL_KSIZE]);
    Cr03 = _mm256_fmadd_pd(Ar05,Br5,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar15,Br5,Cr13);
    Br6 = _mm256_set1_pd(B[6 + 4 * MICROKERNEL_KSIZE]);
    Cr04 = _mm256_fmadd_pd(Ar06,Br6,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar16,Br6,Cr14);
    Br7 = _mm256_set1_pd(B[7 + 5 * MICROKERNEL_KSIZE]);
    Cr05 = _mm256_fmadd_pd(Ar07,Br7,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar17,Br7,Cr15);
    Br0 = _mm256_set1_pd(B[0 + 6 * MICROKERNEL_KSIZE]);
    Cr06 = _mm256_fmadd_pd(Ar00,Br0,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar10,Br0,Cr16);
    Br1 = _mm256_set1_pd(B[1 + 7 * MICROKERNEL_KSIZE]);
    Cr07 = _mm256_fmadd_pd(Ar01,Br1,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar11,Br1,Cr17);

    //// Eighth pass
    Br1 = _mm256_set1_pd(B[1 + 0 * MICROKERNEL_KSIZE]);
    Cr00 = _mm256_fmadd_pd(Ar01,Br1,Cr00);
    Cr10 = _mm256_fmadd_pd(Ar11,Br1,Cr10);
    Br2 = _mm256_set1_pd(B[2 + 1 * MICROKERNEL_KSIZE]);
    Cr01 = _mm256_fmadd_pd(Ar02,Br2,Cr01);
    Cr11 = _mm256_fmadd_pd(Ar12,Br2,Cr11);
    Br3 = _mm256_set1_pd(B[3 + 2 * MICROKERNEL_KSIZE]);
    Cr02 = _mm256_fmadd_pd(Ar03,Br3,Cr02);
    Cr12 = _mm256_fmadd_pd(Ar13,Br3,Cr12);
    Br4 = _mm256_set1_pd(B[4 + 3 * MICROKERNEL_KSIZE]);
    Cr03 = _mm256_fmadd_pd(Ar04,Br4,Cr03);
    Cr13 = _mm256_fmadd_pd(Ar14,Br4,Cr13);
    Br5 = _mm256_set1_pd(B[5 + 4 * MICROKERNEL_KSIZE]);
    Cr04 = _mm256_fmadd_pd(Ar05,Br5,Cr04);
    Cr14 = _mm256_fmadd_pd(Ar15,Br5,Cr14);
    Br6 = _mm256_set1_pd(B[6 + 5 * MICROKERNEL_KSIZE]);
    Cr05 = _mm256_fmadd_pd(Ar06,Br6,Cr05);
    Cr15 = _mm256_fmadd_pd(Ar16,Br6,Cr15);
    Br7 = _mm256_set1_pd(B[7 + 6 * MICROKERNEL_KSIZE]);
    Cr06 = _mm256_fmadd_pd(Ar07,Br7,Cr06);
    Cr16 = _mm256_fmadd_pd(Ar17,Br7,Cr16);
    Br0 = _mm256_set1_pd(B[0 + 7 * MICROKERNEL_KSIZE]);
    Cr07 = _mm256_fmadd_pd(Ar00,Br0,Cr07);
    Cr17 = _mm256_fmadd_pd(Ar10,Br0,Cr17);

    //// Store
    _mm256_store_pd(C + 0 + 0 * MICROKERNEL_ISIZE, Cr00);
    _mm256_store_pd(C + 4 + 0 * MICROKERNEL_ISIZE, Cr10);
    _mm256_store_pd(C + 0 + 1 * MICROKERNEL_ISIZE, Cr01);
    _mm256_store_pd(C + 4 + 1 * MICROKERNEL_ISIZE, Cr11);
    _mm256_store_pd(C + 0 + 2 * MICROKERNEL_ISIZE, Cr02);
    _mm256_store_pd(C + 4 + 2 * MICROKERNEL_ISIZE, Cr12);
    _mm256_store_pd(C + 0 + 3 * MICROKERNEL_ISIZE, Cr03);
    _mm256_store_pd(C + 4 + 3 * MICROKERNEL_ISIZE, Cr13);
    _mm256_store_pd(C + 0 + 4 * MICROKERNEL_ISIZE, Cr04);
    _mm256_store_pd(C + 4 + 4 * MICROKERNEL_ISIZE, Cr14);
    _mm256_store_pd(C + 0 + 5 * MICROKERNEL_ISIZE, Cr05);
    _mm256_store_pd(C + 4 + 5 * MICROKERNEL_ISIZE, Cr15);
    _mm256_store_pd(C + 0 + 6 * MICROKERNEL_ISIZE, Cr06);
    _mm256_store_pd(C + 4 + 6 * MICROKERNEL_ISIZE, Cr16);
    _mm256_store_pd(C + 0 + 7 * MICROKERNEL_ISIZE, Cr07);
    _mm256_store_pd(C + 4 + 7 * MICROKERNEL_ISIZE, Cr17);    
}


/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */

static void do_block_L1(int lda, int M, int N, int K, double* A, double* B, double* C) {
    for (int j = 0; j < N; j+=MICROKERNEL_ISIZE) {
        for (int k = 0; k < K; k+=MICROKERNEL_KSIZE) {
            for (int i = 0; i < M; i+=MICROKERNEL_ISIZE) {
                micro_kernel_8by8(A + i*MICROKERNEL_KSIZE + k * M, B + k*MICROKERNEL_ISIZE + j * K, C + i*MICROKERNEL_ISIZE + j * M);
            }
        }
    }
}

static void do_block_L2(int lda, int M, int N, int K, double* A, double* B, double* C) {
    for (int k = 0; k < K; k += L1_BLOCK_SIZE) {
        for (int j = 0; j < N; j += L1_BLOCK_SIZE) {
            for (int i = 0; i < M; i += L1_BLOCK_SIZE) {
                int M1 = min(L1_BLOCK_SIZE, M - i);
                int N1 = min(L1_BLOCK_SIZE, N - j);
                int K1 = min(L1_BLOCK_SIZE, K - k);
                // Perform individual block dgemm
                do_block_L1(lda, M1, N1, K1, A + i*K1 + k*M, B + k*N1 + j*K, C + i*N1 + j*M);
            }
        }
    }
}

///////////////////////////////////////////////////////////////
////////////////////////// REPACKING //////////////////////////
///////////////////////////////////////////////////////////////

static void populate_block_microk(int lda, int M, int N, double* A, double* newA){
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            newA[i + j*MICROKERNEL_ISIZE] = A[i+j*lda];
        }
    }
    // Pad edges with zeros
    for (int j = 0; j < N; ++j) {
        for (int i = M; i < MICROKERNEL_ISIZE; ++i) {
            newA[i + j*MICROKERNEL_ISIZE] = 0.;
        }
    }    
    for (int j = N; j < MICROKERNEL_KSIZE; ++j) {
        for (int i = 0; i < MICROKERNEL_ISIZE; ++i) {
            newA[i + j*MICROKERNEL_ISIZE] = 0.;
        }
    }
}

static void populate_block_L1(int lda, int M, int N, int newM, int newN, double* A, double* newA){
    for (int j = 0; j < N; j += MICROKERNEL_KSIZE) {
        for (int i = 0; i < M; i += MICROKERNEL_ISIZE) {
            int M1 = min(MICROKERNEL_ISIZE, M - i);
            int N1 = min(MICROKERNEL_KSIZE, N - j);
            populate_block_microk(lda, M1, N1, A + i + j*lda, newA + i*MICROKERNEL_KSIZE + j*newM);
        }
    }
}

static void populate_block_L2(int lda, int M, int N, int newM, int newN, double* A, double* newA){
    for (int j = 0; j < N; j += L1_BLOCK_SIZE) {
        for (int i = 0; i < M; i += L1_BLOCK_SIZE) {
            int M1 = min(L1_BLOCK_SIZE, M - i);
            int N1 = min(L1_BLOCK_SIZE, N - j);
            int newM1 = min(L1_BLOCK_SIZE, newM - i);
            int newN1 = min(L1_BLOCK_SIZE, newN - j);
            populate_block_L1(lda, M1, N1, newM1, newN1, A + i + j*lda, newA + i*newN1 + j*newM);
        }
    }
}

static void create_repacked_copy(int lda, int newlda, double* A, double* newA){
    for (int j = 0; j < lda; j += L2_BLOCK_SIZE) {
        for (int i = 0; i < lda; i += L2_BLOCK_SIZE) {
            int M = min(L2_BLOCK_SIZE, lda - i);
            int N = min(L2_BLOCK_SIZE, lda - j);
            int newM = min(L2_BLOCK_SIZE, newlda - i);
            int newN = min(L2_BLOCK_SIZE, newlda - j);
            populate_block_L2(lda, M, N, newM, newN, A + i + j*lda, newA + i*newN + j*newlda);
        }
    }
}

///////////////////////////////////////////////////////////////
///////////////////// INVERSE REPACKING ///////////////////////
///////////////////////////////////////////////////////////////
static void copy_back_block_microk(int lda, int M, int N, double* A, double* newA){
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            A[i+j*lda] = newA[i + j*MICROKERNEL_ISIZE];
        }
    }
}

static void copy_back_block_L1(int lda, int M, int N, int newM, double* A, double* newA){
    for (int j = 0; j < N; j += MICROKERNEL_KSIZE) {
        for (int i = 0; i < M; i += MICROKERNEL_ISIZE) {
            int M1 = min(MICROKERNEL_ISIZE, M - i);
            int N1 = min(MICROKERNEL_KSIZE, N - j);
            copy_back_block_microk(lda, M1, N1, A + i + j*lda, newA + i*MICROKERNEL_KSIZE + j*newM);
        }
    }
}

static void copy_back_block_L2(int lda, int M, int N, int newM, int newN, double* A, double* newA){
    for (int j = 0; j < N; j += L1_BLOCK_SIZE) {
        for (int i = 0; i < M; i += L1_BLOCK_SIZE) {
            int M1 = min(L1_BLOCK_SIZE, M - i);
            int N1 = min(L1_BLOCK_SIZE, N - j);
            int newM1 = min(L1_BLOCK_SIZE, newM - i);
            int newN1 = min(L1_BLOCK_SIZE, newN - j);
            copy_back_block_L1(lda, M1, N1, newM1, A + i + j*lda, newA + i*newN1 + j*newM);
        }
    }
}

static void copy_back(int lda, int newlda, double* A, double* newA){
    for (int j = 0; j < lda; j += L2_BLOCK_SIZE) {
        for (int i = 0; i < lda; i += L2_BLOCK_SIZE) {
            int M = min(L2_BLOCK_SIZE, lda - i);
            int N = min(L2_BLOCK_SIZE, lda - j);
            int newM = min(L2_BLOCK_SIZE, newlda - i);
            int newN = min(L2_BLOCK_SIZE, newlda - j);
            copy_back_block_L2(lda, M, N, newM, newN, A + i + j*lda, newA + i*newN + j*newlda);
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
    int newlda = (((lda-1) / MICROKERNEL_ISIZE) + 1) * MICROKERNEL_ISIZE; // only handles square microkernels
    double* newA = (double*) _mm_malloc(newlda*newlda*sizeof(double), 64);
    double* newB = (double*) _mm_malloc(newlda*newlda*sizeof(double), 64);
    double* newC = (double*) _mm_malloc(newlda*newlda*sizeof(double), 64);
    //double* newA = malloc(newlda*newlda*sizeof(double));
    //double* newB = malloc(newlda*newlda*sizeof(double));
    //double* newC = malloc(newlda*newlda*sizeof(double));
    create_repacked_copy(lda,newlda,A,newA);
    create_repacked_copy(lda,newlda,B,newB);
    create_repacked_copy(lda,newlda,C,newC);
    
    for (int j = 0; j < newlda; j += L2_BLOCK_SIZE) {
        for (int i = 0; i < newlda; i += L2_BLOCK_SIZE) {
            for (int k = 0; k < newlda; k += L2_BLOCK_SIZE) {
                int M = min(L2_BLOCK_SIZE, newlda - i);
                int N = min(L2_BLOCK_SIZE, newlda - j);
                int K = min(L2_BLOCK_SIZE, newlda - k);
                // Perform individual block dgemm
                do_block_L2(newlda, M, N, K, newA + i*K + k*newlda, newB + k*N + j*newlda, newC + i*N + j*newlda);
            }
        }
    }
    copy_back(lda,newlda,C,newC);
    _mm_free(newA);
    _mm_free(newB);
    _mm_free(newC);
    //printf("computed4\n");
}