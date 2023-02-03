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
#define MICROKERNEL_ISIZE 4
#define MICROKERNEL_KSIZE 4
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) < (b)) ? (b) : (a))

static void micro_kernel_4by4(double* A, double* B, double* C) {
    // Declare
    __m256d Ar00,Ar01,Ar02,Ar03;
    __m256d Br0,Br1,Br2,Br3;
    __m256d Cr00,Cr01,Cr02,Cr03;
    
    Ar00 = _mm256_load_pd(A + 0 + 0 * MICROKERNEL_ISIZE);
    Ar01 = _mm256_load_pd(A + 0 + 1 * MICROKERNEL_ISIZE);
    Ar02 = _mm256_load_pd(A + 0 + 2 * MICROKERNEL_ISIZE);
    Ar03 = _mm256_load_pd(A + 0 + 3 * MICROKERNEL_ISIZE);

    Cr00 = _mm256_load_pd(C + 0 + 0 * MICROKERNEL_ISIZE);
    Cr01 = _mm256_load_pd(C + 0 + 1 * MICROKERNEL_ISIZE);
    Cr02 = _mm256_load_pd(C + 0 + 2 * MICROKERNEL_ISIZE);
    Cr03 = _mm256_load_pd(C + 0 + 3 * MICROKERNEL_ISIZE);

    //// 1 th pass
    Br0 = _mm256_set1_pd(B[0 + 0 * MICROKERNEL_KSIZE]);
    Cr00 = _mm256_fmadd_pd(Ar00,Br0,Cr00);
    Br1 = _mm256_set1_pd(B[1 + 1 * MICROKERNEL_KSIZE]);
    Cr01 = _mm256_fmadd_pd(Ar01,Br1,Cr01);
    Br2 = _mm256_set1_pd(B[2 + 2 * MICROKERNEL_KSIZE]);
    Cr02 = _mm256_fmadd_pd(Ar02,Br2,Cr02);
    Br3 = _mm256_set1_pd(B[3 + 3 * MICROKERNEL_KSIZE]);
    Cr03 = _mm256_fmadd_pd(Ar03,Br3,Cr03);

    //// 2 th pass
    Br1 = _mm256_set1_pd(B[1 + 0 * MICROKERNEL_KSIZE]);
    Cr00 = _mm256_fmadd_pd(Ar01,Br1,Cr00);
    Br2 = _mm256_set1_pd(B[2 + 1 * MICROKERNEL_KSIZE]);
    Cr01 = _mm256_fmadd_pd(Ar02,Br2,Cr01);
    Br3 = _mm256_set1_pd(B[3 + 2 * MICROKERNEL_KSIZE]);
    Cr02 = _mm256_fmadd_pd(Ar03,Br3,Cr02);
    Br0 = _mm256_set1_pd(B[0 + 3 * MICROKERNEL_KSIZE]);
    Cr03 = _mm256_fmadd_pd(Ar00,Br0,Cr03);

    //// 3 th pass
    Br2 = _mm256_set1_pd(B[2 + 0 * MICROKERNEL_KSIZE]);
    Cr00 = _mm256_fmadd_pd(Ar02,Br2,Cr00);
    Br3 = _mm256_set1_pd(B[3 + 1 * MICROKERNEL_KSIZE]);
    Cr01 = _mm256_fmadd_pd(Ar03,Br3,Cr01);
    Br0 = _mm256_set1_pd(B[0 + 2 * MICROKERNEL_KSIZE]);
    Cr02 = _mm256_fmadd_pd(Ar00,Br0,Cr02);
    Br1 = _mm256_set1_pd(B[1 + 3 * MICROKERNEL_KSIZE]);
    Cr03 = _mm256_fmadd_pd(Ar01,Br1,Cr03);

    //// 4 th pass
    Br3 = _mm256_set1_pd(B[3 + 0 * MICROKERNEL_KSIZE]);
    Cr00 = _mm256_fmadd_pd(Ar03,Br3,Cr00);
    Br0 = _mm256_set1_pd(B[0 + 1 * MICROKERNEL_KSIZE]);
    Cr01 = _mm256_fmadd_pd(Ar00,Br0,Cr01);
    Br1 = _mm256_set1_pd(B[1 + 2 * MICROKERNEL_KSIZE]);
    Cr02 = _mm256_fmadd_pd(Ar01,Br1,Cr02);
    Br2 = _mm256_set1_pd(B[2 + 3 * MICROKERNEL_KSIZE]);
    Cr03 = _mm256_fmadd_pd(Ar02,Br2,Cr03);

    //// Store
    _mm256_store_pd(C + 0 + 0 * MICROKERNEL_ISIZE, Cr00);
    _mm256_store_pd(C + 0 + 1 * MICROKERNEL_ISIZE, Cr01);
    _mm256_store_pd(C + 0 + 2 * MICROKERNEL_ISIZE, Cr02);
    _mm256_store_pd(C + 0 + 3 * MICROKERNEL_ISIZE, Cr03);
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
                micro_kernel_4by4(A + i*MICROKERNEL_KSIZE + k * M, B + k*MICROKERNEL_ISIZE + j * K, C + i*MICROKERNEL_ISIZE + j * M);
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
    int maxsize = max(MICROKERNEL_ISIZE,MICROKERNEL_KSIZE);
    int newlda = (((lda-1) / maxsize) + 1) * maxsize;
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