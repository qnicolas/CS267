#include <immintrin.h> //AVX intrinsics for SIMD operations
#include <malloc.h>
#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef L1_BLOCK_SIZE
#define L1_BLOCK_SIZE 48
#endif

#ifndef L2_BLOCK_SIZE
#define L2_BLOCK_SIZE 144
#endif

#ifndef MICROKERNEL_ISIZE
#define MICROKERNEL_ISIZE 8
#define MICROKERNEL_JSIZE 6
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

static void micro_kernel(int M, int K,double* A, double* B, double* C) {
    __m256d Ar0, Ar1;
    __m256d Br;
    __m256d Cr00,Cr01,Cr02,Cr03,Cr04,Cr05,Cr10,Cr11,Cr12,Cr13,Cr14,Cr15;

    Cr00 = _mm256_load_pd(C + 0 + 0 * M);
    Cr10 = _mm256_load_pd(C + 4 + 0 * M);
    Cr01 = _mm256_load_pd(C + 0 + 1 * M);
    Cr11 = _mm256_load_pd(C + 4 + 1 * M);
    Cr02 = _mm256_load_pd(C + 0 + 2 * M);
    Cr12 = _mm256_load_pd(C + 4 + 2 * M);
    Cr03 = _mm256_load_pd(C + 0 + 3 * M);
    Cr13 = _mm256_load_pd(C + 4 + 3 * M);
    Cr04 = _mm256_load_pd(C + 0 + 4 * M);
    Cr14 = _mm256_load_pd(C + 4 + 4 * M);
    Cr05 = _mm256_load_pd(C + 0 + 5 * M);
    Cr15 = _mm256_load_pd(C + 4 + 5 * M);

    for (int k = 0; k<K; ++k){
        Ar0 = _mm256_load_pd(A + 0 + k * M);
        Ar1 = _mm256_load_pd(A + 4 + k * M);

        Br = _mm256_set1_pd(B[k + 0 * K]);
        Cr00 = _mm256_fmadd_pd(Ar0,Br,Cr00);
        Cr10 = _mm256_fmadd_pd(Ar1,Br,Cr10);

        Br = _mm256_set1_pd(B[k + 1 * K]);
        Cr01 = _mm256_fmadd_pd(Ar0,Br,Cr01);
        Cr11 = _mm256_fmadd_pd(Ar1,Br,Cr11);

        Br = _mm256_set1_pd(B[k + 2 * K]);
        Cr02 = _mm256_fmadd_pd(Ar0,Br,Cr02);
        Cr12 = _mm256_fmadd_pd(Ar1,Br,Cr12);

        Br = _mm256_set1_pd(B[k + 3 * K]);
        Cr03 = _mm256_fmadd_pd(Ar0,Br,Cr03);
        Cr13 = _mm256_fmadd_pd(Ar1,Br,Cr13);

        Br = _mm256_set1_pd(B[k + 4 * K]);
        Cr04 = _mm256_fmadd_pd(Ar0,Br,Cr04);
        Cr14 = _mm256_fmadd_pd(Ar1,Br,Cr14);

        Br = _mm256_set1_pd(B[k + 5 * K]);
        Cr05 = _mm256_fmadd_pd(Ar0,Br,Cr05);
        Cr15 = _mm256_fmadd_pd(Ar1,Br,Cr15);
   }
    //// Store
    _mm256_store_pd(C + 0 + 0 * M, Cr00);
    _mm256_store_pd(C + 4 + 0 * M, Cr10);
    _mm256_store_pd(C + 0 + 1 * M, Cr01);
    _mm256_store_pd(C + 4 + 1 * M, Cr11);
    _mm256_store_pd(C + 0 + 2 * M, Cr02);
    _mm256_store_pd(C + 4 + 2 * M, Cr12);
    _mm256_store_pd(C + 0 + 3 * M, Cr03);
    _mm256_store_pd(C + 4 + 3 * M, Cr13);
    _mm256_store_pd(C + 0 + 4 * M, Cr04);
    _mm256_store_pd(C + 4 + 4 * M, Cr14);
    _mm256_store_pd(C + 0 + 5 * M, Cr05);
    _mm256_store_pd(C + 4 + 5 * M, Cr15);
}

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block_L1(int lda, int M, int N, int K, double* A, double* B, double* C) {
    for (int j = 0; j < N; j+=MICROKERNEL_JSIZE) {
        for (int i = 0; i < M; i+=MICROKERNEL_ISIZE) {
            micro_kernel(M, K, A + i, B + j * K, C + i + j * M);
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

static void populate_block_L1(int lda, int M, int N, int newM, int newN, double* A, double* newA){
    //printf("%d\n", newM);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            newA[i + j*newM] = A[i+j*lda];
        }
    }
    // Pad edges with zeros
    for (int j = 0; j < N; ++j) {
        for (int i = M; i < newM; ++i) {
            newA[i + j*newM] = 0.;
        }
    }    
    for (int j = N; j < newN; ++j) {
        for (int i = 0; i < newM; ++i) {
            newA[i + j*newM] = 0.;
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

static void copy_back_block_L1(int lda, int M, int N, int newM, double* A, double* newA){
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            A[i+j*lda] = newA[i + j*newM];
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
    int newlda = (((lda-1) / 24) + 1) * 24; // only handles square microkernels
    double* newA = (double*) _mm_malloc(newlda*newlda*sizeof(double), 32);
    double* newB = (double*) _mm_malloc(newlda*newlda*sizeof(double), 32);
    double* newC = (double*) _mm_malloc(newlda*newlda*sizeof(double), 32);
    create_repacked_copy(lda,newlda,A,newA);
    create_repacked_copy(lda,newlda,B,newB);
    create_repacked_copy(lda,newlda,C,newC);
    for (int i = 0; i < newlda; i += L2_BLOCK_SIZE) {
        for (int j = 0; j < newlda; j += L2_BLOCK_SIZE) {
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