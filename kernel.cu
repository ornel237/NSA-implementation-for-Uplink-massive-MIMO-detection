#include <device_launch_parameters.h>

#include <cuda_runtime.h>

#include <stdio.h>

#include <stdlib.h>

#include <math.h>



#define N 64

#define TILE 16



// ================================

// Kernel multiplication matricielle avec tiling en double précision

// ================================

__global__ void matrixMulTiled(double* A, double* B, double* C, int n) {

    __shared__ double tileA[TILE][TILE];

    __shared__ double tileB[TILE][TILE];



    int row = blockIdx.y * TILE + threadIdx.y;

    int col = blockIdx.x * TILE + threadIdx.x;

    double sum = 0.0;



    for (int m = 0; m < (n + TILE - 1) / TILE; ++m) {

        tileA[threadIdx.y][threadIdx.x] = (row < n && m * TILE + threadIdx.x < n) ? A[row * n + m * TILE + threadIdx.x] : 0.0;

        tileB[threadIdx.y][threadIdx.x] = (col < n && m * TILE + threadIdx.y < n) ? B[(m * TILE + threadIdx.y) * n + col] : 0.0;

        __syncthreads();

        for (int k = 0; k < TILE; ++k)

            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();

    }



    if (row < n && col < n)

        C[row * n + col] = sum;

}



// ================================

// Affichage matrice joliment formatée

// ================================

void printMatrix(double* mat, int n, const char* name) {

    printf("=== %s ===\n", name);

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            printf("%10.6f ", mat[i * n + j]);

            if ((j + 1) % 8 == 0) printf("  ");

        }

        printf("\n");

    }

    printf("\n");

}



// ================================

// Norme Frobenius

// ================================

double normMat(double* A, int n) {

    double sum = 0.0;

    for (int i = 0; i < n * n; i++) sum += A[i] * A[i];

    return sqrt(sum);

}



// ================================

// Erreur Frobenius normalisée (comme MATLAB)

// ================================

double frobeniusErrorMATLAB(double* Aref, double* Aapprox, int n) {

    double sum_diff = 0.0;

    double sum_ref = 0.0;

    for (int i = 0; i < n * n; i++) {

        double diff = Aref[i] - Aapprox[i];

        sum_diff += diff * diff;

        sum_ref += Aref[i] * Aref[i];

    }

    return sqrt(sum_diff) / sqrt(sum_ref);

}



// ================================

// MAIN

// ================================

int main() {

    double h_A[N * N], h_Ainv_exact[N * N];



    // --- Lire matrice A depuis MATLAB ---

    FILE* fp = fopen("matrix_A.txt", "r");

    if (!fp) { printf("Erreur : impossible d'ouvrir matrix_A.txt\n"); return 1; }

    for (int i = 0; i < N; i++)

        for (int j = 0; j < N; j++)

            fscanf(fp, "%lf", &h_A[i * N + j]);

    fclose(fp);

    printMatrix(h_A, N, "Matrice A");



    // --- Lire l'inverse exact MATLAB (si disponible) ---

    fp = fopen("Ainv.txt", "r");

    bool hasExact = false;

    if (fp) {

        hasExact = true;

        for (int i = 0; i < N; i++)

            for (int j = 0; j < N; j++)

                fscanf(fp, "%lf", &h_Ainv_exact[i * N + j]);

        fclose(fp);

    }



    // --- Créer D, Dinv, E ---

    double h_D[N * N], h_Dinv[N * N], h_E[N * N];

    for (int i = 0; i < N; i++) {

        for (int j = 0; j < N; j++) {

            if (i == j) {

                h_D[i * N + j] = h_A[i * N + j];

                h_Dinv[i * N + j] = 1.0 / h_A[i * N + j];

                h_E[i * N + j] = 0.0;

            }
            else {

                h_D[i * N + j] = 0.0;

                h_Dinv[i * N + j] = 0.0;

                h_E[i * N + j] = h_A[i * N + j];

            }

        }

    }

    printMatrix(h_D, N, "Matrice D (diagonale de A)");

    printMatrix(h_Dinv, N, "Matrice Dinv (inverse diagonale)");



    // --- Allocation GPU ---

    double* d_Dinv, * d_E, * d_temp1, * d_temp2, * d_temp3, * d_temp4;

    cudaMalloc((void**)&d_Dinv, N * N * sizeof(double));

    cudaMalloc((void**)&d_E, N * N * sizeof(double));

    cudaMalloc((void**)&d_temp1, N * N * sizeof(double));

    cudaMalloc((void**)&d_temp2, N * N * sizeof(double));

    cudaMalloc((void**)&d_temp3, N * N * sizeof(double));

    cudaMalloc((void**)&d_temp4, N * N * sizeof(double));



    cudaMemcpy(d_Dinv, h_Dinv, N * N * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_E, h_E, N * N * sizeof(double), cudaMemcpyHostToDevice);



    dim3 threads(TILE, TILE);

    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);



    // --- Calcul Neumann série ---

    matrixMulTiled << <blocks, threads >> > (d_Dinv, d_E, d_temp1, N);     // temp1 = Dinv*E

    cudaDeviceSynchronize();



    matrixMulTiled << <blocks, threads >> > (d_temp1, d_Dinv, d_temp2, N); // ordre 2

    cudaDeviceSynchronize();



    matrixMulTiled << <blocks, threads >> > (d_temp1, d_temp1, d_temp3, N); // (Dinv*E)^2

    cudaDeviceSynchronize();

    matrixMulTiled << <blocks, threads >> > (d_temp3, d_Dinv, d_temp3, N);  // ordre 3

    cudaDeviceSynchronize();



    matrixMulTiled << <blocks, threads >> > (d_temp3, d_temp1, d_temp4, N); // (Dinv*E)^3

    cudaDeviceSynchronize();

    matrixMulTiled << <blocks, threads >> > (d_temp4, d_Dinv, d_temp4, N);  // ordre 4

    cudaDeviceSynchronize();



    // --- Copier GPU -> CPU ---

    double h_temp2[N * N], h_temp3[N * N], h_temp4[N * N];

    cudaMemcpy(h_temp2, d_temp2, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_temp3, d_temp3, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_temp4, d_temp4, N * N * sizeof(double), cudaMemcpyDeviceToHost);



    // --- Calcul approximations finales ---

    double h_Ainv2[N * N], h_Ainv3[N * N], h_Ainv4[N * N];

    for (int i = 0; i < N * N; i++) {

        h_Ainv2[i] = h_Dinv[i] - h_temp2[i];

        h_Ainv3[i] = h_Dinv[i] - h_temp2[i] + h_temp3[i];

        h_Ainv4[i] = h_Dinv[i] - h_temp2[i] + h_temp3[i] - h_temp4[i];

    }



    printMatrix(h_Ainv2, N, "Neumann ordre 2");

    printMatrix(h_Ainv3, N, "Neumann ordre 3");

    printMatrix(h_Ainv4, N, "Neumann ordre 4");



    // --- Calcul erreurs Frobenius ---

    if (hasExact) {

        printf("=== Erreurs Frobenius relatives ===\n");

        printf("Ordre 2 : %.6e\n", frobeniusErrorMATLAB(h_Ainv_exact, h_Ainv2, N));

        printf("Ordre 3 : %.6e\n", frobeniusErrorMATLAB(h_Ainv_exact, h_Ainv3, N));

        printf("Ordre 4 : %.6e\n", frobeniusErrorMATLAB(h_Ainv_exact, h_Ainv4, N));

        printf("\n");

    }



    // --- Sauvegarde des matrices pour MATLAB ---

    FILE* f2 = fopen("Ainv2.txt", "w");

    FILE* f3 = fopen("Ainv3.txt", "w");

    FILE* f4 = fopen("Ainv4.txt", "w");

    if (!f2 || !f3 || !f4) {

        printf("Erreur lors de la création des fichiers de sortie.\n");

        return 1;

    }

    for (int i = 0; i < N * N; i++) {

        fprintf(f2, "%.15lf\n", h_Ainv2[i]);

        fprintf(f3, "%.15lf\n", h_Ainv3[i]);

        fprintf(f4, "%.15lf\n", h_Ainv4[i]);

    }

    fclose(f2); fclose(f3); fclose(f4);

    printf("=== Matrices enregistrées : Ainv2.txt, Ainv3.txt, Ainv4.txt ===\n");



    // --- Libération mémoire GPU ---

    cudaFree(d_Dinv); cudaFree(d_E); cudaFree(d_temp1); cudaFree(d_temp2);

    cudaFree(d_temp3); cudaFree(d_temp4);



    return 0;

}