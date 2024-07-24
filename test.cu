#include <iostream>
#include <cstdio>
#include <Eigen/Dense>
#include "svd3_cuda.h"

__global__ void svd_kernel(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33,
                           float* u, float* s, float* v) {
    svd(a11, a12, a13, a21, a22, a23, a31, a32, a33,
        u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8],
        s[0], s[1], s[2],
        v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
}

void print_matrix(const Eigen::Matrix3f& matrix, const std::string& name) {
    std::cout << "Matrix " << name << ":\n" << matrix << std::endl;
}

int main() {
    float a11 = 1.0f, a12 = 2.0f, a13 = 3.0f;
    float a21 = 4.0f, a22 = 5.0f, a23 = 6.0f;
    float a31 = 7.0f, a32 = 8.0f, a33 = 9.0f;

    float h_u[9], h_s[3], h_v[9];
    float *d_u, *d_s, *d_v;

    cudaMalloc(&d_u, 9 * sizeof(float));
    cudaMalloc(&d_s, 3 * sizeof(float));
    cudaMalloc(&d_v, 9 * sizeof(float));

    svd_kernel<<<1, 1>>>(a11, a12, a13, a21, a22, a23, a31, a32, a33, d_u, d_s, d_v);
    cudaDeviceSynchronize();

    cudaMemcpy(h_u, d_u, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_s, d_s, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_u);
    cudaFree(d_s);
    cudaFree(d_v);

    // Create matrices U, S, V from the float arrays
    Eigen::Matrix3f U, S, V;
    U << h_u[0], h_u[1], h_u[2],
         h_u[3], h_u[4], h_u[5],
         h_u[6], h_u[7], h_u[8];

    S << h_s[0], 0, 0,
         0, h_s[1], 0,
         0, 0, h_s[2];

    V << h_v[0], h_v[1], h_v[2],
         h_v[3], h_v[4], h_v[5],
         h_v[6], h_v[7], h_v[8];

    // Print matrices U, S, V using std::cout
    print_matrix(U, "U");
    print_matrix(S, "S");
    print_matrix(V, "V");

    // Reconstruct the original matrix A from USV^T
    Eigen::Matrix3f A_reconstructed = U * S * V.transpose();
    print_matrix(A_reconstructed, "A_reconstructed");

    // Create the original matrix A
    Eigen::Matrix3f A;
    A << a11, a12, a13,
         a21, a22, a23,
         a31, a32, a33;
    print_matrix(A, "A_original");

    // Check orthogonality of U and V
    Eigen::Matrix3f UUT = U * U.transpose();
    Eigen::Matrix3f VVT = V * V.transpose();
    print_matrix(UUT, "U*U^T");
    print_matrix(VVT, "V*V^T");

    // Additional prints using printf
    std::printf("Matrix A (original):\n");
    std::printf("%f %f %f\n%f %f %f\n%f %f %f\n", a11, a12, a13, a21, a22, a23, a31, a32, a33);

    return 0;
}

