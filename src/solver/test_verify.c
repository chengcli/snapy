#include <stdio.h>
#include <math.h>

int main() {
    int n = 5;
    double h = 1.0 / (n + 1);
    double pi = 3.14159265358979323846;
    
    printf("Verification test for -u'' = sin(pi*x), u(0)=u(1)=0\n");
    printf("Expected solution: u(x) = sin(pi*x)/pi²\n\n");
    
    printf("Grid spacing h = %f\n", h);
    printf("Discretization: (2*u[i] - u[i-1] - u[i+1])/h² = sin(pi*x[i])\n");
    printf("Or: A*u = h²*sin(pi*x[i])\n\n");
    
    double u[7];  // Include ghost points
    u[0] = 0.0;  // Left BC
    u[n+1] = 0.0;  // Right BC
    
    // Set exact solution at interior points
    for (int i = 1; i <= n; i++) {
        double xi = i * h;
        u[i] = sin(pi * xi) / (pi * pi);
    }
    
    // Verify the discretization
    printf("i   x[i]     u[i]       Au[i]     h²*sin(pi*x[i])  error\n");
    for (int i = 1; i <= n; i++) {
        double xi = i * h;
        double Au = (2.0 * u[i] - u[i-1] - u[i+1]) / (h * h);
        double rhs = sin(pi * xi);
        printf("%d  %.4f  %.6f  %.6f  %.6f  %.6e\n",
               i, xi, u[i], Au, rhs, Au - rhs);
    }
    
    return 0;
}
