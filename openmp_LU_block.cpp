#include <random>
#include <string.h>
#include <iostream>
#include <chrono>
#include <unistd.h>
#include <omp.h>

#ifndef CACHE_BLOCK_SIZE
#define CACHE_BLOCK_SIZE 64
#endif

#define TILE_SIZE 8

#define A(i, j) A[(i) * (n) + (j)]
#define AA(i, j) AA[(i) * (n) + (j)]
#define L(i, j) L[(j) * (n) + (i)]
#define U(i, j) U[(i) * (n) + (j)]

#define ABS(a) ((a) < 0 ? -(a) : (a))

void test(int n, double *A, double *L, double *U, int *P)
{
    double norm = 0.0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double aij_ = 0.0;
            for (int k = 0; k < n; k++)
            {
                aij_ += L(i, k) * U(k, j);
            }
            double aij = A(P[i], j);
            norm += (aij - aij_) * (aij - aij_);
        }
    }
    std::cout << "L2 norm error (PA - LU): " << norm << "\n";
}

double *AA;

double Linv[TILE_SIZE][TILE_SIZE];
double Uinv[TILE_SIZE][TILE_SIZE];

void LUD_parallel(double *A, double *L, double *U, int *P, int n, int t)
{

// assuming n is divisible by TILE_SIZE
#pragma omp parallel num_threads(t)
    for (int tr = 0; tr < n; tr += TILE_SIZE)
    {
        if (omp_get_thread_num() == 0)
        {
            // std::cout << "ola\n";

            // pivot all the rows that span the tiles in this iteration
            for (int k = tr; k < tr + TILE_SIZE; k++)
            {
                double max = 0.0;
                int k_ = -1;

                for (int i = k; i < n; i++)
                {
                    if (max < ABS(A(i, k)))
                    {
                        max = ABS(A(i, k));
                        k_ = i;
                    }
                }

                if (k_ == -1)
                {
                    std::cout << "invalid argument(singular matrix)\n";
                    exit(0);
                }

                std::swap(P[k], P[k_]);
                for (int i = 0; i < n; i++)
                {
                    std::swap(A(k, i), A(k_, i));
                }
                for (int i = 0; i < k; i++)
                {
                    std::swap(L(k, i), L(k_, i));
                }
            }

            // Cacluate L, U for top left A tile
            // Hopefully everything fits in cache
            for (int k = 0; k < TILE_SIZE; k++)
            {
                U(tr + k, tr + k) = A(tr + k, tr + k);

                for (int i = k + 1; i < TILE_SIZE; i++)
                {
                    L(tr + i, tr + k) = A(tr + i, tr + k) / U(tr + k, tr + k);
                    U(tr + k, tr + i) = A(tr + k, tr + i);
                }
                for (int i = k + 1; i < TILE_SIZE; i++)
                {
                    for (int j = k + 1; j < TILE_SIZE; j++)
                    {
                        A(tr + i, tr + j) -= L(tr + i, tr + k) * U(tr + k, tr + j);
                    }
                }
            }

            // for (int i = 0; i < TILE_SIZE; i++)
            // {
            //     for (int j = 0; j < TILE_SIZE; j++)
            //     {
            //         A(tr + i, tr + j) = AA(tr + i, tr + j);
            //     }
            // }

            // calculate their inverses
            // Hopefully everything fits in cache
            for (int i = 0; i < TILE_SIZE; i++)
            {
                for (int j = 0; j < TILE_SIZE; j++)
                {
                    Linv[i][j] = 0;
                }
            }

            for (int j = 0; j < TILE_SIZE; j++)
            {
                Linv[j][j] = 1.0;
                for (int i = j + 1; i < TILE_SIZE; i++)
                {
                    double s = 0;
                    for (int k = j; k < i; k++)
                    {
                        s -= L(tr + i, tr + k) * Linv[k][j];
                    }
                    Linv[i][j] = s;
                }
            }

            // for (int i = 0; i < TILE_SIZE; i++)
            // {
            //     for (int j = 0; j < TILE_SIZE; j++)
            //     {
            //         double s = 0;
            //         for (int k = 0; k < TILE_SIZE; k++)
            //         {
            //             s += L(tr + i, tr + k) * Linv[k][j];
            //         }
            //         std::cout << s << " ";
            //     }
            //     std::cout << "\n";
            // }

            // calculate their inverses
            // Hopefully everything fits in cache
            for (int i = 0; i < TILE_SIZE; i++)
            {
                for (int j = 0; j < TILE_SIZE; j++)
                {
                    Uinv[i][j] = 0;
                }
            }

            for (int j = 0; j < TILE_SIZE; j++)
            {
                Uinv[j][j] = 1 / U(tr + j, tr + j);
                for (int i = j - 1; i >= 0; i--)
                {
                    double s = 0;
                    for (int k = j; k > i; k--)
                    {
                        s -= U(tr + i, tr + k) * Uinv[k][j];
                    }
                    Uinv[i][j] = s / U(tr + i, tr + i);
                }
            }

            // for (int i = 0; i < TILE_SIZE; i++)
            // {
            //     for (int j = 0; j < TILE_SIZE; j++)
            //     {
            //         double s = 0;
            //         for (int k = 0; k < TILE_SIZE; k++)
            //         {
            //             s += U(tr + i, tr + k) * Uinv[k][j];
            //         }
            //         std::cout << s << " ";
            //     }
            //     std::cout << "\n";
            // }
        }
#pragma omp barrier

        int bb = tr + TILE_SIZE;
        int col_base = tr;
        int row_base;
#pragma omp for
        for (row_base = bb; row_base < n; row_base += TILE_SIZE)
        {
            for (int i = 0; i < TILE_SIZE; i++)
            {
                for (int j = 0; j < TILE_SIZE; j++)
                {
                    // calc L(row_base + i, col_base + j)
                    double s = 0;
                    for (int k = 0; k < TILE_SIZE; k++)
                    {
                        s += A(row_base + i, col_base + k) * Uinv[k][j];
                    }
                    L(row_base + i, col_base + j) = s;
                }
            }
        }

        row_base = tr;

#pragma omp for
        for (col_base = bb; col_base < n; col_base += TILE_SIZE)
        {

            for (int i = 0; i < TILE_SIZE; i++)
            {
                for (int j = 0; j < TILE_SIZE; j++)
                {
                    // calc U(bb + i, col_base + j)
                    double s = 0;
                    for (int k = 0; k < TILE_SIZE; k++)
                    {
                        s += Linv[i][k] * A(row_base + k, col_base + j);
                    }
                    U(row_base + i, col_base + j) = s;
                }
            }
        }

        // for (int i = 0; i < n; i++)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         std::cout << L(i, j) << " ";
        //     }
        //     std::cout << "\n";
        // }
        // std::cout << "---pp-\n";
        // for (int i = 0; i < n; i++)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         std::cout << U(i, j) << " ";
        //     }
        //     std::cout << "\n";
        // }
        // std::cout << "----\n";

        // for (int i = 0; i < n; i++)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         std::cout << AA(i, j) << " ";
        //     }
        //     std::cout << "\n";
        // }
        // std::cout << "----\n";

        // for (int i = 0; i < TILE_SIZE; i++)
        // {
        //     for (int j = 0; j < TILE_SIZE; j++)
        //     {
        //         std::cout << Linv[i][j] << " ";
        //     }
        //     std::cout << "\n";
        // }
        // std::cout << "----\n";

        // for (int i = 0; i < TILE_SIZE; i++)
        // {
        //     for (int j = 0; j < TILE_SIZE; j++)
        //     {
        //         std::cout << Uinv[i][j] << " ";
        //     }
        //     std::cout << "\n";
        // }
        // std::cout << "----\n";

        // std::cout << "printing diff\n";
        // for (int i = 0; i < n; i++)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         double s = 0.0;
        //         for (int k = 0; k < n; k++)
        //         {
        //             s += L(i, k) * U(k, j);
        //         }
        //         double a = AA(P[i], j);
        //         // norm += (a - s) * (a - s);
        //         std::cout << ABS(a - s) << " ";
        //     }
        //     std::cout << "\n";
        // }
        // std::cout << "----\n";

        // std::cout << "(" << rb << ", " << cb << "): " << norm << "\n";

#pragma omp for collapse(2)
        for (int tbr = bb; tbr < n; tbr += TILE_SIZE)
        {
            for (int tbc = bb; tbc < n; tbc += TILE_SIZE)
            {
                for (int i = 0; i < TILE_SIZE; i++)
                {
                    for (int j = 0; j < TILE_SIZE; j++)
                    {
                        double s = 0;
                        for (int k = 0; k < TILE_SIZE; k++)
                        {
                            s += L(tbr + i, tr + k) * U(tr + k, tbc + j);
                            // std::cout << "yoypppoyo " << A(1, 1) << " " << s << " " << L(tbr + i, tr + k) << " " << U(tr + k, tbc + j) << "\n";
                        }
                        // std::cout << "yoyoyo " << A(1, 1) << " " << s << "\n";
                        A(tbr + i, tbc + j) -= s;
                    }
                }
            }
        }

        // std::cout << "-printing A-\n";

        // for (int i = 0; i < n; i++)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         std::cout << A(i, j) << " ";
        //     }
        //     std::cout << "\n";
        // }
        // std::cout << "----\n";
    }
}

int round_up_to_next_p2(int n)
{
    int x = 1;
    while (x < n)
    {
        x *= 2;
    }
    return x;
}

void LU_Decomp(int n, int t, bool check_res)
{
    double lower_bound = 0;
    double upper_bound = 1.0;

    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;
    // re.seed(time(0)); // TODO: fix seed
    re.seed(13);
    n = round_up_to_next_p2(n);

    double *A = new double[n * n];
    double *L = new double[n * n];
    double *U = new double[n * n];
    int *P = new int[n];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A(i, j) = unif(re);
        }
    }

    for (int i = 0; i < n; i++)
    {
        P[i] = i;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                L(i, i) = 1.0;
            }
            else
            {
                L(i, j) = 0.0;
                U(i, j) = 0.0;
            }
        }
    }

    AA = A;

    double *A_ = new double[n * n];
    memcpy(A_, A, sizeof(double) * n * n);

    auto start = std::chrono::high_resolution_clock::now();

    LUD_parallel(A_, L, U, P, n, t);

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "n = " << n << ", thread count: " << t << ", time: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() / 1e6 << "ms\n";

    if (check_res)
        test(n, A, L, U, P);

    delete[] A;
    delete[] A_;
    delete[] L;
    delete[] U;
    delete[] P;
}

int main(int argc, char **argv)
{
    int n = 100, t = 1;
    bool check_res = false;
    if (argc > 1)
    {
        n = atoi(argv[1]);
    }
    if (argc > 2)
    {
        t = atoi(argv[2]);
    }
    if (argc > 3)
    {
        check_res = true;
    }
    LU_Decomp(n, t, check_res);
}
