#include <random>
#include <string.h>
#include <iostream>
#include <chrono>
#include <pthread.h>
#include <unistd.h>
#include <immintrin.h>

#define A(i, j) A[(i) * (n) + (j)]
#define AA(i, j) AA[(i) * (n) + (j)]
#define L(i, j) L[(i) * (n) + (j)]
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

void LUD_serial(int n, double *A, double *L, double *U, int *P)
{
    for (int k = 0; k < n; k++)
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
            std::cout << "Invalid argument(singular matrix)\n";
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

        U(k, k) = A(k, k);

        for (int i = k + 1; i < n; i++)
        {
            L(i, k) = A(i, k) / U(k, k);
            U(k, i) = A(k, i);
        }
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                A(i, j) -= L(i, k) * U(k, j);
            }
        }
    }
}

struct LUDInput
{
    int n;
    int t;
    double *A;
    double *L;
    double *U;
    int *P;
    int tid;
    pthread_barrier_t *barrier;
};

void *LUD_parallel(void *arg)
{
    LUDInput *input = (LUDInput *)arg;
    int n = input->n;
    int t = input->t;
    double *A = input->A;
    double *L = input->L;
    double *U = input->U;
    int *P = input->P;
    int tid = input->tid;
    pthread_barrier_t *barrier = input->barrier;

    for (int k = 0; k < n; k++)
    {
        double max = 0.0;
        int k_ = -1;

        if (tid == 0)
        {
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
            U(k, k) = A(k, k);
            for (int i = k + 1; i < n; i++)
            {
                L(i, k) = A(i, k) / U(k, k);
                U(k, i) = A(k, i);
            }
        }

        pthread_barrier_wait(barrier);

        int total_rows_left = (n - k - 1);
        int work_per_th = (total_rows_left + t - 1) / t;

        int st = k + 1 + tid * work_per_th;
        int ed = std::min(st + work_per_th, n);

        int total_cols_left = (n - k - 1);

#ifndef TILE_SIZE
#define TILE_SIZE 256
#endif

        int block_size = TILE_SIZE;
        int num_blocks = (total_cols_left + block_size - 1) / block_size;

        for (int block = 0; block < num_blocks; block++)
        {
            for (int i = st; i < ed; i++)
            {
                int j_st = k + 1 + block * block_size;
                int j_end = std::min(j_st + block_size, n);
                double c = L(i, k);

                for (int j = j_st; j < j_end; j++)
                {
                    A(i, j) -= c * U(k, j);
                }
            }
        }

        pthread_barrier_wait(barrier);
    }
    return NULL;
}

void LU_Decomp(int n, int t, bool check_res)
{
    double lower_bound = 0;
    double upper_bound = 1.0;

    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;
    re.seed(time(0));

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

    double *A_ = new double[n * n];
    memcpy(A_, A, sizeof(double) * n * n);

    auto start = std::chrono::high_resolution_clock::now();

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, t);

    if (t == 1)
    {
        LUD_serial(n, A_, L, U, P);
    }
    else
    {

        std::vector<pthread_t> threads(t);
        std::vector<LUDInput *> inputs(t);

        for (int i = 0; i < t; i++)
        {
            LUDInput *input = new LUDInput();
            input->n = n;
            input->t = t;
            input->A = A_;
            input->L = L;
            input->U = U;
            input->P = P;
            input->tid = i;
            input->barrier = &barrier;
            inputs[i] = input;

            pthread_create(&threads[i], NULL, LUD_parallel, (void *)inputs[i]);
        }

        for (int i = 0; i < t; i++)
        {
            pthread_join(threads[i], NULL);
        }

        pthread_barrier_destroy(&barrier);
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "n = " << n << ", thread count: " << t << ", time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() / 1e6 << "ms\n";

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
