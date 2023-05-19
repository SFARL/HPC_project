#include <stdio.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <algorithm>
#include <curand.h>
#include <string>
#include <string.h>
#include "utils.h"
using namespace std;

void print_grid(int *grid, int n)
{
    for (int i = 0; i < n * n; i++)
    {
        cout << std::setw(2) << grid[i] << " ";
        if ((i + 1) % n == 0)
            cout << endl;
    }
}

void remove_standard(int *grid, int n, int cnt)
{
    int idx;
    int row;
    int col;
    while (cnt > 0)
    {
        col = rand() % n;
        row = rand() % n;
        idx = row * n + col;

        int temp = grid[idx];
        if (temp != 0)
        {
            grid[idx] = 0;
            cnt--;
        }
    }
}

bool is_safe(int *grid, int n, int row, int col, int num)
{
    int idx;
    // Check if the number is already in the row and col
    for (int i = 0; i < n; i++)
    {
        idx = row * n + i;
        if (grid[idx] == num)
        {
            return false;
        }
        idx = i * n + col;
        if (grid[idx] == num)
        {
            return false;
        }
    }

    // Check if the number is already in the sqrt(n) box
    int root = sqrt(n);
    int box_row = row - row % root;
    int box_col = col - col % root;
    for (int i = box_row; i < box_row + root; i++)
    {
        for (int j = box_col; j < box_col + root; j++)
        {
            idx = i * n + j;
            if (grid[idx] == num)
            {
                return false;
            }
        }
    }
    return true;
}

bool solve_sudoku_cpu(int *grid, int n, int row, int col)
{
    // cout << "row: " << row << " col: " << col << endl;
    if (row == n - 1 && col == n)
    {
        return true;
    }
    if (col == n)
    {
        row++;
        col = 0;
    }

    int idx = row * n + col;
    if (grid[idx] != 0)
    {
        return solve_sudoku_cpu(grid, n, row, col + 1);
    }
    for (int num = n; num > 0; num--)
    {
        // if (col == 20)
        // cout << "row: " << row << " col: " << col << " num: " << num << endl;
        if (is_safe(grid, n, row, col, num))
        {
            // cout << "row: " << row << " col: " << col << " num: " << num << endl;
            // print_grid(grid, n);
            // cout << endl;
            grid[idx] = num;
            if (solve_sudoku_cpu(grid, n, row, col + 1))
            {
                return true;
            }
            grid[idx] = 0; // backtrack
        }
    }
    return false;
}

void build_sudoku(int *grid, int n)
{
    std::mt19937 rng(42);
    int root = sqrt(n);
    for (int i = 0; i < n; i += root)
    {
        int num = std::uniform_int_distribution<int>{1, n}(rng);
        for (int row = 0; row < root; row++)
        {
            for (int col = 0; col < root; col++)
            {
                while (!is_safe(grid, n, row + i, col + i, num))
                {
                    num = std::uniform_int_distribution<int>{1, n}(rng);
                }
                grid[(row + i) * n + col + i] = num;
            }
        }
    }
    solve_sudoku_cpu(grid, n, 0, 0);
}

int cnt_start_board(int *grid, int n, int level, int row, int col)
{
    if (level <= 0)
    {
        return 1;
    }
    if (col == n)
    {
        row++;
        col = 0;
    }
    int idx = row * n + col;
    if (grid[idx] != 0)
    {
        return cnt_start_board(grid, n, level, row, col + 1);
    }
    int res = 0;
    for (int num = n; num > 0; num--)
    {
        // if (col == 20)
        // cout << "row: " << row << " col: " << col << " num: " << num << endl;
        if (is_safe(grid, n, row, col, num))
        {
            // cout << "row: " << row << " col: " << col << " num: " << num << endl;
            // print_grid(grid, n);
            // cout << endl;
            grid[idx] = num;
            res += cnt_start_board(grid, n, level - 1, row, col + 1);
            grid[idx] = 0; // backtrack
        }
    }
    return res;
}

__device__ bool is_safec(int *grid, int n, int index)
{
    if (index < 0 || index >= n * n)
    {
        return false;
    }

    int row = index / n;
    int col = index % n;
    int num = grid[index];
    if (num > n)
    {
        return false;
    }
    for (int i = 0; i < n; i++)
    {
        if (grid[row * n + i] == num && i != col)
        {
            return false;
        }
        if (grid[i * n + col] == num && i != row)
        {
            return false;
        }
    }
    int root = sqrtf(n);
    int box_row = row - row % root;
    int box_col = col - col % root;
    for (int i = box_row; i < box_row + root; i++)
    {
        for (int j = box_col; j < box_col + root; j++)
        {
            int idx = i * n + j;
            if (grid[idx] == num && idx != index)
            {
                return false;
            }
        }
    }
    return true;
}

__global__ void
bfs(int *old_grid, int *new_grid, int total_grid, int *grid_index, int *empty_space, int *empty_cnt, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_grid)
    {
        return;
    }
    bool found = false;
    int start = index * n * n;
    int *grid_now = old_grid + start;
    for (int i = 0; i < n * n && !found; i++)
    {
        if (grid_now[i] != 0)
        {
            continue;
        }
        found = true;

        for (int ava = 1; ava <= n; ava++)
        {
            grid_now[i] = ava;
            if (!is_safec(grid_now, n, i))
            {
                grid_now[i] = 0;
                continue;
            }

            int next_grid_index = atomicAdd(grid_index, 1);
            int empty_index = 0;
            for (int r = 0; r < n; r++)
            {
                for (int c = 0; c < n; c++)
                {
                    new_grid[next_grid_index * n * n + r * n + c] = grid_now[r * n + c];

                    // calculate the new grid empty space.
                    if (grid_now[r * n + c] == 0)
                    {
                        empty_space[next_grid_index * n * n + empty_index] = r * n + c;
                        empty_index++;
                    }
                }
            }
            empty_cnt[next_grid_index] = empty_index;
        }
    }
}

__global__ void cudaBackTrack(int *grid, int *empty_space, int *empty_cnt, bool *solved, int *answer_cuda, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int max_iter = 0;
    int *grid_now, *empty_space_now, empty_cnt_now;
    int empty_index_cnt = 0;
    grid_now = grid + index * n * n;
    empty_space_now = empty_space + index * n * n;
    empty_cnt_now = empty_cnt[index];
    while (empty_index_cnt < empty_cnt_now)
    {
        // printf("index %d empty_index_cnt %d of  empty_cnt_now %d empty_index_cnt %d empty_index_val %d\n", index, empty_index_cnt, empty_cnt_now, empty_space_now[empty_index_cnt], grid_now[empty_space_now[empty_index_cnt]]);
        if (max_iter >= 10000000)
        {
            // printf("index %d exist since over max_iter %d\n", index, 10000000);
            return;
        }
        max_iter++;
        if (*solved)
        {
            // printf("index %d exist cause othter solved\n", index);
            return;
        }
        if (empty_index_cnt < 0)
        {
            // printf("index %d exist cause empty_index_cnt < 0\n", index);
            return;
        }
        // if (empty_space_now[empty_index_cnt] <= 0)
        // {
        //     // printf("index %d exist cause empty_space_now[empty_index_cnt] <= 0\n", index);
        //     return;
        // }
        // atomicAdd(&grid_now[empty_space_now[empty_index_cnt]], 1);
        grid_now[empty_space_now[empty_index_cnt]]++;
        if (is_safec(grid_now, n, empty_space_now[empty_index_cnt]))
        {
            empty_index_cnt++;
        }
        else
        {
            if (grid_now[empty_space_now[empty_index_cnt]] >= n)
            {
                grid_now[empty_space_now[empty_index_cnt]] = 0;
                empty_index_cnt--;
            }
        }
        if (empty_index_cnt == empty_cnt_now)
        {
            *solved = true;
            for (int i = 0; i < n * n; i++)
            {
                answer_cuda[i] = grid_now[i];
            }
            // printf("index %d solved\n", index);
            return;
        }
    }
    return;
}

int main()
{
    int rand_seed[] = {1, 6, 8, 13, 24, 32, 42, 52, 63, 71};
    float drop_rate_3[] = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
    float drop_rate_4[] = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
    int bolck_dims[] = {32, 64, 128, 256};
    int iter_levels[] = {2, 3, 4, 5};
    int root, n;
    int *grid, *mask, *answer;
    double time_taken;
    int rm_cnt;
    float solve_cnt;
    float drop_rate;

    // Test for order 3
    root = 3;
    n = root * root;
    grid = (int *)malloc(n * n * sizeof(int));
    mask = (int *)malloc(n * n * sizeof(int));
    answer = (int *)malloc(n * n * sizeof(int));
    for (int i = 0; i < n * n; i++)
    {
        grid[i] = 0;
    }
    build_sudoku(grid, n);
    memcpy(mask, grid, n * n * sizeof(int));
    int level, block_dim, grim_dim, bfs_cnt, bfs_array_size;                       // cpu
    int *old_grid, *new_grid, *empty_space, *empty_cnt, *grid_index, *answer_cuda; // cuda
    bool *solved, solved_cpu;
    for (int i_level = 0; i_level < 4; i_level++)
    {
        // Different bfs level
        level = iter_levels[i_level];
        for (int i_block = 0; i_block < 4; i_block++)
        {
            // Different block size
            block_dim = bolck_dims[i_block];

            for (int i_drop_rate = 0; i_drop_rate < 6; i_drop_rate++)
            {
                // drop rate
                drop_rate = drop_rate_3[i_drop_rate];
                rm_cnt = n * n * drop_rate;
                solve_cnt = 0;
                time_taken = 0;
                for (int i_rs = 0; i_rs < 10; i_rs++)
                {

                    memcpy(mask, grid, n * n * sizeof(int));
                    srand(rand_seed[i_rs]);
                    remove_standard(mask, n, rm_cnt);
                    bfs_cnt = cnt_start_board(mask, n, level, 0, 0);
                    grim_dim = bfs_cnt / block_dim + 1;
                    time_taken = 0;

                    bfs_array_size = bfs_cnt * n * n * sizeof(int);
                    cudaMalloc(&new_grid, bfs_array_size);
                    cudaMalloc(&old_grid, bfs_array_size);
                    cudaMalloc(&empty_space, bfs_array_size);
                    cudaMalloc(&empty_cnt, bfs_array_size / n / n + 1);
                    cudaMalloc(&grid_index, sizeof(int));
                    cudaMalloc(&solved, sizeof(bool));
                    cudaMalloc(&answer_cuda, n * n * sizeof(int));
                    cudaDeviceSynchronize();
                    cudaMemset(grid_index, 0, sizeof(int));
                    cudaMemset(new_grid, 0, bfs_array_size);
                    cudaMemset(old_grid, 0, bfs_array_size);
                    cudaMemset(empty_space, 0, bfs_array_size);
                    cudaMemset(empty_cnt, 0, bfs_array_size / n / n + 1);
                    cudaMemset(solved, false, sizeof(bool));
                    cudaDeviceSynchronize();
                    cudaMemcpy(old_grid, mask, n * n * sizeof(int), cudaMemcpyHostToDevice);

                    int grid_cnt = 0;
                    int iter = 0;
                    while (iter < level + 1)
                    {
                        cudaMemcpy(&grid_cnt, grid_index, sizeof(int), cudaMemcpyDeviceToHost);
                        if (iter == 0)
                        {
                            grid_cnt = 1;
                        }
                        // printf("iter %d, grid_cnt %d\n", iter, grid_cnt);
                        cudaMemset(grid_index, 0, sizeof(int));
                        cudaDeviceSynchronize();
                        if (iter % 2 == 0)
                        {
                            bfs<<<grim_dim, block_dim>>>(old_grid, new_grid, grid_cnt, grid_index, empty_space, empty_cnt, n);
                        }
                        else
                        {
                            bfs<<<grim_dim, block_dim>>>(new_grid, old_grid, grid_cnt, grid_index, empty_space, empty_cnt, n);
                        }
                        cudaDeviceSynchronize();
                        iter++;
                    }
                    // printf("Order: %d, Level: %d, Grim: %d,Block: %d, Drop rate: %f, Rand seed: %d iter %d, grid_cnt %d and bfs_cnt %d\n", root, level, grim_dim, block_dim, drop_rate, rand_seed[i_rs], iter, grid_cnt, bfs_cnt);
                    if (iter % 2 == 0)
                    {
                        cudaMemcpy(new_grid, old_grid, bfs_array_size, cudaMemcpyDeviceToDevice);
                    }
                    solved_cpu = false;
                    cudaDeviceSynchronize();
                    auto t_start = std::chrono::high_resolution_clock::now();
                    cudaBackTrack<<<grim_dim, block_dim>>>(new_grid, empty_space, empty_cnt, solved, answer_cuda, n);
                    cudaDeviceSynchronize();
                    auto t_end = std::chrono::high_resolution_clock::now();
                    cudaMemcpy(&solved_cpu, solved, sizeof(bool), cudaMemcpyDeviceToHost);
                    if (solved_cpu)
                    {
                        time_taken += std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
                        solve_cnt++;
                    }

                    // for
                    // cudaMemcpy(mask, new_grid, n * n * sizeof(int), cudaMemcpyDeviceToHost);
                    // print_grid(mask, n);
                    // int *test_empty = empty_space + 6 * n * n * sizeof(int);
                    // cudaMemcpy(mask, test_empty, n * n * sizeof(int), cudaMemcpyDeviceToHost);
                    // print_grid(mask, n);

                    cudaFree(new_grid);
                    cudaFree(old_grid);
                    cudaFree(empty_space);
                    cudaFree(empty_cnt);
                    cudaFree(grid_index);
                    cudaFree(solved);
                    cudaFree(answer_cuda);
                }
                printf("Order: %d, level: %d, grim %d, block: %d, drop_rate: %f, solved rate %f time taken: %f\n", root, level, grim_dim, block_dim, drop_rate, solve_cnt / 10.0, time_taken / solve_cnt);
            }
        }
    }
    free(grid);
    free(mask);
    free(answer);

    // Test for order 4
    root = 4;
    n = root * root;
    grid = (int *)malloc(n * n * sizeof(int));
    mask = (int *)malloc(n * n * sizeof(int));
    answer = (int *)malloc(n * n * sizeof(int));
    for (int i = 0; i < n * n; i++)
    {
        grid[i] = 0;
    }
    build_sudoku(grid, n);
    memcpy(mask, grid, n * n * sizeof(int));
    // int level, block_dim, grim_dim, bfs_cnt, bfs_array_size;                       // cpu
    // int *old_grid, *new_grid, *empty_space, *empty_cnt, *grid_index, *answer_cuda; // cuda
    // bool *solved;
    for (int i_level = 0; i_level < 2; i_level++)
    {
        // Different bfs level
        level = iter_levels[i_level];
        for (int i_block = 0; i_block < 4; i_block++)
        {
            // Different block size
            block_dim = bolck_dims[i_block];

            for (int i_drop_rate = 0; i_drop_rate < 6; i_drop_rate++)
            {
                // drop rate
                solve_cnt = 0;
                time_taken = 0;
                drop_rate = drop_rate_4[i_drop_rate];
                rm_cnt = n * n * drop_rate;
                for (int i_rs = 0; i_rs < 10; i_rs++)
                {

                    memcpy(mask, grid, n * n * sizeof(int));
                    srand(rand_seed[i_rs]);
                    remove_standard(mask, n, rm_cnt);
                    bfs_cnt = cnt_start_board(mask, n, level, 0, 0);
                    grim_dim = bfs_cnt / block_dim + 1;
                    time_taken = 0;

                    bfs_array_size = bfs_cnt * n * n * sizeof(int);
                    cudaMalloc(&new_grid, bfs_array_size);
                    cudaMalloc(&old_grid, bfs_array_size);
                    cudaMalloc(&empty_space, bfs_array_size);
                    cudaMalloc(&empty_cnt, bfs_array_size / n / n + 1);
                    cudaMalloc(&grid_index, sizeof(int));
                    cudaMalloc(&solved, sizeof(bool));
                    cudaMalloc(&answer_cuda, n * n * sizeof(int));
                    cudaDeviceSynchronize();
                    cudaMemset(grid_index, 0, sizeof(int));
                    cudaMemset(new_grid, 0, bfs_array_size);
                    cudaMemset(old_grid, 0, bfs_array_size);
                    cudaMemset(empty_space, 0, bfs_array_size);
                    cudaMemset(empty_cnt, 0, bfs_array_size / n / n + 1);
                    cudaMemset(solved, false, sizeof(bool));
                    cudaDeviceSynchronize();
                    cudaMemcpy(old_grid, mask, n * n * sizeof(int), cudaMemcpyHostToDevice);

                    int grid_cnt = 0;
                    int iter = 0;
                    while (iter < level + 1)
                    {
                        cudaMemcpy(&grid_cnt, grid_index, sizeof(int), cudaMemcpyDeviceToHost);
                        if (iter == 0)
                        {
                            grid_cnt = 1;
                        }
                        // printf("iter %d, grid_cnt %d\n", iter, grid_cnt);
                        cudaMemset(grid_index, 0, sizeof(int));
                        cudaDeviceSynchronize();
                        if (iter % 2 == 0)
                        {
                            bfs<<<grim_dim, block_dim>>>(old_grid, new_grid, grid_cnt, grid_index, empty_space, empty_cnt, n);
                        }
                        else
                        {
                            bfs<<<grim_dim, block_dim>>>(new_grid, old_grid, grid_cnt, grid_index, empty_space, empty_cnt, n);
                        }
                        cudaDeviceSynchronize();
                        iter++;
                    }
                    // printf("Order: %d, Level: %d, Grim: %d,Block: %d, Drop rate: %f, Rand seed: %d iter %d, grid_cnt %d and bfs_cnt %d\n", root, level, grim_dim, block_dim, drop_rate, rand_seed[i_rs], iter, grid_cnt, bfs_cnt);
                    if (iter % 2 == 0)
                    {
                        cudaMemcpy(new_grid, old_grid, bfs_array_size, cudaMemcpyDeviceToDevice);
                    }
                    solved_cpu = false;
                    cudaDeviceSynchronize();
                    auto t_start = std::chrono::high_resolution_clock::now();
                    cudaBackTrack<<<grim_dim, block_dim>>>(new_grid, empty_space, empty_cnt, solved, answer_cuda, n);
                    cudaDeviceSynchronize();
                    auto t_end = std::chrono::high_resolution_clock::now();

                    cudaMemcpy(&solved_cpu, solved, sizeof(bool), cudaMemcpyDeviceToHost);
                    if (solved_cpu)
                    {
                        time_taken += std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
                        solve_cnt++;
                    }

                    // for
                    // cudaMemcpy(mask, new_grid, n * n * sizeof(int), cudaMemcpyDeviceToHost);
                    // print_grid(mask, n);
                    // int *test_empty = empty_space + 6 * n * n * sizeof(int);
                    // cudaMemcpy(mask, test_empty, n * n * sizeof(int), cudaMemcpyDeviceToHost);
                    // print_grid(mask, n);

                    cudaFree(new_grid);
                    cudaFree(old_grid);
                    cudaFree(empty_space);
                    cudaFree(empty_cnt);
                    cudaFree(grid_index);
                    cudaFree(solved);
                    cudaFree(answer_cuda);
                }
                printf("Order: %d, level: %d, grim %d, block: %d, drop_rate: %f, solved rate %f time taken: %f\n", root, level, grim_dim, block_dim, drop_rate, solve_cnt / 10.0, time_taken / solve_cnt);
            }
        }
    }
    free(grid);
    free(mask);
    free(answer);

    // Test for order 5
    root = 5;
    n = root * root;
    grid = (int *)malloc(n * n * sizeof(int));
    mask = (int *)malloc(n * n * sizeof(int));
    answer = (int *)malloc(n * n * sizeof(int));
    for (int i = 0; i < n * n; i++)
    {
        grid[i] = 0;
    }
    build_sudoku(grid, n);
    memcpy(mask, grid, n * n * sizeof(int));
    // int level, block_dim, grim_dim, bfs_cnt, bfs_array_size;                       // cpu
    // int *old_grid, *new_grid, *empty_space, *empty_cnt, *grid_index, *answer_cuda; // cuda
    // bool *solved;
    for (int i_level = 0; i_level < 2; i_level++)
    {
        // Different bfs level
        level = iter_levels[i_level];
        for (int i_block = 0; i_block < 4; i_block++)
        {
            // Different block size
            block_dim = bolck_dims[i_block];

            for (int i_drop_rate = 0; i_drop_rate < 4; i_drop_rate++)
            {
                // drop rate
                solve_cnt = 0;
                time_taken = 0;
                drop_rate = drop_rate_4[i_drop_rate];
                rm_cnt = n * n * drop_rate;
                for (int i_rs = 0; i_rs < 10; i_rs++)
                {

                    memcpy(mask, grid, n * n * sizeof(int));
                    srand(rand_seed[i_rs]);
                    remove_standard(mask, n, rm_cnt);
                    bfs_cnt = cnt_start_board(mask, n, level, 0, 0);
                    grim_dim = bfs_cnt / block_dim + 1;
                    time_taken = 0;

                    bfs_array_size = bfs_cnt * n * n * sizeof(int);
                    cudaMalloc(&new_grid, bfs_array_size);
                    cudaMalloc(&old_grid, bfs_array_size);
                    cudaMalloc(&empty_space, bfs_array_size);
                    cudaMalloc(&empty_cnt, bfs_array_size / n / n + 1);
                    cudaMalloc(&grid_index, sizeof(int));
                    cudaMalloc(&solved, sizeof(bool));
                    cudaMalloc(&answer_cuda, n * n * sizeof(int));
                    cudaDeviceSynchronize();
                    cudaMemset(grid_index, 0, sizeof(int));
                    cudaMemset(new_grid, 0, bfs_array_size);
                    cudaMemset(old_grid, 0, bfs_array_size);
                    cudaMemset(empty_space, 0, bfs_array_size);
                    cudaMemset(empty_cnt, 0, bfs_array_size / n / n + 1);
                    cudaMemset(solved, false, sizeof(bool));
                    cudaDeviceSynchronize();
                    cudaMemcpy(old_grid, mask, n * n * sizeof(int), cudaMemcpyHostToDevice);

                    int grid_cnt = 0;
                    int iter = 0;
                    while (iter < level + 1)
                    {
                        cudaMemcpy(&grid_cnt, grid_index, sizeof(int), cudaMemcpyDeviceToHost);
                        if (iter == 0)
                        {
                            grid_cnt = 1;
                        }
                        // printf("iter %d, grid_cnt %d\n", iter, grid_cnt);
                        cudaMemset(grid_index, 0, sizeof(int));
                        cudaDeviceSynchronize();
                        if (iter % 2 == 0)
                        {
                            bfs<<<grim_dim, block_dim>>>(old_grid, new_grid, grid_cnt, grid_index, empty_space, empty_cnt, n);
                        }
                        else
                        {
                            bfs<<<grim_dim, block_dim>>>(new_grid, old_grid, grid_cnt, grid_index, empty_space, empty_cnt, n);
                        }
                        cudaDeviceSynchronize();
                        iter++;
                    }
                    // printf("Order: %d, Level: %d, Grim: %d,Block: %d, Drop rate: %f, Rand seed: %d iter %d, grid_cnt %d and bfs_cnt %d\n", root, level, grim_dim, block_dim, drop_rate, rand_seed[i_rs], iter, grid_cnt, bfs_cnt);
                    if (iter % 2 == 0)
                    {
                        cudaMemcpy(new_grid, old_grid, bfs_array_size, cudaMemcpyDeviceToDevice);
                    }
                    solved_cpu = false;
                    cudaDeviceSynchronize();
                    auto t_start = std::chrono::high_resolution_clock::now();
                    cudaBackTrack<<<grim_dim, block_dim>>>(new_grid, empty_space, empty_cnt, solved, answer_cuda, n);
                    cudaDeviceSynchronize();
                    auto t_end = std::chrono::high_resolution_clock::now();

                    cudaMemcpy(&solved_cpu, solved, sizeof(bool), cudaMemcpyDeviceToHost);
                    if (solved_cpu)
                    {
                        time_taken += std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
                        solve_cnt++;
                    }

                    // for
                    // cudaMemcpy(mask, new_grid, n * n * sizeof(int), cudaMemcpyDeviceToHost);
                    // print_grid(mask, n);
                    // int *test_empty = empty_space + 6 * n * n * sizeof(int);
                    // cudaMemcpy(mask, test_empty, n * n * sizeof(int), cudaMemcpyDeviceToHost);
                    // print_grid(mask, n);

                    cudaFree(new_grid);
                    cudaFree(old_grid);
                    cudaFree(empty_space);
                    cudaFree(empty_cnt);
                    cudaFree(grid_index);
                    cudaFree(solved);
                    cudaFree(answer_cuda);
                }
                printf("Order: %d, level: %d, grim %d, block: %d, drop_rate: %f, solved rate %f time taken: %f\n", root, level, grim_dim, block_dim, drop_rate, solve_cnt / 10.0, time_taken / solve_cnt);
            }
        }
    }
    free(grid);
    free(mask);
    free(answer);
    return 0;
}