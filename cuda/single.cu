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

const int bfs_size = 1024;
// nvcc single.cu -o single.o
// Function to prlong the Sudoku grid
void read_grid(int *grid, int n, string filename)
{
    ifstream file(filename);
    if (file.is_open())
    {
        for (int i = 0; i < n * n; i++)
        {
            file >> grid[i];
        }
        file.close();
    }
    else
    {
        cout << "Error: Unable to open input file." << endl;
    }
}

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
// Function to check if a number is safe to be placed in a cell
bool is_safe(int *grid, int n, int row, int col, int num)
{
    int idx;
    // Check if the number is already in the row
    for (int i = 0; i < n; i++)
    {
        idx = row * n + i;
        if (grid[idx] == num)
        {
            return false;
        }
    }
    // Check if the number is already in the column, can be optimized by data locality when size increase
    for (int i = 0; i < n; i++)
    {
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
void build_sudoku(int *grid, int n)
{
    std::mt19937 rng(42);
    int root = sqrt(n);
    for (int i = 0; i < n; i += root)
    {
        int num = 1;
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
}

__global__ void
cudaBFS(int *old_grid, int *new_grid, int grid_total, int *grid_index, int *empty_space, int *empty_cnt, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("index %d\n", *grid_index);
    // printf("hello from thread %d of block %d\n", threadIdx.x, blockIdx.x);
    while (index < grid_total)
    {
        // printf("index %d of %d grid_total\n", index, grid_total);
        bool found = false;
        int start = index * n * n;
        // from the start of the grid to find next avaiable space
        for (int i = start; i < (start + n * n) && !found; i++)
        {
            // printf("i %d\n", i);
            if (old_grid[i] != 0)
            {
                continue;
            }
            found = true;
            int row = (i - start) / n;
            int col = (i - start) % n;
            // printf("row %d col %d\n", row, col);
            // check avaible number
            for (int ava = 1; ava <= n; ava++)
            {
                // printf("ava %d\n", ava);
                bool avaible = true;
                // check row and col
                for (int j = 0; j < n; j++)
                {
                    if (old_grid[start + row * n + j] == ava)
                    {
                        avaible = false;
                        // printf("r row %d col %d ava %d avaible %d this value %d\n", row, col, ava, avaible, old_grid[start + row * n + j]);
                        break;
                    }
                    if (old_grid[start + j * n + col] == ava)
                    {
                        avaible = false;
                        // printf("c row %d col %d ava %d avaible %d this value %d\n", row, col, ava, avaible, old_grid[start + j * n + col]);
                        break;
                    }
                }

                // check board
                int root = sqrtf(n);
                int box_row = row - row % root;
                int box_col = col - col % root;
                for (int i = box_row; i < box_row + root; i++)
                {
                    for (int j = box_col; j < box_col + root; j++)
                    {
                        int idx = start + i * n + j;
                        if (old_grid[idx] == ava)
                        {
                            avaible = false;
                            // printf("grid row %d col %d ava %d avaible %d grid val %d \n", row, col, ava, avaible, old_grid[idx]);
                            break;
                        }
                    }
                }
                if (!avaible)
                {
                    continue;
                }
                // if avaible, add to new grid
                int next_grid_index = atomicAdd(grid_index, 1);
                // printf("grid_index %d\n", grid_index);
                if (next_grid_index >= bfs_size)
                {
                    // printf("row %d col %d val %d next_grid_index %d\n", row, col, ava, next_grid_index);
                    return;
                }
                int empty_index = 0;
                // This step maybe refact to using old gird index information
                for (int r = 0; r < n; r++)
                {
                    for (int c = 0; c < n; c++)
                    {
                        // printf("r %d c %d\n", r, c);
                        new_grid[next_grid_index * n * n + r * n + c] = old_grid[index * n * n + r * n + c];

                        // calculate the new grid empty space.
                        if (old_grid[index * n * n + r * n + c] == 0 && (r != row || c != col))
                        {
                            empty_space[next_grid_index * n * n + empty_index] = r * n + c;
                            empty_index++;
                        }
                    }
                }
                // printf("row %d col %d val %d empty index %d, next_grid_index %d thread %d of block %d\n", row, col, ava, empty_index, next_grid_index, threadIdx.x, blockIdx.x);
                empty_cnt[next_grid_index] = empty_index;
                // printf("next_grid_index %d empty_index %d\n", next_grid_index, empty_index);
                new_grid[next_grid_index * n * n + row * n + col] = ava;
            }
        }
        index += blockDim.x * gridDim.x;
    }
    // *grid_index = index;
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
    // printf("row %d col %d num %d\n", row, col, num);
    // check row and col
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
        // printf("rc row %d col %d num %d\n", row, col, num);
    }

    // check board
    // for (int i = 0; i < sqrtf(n); i++)
    // {
    //     for (int j = 0; j < sqrtf(n); j++)
    //     {
    //         if (grid[(row / n + i) * n + (col / n + j)] == num && (row / n + i) * n + (col / n + j) != index)
    //         {
    //             return false;
    //         }
    //     }
    // }
    // check board
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
    // printf("grid row %d col %d num %d\n", row, col, num);
    return true;
}

__global__ void cudaBackTrack(int *grid, int grid_total, int *emptySpaces, int *empty_cnt, bool *finished, int *solved, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("finsihed %d BT dealing with index %d of %d grid_total thread %d of block %d\n", (*finished), index, grid_total, threadIdx.x, blockIdx.x);

    // bool finished_now = *finished;
    while (!(*finished) && index < grid_total)
    {

        // printf("dealing with index %d of %d grid_total thread %d of block %d\n", index, grid_total, threadIdx.x, blockIdx.x);
        int *grid_now, *empty_now, empty_cnt_now;
        int empty_index = 0;
        grid_now = grid + index * n * n;
        empty_now = emptySpaces + index * n * n;
        empty_cnt_now = empty_cnt[index];
        // printf("empty_index %d empty cnt now = %d, empty position now:%d, index %d of %d grid_total\n", empty_index, empty_cnt_now, empty_now[empty_index], index, grid_total);
        while (empty_index < empty_cnt_now)
        {
            if (*finished)
            {
                printf("Exit cause finished by others; index %d finished now %d tread %d of block %d\n", index, *finished, threadIdx.x, blockIdx.x);
                return;
            }

            if (empty_index < 0)
            {
                // printf("dealing empty index %d of empty cnt %d with index %d of %d grid_total thread %d of block %d\n", empty_index, empty_cnt_now, index, grid_total, threadIdx.x, blockIdx.x);
                break;
            }
            grid_now[empty_now[empty_index]]++;
            if (is_safec(grid_now, n, empty_now[empty_index]))
            {
                // printf("empty_index %d of solved empty cnt %d value is %d\n hello from thread %d of block %d\n", empty_index, empty_cnt_now, grid_now[empty_now[empty_index]], threadIdx.x, blockIdx.x);
                empty_index++;
            }
            else
            {
                if (grid_now[empty_now[empty_index]] >= n)
                {
                    grid_now[empty_now[empty_index]] = 0;
                    empty_index--;
                }
            }
        }
        if (empty_index == empty_cnt_now)
        {
            // printf("now is empty_index == empty_cnt_now\n");
            // printf("dealing empty index %d of empty cnt %d with index %d of %d grid_total thread %d of block %d\n", empty_index, empty_cnt_now, index, grid_total, threadIdx.x, blockIdx.x);
            // finished_now = true;
            *finished = true;
            for (int i = 0; i < n * n; i++)
            {
                solved[i] = grid_now[i];
            }
            // printf("solved. finished_now %d, finish %d\n", finished_now, *finished);
        }

        index += blockDim.x * gridDim.x;
        // printf("index %d finished next index %d of grid total %d and return %d \n", index - blockDim.x * gridDim.x, index, grid_total, (!(*finished) && index < grid_total));
    }
    // *finished = finished_now;
    // printf("break the loop with index %d thread  %d of block %d \n", index, threadIdx.x, blockIdx.x);
}
void callCudaBFS(int gridDim, int blockDim, int *old_grid, int *new_grid, int grid_total, int *grid_index, int *empty_space, int *empty_cnt, int n)
{
    cudaBFS<<<gridDim, blockDim>>>(old_grid, new_grid, grid_total, grid_index, empty_space, empty_cnt, n);
}

void callCudaBackTrack(int gridDim, int blockDim, int *grid, int grid_total, int *emptySpaces, int *empty_cnt, bool *finished, int *solved, int n)
{
    cudaBackTrack<<<gridDim, blockDim>>>(grid, grid_total, emptySpaces, empty_cnt, finished, solved, n);
}
int main()
{
    // if (argc < 6)
    // {
    //     printf("Usage: ./sequential <int:n> <string:input_file> <string:answer_file> <int:gridDim> <int:blockDim> \n");
    //     abort();
    // }
    // int root = atoi(argv[1]);
    // string input_filename = argv[2];
    // string answer_filename = argv[3];
    // int gridDim = atoi(argv[4]);
    // int blockDim = atoi(argv[5]);
    int root = 4;
    // string input_filename = "../9.txt";
    // string answer_filename = "../9_answer.txt";
    int gridDim = 4;
    int blockDim = 32;
    float mask_rate = 0.5;
    int n = root * root;
    int *grid = (int *)malloc(n * n * sizeof(int));
    // read_grid(grid, n, input_filename);
    build_sudoku(grid, n);
    cout << "Input grid: " << endl;
    print_grid(grid, n);
    // Prepare for the gpu parallel
    int *new_grid, *old_grid;
    int *empty_space, *empty_cnt;
    int *grid_index;
    int grid_total = 1;

    int bfs_array_size = bfs_size * n * n * sizeof(int); // initial bfs search 128 grid
    int *grid_print = (int *)malloc(bfs_array_size);
    // cuda allocate memory
    cudaMalloc(&new_grid, bfs_array_size);
    cudaMalloc(&old_grid, bfs_array_size);
    cudaMalloc(&empty_space, bfs_array_size);
    cudaMalloc(&empty_cnt, bfs_array_size / n / n + 1);
    cudaMalloc(&grid_index, sizeof(int));

    // initialize the grid to 0
    cudaMemset(grid_index, 0, sizeof(int));
    cudaMemset(new_grid, 0, bfs_array_size);
    cudaMemset(old_grid, 0, bfs_array_size);
    cudaMemset(empty_space, 0, bfs_array_size);
    cudaMemset(empty_cnt, 0, bfs_array_size / n / n + 1);

    cudaMemcpy(old_grid, grid, n * n * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(new_grid, grid, n * n * sizeof(int), cudaMemcpyHostToDevice);
    Timer t;
    cudaDeviceSynchronize();
    t.tic();

    callCudaBFS(gridDim, blockDim, old_grid, new_grid, grid_total, grid_index, empty_space, empty_cnt, n);
    cudaDeviceSynchronize();
    int iter = 0;
    while (grid_total < bfs_size)
    {
        if (grid_total == 0)
        {
            printf("error grid total == 0");
            return 0;
        }
        cudaMemcpy(&grid_total, grid_index, sizeof(int), cudaMemcpyDeviceToHost);
        printf("grid_total: %d\n", grid_total);
        cudaMemset(grid_index, 0, sizeof(int));
        printf("grid_total: %d\n", grid_total);

        if (iter % 2 == 0)
        {
            callCudaBFS(gridDim, blockDim, new_grid, old_grid, grid_total, grid_index, empty_space, empty_cnt, n);
        }
        else
        {
            callCudaBFS(gridDim, blockDim, old_grid, new_grid, grid_total, grid_index, empty_space, empty_cnt, n);
        }
        iter += 1;
        printf("iter %d\n", iter);
    }
    cudaDeviceSynchronize();
    double tt = t.toc();
    printf("time to bfs board: %f\n", tt);

    // cudaMemcpy(&grid_total, &bfs_size, sizeof(int), cudaMemcpyDeviceToHost);
    if (iter % 2 == 1)
    {
        printf("in here change iter: %d\n", iter);
        // old_grid = new_grid;
        cudaMemcpy(new_grid, old_grid, bfs_array_size, cudaMemcpyDeviceToDevice);
    }

    // cudaMemcpy(grid_print, old_grid, bfs_array_size, cudaMemcpyDeviceToHost);
    grid_total = bfs_size;
    // for (int i = 0; i < bfs_size; i++)
    // {
    //     printf("grid %d\n", i);
    //     print_grid(grid_print + i * n * n, n);
    // }
    // cudaStreamSynchronize(0);
    bool *finished;
    int *solved;
    cudaMalloc(&solved, n * n * sizeof(int));
    cudaMalloc(&finished, sizeof(bool));

    cudaMemcpy(solved, grid, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(finished, false, sizeof(bool));

    cudaDeviceSynchronize();
    t.tic();

    callCudaBackTrack(gridDim, blockDim, new_grid, grid_total, empty_space, empty_cnt, finished, solved, n);
    cudaDeviceSynchronize();
    tt = t.toc();
    printf("time to back track: %f\n", tt);
    printf("print result\n");
    int *result = (int *)malloc(n * n * sizeof(int));
    memset(result, 0, n * n * sizeof(int));
    cudaMemcpy(result, solved, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    print_grid(result, n);

    /* Mask sudoku and solve it agian*/

    int mask_num = mask_rate * n * n;
    remove_standard(result, n, mask_num);
    printf("Mask sudoku with rate %f: \n", mask_rate);
    print_grid(result, n);
    memcpy(grid, result, n * n * sizeof(int));
    // print_grid(grid, n);
    // grid = result;

    // initialize the grid to 0
    // free(grid_index);
    // int *grid_index;
    // cudaMalloc(&grid_index, sizeof(int));
    cudaMemset(grid_index, 0, sizeof(int));
    cudaMemset(new_grid, 0, bfs_array_size);
    cudaMemset(old_grid, 0, bfs_array_size);
    cudaMemset(empty_space, 0, bfs_array_size);
    cudaMemset(empty_cnt, 0, bfs_array_size / n / n + 1);
    cudaMemcpy(old_grid, result, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(new_grid, result, n * n * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    t.tic();
    grid_total = 1;
    callCudaBFS(gridDim, blockDim, old_grid, new_grid, grid_total, grid_index, empty_space, empty_cnt, n);
    printf("grid_total:%d\n", grid_total);
    // printf("grid_index:%d\n", *grid_index);
    // cudaMemcpy(grid_print, new_grid, bfs_array_size, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < bfs_size; i++)
    // {
    //     printf("grid %d\n", i);
    //     print_grid(grid_print + i * n * n, n);
    // }
    iter = 0;
    cudaDeviceSynchronize();
    while (grid_total < bfs_size)
    {
        if (grid_total == 0)
        {
            printf("error grid total == 0");
            return 0;
        }
        cudaMemcpy(&grid_total, grid_index, sizeof(int), cudaMemcpyDeviceToHost);
        printf("grid_total: %d\n", grid_total);
        cudaMemset(grid_index, 0, sizeof(int));
        printf("grid_total: %d\n", grid_total);

        if (iter % 2 == 0)
        {
            callCudaBFS(gridDim, blockDim, new_grid, old_grid, grid_total, grid_index, empty_space, empty_cnt, n);
        }
        else
        {
            callCudaBFS(gridDim, blockDim, old_grid, new_grid, grid_total, grid_index, empty_space, empty_cnt, n);
        }
        iter += 1;
        printf("iter %d\n", iter);
    }
    cudaDeviceSynchronize();
    tt = t.toc();
    printf("time to bfs board: %f\n", tt);

    // cudaMemcpy(&grid_total, &bfs_size, sizeof(int), cudaMemcpyDeviceToHost);
    if (iter % 2 == 1)
    {
        printf("in here change iter: %d\n", iter);
        // old_grid = new_grid;
        cudaMemcpy(new_grid, old_grid, bfs_array_size, cudaMemcpyDeviceToDevice);
    }

    // cudaMemcpy(grid_print, old_grid, bfs_array_size, cudaMemcpyDeviceToHost);
    grid_total = bfs_size;
    // for (int i = 0; i < bfs_size; i++)
    // {
    //     printf("grid %d\n", i);
    //     print_grid(grid_print + i * n * n, n);
    // }
    // cudaStreamSynchronize(0);

    cudaMemcpy(solved, grid, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(finished, false, sizeof(bool));
    cudaDeviceSynchronize();
    t.tic();

    callCudaBackTrack(gridDim, blockDim, new_grid, grid_total, empty_space, empty_cnt, finished, solved, n);
    cudaDeviceSynchronize();
    tt = t.toc();
    printf("time to back track: %f\n", tt);
    printf("print result\n");

    memset(result, 0, n * n * sizeof(int));
    cudaMemcpy(result, solved, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    print_grid(result, n);

    free(grid);
    free(result);
    free(grid_print);
    cudaFree(new_grid);
    cudaFree(old_grid);
    cudaFree(empty_space);
    cudaFree(empty_cnt);
    cudaFree(grid_index);
    cudaFree(finished);
    cudaFree(solved);
    return 0;
}