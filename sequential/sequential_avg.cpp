#include <stdio.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cstring>

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

bool solve_sudoku(int *grid, int n, int row, int col)
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
        return solve_sudoku(grid, n, row, col + 1);
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
            if (solve_sudoku(grid, n, row, col + 1))
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
    solve_sudoku(grid, n, 0, 0);
}

void remove_standard(int *grid, int n, int cnt)
{
    int idx;
    int row;
    int col;
    // srand(42);
    // srand(time(NULL));
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

// int cnt_start_board(int *grid, int n, int level, int row, int col)
// {
//     if (level <= 0)
//     {
//         return 1;
//     }
//     if (col == n)
//     {
//         row++;
//         col = 0;
//     }
//     int idx = row * n + col;
//     if (grid[idx] != 0)
//     {
//         return cnt_start_board(grid, n, level, row, col + 1);
//     }
//     int res = 0;
//     for (int num = n; num > 0; num--)
//     {
//         // if (col == 20)
//         // cout << "row: " << row << " col: " << col << " num: " << num << endl;
//         if (is_safe(grid, n, row, col, num))
//         {
//             // cout << "row: " << row << " col: " << col << " num: " << num << endl;
//             // print_grid(grid, n);
//             // cout << endl;
//             grid[idx] = num;
//             res += cnt_start_board(grid, n, level - 1, row, col + 1);
//             grid[idx] = 0; // backtrack
//         }
//     }
//     return res;
// }

int main(int argc, char **argv)
{
    int rand_seed[] = {1, 6, 8, 13, 24, 32, 42, 52, 63, 71};
    float drop_rate_3[] = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
    float drop_rate_4[] = {0.2, 0.3, 0.4, 0.5, 0.6};
    int root, n;
    int *grid, *mask;
    double time_taken;
    int solve_cnt, rm_cnt;
    float drop_rate;

    // Test order 3
    root = 3;
    n = root * root;
    time_taken = 0;

    grid = (int *)malloc(n * n * sizeof(int));
    mask = (int *)malloc(n * n * sizeof(int));
    for (int i = 0; i < n * n; i++)
    {
        grid[i] = 0;
    }
    build_sudoku(grid, n);
    memcpy(mask, grid, n * n * sizeof(int));

    for (int i = 0; i < 6; i++)
    {
        solve_cnt = 0;
        drop_rate = drop_rate_3[i];
        rm_cnt = n * n * drop_rate;

        for (int j = 0; j < 10; j++)
        {
            memcpy(mask, grid, n * n * sizeof(int));
            srand(rand_seed[j]);
            remove_standard(mask, n, rm_cnt);
            // print_grid(mask, n);
            // // Test cnt start board
            // printf("Avaiable board for level 1 is %d \n", cnt_start_board(mask, n, 1, 0, 0));
            auto t_start = std::chrono::high_resolution_clock::now();
            if (solve_sudoku(mask, n, 0, 0))
            {
                auto t_end = std::chrono::high_resolution_clock::now();
                solve_cnt++;
                time_taken += std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
            }
        }
        printf("Order: %d, Drop rate: %f, Solve rate: %f, Avg time: %f\n", n, drop_rate, solve_cnt / 10.0, time_taken / solve_cnt);
    }
    free(grid);
    free(mask);

    // Test order 4
    root = 4;
    n = root * root;
    time_taken = 0;

    grid = (int *)malloc(n * n * sizeof(int));
    mask = (int *)malloc(n * n * sizeof(int));
    for (int i = 0; i < n * n; i++)
    {
        grid[i] = 0;
    }
    build_sudoku(grid, n);
    memcpy(mask, grid, n * n * sizeof(int));

    for (int i = 0; i < 5; i++)
    {
        solve_cnt = 0;
        drop_rate = drop_rate_4[i];
        rm_cnt = n * n * drop_rate;
        for (int j = 0; j < 10; j++)
        {
            memcpy(mask, grid, n * n * sizeof(int));
            srand(rand_seed[j]);
            remove_standard(mask, n, rm_cnt);
            auto t_start = std::chrono::high_resolution_clock::now();
            if (solve_sudoku(mask, n, 0, 0))
            {
                auto t_end = std::chrono::high_resolution_clock::now();
                solve_cnt++;
                time_taken += std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
            }
        }
        printf("Order: %d, Drop rate: %f, Solve rate: %f, Avg time: %f\n", n, drop_rate, solve_cnt / 10.0, time_taken / solve_cnt);
    }
    return 0;
}