#define SIZE 16
#define MINIGRIDSIZE 4

int **readInput(char *);
int isValid(int **, int **);
int **solveSudoku(int **);
void print_grid(int **grid);
bool Sudoku_seq(int **grid, int row, int col);
void build_sudoku(int **grid);
void remove_unique(int **grid, int cnt);
void remove_standard(int **grid,int cnt);
bool isok(int **grid, int row, int col, int num);
int eliminations(int **grid);
void twins(int **grid);
void loneranger(int **grid);
void stackinit(int** Grid, struct grid* curr);
void push(struct stack* newelement);
//struct stack* pop();
//struct stack* stackalloc(int index_i, int index_j);
void deletestack(struct stack* mystack);
void assign_possiblevalues(int value, struct grid* curr, int i, int j);
void Sudoku();
int elimination(struct grid* curr);
