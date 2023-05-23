//inclusion files
#include <stdio.h>
#include <stdlib.h>
#include "sudoku.h"
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <iostream>
#include <random>

#define UNASSIGNED 0

//global variables
bool possiblevalues[SIZE][SIZE][SIZE+1];
int **zerogrid;
int stack_sz;
bool solved;
struct stack* solvedgrid;
int** originalGrid ;
struct stack* top;
struct stack* bottom;

//structures
struct grid
{
    int** my_grid;
    bool** assigned; 
    int** possiblevalues;     
};
struct stack
{
    struct grid* Grid;
    struct stack* next;
    int row;
    int col;
};

//functions
//void print_grid(int **grid);
//bool Sudoku_seq(int **grid, int row, int col);
//void build_sudoku(int **grid, int SIZE);
//void remove_unique(int **grid, int SIZE, int cnt);
//void remove_standard(int **grid, int SIZE, int cnt);
//bool isok(int **grid, int row, int col, int num);
//int eliminations(int **grid);
//void twins(int **grid);
//void loneranger(int **grid);
//void stackinit(int** Grid, struct grid* curr);
//void push(struct stack* newelement);
struct stack* pop();
struct stack* stackalloc(int index_i, int index_j);
//void deletestack(struct stack* mystack);
//void assign_possiblevalues(int value, struct grid* curr, int i, int j);
//void Sudoku();
//int elimination(struct grid* curr);


void print_grid(int **grid)
{
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            printf("%d ", grid[i][j]);
        }
        printf("\n");
    }
}

//brute force sudoku without parallel
bool Sudoku_seq(int **grid, int row, int col)
{
     if (row == SIZE - 1 && col == SIZE) return true;
     if (col ==SIZE){
        row ++;
        col = 0;
     }
     if(grid[row][col] != 0) return Sudoku_seq(grid, row, col+1);
     for (int num = SIZE; num > 0; num--){
        if (isok(grid, row, col, num)){
            grid[row][col] = num;
            if (Sudoku_seq(grid, row, col+1)) return true;
            grid[row][col] = 0; //backtrack
        }
     }
     return false;
}

//sudoku builder
void build_sudoku(int **grid)
{
    std::mt19937 rng(6);
    for (int i = 0; i < SIZE; i += MINIGRIDSIZE){
        int num = 1;
        for(int row = 0; row < MINIGRIDSIZE; row++){
            for(int col = 0; col < MINIGRIDSIZE; col++){
                while(!isok(grid, row + i, col+i, num)){
                    num = std::uniform_int_distribution<int>{1, SIZE}(rng);
                }
                grid[row+i][col+i] = num;
            }
        }
    }
    Sudoku_seq(grid, 0,0);
}

void remove_unique(int **grid,int cnt)
{
    int row,col,solution;
    srand(time(NULL));
    while (cnt >0){
        col = rand() %SIZE;
        row = rand() %SIZE;
        cnt--;
        solution = 0;
        int temp = grid[row][col];
        if (temp != 0)
        {
            grid[row][col] = 0;
            for(int num =1; num <= SIZE; num++)
            {
                if(isok(grid, row, col, num) && Sudoku_seq(grid, row, col))
                {
                    solution++;
                }
                if (solution >=2)
                {
                    grid[row][col] = temp;
                    cnt--;
                    break;
                }
            }
        }
    }
}

void remove_standard(int **grid, int cnt)
{
    int row, col, solution;
    srand(time(NULL));
    while (cnt > 0)
    {
        col = rand()%SIZE;
        row = rand()%SIZE;
        int temp = grid[row][col];
        if (temp != 0)
        {
            grid[row][col] = 0;
            cnt--;
        }
    }
}

// check if the variable num is possible at that row and column in  sudoku(grid)
bool isok(int **grid, int row, int col, int num)
{
    //printf("**in isSafe of *** %d %d %d\n",row,col,num);
    int myrow,mycol;
    

    bool alreadyinrow=false;
    // check every row
    for (mycol = 0; mycol < SIZE; mycol++)
        if (grid[row][mycol] == num)
        {
            alreadyinrow=true;
            mycol=SIZE;
        }


    bool alreadyincol=false;
    //check every column
     for (myrow = 0; myrow < SIZE; myrow++)
        if (grid[myrow][col] == num)
        {
            alreadyincol=true;
            myrow=SIZE;
        }

    bool alreadyinbox=false;
    //check every box
    int boxStartRow=row - row % MINIGRIDSIZE;
    int boxStartCol= col - col % MINIGRIDSIZE;
    for (myrow = 0; myrow < MINIGRIDSIZE; myrow++)
        for (mycol = 0; mycol < MINIGRIDSIZE; mycol++)
            if (grid[myrow+boxStartRow][mycol+boxStartCol] == num)
            {
                alreadyinbox=true;
                myrow=MINIGRIDSIZE;
                mycol=MINIGRIDSIZE;
            }
            //return true if it is possible at that particular row and column
    return (!alreadyinrow && !alreadyincol && !alreadyinbox);
}


// Brute force function
//start taking an element from the stack and then run them each parallely
void Sudoku()
{
    //run each of the stack element parallely
    #pragma omp parallel
    {
        // do until all the stack elements are emptied
        while ( top && !solved)
        {
            struct stack* curr = NULL;
            // Lock the stack pop so that only one thread can pop the data structure
            #pragma omp critical
            {
                if (top!=NULL)
                {
                    curr = pop();
                }
            }
            bool cond=false;
            int i,j;
            // find the next grid box element which isn't assigned
            for (i = curr->row ; i < SIZE; i++)
            {
                for (j = 0 ; j <SIZE; j++){
                    if ( curr->Grid->assigned[i][j] == 0)
                    {
                        curr->row = i;
                        curr->col = j;
                        cond=true;
                        i=SIZE;
                        j=SIZE;
                    }
                }
            }
            // if there exist the unassigned numbers in the sudoku puzzle execute
            if (curr && cond)
            {
                int num;
                // note all the possible values in the test
                int test = curr->Grid->possiblevalues[curr->row][curr->col];

                // for each possible value in the box element grid enter to the loop
                for ( num = 1; num <= SIZE ; num++)
                {
                    if (!solved && (test & (1 << (num - 1))))
                    {
                        // create a new data structure for each of the possible values by assigning the values to that box element
                        struct stack* mystack = stackalloc(curr->row, curr->col);
                        int i,j;
                        // copy the values of the parent grid data structure to the child data structure
                        for ( i =0 ; i < SIZE ; i++)
                        {
                            for (j = 0 ; j < SIZE ; j++)
                            {
                                mystack->Grid->assigned[i][j] = curr->Grid->assigned[i][j] ;
                                mystack->Grid->my_grid[i][j] = curr->Grid->my_grid[i][j];
                                mystack->Grid->possiblevalues[i][j]= curr->Grid->possiblevalues[i][j];
                            }
                        }
                        //  now assign the newly possible elements for each of the box element
                        assign_possiblevalues(num, mystack->Grid,mystack->row, mystack->col);
                        if (!solved && elimination(mystack->Grid) == 0){
                            if (!solved &&  isValid(zerogrid,mystack->Grid->my_grid))
                            {
                                // if the sudoku grid is solved then make the solved value to true
                                #pragma omp critical
                                {
                                    if (!solved)
                                    {
                                        solvedgrid = mystack;
                                        solved = true;
                                    }
                                }
                            }
                            // push an element to the stack bottom and while pushing lock it so that no other thread 
                            #pragma omp critical
                            {
                                push(mystack);
                            }
                        }
                    }
                }
            }
            // delete the stack element after the child process are created
            if ( curr!=NULL)
                deletestack(curr);
        }
    }
    return;
}

// assigning values to the stack
void stackinit(int** Grid, struct grid* curr)
{
    int i,j;

    // int values to store the possible values in the virtual binary form
    int* row = (int*) malloc (SIZE*sizeof (int));
    int* column = (int*) malloc (SIZE*sizeof (int));
    int* box = (int*) malloc (SIZE*sizeof (int));

    // initialising each with 0
    for(i=0;i<SIZE;i++){
            row[i]=0;
            column[i]=0;
            box[i]=0;
    }

    // loop to find the virtual binary values of columns, rows and box
    for( i = 0; i < SIZE ; i++)
    {
        for ( j = 0 ; j < SIZE ; j++)
        {
            // assigning the values to the new data structure
            curr->my_grid[i][j] = Grid[i][j];
            // enter the loop if we have an assigned number
            if ( Grid[i][j] != 0){
                // find the virtual binary value of that particular number
                int k = 1 << ( Grid[i][j] - 1);
                curr->assigned[i][j] = 1;
                // adding it to the virtual row and column and box i.e saying that it is occured in that box,row,column
                row[i] |= k;
                column[j] |= k;
                box[(i/MINIGRIDSIZE)*MINIGRIDSIZE + j/MINIGRIDSIZE] |= k;
            }
        }
    }

    // now going to each of the subbox box and assigning the possible values of that particular subbox
    for( i = 0; i < SIZE ; i++)
    {
        for ( j = 0 ; j < SIZE ; j++)
        {
            // if it isn't assigned then find the possible values using the row,column,box value of that particular column values
            if ( curr->assigned[i][j] == 0)
            {
                // finally store the values that are possible in that particular subbox
                int temp = row[i] | column[j] | box[(i/MINIGRIDSIZE)*MINIGRIDSIZE + j/MINIGRIDSIZE] ;
                temp = ~temp;

                // and now filrtering them to make it to the required digits
                temp &= ((1 << SIZE) - 1);
                curr->possiblevalues[i][j] = temp;
            }
            else
            {
                //if it is assigned the make the possible to 0
                curr->possiblevalues[i][j] = 0;
            }
        }
    }
}

// to push the element to stack from the bottom
void push(struct stack* newelement)
{

    if ( !top)
    {
        //no elements in the stack
        top = newelement;
        top->next = NULL;
        bottom = newelement;
    }
    else
    {
        bottom->next = newelement;
        bottom = newelement;
    }
}

// pop the data structure at the top
struct stack* pop()
{
    if (top)
    {
        struct stack* ret = top;
        top = top->next;
        return ret;
    }
    else
    {
        printf("error:trying to pop off from an empty stack\n");
        exit(0);
    }
}

// Dynamic allocation for the data
struct stack* stackalloc(int myrow, int mycol)
{
    struct stack* mystack;
    //mystack = malloc(sizeof(struct stack));
    mystack = (stack *)malloc(sizeof(struct stack));
    mystack->Grid = (grid *)malloc(sizeof(struct grid));
    int row,col;
    mystack->Grid->my_grid = (int**)malloc(SIZE*sizeof(int*));
    for (row=0; row<SIZE; row++)
    {
        mystack->Grid->my_grid[row] = (int*) malloc (SIZE * sizeof (int));
    }
    mystack->Grid->possiblevalues = (int**)malloc(SIZE*sizeof(int*));
    for (row=0; row<SIZE; row++)
    {
        mystack->Grid->possiblevalues[row] = (int*) malloc (SIZE*sizeof (int));
        for (col=0;col<SIZE;col++){
            mystack->Grid->possiblevalues[row][col] = 0;
        }
    }
    mystack->Grid->assigned = (bool**)malloc(SIZE*sizeof(bool*));
    for (row=0; row<SIZE; row++)
    {
        mystack->Grid->assigned[row] = (bool*) malloc (SIZE*sizeof (bool));
        for (col=0;col<SIZE;col++)
        {
            //initialise the grid values to zero
            mystack->Grid->assigned[row][col] = 0;
        }
    }
    mystack->next = NULL;
    mystack->row = myrow;
    mystack->col = mycol;
    return mystack;
}

// to free the stack the data structure
void deletestack(struct stack* mystack){
    int i;
    for (i = 0; i < SIZE; i++) 
    {
        free(mystack->Grid->my_grid[i]);
        free(mystack->Grid->possiblevalues[i]);
        free(mystack->Grid->assigned[i]);
    }
    free(mystack->Grid->my_grid);
    free(mystack->Grid->possiblevalues);
    free(mystack->Grid->assigned);
    free(mystack);
}

// the trick implementation of twins
void twins(int **grid)
{
    
    for (int row = 0; row < SIZE; row++)
    {
        for (int col=0; col<SIZE; col++)
        {
              int nonzero=0;
              int t;
              for(t=1;t<SIZE+1;t++)
              {
                if(possiblevalues[row][col][t])nonzero++;
              }
              if(nonzero>=2)
              {
                    //atleast two possible values
                    //printf("two possible values for row,col %d,%d\n",row,col);
                    //select all possible pairs of twins from these possible values
                    int nonzero_possible[SIZE];
                    int nonzero_index=0;
                    int j;
                    for(j=1;j<=SIZE;j++){
                         if(possiblevalues[row][col][j]){
                            nonzero_possible[nonzero_index]=j;
                            nonzero_index++;
                         }
                     }
                    /* printf("nonzero_possible values of row,col %d,%d\n",row,col);
                     for(j=0;j<nonzero_index;j++){
                         printf("%d ",nonzero_possible[j]);
                     }
                     printf("\n");
                     */
                     int p,q;
                    for(p=0;p<nonzero_index-1;p++){
                        int twin1=nonzero_possible[p];
                        for(q=p+1;q<nonzero_index;q++){
                            int twin2=nonzero_possible[q];
                           // printf("twins:%d,%d\n",twin1,twin2 );
                            int index=0;
                            int nxtcol;
                            int uniquecol;
                            for(nxtcol=0;nxtcol<SIZE;nxtcol++){
                                 if(nxtcol!=col){
                                     //printf("checking col %d\n",nxtcol);
                                     if(possiblevalues[row][nxtcol][twin1] && possiblevalues[row][nxtcol][twin2]){
                                        //twins exist in another cell
                                        //printf("twins found in col %d\n",nxtcol);
                                        index++;
                                        uniquecol=nxtcol;
                                     }
                                     if(index>1)nxtcol=SIZE;
                                 }
                            }
                            if(index==1){
                                bool occur=false;
                                for(nxtcol=0;nxtcol<SIZE;nxtcol++){
                                 if(nxtcol!=col && nxtcol!=uniquecol){
                                     //printf("checking col %d\n",nxtcol);
                                     if(possiblevalues[row][nxtcol][twin1] || possiblevalues[row][nxtcol][twin2]){
                                        //twins exist in another cell
                                        occur=true;  
                                        nxtcol=SIZE;
                                     }
                                    }
                                  }
                                if(!occur){
                                        //printf("yay found twin in row,col %d ,%d\n",row,uniquecol);
                                        //only one other cell that contains same twins
                                        //remove other possible values from these two cells
                                        #pragma omp parallel for
                                        for(int i=1;i<=SIZE;i++){
                                            if(i!=twin1 && i!=twin2){
                                                 possiblevalues[row][col][i]=false;  
                                                 possiblevalues[row][uniquecol][i]=false;  
                                            }
                                        }
                                        
                                }

                            }
                        }
                    }
              }
        }
     }
 

   
    for (int col = 0; col < SIZE; col++){
        for (int row=0; row<SIZE; row++){
              int nonzero=0;
              int t;
              for(t=1;t<SIZE+1;t++){
                if(possiblevalues[row][col][t])nonzero++;
              }
              if(nonzero>=2){
                    //atleast two possible values
                    //printf("two possible values for row,col %d,%d\n",row,col);
                    //select all possible pairs of twins from these possible values
                    int nonzero_possible[SIZE];
                    int nonzero_index=0;
                    int j;
                    for(j=1;j<=SIZE;j++){
                         if(possiblevalues[row][col][j]){
                            nonzero_possible[nonzero_index]=j;
                            nonzero_index++;
                         }
                     }
                    /* printf("nonzero_possible values of row,col %d,%d\n",row,col);
                     for(j=0;j<nonzero_index;j++){
                         printf("%d ",nonzero_possible[j]);
                     }
                     printf("\n");
                     */
                     int p,q;
                    for(p=0;p<nonzero_index-1;p++){
                        int twin1=nonzero_possible[p];
                        for(q=p+1;q<nonzero_index;q++){
                            int twin2=nonzero_possible[q];
                           // printf("twins:%d,%d\n",twin1,twin2 );
                            int index=0;
                            int nxtcol;
                            int uniquecol;
                            for(nxtcol=0;nxtcol<SIZE;nxtcol++){
                                 if(nxtcol!=col){
                                     //printf("checking col %d\n",nxtcol);
                                     if(possiblevalues[row][nxtcol][twin1] && possiblevalues[row][nxtcol][twin2]){
                                        //twins exist in another cell
                                        //printf("twins found in col %d\n",nxtcol);
                                        index++;
                                        uniquecol=nxtcol;
                                     }
                                     if(index>1)nxtcol=SIZE;
                                 }
                            }
                            if(index==1){
                                bool occur=false;
                                for(nxtcol=0;nxtcol<SIZE;nxtcol++){
                                 if(nxtcol!=col && nxtcol!=uniquecol){
                                     //printf("checking col %d\n",nxtcol);
                                     if(possiblevalues[row][nxtcol][twin1] || possiblevalues[row][nxtcol][twin2]){
                                        //twins exist in another cell
                                        occur=true;  
                                        nxtcol=SIZE;
                                     }
                                    }
                                  }
                                if(!occur){
                                        //printf("yay found twin in row,col %d ,%d\n",row,uniquecol);
                                        //only one other cell that contains same twins
                                        //remove other possible values from these two cells

                                        int i;
                                        #pragma omp parallel for
                                        for(i=1;i<=SIZE;i++){
                                            if(i!=twin1 && i!=twin2){
                                                 possiblevalues[row][col][i]=false;  
                                                 possiblevalues[row][uniquecol][i]=false;  
                                            }
                                        }
                                }

                            }
                        }
                    }
              }
        }
     }
 
    /*
    printf("************************GRID AFTER twins***********************\n");
    int i,j;
    for (i=0;i<SIZE;i++)
    {
        for (j=0;j<SIZE;j++)
            printf("%d ",grid[i][j]);
        printf("\n");
    }
    printf("*********************************************************\n");
   */
 

     
    return;

}


int eliminations(int **grid){ 
      
    int row,col;
    int eliminations=0;
    #pragma omp parallel for num_threads(SIZE*SIZE) collapse(2)
    for (row = 0; row < SIZE; row++){
        for (col = 0; col < SIZE; col++){
                if(grid[row][col]==UNASSIGNED)
                {
                    int i;
                    int index;
                    int num_possible=0;
                    for (i = 0; i <= SIZE; i++)
                    {
                         if(possiblevalues[row][col][i])
                         {
                               index=i;
                               num_possible++;
                          }
                          if(num_possible>1)
                            i=SIZE;
                    }
                    if(num_possible==1)
                    {
                      //  printf("only one possible value\n");
                        eliminations++;
                        grid[row][col]=index;
                        possiblevalues[row][col][index]=true;

                        #pragma omp parallel
                        {
                            #pragma omp sections
                            {

                                #pragma omp section
                                {
                                    #pragma omp parallel for
                                    for(int i=0;i<=SIZE;i++){
                                        if(i!=index)
                                            possiblevalues[row][col][i]=false;
                                    }
                                }

                                #pragma omp section
                                {

                                    #pragma omp parallel for
                                    for(int myrow=0;myrow<SIZE;myrow++){
                                        if(myrow!=row)
                                        possiblevalues[myrow][col][index]=false;
                                    }
                                }

                                #pragma omp section
                                {
                                    #pragma omp parallel for
                                    for(int mycol=0;mycol<SIZE;mycol++)
                                    {
                                        if(mycol!=col)
                                        possiblevalues[row][mycol][index]=false;
                                    }
                                }    


                                #pragma omp section
                                {
                                    int boxStartRow=row - row % MINIGRIDSIZE;
                                    int boxStartCol= col - col % MINIGRIDSIZE;
                                    #pragma omp parallel for
                                    for (int myrow = 0; myrow < MINIGRIDSIZE; myrow++)
                                          for (int mycol = 0; mycol < MINIGRIDSIZE; mycol++)
                                                 if (myrow!=row && mycol!=col)
                                                 {
                                                          possiblevalues[myrow+boxStartRow][mycol+boxStartCol][index]=false;
                                                 }
                                }
                            }             
                        }             
                    }
              }
        }
    }
    
   /* printf("************************GRID AFTER ELIMINATION***********************\n");
    int i,j;
    for (i=0;i<SIZE;i++)
    {
        for (j=0;j<SIZE;j++)
            printf("%d ",grid[i][j]);
        printf("\n");
    }
    printf("*********************************************************\n");
    */
    
    return eliminations;
}

void loneranger(int **grid){
    
    int row,col;
    for (row = 0; row < SIZE; row++)
    {
        int num;
        for(num=1;num<=SIZE;num++)
        {
             int index_row=0;
             int uniquecol;
             for (col = 0; col < SIZE; col++)
             {
                    if(possiblevalues[row][col][num])
                    {
                        uniquecol=col;
                        index_row++;
                    }
                if(index_row>1)
                    col=SIZE;
             }
             if(index_row==1)
             {
                //printf("loneranger found in row %d , %d , %d\n",row,uniquecol,num);
                grid[row][uniquecol]=num;
                possiblevalues[row][uniquecol][num]=true;
                
                #pragma omp parallel
                {
                    #pragma omp sections
                        {

                            #pragma omp section
                                {
                                    #pragma omp parallel for
                                    for(int i=0;i<=SIZE;i++)
                                    {
                                        if(i!=num)
                                            possiblevalues[row][uniquecol][i]=false;
                                    }
                                }
                            #pragma omp section
                                {
                                    #pragma omp parallel for
                                    for(int myrow=0;myrow<SIZE;myrow++)
                                    {
                                                if(myrow!=row)
                                                possiblevalues[myrow][uniquecol][num]=false;
                                    }
                                }

                            #pragma omp section
                                {
                                    #pragma omp parallel for
                                    for(int mycol=0;mycol<SIZE;mycol++)
                                    {
                                                if(mycol!=uniquecol)
                                                possiblevalues[row][mycol][num]=false;
                                    }
                                }
                                
                            #pragma omp section
                                {    

                                    int boxStartRow=row - row % MINIGRIDSIZE;
                                    int boxStartCol= uniquecol - uniquecol % MINIGRIDSIZE;
                                    #pragma omp parallel for
                                    for (int myrow = 0; myrow < MINIGRIDSIZE; myrow++)
                                        for (int mycol = 0; mycol < MINIGRIDSIZE; mycol++)
                                            if (myrow!=row && mycol!=uniquecol)
                                            {
                                                 possiblevalues[myrow+boxStartRow][mycol+boxStartCol][num]=false;
                                            }
                                }            
                        }
                }        
             }
             int index_col=0;
             int uniquerow;
             for (col = 0; col < SIZE; col++)
             {
                    if(possiblevalues[col][row][num])
                    {
                        uniquerow=col;
                        index_col++;
                    }
                if(index_col>1)
                    col=SIZE;
             }
             if(index_col==1)
             {
                //printf("loneranger found in column %d , %d , %d\n",uniquerow,row,num);
                grid[uniquerow][row]=num;
                possiblevalues[uniquerow][row][num]=true;
                
                #pragma omp parallel 
                {
                    #pragma omp sections
                    {
                        #pragma omp section
                        {
                            #pragma omp parallel for
                            for(int i=0;i<=SIZE;i++)
                            {
                                if(i!=num)
                                    possiblevalues[uniquerow][row][i]=false;
                            }
                        }
                        

                        #pragma omp section
                        {
                            #pragma omp parallel for
                            for(int myrow=0;myrow<SIZE;myrow++)
                            {
                                        if(myrow!=uniquerow)
                                            possiblevalues[myrow][row][num]=false;
                            }
                        }


                        #pragma omp section
                        {
                            #pragma omp parallel for
                            for(int mycol=0;mycol<SIZE;mycol++)
                            {
                                        if(mycol!=row)
                                            possiblevalues[uniquerow][mycol][num]=false;
                            }
                        }
                        #pragma omp section
                        {
                            
                            int boxStartRow=uniquerow - uniquerow % MINIGRIDSIZE;
                            int boxStartCol= row - row % MINIGRIDSIZE;
                           #pragma omp parallel for
                            for (int myrow = 0; myrow < MINIGRIDSIZE; myrow++)
                                for (int mycol = 0; mycol < MINIGRIDSIZE; mycol++)
                                    if (myrow!=uniquerow && mycol!=row)
                                    {
                                         possiblevalues[myrow+boxStartRow][mycol+boxStartCol][num]=false;
                                    }
                        }            
                               
                    }            
                }       
             }
        }
    }

    /*
    printf("************************GRID AFTER loneranger***********************\n");
    int i,j;
    for (i=0;i<SIZE;i++){
        for (j=0;j<SIZE;j++)
            printf("%d ",grid[i][j]);
        printf("\n");
    }
    printf("*********************************************************\n");
    */
    
    return;
}


//elimination for stack data structure
int elimination(struct grid* curr)
{
    
    int row,col;
    //
    for( row = 0; row < SIZE ; row++){
        for ( col = 0 ; col < SIZE ; col++){
            if ( !(curr->assigned[row][col])){
                if ( curr->possiblevalues[row][col] == 0 )
                    return -1; 
                else
                {
                    if ( (curr->possiblevalues[row][col] & (curr->possiblevalues[row][col] - 1)) == 0 ){
                        int k = 0;
                        do{
                            k++;
                            curr->possiblevalues[row][col]>>=1;
                        }while (curr->possiblevalues[row][col]);
                        assign_possiblevalues(k,curr,row,col);
                        col = SIZE;
                        row=-1;
                   }
                 }
            }
        }
    }
    
    return 0;
}

//assign possible values of the stack data structure
void init_possiblevalues(int **grid){
    //we can run this in parallel as each inside function don't depend on the other
    #pragma omp parallel for
    for (int row = 0; row < SIZE; row++)
    {
        for (int col = 0; col < SIZE; col++)
        {
            // initialise all the possible values to false
            for(int i=0;i<SIZE+1;i++)
            {
                    possiblevalues[row][col][i]=false;
            }
            // assign true to the assigned value
            if(grid[row][col] != UNASSIGNED)
            {
                 possiblevalues[row][col][grid[row][col]]=true;
            }
            else
            {   
                // for not assigned sub boxes check the isok function and then assign true to that
                for (int num = 1; num <= SIZE; num++)
                {
                        if(isok(grid,row,col,num))
                        {
                               possiblevalues[row][col][num]=true;
                        }
                 }
            }
        }
    }
}

// assign the possible value to that value in that row and column element
void assign_possiblevalues(int value, struct grid* curr, int row, int col)
{
    // assign the values and it's other data structures
    curr->my_grid[row][col] = value;
    curr->assigned[row][col] = 1;
    curr->possiblevalues[row][col] = 0;
    int temp = ~(1 << ( value - 1));
    int ind;
    //now deleting the possible value at the index of row column 
    for (ind = 0 ; ind < SIZE ; ind++)
    {

        curr->possiblevalues[row][ind] &= temp;
        curr->possiblevalues[ind][col] &= temp;
    }
   /* int myrow,mycol;
    int boxStartRow=row - row % MINIGRIDSIZE;
    int boxStartCol= col - col % MINIGRIDSIZE;
    for (myrow = 0; myrow < MINIGRIDSIZE; myrow++)
        for (mycol = 0; mycol < MINIGRIDSIZE; mycol++)
              curr->possiblevalues[myrow+boxStartRow][mycol+boxStartCol] &= temp;*/
            //deleting the paticular  box possibilities
    int boxStartRow = (row/MINIGRIDSIZE) * MINIGRIDSIZE;
    int boxStartCol = (col/MINIGRIDSIZE) * MINIGRIDSIZE;
    int i,j;
    for (i = boxStartRow; i < boxStartRow + MINIGRIDSIZE; i++)
        for (j = boxStartCol; j < boxStartCol+ MINIGRIDSIZE; j++)
            curr->possiblevalues[i][j] &= temp;
}
int** solveSudoku(int **original_Grid)
{
    int i,j,num_threads;
    zerogrid= (int**)malloc(sizeof(int*)*SIZE);
    for (i=0;i<SIZE;i++)
    {
        zerogrid[i] = (int*)malloc(sizeof(int)*SIZE);
        for (j=0;j<SIZE;j++)
            zerogrid[i][j] = 0;
    }
    if(SIZE==9||SIZE==16 || SIZE==25)
    {
       init_possiblevalues(original_Grid);
       int x;
         /*
     int update=true;
      while(update)

      {

       if(elimination(grid)>0)   //If atleast one elimination is possible, then try for loneranger, twins, triplets  and check for eliminaiton
        update=true;
       else          //If no elimination is possible, then try for loneranger, twins, triplets and go out of loop
        update=false;

       loneranger(grid);
       twins(grid);
       //triplets(grid);

      }

          */
         
        for (bool isupdate=true;isupdate;isupdate= (x>0) ?true:false )
        {
                    #pragma omp parallel 
                    {
                        #pragma omp sections
                        { 
                                #pragma omp section
                                {   
                                    
                                    x=eliminations(original_Grid);
                                    
                                }
                                #pragma omp section
                                {
                                    loneranger(original_Grid);
                                    
                                }

                                #pragma omp section
                                {
                                    
                                    twins(original_Grid);

                                    
                                } 

                
                       }
                    }
        }

          
     }

    solvedgrid = NULL;
    solved = false;
    top = NULL;
    struct stack* curr;
    curr = stackalloc(0,0);
    stackinit(original_Grid, curr->Grid);
    int update;
    update = elimination(curr->Grid);
    /*printf("update now is %d\n", update);
    for (i = 0; i < SIZE; i++){
        for(j = 0; j < SIZE; j++){
            printf("%d",curr->Grid->my_grid[i][j]);
        printf("\n");    
        }
    }*/
    if ( update == -1)
    {
        return original_Grid;
    }
    else
    {
        if (isValid(zerogrid,curr->Grid->my_grid))
        {
            return curr->Grid->my_grid;
        }
        else
        {
            push(curr);
            num_threads = 1;
            #pragma omp parallel
            {
            num_threads = omp_get_num_threads();
            //printf("number of threads are %d\n", num_threads);
            }
            
            update=0;

            for(stack_sz=1;top!=NULL &&  stack_sz < num_threads;stack_sz--)
            {
                        //printf("stack size is %d\n", stack_sz);

                        struct stack* curr = pop();
                        bool cond=false;
                        int i,j;
                        for (i = curr->row ; i < SIZE; i++)
                        {
                            for (j = 0 ; j <SIZE; j++)
                            {
                                if ( curr->Grid->assigned[i][j] == 0)
                                {
                                    curr->row = i;
                                    curr->col = j;
                                    cond=true;
                                    i=SIZE;
                                    j=SIZE;
                                }
                            }
                        }
                        if (cond)
                        {
                            int num;
                            int test = curr->Grid->possiblevalues[curr->row][curr->col];
                            for ( num = 1; num <= SIZE ; num++)
                            {
                                if (test & (1 << (num - 1)))
                                {
                                    struct stack* temp = stackalloc(curr->row, curr->col);
                                    for ( i =0 ; i < SIZE ; i++)
                                    {
                                        for (j = 0 ; j < SIZE ; j++)
                                        {
                                            temp->Grid->assigned[i][j] = curr->Grid->assigned[i][j] ;
                                            temp->Grid->my_grid[i][j] = curr->Grid->my_grid[i][j];
                                            temp->Grid->possiblevalues[i][j]= curr->Grid->possiblevalues[i][j];
                                        }
                                    }
                                    assign_possiblevalues(num, temp->Grid,temp->row, temp->col);
                                    if (elimination(temp->Grid) == 0)
                                    {
                                        if (isValid(zerogrid,temp->Grid->my_grid))
                                        {
                                            solvedgrid = temp;
                                            return solvedgrid->Grid->my_grid;
                                        }
                                        push(temp);
                                        stack_sz++;
                                    }
                                }
                            }
                        }
                        deletestack(curr);
            }
            
            if ( update == 0 && stack_sz == 0)
            {
                return original_Grid;
            }
            else
            { 
                    //printf("stack size is now %d\n", stack_sz);
                    Sudoku();
                    if ( !solved)
                    {
                        return original_Grid;
                    }
                    else
                    {
                        return solvedgrid->Grid->my_grid;
                    }
                
               }
        }
  }
}
