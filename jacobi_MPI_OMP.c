#include<stdlib.h>
#include<omp.h>
#include<stdio.h>
#include<math.h>
#include<mpi.h>
#include<string.h>

#define N 1000 //rows of matrix
#define NUM_ITERATION 1000 //the number of iterations
int num_procs;
int myId;
//Ax = b
int init_data(int * A,double *x,int *b){

    if(!A||!b||!x)return -1;
    memset(A,0,sizeof(int)*N*N);
    //init Matrix A
    for(int i = 0;i < N;++i){
        for (int j = i; j < N; j++)
        {
            A[i*N+j] = i==j?(i+1)*(i+1):(int)(j*1.0/i)%5;
        }
        
    }
    //init Vector b; b = A*x; x = range(1:N+1)
    for(int i = 0;i < N;++i){
        int sum = 0;
        for (int j = 0; j < N; j++)
        {
            sum+=A[i*N+j]*(j+1);
        }
        b[i] = sum;
    }
    //init Vector x; x[i] = 1;
    for (int i = 0; i < N; i++)
    {
        x[i] = 1;
    }
}
void jacobi(int *A,double *x,int *b,int * proc_blocksize,int *offset){
    double * x0 = x;// array x0 store the data from previous  iteration
    double * x1 = (double*)malloc(sizeof(double)*N);
    int blocksize = proc_blocksize[myId];
    int start = offset[myId];
    //start iteration
    for (int itr = 0; itr < NUM_ITERATION; itr++)
    {   
        #pragma parallel for
        for (int i = start; i < start+blocksize; i++)
        {
            double sum = 0;
            int rows = i-start;
            for (int j = 0; j < N; j++)
            {
                sum+=i==j?0:A[(rows)*N+j]*x0[j];
            }
            x1[i] = (b[rows]-sum)/(A[rows*N+i]);
        }
        //communicate and refresh vector x
        MPI_Allgatherv(x1+offset[myId],proc_blocksize[myId],MPI_DOUBLE,x0,proc_blocksize,offset,MPI_DOUBLE,MPI_COMM_WORLD);
        /*//debug code
        if(!myId)
        for (int i = 0; i < N; i++)
        {
            printf("x[%d] %lf\n",i,x0[i]);
        }*/
        
    }
}
//print A,x,b
void dump(int *A,double *x,int *b){
    printf("Matrix A\n");
    for(int i = 0; i < N ; ++i){
        for (int j = 0; j < N; j++)
        {
           printf("%d ",A[i*N+j]);
        }
        printf("\n");
        
    }
    printf("Vector x\n");
    for (int j = 0; j < N; j++)
     {
        printf("%lf ",x[j]);
     }
    printf("\n");
    printf("Vector b\n");
    for (int j = 0; j < N; j++)
    {
        printf("%d ",b[j]);
    }
    printf("\n");
}

int main(int argc, char* argv[]){

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myId);
    int proc_blocksize[num_procs];//store the data range in vector x and b per process, 
                                 // such as  if N = 10,num of process = 3 then the proc_blocksize[0] = 3,proc_blocksize[2] = 4
    int offset[num_procs];
    int A_elements_per_proc[num_procs];//store the data range in matrix A per process, 
    int offset_A[num_procs];

    for (int i = 0; i < num_procs; i++)
    {
        proc_blocksize[i]=i==num_procs-1?(N-N/num_procs*(num_procs-1)):N/num_procs;
        offset[i]=i*(N/num_procs);
        A_elements_per_proc[i]=proc_blocksize[i]*N;
        offset_A[i] = i*N*(N/num_procs);
    }
 
    
    
    int * A;
    double * x;
    int * b;

    int * _A;
    int * _b;
    x = (double*)malloc(sizeof(double)*N);
    double * _x = (double*)malloc(sizeof(double)*N);
    if(!myId){
        A = (int*)malloc(sizeof(int)*N*N);
        b = (int*)malloc(sizeof(int)*N);
        init_data(A,x,b);
    }
    _A = (int*)malloc(sizeof(int)*proc_blocksize[myId]*N);
    _b = (int*)malloc(sizeof(int)*proc_blocksize[myId]);
    //broadcast vector x
    MPI_Bcast(x,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
    //printf("myId %d x[0] %lf\n",myId,x[0]);
    //scatter A
    //if(!myId)dump(A,x,b);
    MPI_Scatterv(b,proc_blocksize,offset,MPI_INT,_b,proc_blocksize[myId],MPI_INT,0,MPI_COMM_WORLD);
    //printf("myId %d b[0] %d b end %d\n",myId,_b[0],_b[proc_blocksize[myId]-1]);
    MPI_Scatterv(A,A_elements_per_proc,offset_A,MPI_INT,_A,A_elements_per_proc[myId],MPI_INT,0,MPI_COMM_WORLD);
    //printf("myId %d A[0] %d A end %d\n",myId,_A[0],_A[A_elements_per_proc[myId]-1]);
    /*x[offset[myId]] = myId; 
    MPI_Allgatherv(x+offset[myId],proc_blocksize[myId],MPI_DOUBLE,_x,proc_blocksize,offset,MPI_DOUBLE,MPI_COMM_WORLD);
   if(!myId)dump(A,_x,b);*/


    jacobi(_A,x,_b,proc_blocksize,offset);
    MPI_Finalize();
   // dump(A,x,b);
    printf("x[2] = %lf \n",x[N-1]);

}