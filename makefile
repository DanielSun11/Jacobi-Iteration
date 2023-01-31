SRC=jacobi_MPI_OMP.c
OBJ=jacobi
CC=mpicc
CFLAGS = -g

$(OBJ):$(SRC)
	$(CC) $(SRC) -o $(OBJ) -fopenmp
debug_compile:$(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(OBJ)

run:
	mpirun -np 3 ./$(OBJ)
debug:debug_compile
	mpirun -np 4 gdb ./$(OBJ) : -n X-1 ./a.out
.PHONY : clean
clean:
	rm $(OBJ)