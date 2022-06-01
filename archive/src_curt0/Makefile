
#
# Author: Mohammed Al Farhan
# Email:	mohammed.farhan@kaust.edu.sa
#

all: clean main

.PHONY: clean

CFLAGS = -O3 -qopenmp -Wall -Wextra -qopt-report=5
CFLAGS += -fargument-noalias -restrict -qopt-streaming-stores=always
CFLAGS += -xMIC-AVX512
CFLAGS += -D__USE_EDGE_COLORING
CFLAGS += -D__USE_METIS
CFLAGS += -D__USE_MEMKIND -D__USE_POSIX_HBW
CFLAGS += -fgnu89-inline

#CFLAGS += -D__USE_SKX
#CFLAGS += -xCORE-AVX512

#CFLAGS += -g -gdwarf4
#CFLAGS += -D__USE_POSIX -D__USE_POSIX_MEMMEMALIGN

CC 	= icc

SRC =	main.c mesh.c allocator.c fio.c edges.c index.c nodes.c
SRC += boundaries.c csr.c subdomains.c weights.c kernel.c force.c
SRC	+= p_kernel.c fill.c ktime.c guess.c
SRC	+= edge_coloring.c
SRC += sbface_coloring.c
SRC += timestep.c
SRC += grad.c
SRC += flux_avx512.c

#SRC += flux.c
#SRC += timestep_avx512.c
#SRC += grad_avx512.c

OBJ	=	$(SRC:.c=.o)

LNK = ${PETSC_LIB} -mkl -mkl=parallel
LNK += -lpthread
LNK += -lmemkind

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

main: $(OBJ)
	${CC} $(CFLAGS) -o main.out $(OBJ) $(LIBS) $(CFLAGS) $(LNK)
	${RM} *.o

clean::
	${RM} *.out *.o *.optrpt
