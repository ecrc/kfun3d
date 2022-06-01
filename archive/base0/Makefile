
all: clean main run

.PHONY: clean run

include Makefile.conf.cpu
#include Makefile.conf.mic

main: main.o $(FLOW_OBJS) chkopts
	-$(CLINKER) -o main.out main.o $(COMPFLAGS) $(LIBS)
	-${RM}	*.o

run:
	export OMP_NUM_THREADS=16
	export KMP_AFFINITY=verbose, scatter
	./main.out -options_file petsc.opt -nThreads 16

clean::
	${RM} *.out
	${RM} *.o
