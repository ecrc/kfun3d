all: clean matrix main
matrix: matrix_build matrix_lib
.PHONY: clean matrix_build matrix_lib

##########################################################
##########################################################

ENV = DEBUG
COMP = GNU
ISA =
ARCH =
MEM =
METIS_PATH = ./metis
INCS_PATH = ./inc

ifeq (${EXEC},KNL-PERF)
ENV = PERF
COMP = INTEL
ISA = AVX512
ARCH = KNL
MEM = HBW
else ifeq (${EXEC},SKY-PERF)
ENV = PERF
COMP = INTEL
ISA = AVX512
ARCH = SKY
MEM = 
else ifeq (${EXEC},HSX-PERF)
ENV = PERF
COMP = INTEL
ISA = 
ARCH = 
MEM = 
endif

##########################################################
##########################################################

INCS = -I${METIS_PATH}/inc/ -I${INCS_PATH}
LIBS = -L${METIS_PATH}/lib/ -lmetis -lm
LCXX = ./matrix.a -lstdc++
SRCS =	src/main.c
SRCS += src/mesh.c
SRCS += src/allocator.c
## SRCS += src/fio.c
SRCS += src/edges.c
SRCS += src/index.c
SRCS += src/nodes.c
SRCS += src/guess.c
SRCS += src/boundaries.c
SRCS += src/csr.c
SRCS += src/subdomains.c
SRCS += src/weights.c
SRCS += src/bench.c
SRCS += src/kernel.c
SRCS += src/fill.c
SRCS += src/edge_coloring.c
SRCS += src/sbface_coloring.c
ifeq (${ISA},AVX512)
	SRCS += src/timestep_avx512.c
	SRCS += src/flux_avx512.c
else
	SRCS += src/timestep.c
	SRCS += src/flux.c
endif
SRCS += src/ilu.c
SRCS += src/sptrsv.c
SRCS += src/utils.c
SRCS += src/forces.c

OBJS	=	$(SRCS:.c=.o)

ifeq (${COMP},GNU)
	FLGS += -fopenmp
	CC = gcc
	CXX = g++
else ifeq (${COMP},LLVM)
	FLGS += -fopenmp=libomp
	CC = clang
	CXX = clang++
else ifeq (${COMP},INTEL)
	FLGS += -qopenmp -qopt-report=5
	CC = icc
	CXX = icpc
	LIBS += -mkl -mkl=parallel
endif

ifeq (${ENV},DEBUG)
	ifneq (${COMP},$(filter $(COMP),INTEL LLVM))
		FLGS += -O0 -Wall -g -ggdb -Wextra -Werror -Wswitch-enum
		FLGS += -Wuninitialized -Wswitch-default -Wconversion
		FLGS += -Wno-error=unused-variable -Wno-div-by-zero
		FLGS += -gdwarf-4 -Wempty-body -Wenum-compare -Wno-endif-labels
		FLGS +=  -feliminate-unused-debug-symbols -Wdouble-promotion
		FLGS += -Wfloat-equal -Wfloat-equal -gno-strict-dwarf
		FLGS += -pedantic-errors -w -Wunused-variable -Wformat
		FLGS += -Waddress -Waggregate-return -gstrict-dwarf -Wunused-function
		FLGS += -Wno-aggressive-loop-optimizations -Wunreachable-code
		FLGS += -Warray-bounds -Wno-attributes -Wno-attribute-alias
		FLGS += -Wno-builtin-declaration-mismatch -Wno-builtin-macro-redefined
		FLGS += -Wunused-but-set-parameter -Wunused-but-set-variable
		FLGS += -Wvariadic-macros -fdebug-prefix-map=old=new -fno-merge-debug-strings
		FLGS += -Wwrite-strings -Wvla  -fno-dwarf2-cfi-asm -Wfloat-equal
		FLGS += -Wmissing-format-attribute -Wswitch -Wunused-label
		FLGS += -Wsync-nand -Wsystem-headers -Wunknown-pragmas -Wunused
		FLGS += -Wtrampolines -Wtrigraphs -Wtype-limits -Wundef -Wuninitialized
		FLGS += -Wunused-macros -Wunused-parameter -Wno-unused-result -Wunused-value
		FLGS += -Wclobbered -Wcomment -Wcoverage-mismatch -Wno-cpp -femit-class-debug-always
		FLGS += -Wno-attribute-warning -fno-eliminate-unused-debug-types
		FLGS += -Wno-deprecated -Wno-deprecated-declarations -Wdisabled-optimization
		FLGS += -Wno-format-contains-nul -Wno-format-extra-args -Wformat-nonliteral
		FLGS += -fgnu89-inline -femit-struct-debug-baseonly -femit-struct-debug-reduced
		FLGS += -fvar-tracking -fvar-tracking-assignments -fargument-noalias
		# Options that do not work with GCC major version <= 6
		GCCVERSIONGTEQ4 := $(shell expr `gcc -dumpversion | cut -f1 -d.` \>= 6)
		ifeq "$(GCCVERSIONGTEQ4)" "1"
			FLGS += -gcolumn-info -gno-column-info -grecord-gcc-switches -gno-record-gcc-switches
			FLGS += -Wpedantic -Walloc-zero -Wdate-time -Wexpansion-to-defined -gsplit-dwarf
			FLGS += -Wtautological-compare -Wvector-operation-performance -Whsa
			FLGS += -Wduplicated-branches -Wduplicated-cond -Wdangling-else -Wfatal-errors
			FLGS += -Wbool-compare -Wbool-operation -fdebug-types-section -Walloca
			FLGS += -Wswitch-bool -Wswitch-unreachable -Wunused-const-variable
			FLGS += -Wsuggest-final-types -Wsuggest-final-methods -Wunused-local-typedefs
		endif
	else
		FLGS += -O0 -Wall -g
	endif
else ifeq (${ENV},PERF)
	FLGS += -O3
endif

ifeq (${MEM},TCMALLOC)
	FLGS += -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
	LIBS += -ltcmalloc
endif

ifeq (${ISA},AVX512)
	FUN3D_FLGS += -DBUCKET_SORT
ifeq (${ARCH},SKY)
	FLGS += -xCORE-AVX512
	FUN3D_FLGS += -DARCH_SKY
else ifeq (${ARCH},KNL)
	FLGS += -xMIC-AVX512
ifeq (${MEM},HBW)
	FUN3D_FLGS += -DPOSIX_HBW
	LIBS += -lpthread -lmemkind
endif
endif
endif

CFLAGS = $(FLGS) $(INCS) ${FUN3D_FLGS}
CXXFLAGS = $(FLGS) -I${METIS_PATH}/inc/

matrix_build: matrix/init.cpp \
	matrix/edmond.cpp \
	matrix/inverse.cpp \
	matrix/matrix.cpp \
	matrix/sptrsv.cpp \
	matrix/ilu.cpp \
	matrix/wrapper.cpp
	${CXX} ${CXXFLAGS} -c matrix/init.cpp \
	matrix/edmond.cpp \
	matrix/inverse.cpp \
	matrix/matrix.cpp \
	matrix/sptrsv.cpp \
	matrix/ilu.cpp \
	matrix/wrapper.cpp

matrix_lib:
	ar rvs matrix.a init.o \
	edmond.o \
	inverse.o \
	matrix.o \
	sptrsv.o \
	ilu.o \
	wrapper.o

main: $(OBJS)
	${CC} $(CFLAGS) -o kfun3d.out $(OBJS) ${LCXX} $(LIBS)
	${RM} *.o src/*.o *.a

valgrind:
	valgrind \
	--verbose \
	--leak-check=full \
	--leak-resolution=high \
	--show-leak-kinds=all \
	--show-mismatched-frees=yes \
	./kfun3d.out -m A -t 1

cachegrind:
	valgrind --tool=cachegrind \
	./kfun3d.out -m A -t 1

run_serial:
	./kfun3d.out -m A -t 1

run_parallel:
	./kfun3d.out -m B -t 20

clean:
	${RM} *.out *.o *.a src/*.o \
		*.optrpt src/*.optrpt \
		*.dat src/*.dwo *.dwo *.err *.res
