all: clean flow

USERCPPFLAGS    = 	-DINTERLACING -DBLOCKING -qopenmp -O3 -xMIC-AVX512
USERCPPFLAGS		+=	-DHAVE_EDGE_COLORING
CPPFLAGS        = $(USERCPPFLAGS) $(FCONF)
FPPFLAGS      	= $(USERCPPFLAGS)
EXAMPLESC				= grid_info.c
EXAMPLESF       = user.F userJac.F 
FLOW_OBJSF      = $(EXAMPLESF:.F=.o)
FLOW_OBJSC			= $(EXAMPLESC:.c=.o)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

flow: flow.o $(FLOW_OBJSC) $(FLOW_OBJSF) chkopts
	-$(CLINKER) -o flow.out flow.o $(FLOW_OBJSC) $(FLOW_OBJSF) $(USERCPPFLAGS) $(PETSC_SNES_LIB)
	${RM}	flow.o
	${RM}	grid_info.o
	${RM}	user.o
	${RM}	userJac.o

clean::
	${RM} *.out *.o

run:
	./incomp_euler_flow_driver_mic.out -options_file petsc.opt -max_threads 60

run_60:
	./flow_driver.out -options_file petsc.opt -max_threads 60 -target_id 0

run_240:
	./flow_driver.out -options_file petsc.opt -max_threads 240 -target_id 1
