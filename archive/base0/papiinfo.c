
#ifdef HW_COUNTER

#include "defs.h"

int 
initPAPI()
{

  if(PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
  {
    printf("ERROR: initializing PAPI library\n");
    return 1;
  }

  eventset = PAPI_NULL;
  handler = 0;
  if(PAPI_create_eventset(&eventset) != PAPI_OK)
  {
    printf("ERROR: Creating PAPI event est\n");
    return 1;
  }

  int eventcodes;

  eventcodes  = PAPI_FP_INS; 
/*
  eventcodes[1]   = PAPI_INT_INS;
  eventcodes[2]   = PAPI_TLB_DM;
  eventcodes[3]   = PAPI_TLB_IM;
  eventcodes[4]   = PAPI_L2_LDM;
  eventcodes[5]   = PAPI_BR_MSP;
  eventcodes[6]   = PAPI_TOT_INS;
  eventcodes[7]   = PAPI_LD_INS;
  eventcodes[8]   = PAPI_SR_INS;
  eventcodes[9]   = PAPI_BR_INS;
  eventcodes[10]  = PAPI_VEC_INS;
  eventcodes[11]  = PAPI_TOT_CYC;
  eventcodes[12]  = PAPI_L1_DCA;  
  eventcodes[13]  = PAPI_L1_ICA;
*/

  if(PAPI_add_event(eventset, eventcodes) != PAPI_OK)
  {
    printf("ERROR: Adding PAPI events\n");
    return 1;
  }

  return 0;
}

#endif
