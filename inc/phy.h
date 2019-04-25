#ifndef __FUN3D_INC_PHY_H
#define __FUN3D_INC_PHY_H

#include <math.h>
/* 3 is the angle of attack */
#define B 15                      /* Artificial compressibility (Beta) */
#define P 1                       /* Pressure */
#define U cos(3 / (180 / M_PI))   /* Velocity U */
#define V sin(3 / (180 / M_PI))   /* Velocity V */
#define W 0                       /* Velocity W */

#endif /* __FUN3D_INC_PHY_H */