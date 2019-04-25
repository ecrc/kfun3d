#ifndef __FUN3D_INC_BIT_MAP_H
#define __FUN3D_INC_BIT_MAP_H

#include <math.h>
#include <stddef.h>

#define I2B(n) ((size_t) ceil((double) n / CHAR_BIT))

#define IOFF(v) (v / CHAR_BIT)
#define BOFF(v) (v % CHAR_BIT)

#define BSET(b, i) (b |= (unsigned char) (1 << i))
#define BCLR(b, i) (b &= (~(1 << i)))
#define BGET(v, b, i) (v = ((b & (1 << i)) != 0))

#endif /* __FUN3D_INC_BIT_MAP_H */