
/*
	Author: Mohammed Al Farhan
	Email:	mohammed.farhan@kaust.edu.sa
*/

/* #DEF(bmap[IOFF(i, RDDN(n))], BOFF(i, RDDN(n))) */

#ifndef __BIT_MAP_H
#define __BIT_MAP_H

#define I2B(n)        ((size_t) ceil((double) n / CHAR_BIT))

#define IOFF(v)       (v / CHAR_BIT)
#define BOFF(v)       (v % CHAR_BIT)

#define BSET(b, i)    (b |= (1 << i))
#define BCLR(b, i)    (b &= (~(1 << i)))
#define BGET(v, b, i) (v = ((b & (1 << i)) != 0))

#endif
