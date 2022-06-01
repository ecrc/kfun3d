#include <stdarg.h>
#include <stdio.h>
#include "utils.h"

#ifdef DEBUG
void
dprint(const char *format, ...)
{
  va_list arg;
  
  va_start(arg, format);
  fprintf(stdout, "%s", ANSI_COLOR_RED);
  fprintf(stdout, "\tDEBUG >> ");
  const int done = vfprintf(stdout, format, arg);
  fprintf(stdout, "%s\n", ANSI_COLOR_RESET);
  va_end(arg);
}
#endif
