#include <string.h>
#include <unistd.h>
#include "log.h"

int
main()
{
  std::string stages[3] = {"FLUX", "SPARSE", "INITIALIZATION"};
  fun3d::Log bench(3, stages);

  bench.start("FLUX");
  usleep(10000000);
  bench.stop("FLUX");

  bench.start("FLUX");
  usleep(10000000);
  bench.stop("FLUX");

  bench.start("INITIALIZATION");
  usleep(1000000);
  bench.stop("INITIALIZATION");

  bench.print();

  return 0;
}