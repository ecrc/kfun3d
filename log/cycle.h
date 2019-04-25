#ifndef __FUN3D_INC_CYCLE_H
#define __FUN3D_INC_CYCLE_H

namespace fun3d
{
  class Cycle
  {
    typedef struct cycle_t {
      long long int count;
      unsigned long long int total;
    } cycle;
  private:
    cycle *data;
    void reset();
    void set();
  public:
    Cycle();
    ~Cycle();
    void start();
    void stop();
    unsigned long long int get();
    void clear();
  }; /* class Cycle */
}; /* namespace fun3d */

#endif /* __FUN3D_INC_CYCLE_H */