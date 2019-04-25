#ifndef __FUN3D_INC_TIMER_H
#define __FUN3D_INC_TIMER_H

namespace fun3d
{
  class Timer
  {
    typedef struct timer_t {
      struct timespec count;
      struct timespec total;
    } timer;
  private:
    timer *data;
    void reset();
    void set();
  public:
    Timer();
    ~Timer();
    void start();
    void stop();
    double get();
    void clear();
  }; /* class Timer */
}; /* namespace fun3d */

#endif /* __FUN3D_INC_TIMER_H */