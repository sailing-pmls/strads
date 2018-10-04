#ifndef DSFMT_HPP
#define DSFMT_HPP

#define DSFMT_MEXP 19937
#include "dSFMT.h"

#define DEFAULT_SEED 56789U
#define DEFAULT_JUMP 1<<30

struct DSFMT {
  DSFMT(uint32_t seed = DEFAULT_SEED);
  void jump_state(uint64_t steps = DEFAULT_JUMP);
  void discard(uint64_t steps) { jump_state(steps>>1); }
  double uniform_real();     /* half closed [0, 1) */
  double standard_normal();  /* box muller as in gnu C++11 impl */

  dsfmt_t state;
  double _saved;
  bool _saved_available;
};

#endif
