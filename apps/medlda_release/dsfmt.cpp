#include "dsfmt.hpp"
#include "dSFMT-jump.h"
#include "dsfmt_polys.c"
#include <cmath>
#include <iostream>

DSFMT::DSFMT(uint32_t seed) {
  dsfmt_init_gen_rand(&state, seed);
  _saved = NAN;
  _saved_available = false;
};

void DSFMT::jump_state(uint64_t steps) {
  for (int b=0; b < 64; b++)
    if ((steps>>b) & 1)
      dSFMT_jump(&state, JUMP_19937_2POWS[b]);
};

double DSFMT::uniform_real() {
  return dsfmt_genrand_close_open(&state);
};

double DSFMT::standard_normal() {
  if (_saved_available) {
    _saved_available = false;
    return _saved;
  }
  _saved_available = true;
  double x, y, r2;
  do {
    x = 2.0 * dsfmt_genrand_open_open(&state) - 1.0;
    y = 2.0 * dsfmt_genrand_open_open(&state) - 1.0;
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0.0);

  const double mult = std::sqrt(-2 * std::log(r2) / r2);
  _saved = y * mult;
  return x * mult;
};
