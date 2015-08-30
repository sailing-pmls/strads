#include "param.hpp"

DEFINE_string(input, "", "input file in libsvm format");
DEFINE_int64(rsv_verify, 0, " the number of samples to reserve for verification");
DEFINE_double(C, 0.1, " penalty parameter > 0 ");
DEFINE_string(loss, "l1", " specify type of loss function, l1 and l2");
DEFINE_int64(max_iter, 100, " maximum iterations ");
DEFINE_double(epsilon, 0.01, " epsilon ");
DEFINE_string(svmout, "", "out file in output directory");
DEFINE_int32(parallels, 1, "the number of parameters to update in parallel");
