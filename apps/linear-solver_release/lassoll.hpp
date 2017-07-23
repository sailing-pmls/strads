#ifndef _LASSOLL_HPP_
#define _LASSOLL_HPP_

#include <gflags/gflags.h>
#include <vector>

void *coordinator_mach(void *);
void *worker_mach(void *);
void *scheduler_mach(void *);


// definition is in lassoll.hpp 
DECLARE_bool(output_hdfs_use);
DECLARE_string(outputfile_coeff);
DECLARE_string(logfile);
DECLARE_int64(logfreq);
DECLARE_int64(threads);

// non-mandotory but highly recommend to tune 
DECLARE_int64(scheduler);
DECLARE_int64(threads_per_scheduler);

DECLARE_int64(schedule_size);

DECLARE_bool(weight_sample);
DECLARE_bool(dep_check);

// mandatory to set by user since it has problem specific meaning 
DECLARE_double(lambda);
DECLARE_int64(max_iter);
DECLARE_bool(input_hdfs_use);
DECLARE_string(data_xfile); 
DECLARE_string(data_yfile); 
DECLARE_int64(columns); 
DECLARE_int64(samples);
DECLARE_int64(pipeline);
DECLARE_double(bw);
DECLARE_double(infthreshold);

DECLARE_bool(weight_sampling);
DECLARE_bool(check_interference);
DECLARE_string(algorithm);
DECLARE_int64(nzcount);

DECLARE_int64(schedulers); // from system layer

#endif 
