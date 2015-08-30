/**********************************************************
   @author: Jin Kyu Kim (jinkyuk@cs.cmu.edu)
   @project: STRADS: A ML Distributed Scheduler Framework 
***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <sys/time.h>
#include <string>
#include <iostream>
#include <pthread.h>
#include <mpi.h>
#include <assert.h>

#include "strads/include/strads-macro.hpp"
#include "strads/include/common.hpp"
//#include "ds/dshard.hpp"
//#include "ds/binaryio.hpp"
#include "strads/netdriver/comm.hpp"
//#include "com/rdma/rdma-common.hpp"
#if defined(INFINIBAND_SUPPORT)
#include "strads/netdriver/rdma/rdma-common.hpp"
#else
#include "strads/netdriver/zmq/zmq-common.hpp"
#endif

#include "ll-scheduler.hpp"
#include <glog/logging.h>
#include "strads/include/indepds.hpp"

//#include "cd-util.hpp"
#include "cd-train.hpp"

using namespace std;

// main thread of scheduler machine 
// - aggregator/collector of multiple scheduling threads in a machine 
// - fork scheduling threads, communication with dispatcher  
void *scheduler_mach(void *arg){
  // SVM: scheduler is merged into coordinator in order to implement Active Set algorithm 
  return NULL;
}
