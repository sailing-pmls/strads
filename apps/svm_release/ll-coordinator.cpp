/***********************************************************
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
#include "strads/netdriver/comm.hpp"
#include "strads/netdriver/zmq/zmq-common.hpp"

#include "ll-coordinator.hpp"
#include <glog/logging.h>
#include "strads/util/utility.hpp"
#include "strads/include/indepds.hpp"

#include "lassoll.hpp" 
#include "cd-util.hpp"
//#include "cd-train.hpp"

#include "linear-svm.hpp"
#include "param.hpp"
#include "lasso.pb.hpp"

using namespace std;

void *coordinator_mach(void *arg){

  sharedctx *ctx = (sharedctx *)arg;
  strads_msg(ERR, "[coordinator-machine] rank(%d) boot up coordinator-mach \n", ctx->rank);

  //  modify the following if you do not user pure star topology
  strads_msg(OUT, "\t\tCoordinator(%d) has recvport(%lu) sendport(%lu)\n",
	     ctx->rank, ctx->star_recvportmap.size(), ctx->star_sendportmap.size()); 
  strads_msg(OUT, "@@@@@ sched machines (%d)  worker machines (%d) \n", ctx->m_sched_machines, ctx->m_worker_machines);
  strads_msg(OUT, "@@@@@ sched: recvport (%ld) send port(%ld)  \n", ctx->scheduler_recvportmap.size(), ctx->scheduler_sendportmap.size());
  strads_msg(OUT, "@@@@@ worker: recvport (%ld) worker port(%ld)  \n", ctx->worker_recvportmap.size(), ctx->worker_sendportmap.size());
 
  assert(ctx->scheduler_recvportmap.size() == (unsigned long)(ctx->m_sched_machines));
  assert(ctx->scheduler_sendportmap.size() == (unsigned long)(ctx->m_sched_machines));
  assert(ctx->worker_recvportmap.size() == (unsigned long)(ctx->m_worker_machines));
  assert(ctx->worker_sendportmap.size() == (unsigned long)(ctx->m_worker_machines));


  // coordinator handler
  dualcd_svm handler(ctx->rank, ctx->m_worker_machines);
  handler.dist_read_res(FLAGS_input, FLAGS_rsv_verify);

  long samples(0), feats(0);
  for(int i=0; i<ctx->m_worker_machines; i++){
    int length=-1;
    void *buf = ctx->sync_recv(src_worker, i, &length);
    assert(length > 0);
    string stringbuffer((char *)buf, length);
    stradsvm::bcwmsg msg;
    msg.ParseFromString(stringbuffer);
    long local_feats = msg.maxfeat();
    long local_samples = msg.samples();
    strads_msg(OUT, "from worker %d : local_feats (%ld) local_samples (%ld) \n", 
	       i, local_feats, local_samples);
    if(i==0){
      samples = local_samples;
      feats = local_feats;
    }
    assert(samples == local_samples);
    assert(feats == local_feats);      
  }
  handler.set_m_l(samples);
  handler.set_m_m(feats);

  handler.train_coordinator(ctx);

  return NULL;
}
