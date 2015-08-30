#include <atomic>
#include <assert.h>

#include "ll-worker.hpp"
//#include "cd-train.hpp"
//#include "strads/ds/spmat.hpp"
#include "cd-util.hpp"
#include <strads/netdriver/comm.hpp>
#include <strads/include/indepds.hpp>

#include "linear-svm.hpp"
#include "param.hpp"
#include <string>
#include "lasso.pb.hpp"
using namespace std;

void *worker_mach(void *arg){

  sharedctx *ctx = (sharedctx *)arg;
  LOG(INFO) << "[worker " << ctx->rank << "]" << " boot up out of " << ctx->m_worker_machines << " workers " << endl; 
  assert(ctx->star_recvportmap.size() == 1 and ctx->star_sendportmap.size() == 1);

  auto ps = ctx->star_sendportmap.begin();
  _ringport *sport = ps->second;
  context *send_ctx = sport->ctx;

  // create svm trainer 
  dualcd_svm handler(ctx->rank, ctx->m_worker_machines); // worker handler

  // read partition of data that is divided by n workers 
  handler.dist_read_data(FLAGS_input, FLAGS_rsv_verify);

  // report max feats and samples to the coordinator 
  stradsvm::bcwmsg maxmsg;
  maxmsg.set_src(ctx->rank);
  maxmsg.set_samples(handler.get_m_l());
  maxmsg.set_maxfeat(handler.get_m_m());
  string *buffer = new string;
  maxmsg.SerializeToString(buffer);
  while(send_ctx->push_entry_outq((char *)buffer->c_str(), buffer->size()));

  handler.train_worker(ctx);

  LOG(INFO) << "[worker " << ctx->rank << "] terminate job" << std::endl;
  return NULL;
}


  //    void *msg = recv_ctx->pull_entry_inq();
  //    while(send_ctx->push_entry_outq(result, sizeof(mbuffer)));
