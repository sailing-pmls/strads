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
#include <numa.h>

#include "strads-macro.hpp"
#include "common.hpp"
#include "ds/dshard.hpp"
#include "ds/binaryio.hpp"
#include "com/comm.hpp"
//#include "com/rdma/rdma-common.hpp"
#if defined(INFINIBAND_SUPPORT)

#include "com/rdma/rdma-common.hpp"

#else

#include "com/zmq/zmq-common.hpp"

#endif

#include "coordinator/coordinator.hpp"
#include <glog/logging.h>
#include "utility.hpp"
#include "indepds.hpp"
#include "scheduler/scheduler.hpp"

using namespace std;


pthread_barrier_t  cbarrier; // barrier synchronization object  

/* declaration of local functions */
int  _get_tosend_machinecnt(sharedctx *ctx, machtype mtype);
void _scheduler_start_remote(sharedctx *ctx, double *weights, uint64_t wsize, bool rflag);
void _load_dshard_remote(sharedctx *ctx, userfn_entity *fne, ppacketmgt *cmdmgt);
int  _get_tosend_machinecnt(sharedctx *ctx, machtype mtype);
int  _get_totalthreads_formachtype(sharedctx *ctx, machtype mtype);
static double _soft_threshold(double sum, double lambda);
void get_object_first_half(sharedctx *ctx, int64_t cmdid);
double get_object_second_half(sharedctx *ctx);
idval_pair *receive_extended_packet(sharedctx *ctx, int src, int64_t len, double *wsquare);

/* end of local function declarations */
// valid for star topology
static void _send_to_scheduler(sharedctx *ctx, mbuffer *tmpbuf, int len, int schedmid){
  while(ctx->scheduler_sendportmap[schedmid]->ctx->push_entry_outq((void *)tmpbuf, len)); 
  //  free(tmpbuf);
}

#if 0 
// valid for star topology
// *********** CAVEAT: If COM stack frees buffer, you can not use broadcase here without copy overhead.... 
static void _broadcast_to_workers(sharedctx *ctx, mbuffer *tmpbuf, int len){
  for(int i=0; i < ctx->m_worker_machines; i++){
    while(ctx->worker_sendportmap[i]->ctx->push_entry_outq((void *)tmpbuf, len));
  }
// *********** CAVEAT: If COM stack frees buffer, you can not use broadcase here without copy overhead.... 
}
// NOT USED ANY MORE
#endif 

static void _mcopy_broadcast_to_workers(sharedctx *ctx, mbuffer *tmpbuf, int len, int64_t cmdid){

  for(int i=0; i < ctx->m_worker_machines; i++){
    mbuffer *tmp = (mbuffer *)calloc(1, sizeof(mbuffer));
    memcpy(tmp, tmpbuf, sizeof(mbuffer));
    tmp->cmdid = cmdid;
    while(ctx->worker_sendportmap[i]->ctx->push_entry_outq((void *)tmp, len));
  }
  free(tmpbuf);
// *********** CAVEAT: If COM stack frees buffer, you can not use broadcase here without copy overhead.... 
}

// valid for star topology
static void _send_to_workers(sharedctx *ctx, mbuffer *tmpbuf, int len, int workermid){
  while(ctx->worker_sendportmap[workermid]->ctx->push_entry_outq((void *)tmpbuf, len)); 
  //  free(tmpbuf);
}

void _send_weight_update(sharedctx *ctx, int schedmid, int gthrdid, idval_pair *idvalp, int entrycnt){

  mbuffer *mbuf = (mbuffer *)calloc(1, sizeof(mbuffer)); 
  mbuf->msg_type = SYSTEM_SCHEDULING;
  int64_t dlen = USER_MSG_SIZE - sizeof(schedhead);    
  schedhead *schedhp = (schedhead *)mbuf->data;
  schedhp->type = SCHED_UW;
  schedhp->sched_mid = schedmid;
  schedhp->sched_thrdgid = gthrdid;
  schedhp->entrycnt = entrycnt;
  assert((int64_t)(entrycnt *sizeof(idval_pair)) <= dlen);
  idval_pair *amo_ldvalp = (idval_pair *)((uintptr_t)schedhp + sizeof(schedhead));
  for(int64_t k=0; k < entrycnt; k++){
    amo_ldvalp[k].id = idvalp[k].id;
    amo_ldvalp[k].value = idvalp[k].value;
  }
  _send_to_scheduler(ctx, mbuf, sizeof(mbuffer), schedmid); // once sent through com stack, it will be realeased

}

void user_func_make_dispatch_msg(void *buf, work_ptype wtype, TASK_ENTRY_TYPE *tasks, int64_t taskcnt, idval_pair *stats, int64_t statcnt, int64_t bufsize){

  // buf size : maximun byte size inclduing uobj head 
  assert (((uintptr_t)buf) % sizeof(int) == 0); // checking buf address sanity checking. 

  if( int64_t(sizeof(uobjhead) + sizeof(TASK_ENTRY_TYPE)*taskcnt + sizeof(idval_pair)*statcnt) >= bufsize){
    LOG(FATAL) << "mbuffer packet size is too small to handle the current phase. Increase mbuffer size" << endl;
  }
  assert( int64_t(sizeof(uobjhead) + sizeof(TASK_ENTRY_TYPE)*taskcnt + sizeof(idval_pair)*statcnt) <= bufsize);
  // buf alingnment, buf size sanity checking 

  uobjhead *uobjhp = (uobjhead *)buf;
  uobjhp->task_cnt = taskcnt;
  uobjhp->stat_cnt = statcnt;

  TASK_ENTRY_TYPE *ids = (TASK_ENTRY_TYPE *)((uintptr_t)uobjhp + sizeof(uobjhead));
  for(int64_t k=0; k< taskcnt; k++){

    strads_msg(INF, "\t\tTask [%ld] = id %ld  coeff (%lf) \n", k, tasks[k].id, tasks[k].value);

    ids[k].id = tasks[k].id;
    ids[k].value = tasks[k].value;
  }

  idval_pair *idvalp = (idval_pair *)((uintptr_t)ids + sizeof(TASK_ENTRY_TYPE)*taskcnt);
  for(int64_t k=0; k< statcnt; k++){
    strads_msg(INF, "\t\tSTAT [%ld] = id %ld  value-delta (%lf) \n", k, stats[k].id, stats[k].value);
    idvalp[k].id = stats[k].id;
    idvalp[k].value = stats[k].value;
  }
  return;
}

void _load_dshard_coordinator_local(sharedctx *ctx, userfn_entity *fne, ppacketmgt *cmdmgt){

  strads_msg(ERR, "LOAD DSHARD COORDINATOR LOCAL @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");

  const char *sp = fne->m_strfn.c_str();
  const char *type = fne->m_strtype.c_str(); // row / column major 
  //  const char *pscheme = fne->m_strpscheme.c_str(); // partitioning scheme..
  const char *strmachtype = fne->m_strmachtype.c_str(); // on what machines, data will be partitione
  const char *stralias = fne->m_stralias.c_str();
  machtype mtype = fne->m_mtype;
  if(mtype != m_coordinator){
    return;
  }else{
    // loading or allocating dshard here and associate it with sharedctx dshard map. 
    // in current design, data loading is not allowed in the coordinator.
    // only model data structure creation is allowed.
    strads_msg(ERR, "[coordinator load dshard local] file to load: [%s] assigned to [%s]\n", sp, strmachtype);
    assert(strcmp(strmachtype, "coordinator")==0);

    //    assert(strcmp(sp, "gen-model")==0);    
    if(strcmp(sp, "gen-model")==0){    
#if defined(SVM_COL_PART)
      int64_t modelsize = ctx->m_params->m_up->m_samples;
#else
      int64_t modelsize = ctx->m_params->m_sp->m_modelsize;
#endif
      uint64_t rows=1;
      assert(modelsize > 0);
      uint64_t cols = (uint64_t)modelsize;
      assert(strcmp(type, "d2dmat") == 0);
      class dshardctx *dshard;;
      dshard = new dshardctx(rows, cols, d2dmat, sp, stralias);

      dshard->m_dmat.resize(dshard->m_rows, dshard->m_cols);

      rangectx *rctx = new rangectx;

      rctx->r_start = 0;
      rctx->r_end = rows-1;
      rctx->r_len = rctx->r_end - rctx->r_start + 1;
      rctx->c_start = 0; // since row partition, each machine cover all columns
      rctx->c_end = cols-1;
      rctx->c_len = rctx->c_end - rctx->c_start+1;

      dshard->set_range(rctx); // Don't forget this 

      ctx->register_shard(dshard);
      ctx->m_params->m_up->bind_func_param(dshard->m_alias, (void *)dshard);

    }else{     

#if 1
      // CAVEAT this whole block is dedicated for SVM only. 
      // TODO : think about more flexible support of data loading / data type / and ....
      uint64_t rows, cols, nz;
      const char *pscheme = fne->m_strpscheme.c_str(); // partitioning scheme..
      assert(strcmp(pscheme, "NO")==0);

      iohandler_spmat_pbfio_read_size(sp, &rows, &cols, &nz); 

      //      class dshardctx *dshard;
      //      dshard = new dshardctx(rows, cols, d2dmat, sp, stralias);
      //      dshard->m_dmat.resize(dshard->m_rows, dshard->m_cols);
      //      ctx->register_shard(dshard);
      //      ctx->m_params->m_up->bind_func_param(dshard->m_alias, (void *)dshard);
      dshardctx *dshard = new dshardctx(rows, cols, d2dmat, sp, stralias);

      rangectx *rctx = new rangectx;

      dshard->m_dmat.resize(dshard->m_rows, dshard->m_cols);
      dshard->m_atomic.resize(dshard->m_rows);

      rctx->r_start = 0;
      rctx->r_end = rows-1;
      rctx->r_len = rctx->r_end - rctx->r_start + 1;
      rctx->c_start = 0; // since row partition, each machine cover all columns
      rctx->c_end = cols-1;
      rctx->c_len = rctx->c_end - rctx->c_start+1;

      strads_msg(ERR, "[Coordinator] Load Y matrix m_rows(%ld) m_cols(%ld) from a File(%s) rctx->r_start (%ld) r_end (%ld)  c_start(%ld)  c_end(%ld)\n", 
		 dshard->m_rows, dshard->m_cols, sp, rctx->r_start, rctx->r_end, rctx->c_start, rctx->c_end);
      // CAVEAT: bind range ctx with dshard context. This is very important. Don't forget
      dshard->set_range(rctx); // Don't forget this 

      iohandler_spmat_pbfio_partialread(dshard, false, ctx->rank);

      strads_msg(ERR, "[Coordinator] Finish load Y matrix fromaFile(%s)rctx->r_start(%ld)r_end(%ld)c_start(%ld)c_end(%ld)\n", 
		 sp, rctx->r_start, rctx->r_end, rctx->c_start, rctx->c_end);


      ctx->register_shard(dshard);

      strads_msg(ERR, "[Coordinator] register shard \n");

      ctx->m_params->m_up->bind_func_param(dshard->m_alias, (void *)dshard);
      strads_msg(ERR, "[Coordinator] bind dshard with function \n");
#endif
      //      assert(0);
     
    }
  }
}


void *scheduling_emulator(int64_t taskid){

  mbuffer *mbuf = (mbuffer *)calloc(sizeof(mbuffer), 1);
  mbuf->msg_type = SYSTEM_SCHEDULING;
  schedhead *schedhp = (schedhead *)mbuf->data;	
  schedhp->entrycnt = 1;
  schedhp->sched_thrdgid = 0;
  schedhp->sched_mid = 0;
  schedhp->type = SCHED_PHASE;

  strads_msg(INF, "New Phase: schemid(%d) schedthrdgid(%d) entrycnt(%ld)\n", 
	     schedhp->sched_mid, schedhp->sched_thrdgid, schedhp->entrycnt);
  int64_t *amo_gtaskids =  (int64_t *)((uintptr_t)schedhp + sizeof(schedhead));
  amo_gtaskids[0] = taskid;
  return (void *)mbuf;
}

void scheduling_release_buffer(void *buf){
  free(buf);
}



void *coordinator_mach(void *arg){
  sharedctx *ctx = (sharedctx *)arg;
  strads_msg(ERR, "[coordinator-machine] rank(%d) boot up coordinator-mach \n", ctx->rank);
  assert(ctx->m_coordinator_mid >=0);

  /////////////// TEMPORARY ////////////////////////////////////////////////////////
  //  string strcoeff("coeffRow");
  //  dshardctx *coeffshard = ctx->get_dshard_with_alias(strcoeff);
  //  assert(coeffshard != NULL);
  //  double **betatmp = coeffshard->m_dmat.m_mem;
  //  double *beta = betatmp[0]; 


  int thrds = ctx->m_params->m_sp->m_thrds_per_coordinator;

  pthread_barrier_init (&cbarrier, NULL, thrds+1); // +1 for the main thread

  strads_msg(ERR, "[coordinator-mach] pthread-barrier with %d threads", thrds+1);


  coordinator_threadctx **sthrds = (coordinator_threadctx **)calloc(MAX_SCHEDULER_THREAD, sizeof(coordinator_threadctx *));
  // TODO : replace with user configuratio(system configuration)

  assert(thrds > 0);
  for(int i=0; i<thrds; i++){    
    sthrds[i] = new coordinator_threadctx(ctx->rank, ctx->m_coordinator_mid, i, ctx);
  }
 
  //  int gcthrd = thrds;
  //  for(int i=thrds; i<thrds+1; i++){    
  //    sthrds[i] = new coordinator_threadctx(ctx->rank, ctx->m_coordinator_mid, i, ctx, true);
  //  }

  // TODO: modify the following if you do not user pure star topology
  strads_msg(ERR, "\t\tCoordinator(%d) has recvport(%lu) sendport(%lu)\n",
	     ctx->rank, ctx->star_recvportmap.size(), ctx->star_sendportmap.size()); 
  assert(ctx->scheduler_recvportmap.size() == (unsigned long)(ctx->m_sched_machines));
  assert(ctx->scheduler_sendportmap.size() == (unsigned long)(ctx->m_sched_machines));
  assert(ctx->worker_recvportmap.size() == (unsigned long)(ctx->m_worker_machines));
  assert(ctx->worker_sendportmap.size() == (unsigned long)(ctx->m_worker_machines));


  double *weights = (double *)calloc(ctx->m_params->m_sp->m_modelsize, sizeof(double));

  _scheduler_start_remote(ctx, weights, ctx->m_params->m_sp->m_modelsize, false);     
  free(weights);

  // JUST TO TEST CORRECTNESS OF INIT ....... 
  strads_msg(ERR, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ infinite loop @@@@@@@@ \n");

  //  while(1){};

  // 10, 10 : max number of pending system command, max number of pending user commands 
  ppacketmgt *cmdmgt = new ppacketmgt(10, 10, (*ctx));
  /* ussage of cmdmgt 
    long cmdid = cmdmgt->get_cmdclock();
    cmdmgt->push_cmdq(cmdid, m_worker); */

  // sanity checking on input parameters - input data and their mahcine placement 
  for(auto p : ctx->m_params->m_up->m_fnmap){   

    strads_msg(ERR, "#############################################################################\n ");

    userfn_entity *fne = p.second;      
    machtype mtype = fne->m_mtype;
    if(mtype != m_coordinator){

      strads_msg(ERR, "\t\t REMOTE START ENTER @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n ");
      _load_dshard_remote(ctx, fne, cmdmgt); // m_scheduler or m_worker : create ds and upload data in remote machine(s)
      strads_msg(ERR, "\t\t REMOTE END EXIT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n ");

    }else if(mtype == m_coordinator){

      strads_msg(ERR, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ _load_dshard_coordinator_local is called \n");
      _load_dshard_coordinator_local(ctx, fne, cmdmgt); // create ds and upload data in my local machine
      strads_msg(ERR, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ _load_dshard_coordinator_local ENDS \n");

    }else{
      LOG(FATAL) << " Not Supported Mach Type " << endl;
    }
  }

  strads_msg(ERR, "EEEEEEEEEEEE@@@@@@@@@@@EEEEEEEEEEEEE@@@@@@@@@@@@EEEEEEEEEEEE@@@@@@@@@@@@@@@\n");

#if 1 
  /////////////// TEMPORARY ////////////////////////////////////////////////////////
  string strcoeff("coeffRow");
  dshardctx *coeffshard = ctx->get_dshard_with_alias(strcoeff);
  assert(coeffshard != NULL);
  double **betatmp = coeffshard->m_dmat.m_mem;
  double *beta = betatmp[0]; 
#endif

  strads_msg(ERR, "@@@@COEFFROW Is loaded verified in the main thread.......@@@@@@@@@@@@@@@@@@@\n");
  pthread_barrier_wait (&cbarrier);
  strads_msg(ERR, "@@@@[coordinator-mach] Barrier pass !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");


#if defined(SVM_COL_PART)
  for(int64_t i=0; i < ctx->m_params->m_up->m_samples; i++){
    assert(beta[i] == 0.0); 
  }
  int64_t modelsize = ctx->m_params->m_up->m_samples;
  //  double *betadiff = (double *)calloc(ctx->m_params->m_sp->m_modelsize, sizeof(double));
  assert(modelsize == ctx->m_weights_size);
  //  double *betadiff = ctx->m_weights;
#else
  for(int64_t i=0; i < ctx->m_params->m_sp->m_modelsize; i++){
    assert(beta[i] == 0.0); 
  }
  int64_t modelsize = ctx->m_params->m_sp->m_modelsize;
  //  double *betadiff = (double *)calloc(ctx->m_params->m_sp->m_modelsize, sizeof(double));
  assert(modelsize == ctx->m_weights_size);
#endif

  double lambda = ctx->m_params->m_up->m_beta;
  strads_msg(ERR, "@@@@@@@@  LAMBDA %lf  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n", lambda);
  /////////////////////////////////////////////////////////////////////////////////

  // wait until all shard command are acked from workers / scheduler machines 
  while(1){
    for(int i = 0 ; i < ctx->m_worker_machines; i++){
      void *buf = ctx->worker_recvportmap[i]->ctx->pull_entry_inq();       
      if(buf != NULL){	  
	mbuffer *mbuf = (mbuffer *)buf;
	long cmdid = cmdmgt->mark_oneack(mbuf);
	cmdmgt->print_syscmddoneq();
	ctx->worker_recvportmap[i]->ctx->release_buffer((void *)buf);
	cmdmgt->check_cmddone(cmdid); // check and release since mark_oneack is already checked  

      }	
    }

    for(int i = 0 ; i < ctx->m_sched_machines; i++){
      void *buf = ctx->scheduler_recvportmap[i]->ctx->pull_entry_inq();       
      if(buf != NULL){	  
	mbuffer *mbuf = (mbuffer *)buf;
	long cmdid = cmdmgt->mark_oneack(mbuf);
	cmdmgt->print_syscmddoneq();
	ctx->scheduler_recvportmap[i]->ctx->release_buffer((void *)buf);
	cmdmgt->check_cmddone(cmdid); // check and release since mark_oneack is already checked  
	// do not use assert .. here. Think about pending q mechanism. 
	// only when there is no pending mach across all cluster for a given cmd, cmd is moved from 
	// pending q to done queue. Here we sent one cmd id to two machines , 	
      }	
    }
    if((cmdmgt->get_syscmddoneq_size() == 0) && (cmdmgt->get_syscmdpendq_size() == 0)){ // if all shard commanda are done;
      break;
    }
  }

  strads_msg(ERR, "[coordinator] distributed sharding is done and got confirmation\n");

  weights = (double *)calloc(ctx->m_params->m_sp->m_modelsize, sizeof(double));
  _scheduler_start_remote(ctx, weights, ctx->m_params->m_sp->m_modelsize, true);     
  free(weights);


  strads_msg(ERR, "[coordinator] Send initial weight information to all schedulers to activate them\n");
  int rclock=0; // for round robin for scheduler 
  unordered_map<int64_t, idmvals_pair *>*retmap; 
  int64_t iteration=0;
  uint64_t stime = timenow();
  int64_t pending_iteration = 0;

  int64_t staleness = ctx->m_params->m_sp->m_pipelinedepth;


  //  int64_t switch_iter=(ctx->m_params->m_sp->m_modelsize/ctx->m_params->m_sp->m_maxset)*1.2 ;
  int64_t switch_iter=(ctx->m_params->m_sp->m_modelsize/ctx->m_params->m_sp->m_maxset)*3 ;

  strads_msg(ERR, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Staleness : %ld  (scheduling switch pt : %ld\n", 
	     staleness, switch_iter);

  int64_t logid=0;
  timers timer(10, 10);

#if defined(SVM_COL_PART) 
  // only for SVM debugging code. 
  // even for SVM, get rid of this after debugging.
  int64_t debug_task_clock=0;
  int64_t debutg_task_size = ctx->m_params->m_up->m_samples; 
#endif

  while(1){ // grand while loop 
    // CAVEAT: there should no no blocking command / func inside this grand while loop
    //         in order to work in asyncrhonous way    
    while(1){ // inner intinifite loop 

      // DEBUGGIGNG 
      void *buf = ctx->scheduler_recvportmap[rclock]->ctx->pull_entry_inq();       
      //void *buf = scheduling_emulator(debug_task_clock);

#if defined(SVM_COL_PART) 
      debug_task_clock++;
      debug_task_clock = debug_task_clock% debutg_task_size;
#endif

      if(buf != NULL){	
	timer.set_stimer(0);
	if(iteration > 0){
	  void *retcmd=NULL;
	  while(1){
	    retcmd = sthrds[0]->get_entry_outq();
	    if(retcmd != NULL){
	      coord_cmd *cmd = (coord_cmd *)retcmd;	  
	      stret *mret = (stret *)cmd->m_result;
	      retmap = mret->retmap;
	      pending_iteration++;
	      break;
	    }else{
	      int64_t gap = iteration - pending_iteration; 
	      if(gap <= staleness){
		retmap = new unordered_map<int64_t, idmvals_pair *>;
		assert(retmap->size() == 0);
		break;
	      }
	    }
	  }
	}else if(iteration == 0){
	  retmap = new unordered_map<int64_t, idmvals_pair *>;
	  assert(retmap->size() == 0);
	}else{
	  assert(0);	
	}

	timer.set_etimer(0);
	timer.set_stimer(1);

#if !defined(NO_WEIGHT_SAMPLING)
	if(iteration == switch_iter){
	  int toflush = 0;
	  strads_msg(ERR, "I will retry SCHEDULERS \n");
	  int schedmachs = ctx->m_params->m_sp->m_schedulers;
	  int thrds_per_sched = ctx->m_params->m_sp->m_thrds_per_scheduler;
	  int schedthrds = schedmachs * thrds_per_sched;
	  while(1){
	    for(int m=0; m < schedmachs; m++){
	      void *buf = ctx->scheduler_recvportmap[m]->ctx->pull_entry_inq();       
	      if(buf != NULL){	  
		ctx->scheduler_recvportmap[m]->ctx->release_buffer(buf);
		toflush++;
	      }
	    }	    
	    if(toflush == schedthrds-1)
	      break;
	  }
	  strads_msg(ERR, "I could Restart SCHEDULERS \n");
	  double *weights = (double *)calloc(ctx->m_params->m_sp->m_modelsize, sizeof(double));
	  for(int64_t mi=0; mi < modelsize; mi++){
	    weights[mi] = beta[mi];
	  }
	  _scheduler_start_remote(ctx, weights, ctx->m_params->m_sp->m_modelsize, true);     
	  free(weights);

	} // if(iteration == 12000 ) ... reset scheduler 
#endif 

	/* this buf contains SCHED_PHASE message */
	mbuffer *mbuf = (mbuffer *)buf;
	assert(mbuf->msg_type == SYSTEM_SCHEDULING);
	schedhead *schedhp = (schedhead *)mbuf->data;	
	int entrycnt = schedhp->entrycnt;
	int gthrdid = schedhp->sched_thrdgid;
	int schedmid = schedhp->sched_mid;
	assert(schedhp->type == SCHED_PHASE);

	strads_msg(INF, "New Phase: schemid(%d) schedthrdgid(%d) entrycnt(%ld)\n", 
		   schedhp->sched_mid, schedhp->sched_thrdgid, schedhp->entrycnt);
	int64_t *amo_gtaskids =  (int64_t *)((uintptr_t)schedhp + sizeof(schedhead));

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Dispatch tasks from scheduling 
	// USER code is involved to make a task message 
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	mbuffer *task = (mbuffer *)calloc(1, sizeof(mbuffer));
	task->msg_type = USER_UPDATE;
	workhead *workhp =  (workhead *)task->data;
	workhp->type = WORK_PARAMUPDATE;
	uobjhead *uobjhp = (uobjhead *)((uintptr_t)workhp + sizeof(workhead));
	//uobjhp points to buffer to hold new phase information orgainized by make_dispatch_msg .. function

	// fill out status update from the previous k-1 clock 
	// TODO : SWITICHING TO ASYNC, BE CAREFUL  ////////////////////////////////////////////////
	int64_t prevbetacnt = retmap->size();
	idval_pair *idvalp = NULL;	 

	timer.set_etimer(1);	
	timer.set_stimer(2);

	if(retmap->size() != 0){
	  idvalp = (idval_pair *)calloc(prevbetacnt, sizeof(idval_pair));	 
	  int progress=0;
	  for(auto p : *retmap){
	    idvalp[progress].id = p.second->id;
	    //	    idvalp[progress].value = betadiff[p.second->id];
	    idvalp[progress].value = p.second->sqpsum;
	    strads_msg(INF, "@@@@@@@@@@@ BETADIFF[%ld] = %lf\n", p.second->id, idvalp[progress].value); 
	    // TODO WHEN DO ASYNC >> BE CAREFUL CAVEAT 
	    // CAVEAT
	    progress++;
	  }
	}

	//	strads_msg(ERR, "\n\n");

	timer.set_etimer(2);
#if 1
	timer.set_stimer(3);

	for(auto p : *retmap){
	  free(p.second);
	}


	timer.set_etimer(3);
	timer.set_stimer(4);

	retmap->erase(retmap->begin(), retmap->end());       
	assert(retmap->size() == 0);	

	delete retmap;

	timer.set_etimer(4);
#endif 
	//	coord_cmd *gccmd = new coord_cmd(m_coord_gc, retmap, iteration);
	//	sthrds[gcthrd]->put_entry_inq((void *)gccmd);

	//	retmap=NULL;
	/////////////////////////////////////////////////////////////////////////////
	// TODO : change idvalp, and entry cnt as well when you replace above simulation code 
	// CAVEAT : dont forget that.

	timer.set_stimer(5);

	idval_pair *tasks = (TASK_ENTRY_TYPE *)calloc(entrycnt, sizeof(TASK_ENTRY_TYPE));
	for(int i=0; i < entrycnt; i++){
	  int64_t id = amo_gtaskids[i];
	  tasks[i].id = amo_gtaskids[i];
	  //	  tasks[i].value = util_get_double_random(0, 1.0); // beta[tasks[i].id] TODO from beta list
	  tasks[i].value = beta[id] ; // beta[tasks[i].id] TODO from beta list 
	}

	timer.set_etimer(5);
	timer.set_stimer(6);

	strads_msg(INF, "[COORDINATOR MAIN]: iteration(%ld)  size (%dd)\n", iteration, entrycnt);

      	user_func_make_dispatch_msg((void *)uobjhp, WORK_STATUPDATE, tasks, entrycnt, idvalp, prevbetacnt, 
				    USER_MSG_SIZE - sizeof(workhead));  

	int64_t scmdid = cmdmgt->get_cmdclock();       

	// timing log
	ctx->m_tloghandler->write_cmdevent_start_log(scmdid, 0, timenow());

	_mcopy_broadcast_to_workers(ctx, task, sizeof(mbuffer), scmdid);


	// memory leak remedy
	free(tasks);
	if( prevbetacnt > 0)
	  free(idvalp);
	// memory leak remedy

	// debugging code :
	if(entrycnt < 40){
	  strads_msg(INF, "[coordinator] Abnormal : entry cnt from the scheduler is less than 50, entrycnt: %d\n", entrycnt);
	}

	timer.set_etimer(6);
	timer.set_stimer(7);

	coord_cmd *cmd = new coord_cmd(m_coord_paramupdate, gthrdid, schedmid, entrycnt, amo_gtaskids, iteration, scmdid);
	sthrds[0]->put_entry_inq((void *)cmd);
	iteration++;

	// DEBUGGING
	ctx->scheduler_recvportmap[rclock]->ctx->release_buffer((void *)buf);
	//scheduling_release_buffer((void *)buf);

	rclock++;
	rclock = rclock % ctx->m_sched_machines;

	if(iteration % ctx->m_params->m_up->m_logfreq == 0){
#if 1
	  int64_t nz=0;
	  for(int64_t i =0; i < modelsize; i++){
	    if(beta[i] != 0)
	      nz++;
	  }
	  int64_t scmdid = cmdmgt->get_cmdclock();
	  ctx->m_tloghandler->write_cmdevent_start_log(scmdid, 0, timenow());
	  get_object_first_half(ctx, scmdid);
	  coord_cmd *cmd = new coord_cmd(m_coord_object, iteration, scmdid);
	  sthrds[0]->put_entry_inq((void *)cmd);
	  uint64_t etime = timenow();
	  strads_msg(ERR,"Hey %ld processed  elapsedtime : %lf second, NZ : %ld \n", 
		     iteration, (etime - stime)/1000000.0, nz);
	  ctx->write_log(logid++, (etime - stime));
#else
	  uint64_t etime = timenow();
	  strads_msg(ERR,"Hey %ld processed  elapsedtime : %lf second logid(%ld)\n",
		     iteration, (etime - stime)/1000000.0, logid);
#endif
	}

	if(iteration == ctx->m_params->m_up->m_iter){
	  strads_msg(ERR, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Congratulation ! Finishing task. see log file \n");	
	  sleep(5);
	  ctx->flush_log();	  
	  // timing log
	  ctx->m_tloghandler->flush_hdd();	
	  exit(0);
	}	  	
	timer.set_etimer(7);
	timer.update_elapsedtime(0, 7);
	if(iteration % ctx->m_params->m_up->m_logfreq == 0)
	  timer.print_elapsedtime(0, 7);

      } // buf != NULL









    } // while (1) -- inner infinite loop 

  } // end of outer grand while loop

  return NULL;
}




// created by scheduler_mach thread
// a thread in charge of one partition of whole task set 
// - run specified scheduling(weight sampling/dependency checking 
// - send results to the scheduler machine thread 
void *coordinator_thread(void *arg){ 

  coordinator_threadctx *ctx = (coordinator_threadctx *)arg; // this pointer of scheduler_threadctx class  
  strads_msg(ERR, "[Coordinator-thread] rank(%d) coordinatormach(%d) threadid(%d)\n", 
	     ctx->get_rank(), ctx->get_coordinator_mid(), ctx->get_coordinator_thrdid());

  sharedctx *shctx = ctx->m_shctx;
  int64_t iteration = 0;

  int64_t logid=0;

  int64_t updatedsofar=0;
  int64_t validcnt = 0;
  double totaldelta = 0.0;


  strads_msg(ERR, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ START Filling unordered map.....    \n\n");
  unordered_map<int64_t, int64_t>second_chance;
  int64_t modelsize = shctx->m_params->m_sp->m_modelsize;
  for(int64_t i=0; i < modelsize; i++){
    second_chance[i] =i;
  }

  strads_msg(ERR, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ END Filling unordered map.....    \n\n");


#if 0 
  // TEST CODE 
  idmvals_pair **pool = (idmvals_pair **)calloc(5000000, sizeof(idmvals_pair*));
  for(int pidx = 0; pidx<5000000; pidx++){
    pool[pidx] = (idmvals_pair *)calloc(1, sizeof(idmvals_pair));
    assert(pool[pidx]);
    memset(pool[pidx], 0x0, sizeof(idmvals_pair));
  }
  //  int poolidx = 0;
  // TEST CODE 
#endif 

#if !defined(NO_WEIGHT_SAMPLING)
  int64_t switch_iter=(shctx->m_params->m_sp->m_modelsize/shctx->m_params->m_sp->m_maxset)*1.2 ;
#endif 

  pthread_barrier_wait (&cbarrier);
  strads_msg(ERR, "@@@@ [coordinator-thread] Barrier pass !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  \n");

  string strCoeff("coeffRow");
  dshardctx *dshardCoeff = shctx->get_dshard_with_alias(strCoeff);
  assert(dshardCoeff);

  double **betatmp = dshardCoeff->m_dmat.m_mem;
  assert(betatmp);

  double *beta = betatmp[0];
  assert(beta);

  timers timer(20, 20);


  strads_msg(ERR, "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n\n");


  int counter = 0;

  while(1){
    void *rcmd = ctx->get_entry_inq_blocking(); // if inq is empty, this thread will be blocked until inq become non-empty 

    timer.set_stimer(10);

    coord_cmd *cmd = (coord_cmd *)rcmd;

#if !defined(SVM_COL_PART)
    if(cmd->m_type == m_coord_object){
      double psum = get_object_second_half(shctx);      
      double objectvalue = execute_user_object_server(shctx->m_params->m_up->m_funcmap, "getobject_server", psum, (void *)shctx);


      // MASTER LOG FILE WRITING 
      //      shctx->write_log(logid++, objectvalue);
      shctx->write_log_total(logid++, objectvalue, updatedsofar, validcnt, totaldelta);

      // timing log
      shctx->m_tloghandler->write_cmdevent_end_log(cmd->m_cmdid, 0, timenow());
      continue;
    }

#else 

    // in case of SVM, just call application specific code here.. temporarily
    if(cmd->m_type == m_coord_object){

      int64_t len = shctx->m_params->m_up->m_samples;
      string strY("Ycoordinator");
      dshardctx *dshardY = shctx->get_dshard_with_alias(strY);

      string strCoeff("coeffRow");
      dshardctx *dshardCoeff = shctx->get_dshard_with_alias(strCoeff);

      assert(dshardY);
      assert(dshardCoeff);

      double **Ycord = dshardY->m_dmat.m_mem;
      // access example Ycord[tmprow][0]

      double **alpha = dshardCoeff->m_dmat.m_mem;

      double wsquare=0;
      double *wrvector = (double *)calloc(sizeof(double), len);
      for(int wmid=0; wmid < shctx->m_worker_machines; wmid++){

	double partialwsq=0;
	idval_pair *partial_idvalp = receive_extended_packet(shctx, wmid, len, &partialwsq);

	for(int64_t j=0; j<len; j++){	  	
	  //	  wrvector[j] += Ycord[j][0]*partial_idvalp[j].value;
	  wrvector[j] += partial_idvalp[j].value;
	  assert(partial_idvalp[j].id == j);	 
	} 
	free(partial_idvalp);
	wsquare += partialwsq;
      }
     
      strads_msg(ERR, "COORDINATOR : wsquare : %10.10lf \n", wsquare);

      int64_t acc = 0;
      for(int64_t j=0; j<len; j++){	  		
	int pred = (wrvector[j] > 0) ? +1 : -1;
	if (pred == Ycord[j][0])
	  ++acc;
      }

      //      for(int64_t j=0; j<len; j++){
      //strads_msg(ERR, "@@@@@@@@@@@@ COORDINATOR : wrvecto(1 - y[i]WR) [%ld]  : %10.10lf      info (Y[%ld] = %10.10lf\n", 
      //		   j, wrvector[j], j, Ycord[j][0]);
      //      } 

      for(int64_t j=0; j<len; j++){
	//	wrvector[j] = 1 - wrvector[j];
	wrvector[j] = 1 - Ycord[j][0]*wrvector[j];
	//	strads_msg(ERR, "COORDINATOR : wrvecto(1 - y[i]WR) [%ld]  : %10.10lf      info (Y[%ld] = %10.10lf\n", 
	//		   j, wrvector[j], j, Ycord[j][0]);
      } 

      double hingeloss=0;
      for(int64_t j=0; j<len; j++){
	hingeloss += max(wrvector[j], 0.0);
      } 

      strads_msg(INF, "hihge loss : %lf  wsquare : %lf  Acc  %ld AC percent (%lf) \n", 
		 hingeloss, wsquare, acc, ((acc*1.0)/len)*100.0);

      double reg = 0.5*wsquare;    

      strads_msg(ERR, "@@@ Coordinator reg : %10.10lf\n", reg);
      
      double primal_obj = reg + shctx->m_params->m_up->m_cost*hingeloss;
      double alphasum=0;
      for(int64_t i=0; i<len; i++){
	alphasum += alpha[0][i];
      }

      double dual_obj = alphasum - reg;

      strads_msg(ERR, "primal_obj(%lf)   dual_obj(%lf)  primal_dual_gap(%lf)  hihge loss : %lf  wsquare : %lf  Acc  %ld AC percent (%lf) \n", 
		 primal_obj, dual_obj, (primal_obj - dual_obj), hingeloss, wsquare, acc, ((acc*1.0)/len)*100.0);
     
      free(wrvector);     
      // get Ycoord 
      shctx->write_log(logid++, hingeloss);
      shctx->m_tloghandler->write_cmdevent_end_log(cmd->m_cmdid, 0, timenow());
      continue;
    }
#endif 


    //#if !defined(NO_WEIGHT_SAMPLING)
    double *betadiff = shctx->m_weights;
    //#endif

    stmwork *mwork = (stmwork *)cmd->m_work;
    // DEBUGGING
    int gthrdid = mwork->gthrdid;
    int schedmid = mwork->schedmid;

    int64_t entrycnt = mwork->entrycnt;
    int64_t *amo_gtaskids = mwork->amo_gtaskids;
    //  pass amo_gtaskids to the second half thread
    unordered_map<int64_t, idmvals_pair *> *retmap = new unordered_map<int64_t, idmvals_pair *>; 
    timer.set_etimer(10);
    timer.set_stimer(11);

    for(int i=0; i < entrycnt; i++){
      int64_t id = amo_gtaskids[i];
      //
      idmvals_pair *tmp = (idmvals_pair *)calloc(1, sizeof(idmvals_pair)); 
      //TEST CODE 

      //idmvals_pair *tmp = (idmvals_pair *)pool[poolidx++];
      // it should be calloc since all entry should be zero  
      retmap->insert(std::pair<int64_t, idmvals_pair*>(id, tmp));
    }

    timer.set_etimer(11);
    timer.set_stimer(12);



    //    strads_msg(ERR, "[COORDINATOR THREAD] iteration(%d)  size (%ld)\n", counter, retmap->size());



    execute_user_aggregator(shctx->m_params->m_up->m_funcmap, "aggregator", *retmap,  (void *)shctx);       
    // do aggregation 

    timer.set_etimer(12);
    timer.set_stimer(13);

    // make new task status  back to the scheduler 
    uint64_t currentbetacnt = retmap->size();
    assert((int64_t)currentbetacnt == entrycnt);

    //int64_t updatedsofar=0;
    //  int64_t validcnt = 0;
    //    double totaldelta = 0.0;
    updatedsofar += currentbetacnt;

    if(retmap->size() != 0){
      int progress=0;
      for(auto p : *retmap){

	double absdelta = fabs(betadiff[p.second->id]);
	if(absdelta != 0.0){
	  validcnt ++;
	  totaldelta += absdelta;
	}

	if(second_chance.find(p.second->id) != second_chance.end()){	    
	  // do not touch results 
	  second_chance.erase(p.second->id);
	}else{
	  // not in the second chance, and zero become non-zero. 
	  // it's not safe to believe that. 
	  // so put that on the second chance. 
	  // if it become non zero again 
	  // believe it. 

#if !defined(NOREVOKING)
	  ////////////// this if statmenet is core for revoking /////////////////////////
	  if(p.second->psum == 0.0 and p.second->sqpsum != 0.0){
	    // THIS GUY NEED REVOKE ACTION. ZERO BECOME NON ZERO... 
	    beta[p.second->id] = 0.0;
	    betadiff[p.second->id] = betadiff[p.second->id]*5;	  
	    second_chance.insert(std::pair<int64_t, int64_t>(p.second->id, p.second->id));
	  }
	  //	  assert(0);
	  ////////////////////////////////////////////////////////////////////////////////
#endif

	}

	//	    idvalp[progress].id = p.second->id;
	//	    idvalp[progress].value  = betadiff[p.second->id];
	shctx->m_idvalp_buf[progress].id = p.second->id;
#if !defined(NO_WEIGHT_SAMPLING)
	if(iteration > switch_iter){
	  shctx->m_idvalp_buf[progress].value = betadiff[p.second->id];	


	  if(iteration % 10000 == 0){ // for debugging purpose, every 100 iterations, try to delta 
	    strads_msg(ERR, " @@@ iteration[%ld] ID [%ld]   BETA[%1.15lf] Beta-DIFF [%1.15lf] \n", 
		       iteration,
		       p.second->id, 
		       beta[p.second->id],
		       betadiff[p.second->id]);
	  }


	  if(p.second->id == 14760)
	    strads_msg(ERR, " @@@ ID [%ld]   Beta [%lf] \n", 
		       p.second->id, 
		       betadiff[p.second->id]);

	}else{
	  shctx->m_idvalp_buf[progress].value = 0.0;
	}
#else
	shctx->m_idvalp_buf[progress].value = 0.0;
	  if(p.second->id == 14760)
	    strads_msg(ERR, " @@@ ID [%ld]   Beta [%lf] \n", 
		       p.second->id, 
		       betadiff[p.second->id]);	
#endif
	// TODO WHEN DO ASYNC >> BE CAREFUL CAVEAT CAVEAT
	progress++;
      }
    }

    timer.set_etimer(13);
    timer.set_stimer(14);


    //DEBUGGING
    _send_weight_update(shctx, schedmid,  gthrdid, shctx->m_idvalp_buf,  currentbetacnt);       		      


    iteration++;
    //    ctx->scheduler_recvportmap[rclock]->ctx->release_buffer((void *)buf);
    /******************************************************************************************
	  Do not use assert .. here. Think about pending q mechanism. 
	  only when there is no pending mach across all cluster for a given cmd, cmd is moved from 
	  pending q to done queue. Here we sent one cmd id to two machines , 	
    *******************************************************************************************/
    timer.set_etimer(14);
    timer.set_stimer(15);

    int64_t scmdid = cmd->m_cmdid;
    shctx->m_tloghandler->write_cmdevent_end_log(scmdid, 0, timenow());

    delete cmd;

    timer.set_etimer(15);

    timer.set_stimer(16);


    coord_cmd *scmd = new coord_cmd(m_coord_paramupdate, retmap, iteration, scmdid);
    ctx->put_entry_outq((void *)scmd);

    timer.set_etimer(16);


    timer.update_elapsedtime(10, 16);

    if(iteration % 1000 == 0){

      timer.print_elapsedtime(10, 16);
    }


    counter++;



  }
  return NULL;
}


void  get_object_first_half(sharedctx *ctx, int64_t cmdid){
  mbuffer *task = (mbuffer *)calloc(1, sizeof(mbuffer));
  task->msg_type = USER_PROGRESS_CHECK;
  workhead *workhp =  (workhead *)task->data;
  workhp->type = WORK_OBJECT;
  //  uobjhead *uobjhp = (uobjhead *)((uintptr_t)workhp + sizeof(workhead));


  strads_msg(ERR, "[coordinator] send object calc command to all worker though mcopy broadcast \n");
  _mcopy_broadcast_to_workers(ctx, task, sizeof(mbuffer), cmdid);

  return;
}


double get_object_second_half(sharedctx *ctx){

  double psum =0;
  for(int i=0; i < ctx->m_worker_machines; i++){
    while(1){
      void *buf = ctx->worker_recvportmap[i]->ctx->pull_entry_inq();	   
      if(buf != NULL){	  	  
	mbuffer *tmpbuf = (mbuffer *)buf;
	workhead *retworkhp = (workhead *)tmpbuf->data;
	assert(retworkhp->type == WORK_OBJECT);	    
	uobjhead *retuobjhp = (uobjhead *)((uintptr_t)retworkhp + sizeof(workhead));
	assert(retuobjhp->task_cnt == 1);
	idval_pair *idvalp = (idval_pair *)((uintptr_t)retuobjhp + sizeof(uobjhead));
	assert(idvalp[0].id == -1);       
	psum += idvalp[0].value;
	ctx->worker_recvportmap[i]->ctx->release_buffer(buf); // dont' forget 	   
	break;
      }	
    }
  }       
  strads_msg(ERR, "COORDINATOR GOT  OBJECT  PARTIAL RESULTS FROM ALL WORKERS \n");
  return psum;
}



idval_pair *receive_extended_packet(sharedctx *ctx, int src, int64_t len, double *wsquare){

  strads_msg(INF, "[coordinator] receive extended packet with src(%d) is called \n", src);

  idval_pair *idvalp = (idval_pair *)calloc(sizeof(idval_pair), len);
  int64_t progress=0;
  double partial_weightdot=0;

  while(1){
    void *buf = ctx->worker_recvportmap[src]->ctx->pull_entry_inq();	   

    if(buf != NULL){	  	  
     
      mbuffer *tmpbuf = (mbuffer *)buf;
      workhead *retworkhp = (workhead *)tmpbuf->data;
      assert(retworkhp->type == WORK_OBJECT);	    
      uobjhead *retuobjhp = (uobjhead *)((uintptr_t)retworkhp + sizeof(workhead));
      int64_t cnt = retuobjhp->stat_cnt;     
      idval_pair *r_idvalp = (idval_pair *)((uintptr_t)retuobjhp + sizeof(uobjhead));

      int64_t remain = tmpbuf->remain;

      if(remain > 0){
      
	for(int64_t i=0; i<cnt; i++){
	  idvalp[progress].id = r_idvalp[i].id;
	  idvalp[progress].value = r_idvalp[i].value;       

	  if(idvalp[progress].id == 0 or idvalp[progress].id == (len-1) ){
	    strads_msg(ERR, "[coordinator ] got obje sum infom for id (%ld) == %lf from src (%d) remain(%ld)\n", 
		       idvalp[progress].id, idvalp[progress].value, src, remain); 
	  }
	  progress++;
	}
      }else{

	strads_msg(ERR, "remain should be zero (%ld) id[0]->id (%ld)\n",
		   remain, r_idvalp[0].id);

	assert(r_idvalp[0].id == -1);
	partial_weightdot = r_idvalp[0].value;
      }

      ctx->worker_recvportmap[src]->ctx->release_buffer(buf); // dont' forget 	   

      strads_msg(INF, "\t[coordinator] receive extended packet with src(%d) remain(%ld) \n", src, remain);
      if(remain == 0)
	break;
    }	
  }

  *wsquare = partial_weightdot;
  
  assert(progress == len);

  strads_msg(ERR, "[coordinator] finish receiving partisl results from a src(%d) wsquaresum : %lf\n", src, *wsquare);
  return idvalp;
}






// check file name, matrix format  and where the file need to be uploaded, 
// send dshard command to remote machines 
int _get_tosend_machinecnt(sharedctx *ctx, machtype mtype){
  int ret;
  if(mtype == m_worker){
    ret = ctx->m_worker_machines;
  }else if(mtype == m_scheduler){
    ret = ctx->m_sched_machines;
  }else if(mtype == m_coordinator){
    ret = 1;
  }else{
    LOG(FATAL) << "get tosend machinecnt : not supported type " << endl;
  }
  return ret;
}

// check file name, matrix format  and where the file need to be uploaded, 
// send dshard command to remote machines 
int _get_totalthreads_formachtype(sharedctx *ctx, machtype mtype){
  int ret;  
  if(mtype == m_worker){
    ret = ctx->m_worker_machines * ctx->m_params->m_sp->m_thrds_per_worker ;
  }else if(mtype == m_scheduler){
    ret = ctx->m_sched_machines * ctx->m_params->m_sp->m_thrds_per_scheduler;
  }else if(mtype == m_coordinator){
    ret = 1*ctx->m_params->m_sp->m_thrds_per_coordinator;
  }else{
    LOG(FATAL) << "get tosend machinecnt : not supported type " << endl;
  }
  return ret;
}

void _load_dshard_remote(sharedctx *ctx, userfn_entity *fne, ppacketmgt *cmdmgt){
  const char *sp = fne->m_strfn.c_str();
  const char *type = fne->m_strtype.c_str(); // row / column major 
  const char *pscheme = fne->m_strpscheme.c_str(); // partitioning scheme..
  const char *strmachtype = fne->m_strmachtype.c_str(); // on what machines, data will be partitione
  const char *stralias = fne->m_stralias.c_str();
  machtype mtype = fne->m_mtype;
  strads_msg(ERR, "[coordinator load dshard remote] file to load: [%s] assigned to [%s]\n", sp, strmachtype);
  uint64_t rows, cols, nz;





  if(strcmp(sp, "empty")!= 0){
    iohandler_spmat_pbfio_read_size(sp, &rows, &cols, &nz); 
  }else{

    // now only SVM use empty type... 
    //    rows = ctx->m_params->m_up->m_samples;
    //    cols = 1;
    // TODO : Generalize this. 
    if(strcmp(stralias, "WeightRow") == 0){
      rows = ctx->m_params->m_up->m_empty7_size; // for SVM
    }else if (strcmp(stralias, "Weight")==0){
      rows = ctx->m_params->m_up->m_empty5_size; // for SVM 
    }else if (strcmp(stralias, "Ax") == 0){ // for Logistic Regression 
      rows = ctx->m_params->m_up->m_samples;
    }

    cols = 1;
    strads_msg(ERR, "@@@@@ Warning : empty type data structure is initialized with row = user parameter m_empty5_size, col = 1\n");
    // TODO : later, allow users to set their in memory data size 
  }

  strads_msg(ERR, "\t\tdata summary: rows(%ld) cols(%ld) nz(%ld)\n", rows, cols, nz);
  // row major format since row major partitioning  
  dshardctx *dshard=NULL;
  shard_info *shardmaps;


  if( (strcmp(type, "rmspt")== 0) || (strcmp(type, "rvspt")== 0) ){
    //    dshard = new dshardctx(rows, cols, rmspt, sp);


    strads_msg(ERR, "[coordinatoor split revoke ] RVSPT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n");

    if(strcmp(type, "rmspt")==0){
      dshard = new dshardctx(rows, cols, rmspt, sp, stralias);
    }else if(strcmp(type, "rvspt")==0){
      dshard = new dshardctx(rows, cols, rvspt, sp, stralias);
    }

    shardmaps = new shard_info;
    int machines = _get_tosend_machinecnt(ctx, mtype); // currently, data assignment support scheduler / worker grain . 
    // TODO : in future, support more flexible data (partition) / machine mapping 
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //    dshard_make_finepartition(rows, machines, shardmaps->finemap, ctx->rank);
    map<int, range *>tmap;
    int finepartitions = _get_totalthreads_formachtype(ctx, mtype);
    if(strcmp(pscheme, "row") == 0){
      dshard_make_finepartition(rows, finepartitions, tmap, ctx->rank, true);
    }else if(strcmp(pscheme, "col") == 0){
      dshard_make_finepartition(cols, finepartitions, tmap, ctx->rank, true);      
    }else{
      assert(0);
    }
    dshard_make_superpartition(machines, tmap, shardmaps->finemap, ctx->rank, true);
    // CAVEAT : Must keep two level here 
    //          Since potential data/model partitioning is already done in 
    //          sharedctx method using two level method. 
    //          If one level method used here, in some cases, 
    //           there would be slight boundary mismatch due to the c precision error. 
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  }else if( (strcmp(type, "cmspt")==0) || (strcmp(type, "cvspt")==0)){
    //    dshard = new dshardctx(rows, cols, cmspt, sp);


    strads_msg(ERR, "[coordinatoor split revoke ] CVSPT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n");

    if(strcmp(type, "cmspt")==0){
      dshard = new dshardctx(rows, cols, cmspt, sp, stralias);
    }else{
      dshard = new dshardctx(rows, cols, cvspt, sp, stralias);
    }
      
    shardmaps = new shard_info;
    int machines = _get_tosend_machinecnt(ctx, mtype);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //    dshard_make_finepartition(cols, machines, shardmaps->finemap, ctx->rank);
    map<int, range *>tmap;
    int finepartitions = _get_totalthreads_formachtype(ctx, mtype);

    if(strcmp(pscheme, "row") == 0){
      dshard_make_finepartition(rows, finepartitions, tmap, ctx->rank, true);
    }else if(strcmp(pscheme, "col") == 0){
      dshard_make_finepartition(cols, finepartitions, tmap, ctx->rank, true);      
    }else{
      assert(0);
    }
    //    dshard_make_finepartition(cols, finepartitions, tmap, ctx->rank);
    dshard_make_superpartition(machines, tmap, shardmaps->finemap, ctx->rank, true);
    // CAVEAT : Must keep two level here 
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  }else if (strcmp(type, "d2dmat") == 0) {

    strads_msg(ERR, "\t\t[coordinatoor split revoke ] D2DMAT @@@ - now 0000000000000000000000000000000000  \n");

    dshard = new dshardctx(rows, cols, d2dmat, sp, stralias);

    strads_msg(ERR, "[coordinatoor split revoke ] D2DMAT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@P---------------------1 \n");

    shardmaps = new shard_info;

    strads_msg(ERR, "[coordinatoor split revoke ] D2DMAT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@P0 \n");

    int machines = _get_tosend_machinecnt(ctx, mtype);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //    dshard_make_finepartition(cols, machines, shardmaps->finemap, ctx->rank);
    map<int, range *>tmap;
    int finepartitions = _get_totalthreads_formachtype(ctx, mtype);

    strads_msg(ERR, "[coordinatoor split revoke ] D2DMAT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@P1 \n");

    if(strcmp(pscheme, "row") == 0){
      dshard_make_finepartition(rows, finepartitions, tmap, ctx->rank, true);
      dshard_make_superpartition(machines, tmap, shardmaps->finemap, ctx->rank, true);
    }else if(strcmp(pscheme, "col") == 0){
      dshard_make_finepartition(cols, finepartitions, tmap, ctx->rank, true);      
      dshard_make_superpartition(machines, tmap, shardmaps->finemap, ctx->rank, true);
    }else if(strcmp(pscheme, "NO") == 0){
      // do nothing sicne it is not supposed to be partitioned. 
    }else{
      
      assert(0);
    }

    //    dshard_make_superpartition(machines, tmap, shardmaps->finemap, ctx->rank, true);
    // CAVEAT : Must keep two level here 
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    strads_msg(ERR, "[coordinatoor split revoke ] D2DMAT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@END \n");

  }else{
    LOG(FATAL) << "user data type that is not supported " << type << endl;
  }







  long cmdid = cmdmgt->get_cmdclock();
  cmdmgt->push_cmdq(cmdid, mtype);
  int machines = _get_tosend_machinecnt(ctx, mtype);
  for(int k=0; k < machines; k++){
    rangectx *rctx = new rangectx;
    if(strcmp(pscheme, "row")== 0){
      rctx->r_start = shardmaps->finemap[k]->start;
      rctx->r_end = shardmaps->finemap[k]->end;
      rctx->r_len = rctx->r_end - rctx->r_start + 1;
      rctx->c_start = 0; // since row partition, each machine cover all columns
      rctx->c_end = cols-1;
      rctx->c_len = rctx->c_end - rctx->c_start+1;
    } else if( strcmp(pscheme, "col")==  0){
      rctx->r_start = 0; // since column partition, each machine cover all rows 
      rctx->r_end = rows-1;
      rctx->r_len = rctx->r_end - rctx->r_start + 1;
      rctx->c_start = shardmaps->finemap[k]->start;
      rctx->c_end = shardmaps->finemap[k]->end;
      rctx->c_len = rctx->c_end - rctx->c_start + 1;
      strads_msg(ERR, "@@@@@ i[%d] c_start(%ld) c_end(%ld)\n", 
		 k, rctx->c_start, rctx->c_end);
    }else if(strcmp(pscheme, "NO") == 0){
      rctx->r_start = 0; // since column partition, each machine cover all rows 
      rctx->r_end = rows-1;
      rctx->r_len = rctx->r_end - rctx->r_start + 1;
      rctx->c_start = 0;
      rctx->c_end = cols-1;
      rctx->c_len = rctx->c_end - rctx->c_start + 1;

      strads_msg(ERR, "@@@@@ i[%d] c_start(%ld) c_end(%ld)\n", 
		 k, rctx->c_start, rctx->c_end);
    }else{
      assert(0); // not yet supported format.
    }

    strads_msg(ERR, "[coordinator -- load remote ]  rctx->r_start (%ld) r_end (%ld)\n", 
	       rctx->r_start, rctx->r_end);
    // CAVEAT: bind range ctx with dshard context. This is very important. Don't forget
    dshard->set_range(rctx); // Don't forget this 
    // copy dshard's partitioned range info into mini dshard for marshalling
    mini_dshardctx *mshard = new mini_dshardctx(dshard->m_rows, dshard->m_cols, dshard->m_mpartid, 
						dshard->m_type, dshard->m_fn, &dshard->m_range, 
						dshard->m_finecnt, dshard->m_supercnt, dshard->m_alias); 

    mbuffer *mbuf = (mbuffer *)calloc(1, sizeof(mbuffer));
    sys_packet *spacket = (sys_packet *)mbuf->data;
    mbuf->msg_type = SYSTEM_DSHARD;
    memcpy((void *)spacket->syscmd, (void *)mshard, sizeof(mini_dshardctx));

    mbuf->cmdid = cmdid;

    if(mtype == m_worker){
      _send_to_workers(ctx, mbuf, sizeof(mbuffer), k);
    }else if(mtype == m_scheduler){
      _send_to_scheduler(ctx, mbuf, sizeof(mbuffer), k);
    }else{
      LOG(FATAL) << " load_dshard_remote : not yet supported machien type" << endl;
    }

    delete mshard;

    // TODO URGENT : think about memory allocation/leak
    // TODO : keep dshard information. 
  } // for(int k = 0 ; k < machines 

  delete dshard;

}

/* Caller: coordinator only
   coordiantor will call this function to let remote scheduler machines be ready for receiving 
   the following initial weight information and start their service
*/
void _scheduler_start_remote(sharedctx *ctx, double *weights, uint64_t wsize, bool rflag){
  int machines = _get_tosend_machinecnt(ctx, m_scheduler);
  strads_msg(ERR, "Machines : Scheduler %d \n", machines);
  for(int i = 0; i < machines; i++){
    mbuffer *mbuf = (mbuffer *)calloc(1, sizeof(mbuffer));
    mbuf->msg_type = SYSTEM_SCHEDULING;
    mbuf->src_rank = ctx->rank;
    schedhead *schedhp = (schedhead *)mbuf->data;

    int64_t dlen = USER_MSG_SIZE - sizeof(schedhead);    


    if(rflag == true){
      schedhp->type = SCHED_RESTART;
    }else{
      schedhp->type = SCHED_START;
    }

    schedhp->sched_mid = i;
    schedhp->sched_thrdgid = -1; // not for a specific threads, it's for a scheduler machine
    schedhp->entrycnt = 1;
    schedhp->dlen = sizeof(sched_start_p);

    sched_start_p *amo = (sched_start_p *)((uintptr_t)schedhp + sizeof(schedhead));
    auto p = ctx->m_tmap.schmach_tmap.find(i);
    assert(p != ctx->m_tmap.schmach_tmap.end());

    int64_t taskcnt = p->second->end - p->second->start +1;
    int64_t start = p->second->start; 
    int64_t end = p->second->end; 
    int64_t entry_per_chunk = dlen / (2*sizeof(idval_pair));
    int64_t chunks = taskcnt / entry_per_chunk;
   
    if( taskcnt % entry_per_chunk == 0){
      // no remain in amo->chunks = (taskcnt)*(2*sizeof(idval_pair)) / dlen
    }else{
      chunks++; // since there was remain in the  amo->chunks = (taskcnt)*(2*sizeof(idval_pair)) / dlen;
    }

    amo->taskcnt = taskcnt;
    amo->start = start;
    amo->end = end;
    amo->chunks = chunks;    
    strads_msg(ERR, "[Coordinator start remote] for %d schedmach start(%ld) end(%ld) chunks(%ld)\n", 
	       i, start, end, chunks);

    _send_to_scheduler(ctx, mbuf, sizeof(mbuffer), i); // once sent through com stack, it will be realeased

    int64_t progress=0;
    assert(entry_per_chunk * chunks >= taskcnt);
    for(int64_t ci=0; ci < chunks; ci++){      
      mbuf = (mbuffer *)calloc(1, sizeof(mbuffer));
      mbuf->msg_type = SYSTEM_SCHEDULING;
      mbuf->src_rank = ctx->rank;
      schedhp = (schedhead *)mbuf->data;
      idval_pair *pairs = (idval_pair *)((uintptr_t)schedhp + sizeof(schedhead));
      int64_t entrycnt = 0;
      for(int64_t ei=0; ei < entry_per_chunk; ei++){       
	pairs[ei].value = weights[start + ci*entry_per_chunk + ei];
	pairs[ei].id = start + ci*entry_per_chunk + ei;
	progress++;
	entrycnt++;

	//	strads_msg(ERR, "Coordinator pairs[%ld].id : %ld  \n", ei, pairs[ei].id);

	if(progress == taskcnt){
	  break;
	}
      }
      schedhp->type = SCHED_INITVAL;
      schedhp->sched_mid = i;
      schedhp->sched_thrdgid = -1;
      schedhp->entrycnt = entrycnt;
      schedhp->dlen = entrycnt*sizeof(idval_pair); 
      //      strads_msg(ERR, "Coordinator entrycnt  : %ld  \n", entrycnt);     
      _send_to_scheduler(ctx, mbuf, sizeof(mbuffer), i); // once sent through com stack, it will be realeased
      //      while(1);
      usleep(100);
    }

    void *buf;
    while(1){
      buf = ctx->scheduler_recvportmap[i]->ctx->pull_entry_inq();       
      if(buf != NULL){
	break;
      }
    }
    if(buf != NULL){	  
      mbuffer *mbuf = (mbuffer *)buf;
      assert(mbuf->msg_type == SYSTEM_SCHEDULING);
      schedhead *schedhp = (schedhead *)mbuf->data;

      assert(schedhp->type == SCHED_START_ACK);

      sched_start_p *amo = (sched_start_p *)((uintptr_t)schedhp + sizeof(schedhead));      
      int64_t rtaskcnt = amo->taskcnt;
      if(rtaskcnt != taskcnt){ // sent taskcnt in coordinator and received taskcnt in remote side not match
	LOG(FATAL) << "Mismatch of taskcnt in remote side Expected: "<< taskcnt << " received: " << rtaskcnt << endl;
      }else{
	strads_msg(ERR, "\t\t init weight receive ACK from sched_mid(%d)\n", i);
      }
      ctx->scheduler_recvportmap[i]->ctx->release_buffer((void *)buf);
    }	
    strads_msg(ERR, "\tinit weight to sched_mid(%d) -- done \n", i);
  }  
}



/* soft_thrd, multiresidual  for workers */
static double _soft_threshold(double sum, double lambda){
  double res;
  if(sum >=0){
    if(sum > lambda){
      res = sum - lambda;
    }else{
      res = 0;
    }
  }else{
    if(sum < -lambda){
      res = sum + lambda;
    }else{
      res = 0;
    }
  }
  return res;
}

// created by scheduler_mach thread
// a thread in charge of one partition of whole task set 
// - run specified scheduling(weight sampling/dependency checking 
// - send results to the scheduler machine thread 
void *coordinator_gc(void *arg){ 
  coordinator_threadctx *ctx = (coordinator_threadctx *)arg; // this pointer of scheduler_threadctx class  
  strads_msg(ERR, "[Coordinator-GC] rank(%d) coordinatormach(%d) threadid(%d)\n", 
	     ctx->get_rank(), ctx->get_coordinator_mid(), ctx->get_coordinator_thrdid());
  //  sharedctx *shctx = ctx->m_shctx;
  //  int64_t iteration = 0;
  //  int64_t logid=0;
  //  timers timer(20, 20);
  unordered_map<int64_t, idmvals_pair *>*retmap; 
  while(1){
    void *rcmd = ctx->get_entry_inq_blocking(); // if inq is empty, this thread will be blocked until inq become non-empty 
    coord_cmd *cmd = (coord_cmd *)rcmd;
    if(cmd->m_type == m_coord_gc){


      stret *mret = (stret *)cmd->m_result;
      retmap = mret->retmap;
      assert(retmap);


      //      strads_msg(ERR, "GC is called, retmap size : %ld\n", retmap->size());

      for(auto p : *retmap){
	free(p.second);
      }
      retmap->erase(retmap->begin(), retmap->end());
      delete retmap;

      delete cmd;
    }else{
      assert(0);
    }
  }
  return NULL;
}
