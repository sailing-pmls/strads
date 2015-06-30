
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
#include "com/comm.hpp"
#include "ds/dshard.hpp"
#include "ds/binaryio.hpp"
//#include "com/rdma/rdma-common.hpp"

#if defined(INFINIBAND_SUPPORT)

#include "com/rdma/rdma-common.hpp"

#else

#include "com/zmq/zmq-common.hpp"

#endif


#include "worker/worker.hpp"
#include <glog/logging.h>

using namespace std;

pthread_barrier_t   barrier; // barrier synchronization object

void *process_worker_cmd(sharedctx *ctx, mbuffer *mbuf, worker_threadctx **sthrds, context *recv_ctx, context *send_ctx, timers &timer){
  if(mbuf == NULL)
    return NULL; // since previous handler processes that and set mbuf to NULL

  assert(mbuf->msg_type != USER_PROGRESS_CHECK); 
  // this should be handled in previous process object cmd handler 

  int maxthrds = ctx->m_params->m_sp->m_thrds_per_worker;

  int64_t cmdid = mbuf->cmdid;
  
  if(mbuf->msg_type == USER_UPDATE){  

    workhead *workhp = (workhead *)mbuf->data;
    if(workhp->type == WORK_STATUPDATE){  // in cd case, mostly, residual update 
      // skip this. I merge residual update with parameter update 
    } else if(workhp->type == WORK_PARAMUPDATE){ // weight update   
      strads_msg(INF, "Rank(%d) got param update cmd from coordinator \n", ctx->rank);      

      timer.set_stimer(2);

      workhead *workhp = (workhead *)mbuf->data;
      uobjhead *uobjhp = (uobjhead *)((uintptr_t)workhp + sizeof(workhead));
      int64_t taskcnt = uobjhp->task_cnt;
      int64_t statcnt = uobjhp->stat_cnt;      
      TASK_ENTRY_TYPE *ids = (TASK_ENTRY_TYPE *)((uintptr_t)uobjhp + sizeof(uobjhead));
      idval_pair *idvalp = (idval_pair *)((uintptr_t)ids + taskcnt*sizeof(TASK_ENTRY_TYPE)); 
      //      int64_t freethrds = ctx->get_freethrdscnt();
      int64_t freethrds = maxthrds;

      int64_t share = taskcnt / freethrds;
      int64_t remain = taskcnt % freethrds;
      int64_t progress = 0;

      timer.set_etimer(2);
      timer.set_stimer(3);
      int th=0;

      work_cmd **cmd = (work_cmd **)calloc(freethrds, sizeof(work_cmd*));

      for(th=0; th<freethrds; th++){
	//	work_cmd *cmd = new work_cmd(m_work_paramupdate);
	cmd[th] = new work_cmd(m_work_paramupdate);

	cmd[th]->m_cmdid = cmdid;

	void *tmp = (void *)calloc(1, sizeof(uobjhead) + sizeof(TASK_ENTRY_TYPE)*taskcnt + sizeof(idval_pair)*statcnt);	
	uobjhead *uobjhp = (uobjhead *)tmp;
	TASK_ENTRY_TYPE *cids = (TASK_ENTRY_TYPE *)((uintptr_t)uobjhp + sizeof(uobjhead));       
	cmd[th]->m_work = tmp;	
	int64_t tmpcnt = 0;
	int64_t tmpshare = share;
	if(remain > 0){
	  tmpshare++;
	  remain--;
	}
	for(int64_t i=0; i < tmpshare; i++){
	  cids[i].value = ids[progress].value; // id's current coefficient 
	  cids[i].id = ids[progress].id;	 
	  progress++;

	  tmpcnt++;

	  if(progress == taskcnt)	   
	    break;
	}
	uobjhp->task_cnt = tmpcnt;
	if(progress == taskcnt)	   
	  break;

      }


      if((th + 1 ) != freethrds){

	strads_msg(ERR, "th (%d) freethrds(%ld)  progress(%ld)  taskcnt(%ld) statcnt(%ld) \n", 
		   th, freethrds, progress, taskcnt, statcnt);
      }



      assert((th+1) == freethrds);      
      assert(progress == taskcnt);

      share = statcnt / freethrds;
      remain = statcnt % freethrds;
      progress = 0;

      for(th=0; th<freethrds; th++){
	void *tmp = cmd[th]->m_work;
	uobjhead *uobjhp = (uobjhead *)tmp;
	int64_t stat_start = uobjhp->task_cnt;

	TASK_ENTRY_TYPE *cids = (TASK_ENTRY_TYPE *)((uintptr_t)uobjhp + sizeof(uobjhead));              
      	idval_pair *cidvalp = (idval_pair *)((uintptr_t)cids + stat_start*sizeof(TASK_ENTRY_TYPE)); 

	int64_t tmpcnt = 0;
	int64_t tmpshare = share;
	if(remain > 0){
	  tmpshare++;
	  remain--;
	}
	for(int64_t i=0; i < tmpshare; i++){
	  cidvalp[i].value = idvalp[progress].value; // id's current coefficient 
	  cidvalp[i].id = idvalp[progress].id;	 
	  progress++;
	  tmpcnt++;
	  if(progress == statcnt)	   
	    break;
	}
	uobjhp->stat_cnt = tmpcnt;      
	if(progress == statcnt)	   
	  break;	
      }
      assert(progress == statcnt);
      assert((th + 1) == freethrds || statcnt == 0);

      for(th=0; th<freethrds; th++){
	sthrds[th]->put_entry_inq((void*)cmd[th]);       
      }
      // scatter the task ........ 
    } else if(workhp->type == WORK_OBJECT){ // weight update   
      assert(0); // this should be handled in the previous process obj cmd handler       
    }else{
      LOG(FATAL) << "[worker] USER_WORKER msg contains non-support sched-type " << endl;
    }
    recv_ctx->release_buffer((void *)mbuf); // don't forget this           
    timer.set_etimer(3);

    return NULL;
  } 
  return mbuf ;
}

// for lasso / logistic regression 
void *first_half_process_obj_cmd(sharedctx *ctx, mbuffer *mbuf, worker_threadctx **sthrds, context *recv_ctx, context *send_ctx, void **retbuf){
  if(mbuf == NULL)
    return NULL; // since previous handler processes that and set mbuf to NULL

  if(mbuf->msg_type != USER_PROGRESS_CHECK)
    return mbuf;

  if(mbuf->msg_type == USER_PROGRESS_CHECK){  

    workhead *workhp = (workhead *)mbuf->data;
    assert(workhp->type == WORK_OBJECT);  
    double pobjsum = execute_user_object(ctx->m_params->m_up->m_funcmap, "getobject", (void *)ctx);  
    recv_ctx->release_buffer((void *)mbuf); // don't forget this                  
    mbuffer *mtmp = (mbuffer *)calloc(1, sizeof(mbuffer));
    mtmp->msg_type = USER_PROGRESS_CHECK;    
    workhead *retworkhp = (workhead *)mtmp->data;
    retworkhp->type = WORK_OBJECT;
    uobjhead *uobjhp = (uobjhead *)((uintptr_t)retworkhp + sizeof(workhead));
    uobjhp->task_cnt = 1;
    idval_pair *idvalp = (idval_pair *)((uintptr_t)uobjhp + sizeof(uobjhead));
    idvalp[0].id = -1;
    idvalp[0].value = pobjsum;    
    mtmp->cmdid = mbuf->cmdid;
    *retbuf = mtmp; // pass this to the second half and let it send to coordinator without any modification

  }
  return NULL;
}

// for SVM
void *first_half_process_obj_cmd(sharedctx *ctx, mbuffer *mbuf, worker_threadctx **sthrds, context *recv_ctx, context *send_ctx, idval_pair **idvalp, int64_t len, double *wsquare){


  double weightsquare;
  if(mbuf == NULL)
    return NULL; // since previous handler processes that and set mbuf to NULL

  if(mbuf->msg_type != USER_PROGRESS_CHECK)
    return mbuf;

  if(mbuf->msg_type == USER_PROGRESS_CHECK){  
    workhead *workhp = (workhead *)mbuf->data;
    assert(workhp->type == WORK_OBJECT);  

    strads_msg(ERR, "[worker] Rank(%d) start object calc. call user func \n", ctx->rank);

    idval_pair *idvalp_assigned = (idval_pair *)calloc(sizeof(idval_pair), len);
    weightsquare = execute_user_object(ctx->m_params->m_up->m_funcmap, "getobject", (void *)ctx, idvalp_assigned, len);     
    *idvalp = idvalp_assigned;

    strads_msg(ERR, "[worker] Rank(%d) finish object calc. call user func \n", ctx->rank);

    recv_ctx->release_buffer((void *)mbuf); // don't forget this                  
  }
  *wsquare = weightsquare;
  return NULL;
}






void *worker_mach(void *arg){

  sharedctx *ctx = (sharedctx *)arg;
  strads_msg(ERR, "[worker-machine] rank(%d) boot up worker-mach (%d) \n", ctx->rank, ctx->m_worker_mid);
  assert(ctx->m_worker_mid >=0);
  worker_threadctx **sthrds = (worker_threadctx **)calloc(MAX_SCHEDULER_THREAD, sizeof(worker_threadctx *));
  int thrds = ctx->m_params->m_sp->m_thrds_per_worker;  

  pthread_barrier_init (&barrier, NULL, thrds);

  //  pthread_mutex_t pmutex = PTHREAD_MUTEX_INITIALIZER;
  //  int64_t freethrds;


  for(int i=0; i<thrds; i++){    
    //    sthrds[i] = new worker_threadctx(ctx->rank, ctx->m_worker_mid, i);
    sthrds[i] = new worker_threadctx(ctx->rank, ctx->m_worker_mid, i, &ctx->m_freethrds_lock, 
				     &ctx->m_freethrds, ctx->m_params, ctx->m_worker_machines, ctx);
  }


  whalf_threadctx *whalfthrds = new whalf_threadctx(ctx->rank, ctx->m_worker_mid, 0, &ctx->m_freethrds_lock, 
						    &ctx->m_freethrds, ctx->m_params, ctx->m_worker_machines, ctx, sthrds);


  // TODO: modify the following if you do not user pure star topology
  strads_msg(ERR, "[worker] rank(%d) worker_mid(%d) has recvport(%lu) sendport(%lu)\n",
	     ctx->rank, ctx->m_worker_mid, ctx->star_recvportmap.size(), ctx->star_sendportmap.size()); 

  sleep(5); // wait for all worker threads to be created 
  strads_msg(ERR, "\t\t Rank(%d) got (%ld) free threads (%d) remove infinit loop\n", 
	     ctx->rank, ctx->get_freethrdscnt(), ctx->rank);

  assert(ctx->star_recvportmap.size() == 1);
  assert(ctx->star_sendportmap.size() == 1);
  auto pr = ctx->star_recvportmap.begin();
  _ringport *rport = pr->second;
  context *recv_ctx = rport->ctx;
  auto ps = ctx->star_sendportmap.begin();
  _ringport *sport = ps->second;
  context *send_ctx = sport->ctx;


  //  int64_t iterations=0;

#if defined(SVM_COL_PART)
  idval_pair *idvalp =  NULL; //(idval_pair *)calloc(sizeof(idval_pair), ctx->m_params->m_up->m_params);
  int64_t len = ctx->m_params->m_up->m_samples;
#endif  

  timers timer(10, 10);

  while (1){
    void *msg = recv_ctx->pull_entry_inq();
    if(msg != NULL){

      mbuffer *mbuf = (mbuffer *)msg;     

      int64_t cmdid = mbuf->cmdid ;

      mbuf = (mbuffer *)process_common_system_cmd(ctx, mbuf, recv_ctx, send_ctx);  // done by machine agent (worker_mach)
      // if the command was machine-wide system command, it should be handled by this handler 
      // and should not be passed to the following user update command handler.
      if(mbuf == NULL)
	continue;

      //      mbuf = (mbuffer *)process_obj_cmd(ctx, mbuf, sthrds, recv_ctx, send_ctx); // done by worker machine 


      message_type msgtype = mbuf->msg_type;

#if !defined(SVM_COL_PART)
      void *retbuffer = NULL;
      mbuf = (mbuffer *)first_half_process_obj_cmd(ctx, mbuf, sthrds, recv_ctx, send_ctx, &retbuffer); // done by worker machine 
      if(mbuf == NULL){
	
	assert(retbuffer != NULL);
	// pass mbuffer that holds partial object value in this local machine to the second half thread 
	// and let it send that to the coordinator
	int64_t temptaskcnt = 0;
	work_cmd *cmd = new work_cmd(m_work_object, temptaskcnt, cmdid, msgtype); 
	cmd->m_work = retbuffer;
	whalfthrds->put_entry_inq((void*)cmd);            
	continue;
      }
#else
      // SVM case 
      double wsquare;
      mbuf = (mbuffer *)first_half_process_obj_cmd(ctx, mbuf, sthrds, recv_ctx, send_ctx, &idvalp, len, &wsquare); // done by worker machine 
      if(mbuf == NULL){
	//	assert(retbuffer != NULL);
	// pass mbuffer that holds partial object value in this local machine to the second half thread 
	// and let it send that to the coordinator
	int64_t temptaskcnt = 0;
	work_cmd *cmd = new work_cmd(m_work_object, temptaskcnt, cmdid, msgtype); 
	cmd->m_work = (void*)idvalp;
	cmd->m_svm_obj_tmp = wsquare;
	//cmd->m_work = retbuffer;
	whalfthrds->put_entry_inq((void*)cmd);            
	continue;
      }
#endif
      //      message_type msgtype = mbuf->msg_type;
      msgtype = mbuf->msg_type;
      // only for debugging and sanity checking purpose	
      workhead *workhp = (workhead *)mbuf->data;
      uobjhead *temphp = (uobjhead *)((uintptr_t)workhp + sizeof(workhead));
      int64_t temptaskcnt = temphp->task_cnt;

      timer.set_stimer(0);


      ctx->m_tloghandler->write_cmdevent_start_log(cmdid, 0, timenow());

      mbuf = (mbuffer *)process_worker_cmd(ctx, mbuf, sthrds, recv_ctx, send_ctx, timer); 
      // done by helper threads (worker threads)

      assert(mbuf == NULL);
      timer.set_etimer(0);

      timer.set_stimer(1);
      /* collect the results from the worker threads and */
      work_cmd *cmd = new work_cmd(m_work_paramupdate, temptaskcnt, cmdid, msgtype); 

      ctx->m_tloghandler->write_cmdevent_end_log(cmdid, 0, timenow());

      whalfthrds->put_entry_inq((void*)cmd);            

    }// if (msg != NULL) ..
  } // end of while(1)
  return NULL;
}

// created by scheduler_mach thread
// a thread in charge of one partition of whole task set 
// - run specified scheduling(weight sampling/dependency checking 
// - send results to the scheduler machine thread 
void *worker_thread(void *arg){ 

  worker_threadctx *ctx = (worker_threadctx *)arg; // this pointer of scheduler_threadctx class  
  strads_msg(ERR, "[worker-thread] rank(%d) workermach(%d) threadlid(%d) threadgid(%d) \n", 
	     ctx->get_rank(), ctx->get_worker_mid(), ctx->get_worker_thrdlid(), ctx->get_worker_thrdgid());

  sharedctx *shctx = ctx->get_shctx();
  timers timer(10, 10);
  int64_t iterations=0;

  while(1){
    void *rcmd = ctx->get_entry_inq_blocking(); // if inq is empty, this thread will be blocked until inq become non-empty 
    work_cmd *cmd = (work_cmd *)rcmd;
    uobjhead *uobjhp = (uobjhead *)cmd->m_work;
    timer.set_stimer(0);    
    //    TASK_ENTRY_TYPE *cids = (TASK_ENTRY_TYPE *)((uintptr_t)uobjhp + sizeof(uobjhead));     
    //    int64_t taskcnt = uobjhp->task_cnt;
    //    int64_t statcnt = uobjhp->stat_cnt;
    //    TASK_ENTRY_TYPE *ids = (TASK_ENTRY_TYPE *)((uintptr_t)uobjhp + sizeof(uobjhead));   
    //idval_pair *idvalp = (idval_pair *)((uintptr_t)ids + sizeof(TASK_ENTRY_TYPE)*taskcnt);
    //    for(int64_t i=0; i < statcnt; i++){
    //      strads_msg(ERR, " WTHRD GID(%d)  stat i(%ld) = %ld\n", ctx->get_worker_thrdgid(), i, idvalp[i].id);
    //    }
    //    strads_msg(ERR, " Worker Rank(%d) Mid(%d) Thread gid(%d) got %ld task cnt cids[0] = %ld statcnt : %ld\n", 
    //	       ctx->get_rank(), ctx->get_worker_mid(), ctx->get_worker_thrdgid(), taskcnt, cids[0].id, statcnt);
    //    strads_msg(ERR, "Start STAT UPDATE\n");
    //    execute_user_func(ctx->m_params->m_up->m_funcmap, "statupdate", (void *)uobjhp, (void *)ctx);  
    shctx->m_tloghandler->write_cmdevent_start_log(cmd->m_cmdid, 1*10 + ctx->get_worker_thrdlid(), timenow());
    execute_user_func(ctx->m_params->m_up->m_funcmap, "statupdate", (void *)uobjhp, (void *)ctx, timer);  
    timer.set_etimer(0);
    timer.set_stimer(1);
    // TEST PURPOSE ONLY     
#if defined(TEST_ATOMIC)
    pthread_barrier_wait (&barrier);
#else
    pthread_barrier_wait (&barrier);
#endif

    timer.set_etimer(1);    
    timer.set_stimer(2);
    //    void *ret = execute_user_func(ctx->m_params->m_up->m_funcmap, "update", (void *)uobjhp, (void *)ctx);
    //    strads_msg(ERR, "Start UPDATE\n");
    void *ret = execute_user_func(ctx->m_params->m_up->m_funcmap, "update", (void *)uobjhp, (void *)ctx, timer);     

    timer.set_etimer(2);
    timer.set_stimer(3);
    delete cmd; // not free, it should be delete 



    work_cmd *bcmd = new work_cmd(m_work_paramupdate);
    bcmd->m_result = ret; // hook partial result to returning command 
    //    strads_msg(ERR, " Done at worker thrd gid (%d)\n", ctx->get_worker_thrdgid());
    ctx->put_entry_outq((void *)bcmd);
    timer.set_etimer(3);
    shctx->m_tloghandler->write_cmdevent_end_log(cmd->m_cmdid, 1*10 + ctx->get_worker_thrdlid(), timenow());

    timer.update_elapsedtime(0, 7);
    if(iterations++ % 1000 == 0){
      //      timer.print_elapsedtime(0, 7, ctx->get_worker_thrdgid());
    }
    // get parial sum from this call and forward it to the worker-machine            
  }
  return NULL;
}




void *whalf_thread(void *arg){ 
  whalf_threadctx  *ctx = (whalf_threadctx *)arg; // this pointer of scheduler_threadctx class  
  strads_msg(ERR, "[worker Half machine - thread] rank(%d) workermach(%d) threadlid(%d) threadgid(%d) \n", 
	     ctx->get_rank(), ctx->get_worker_mid(), ctx->get_worker_thrdlid(), ctx->get_worker_thrdgid());  
  sharedctx *shctx = ctx->get_shctx();
  auto ps = shctx->star_sendportmap.begin();
  _ringport *sport = ps->second;
  context *send_ctx = sport->ctx;
  timers timer(10, 10);
  int64_t iterations=0;

  worker_threadctx **sthrds = ctx->get_sthrds();

  while(1){
    void *rcmd = ctx->get_entry_inq_blocking(); 
    // if inq is empty, this thread will be blocked until inq become non-empty 
    work_cmd *cmd = (work_cmd *)rcmd;
    //    uobjhead *uobjhp = (uobjhead *)cmd->m_work;
#if !defined(SVM_COL_PART)
    if(cmd->m_type == m_work_object){
      assert(cmd->m_work != NULL);
      mbuffer *result = (mbuffer *)cmd->m_work;
      while(send_ctx->push_entry_outq(result, sizeof(mbuffer)));
      //      delete(cmd);
      continue;
    }
#else
    // TODO : this is SVM specific code, 
    // when communication abstraction is done, get rid of this.
    if(cmd->m_type == m_work_object){
      assert(cmd->m_work != NULL);
      int64_t len = shctx->m_params->m_up->m_samples;
      idval_pair *idvalp = (idval_pair*)cmd->m_work;
      int64_t payloadsize = USER_MSG_SIZE - sizeof(workhead) - sizeof(uobjhead) - 64;
      // 64 bytes : for slack space 
      int pair_per_packet = payloadsize / sizeof(idval_pair);
      int packet2send = len / pair_per_packet;
      int packetsent=0;
      if(len % pair_per_packet != 0){
	packet2send++;
      }            
      packet2send++; // for sending wsquare sum information 
      int64_t progress = 0;
      for(int i=0; i<packet2send-1; i++){
	mbuffer *mtmp = (mbuffer *)calloc(1, sizeof(mbuffer));
	mtmp->msg_type = USER_PROGRESS_CHECK;
	workhead *retworkhp = (workhead *)mtmp->data;
	retworkhp->type = WORK_OBJECT;
	uobjhead *uobjhp = (uobjhead *)((uintptr_t)retworkhp + sizeof(workhead));
	uobjhp->task_cnt = 0;
	idval_pair *packet_idvalp = (idval_pair *)((uintptr_t)uobjhp + sizeof(uobjhead));

	int statcnt = 0;
	for(int j=0; j<pair_per_packet; j++){
	  if(progress == len)
	    assert(0);	  
	  packet_idvalp[j].id = idvalp[progress].id;
	  packet_idvalp[j].value = idvalp[progress].value;
	  statcnt++;
	  progress++;
	  if(progress == len)
	    break;
	}
	uobjhp->stat_cnt = statcnt;
	mtmp->remain = packet2send -i - 1; 
	packetsent++;
	strads_msg(INF, "[worker] send %d messages(remain : %ld) \n", packetsent, mtmp->remain); 
	while(send_ctx->push_entry_outq((void *)mtmp, sizeof(mbuffer)));
	//	free(mtmp);
      }
      free(idvalp);


      assert(progress == len);
      assert(packetsent == packet2send-1);      

      mbuffer *mtmp = (mbuffer *)calloc(1, sizeof(mbuffer));
      mtmp->msg_type = USER_PROGRESS_CHECK;
      workhead *retworkhp = (workhead *)mtmp->data;
      retworkhp->type = WORK_OBJECT;
      uobjhead *uobjhp = (uobjhead *)((uintptr_t)retworkhp + sizeof(workhead));
      uobjhp->stat_cnt = 1;
      uobjhp->task_cnt = 0;
      idval_pair *packet_idvalp = (idval_pair *)((uintptr_t)uobjhp + sizeof(uobjhead));

      packet_idvalp[0].id = -1; // id no... 
      packet_idvalp[0].value = cmd->m_svm_obj_tmp;
      
      mtmp->remain = 0;
      packetsent++;

      assert(packetsent == packet2send);
      strads_msg(ERR, "[worker] send %d messages(remain : %ld) \n", packetsent, mtmp->remain); 

      while(send_ctx->push_entry_outq((void *)mtmp, sizeof(mbuffer)));
      //      free(mtmp);
      delete(cmd);
      continue;
    } // cmd->m_type == m_work_object
#endif

    shctx->m_tloghandler->write_cmdevent_start_log(cmd->m_cmdid, 2*10, timenow());

    int64_t temptaskcnt = cmd->m_temptaskcnt;
    assert(temptaskcnt != -1);
    int64_t cmdid = cmd->m_cmdid;
    message_type msgtype = cmd->m_msgtype;

    mbuffer *result = (mbuffer *)calloc(1, sizeof(mbuffer));      
    assert(result);

    result->msg_type = msgtype;
    // CAVEAT TODO : when apply stalness .. be careful this. it was taken above  

    workhead *retworkhp = (workhead *)result->data;
    retworkhp->type = WORK_PARAMUPDATE;
    uobjhead *retobjhp = (uobjhead *)((uintptr_t)retworkhp + sizeof(workhead));
    idmvals_pair *idmvalp = (idmvals_pair *)((uintptr_t)retobjhp + sizeof(uobjhead));     

    int maxthrds = shctx->m_params->m_sp->m_thrds_per_worker;
    int donethrdcnt=0;
    int64_t progress = 0;

    //    while(donethrdcnt < maxthrds){
    for(int th=0; th<maxthrds; th++){
      while(1){
	void *ret = sthrds[th]->get_entry_outq();
	// TODO URGENT TODO : DO SOME MACHINE WISE AGGREGATION WORK HERE
	if(ret != NULL){
	  donethrdcnt++;
	  // TODO : THIS IS ONLY FOR UPDATE. 
	  // TODO : add identifying code for objet function ..
	  work_cmd *tmpcmd = (work_cmd *)ret;
	  if(tmpcmd->m_type == m_work_paramupdate){
	     
	    uobjhead *rpobjhp = (uobjhead *)tmpcmd->m_result;
	    int64_t rptaskcnt = rpobjhp->task_cnt;

	    strads_msg(INF, "Rank(%d) Worker-mid(%d) get partially results : %ld \n", 
		       shctx->rank, shctx->m_worker_mid, rptaskcnt);

	    idmvals_pair *pidmvalp = (idmvals_pair *)((uintptr_t)rpobjhp + sizeof(uobjhead));
	    for(int64_t i = 0; i  < rptaskcnt; i++){
	      idmvalp[progress].id = pidmvalp[i].id;
	      idmvalp[progress].psum = pidmvalp[i].psum;
	      idmvalp[progress].sqpsum = pidmvalp[i].sqpsum;
	      progress++;
	    }
	  }else if(tmpcmd->m_type == m_work_object){
	    LOG(FATAL) << "not supported yet. " << endl;
	  }else{
	    LOG(FATAL) << "Fatal : worker . not supported partiam sum type \n" << endl;
	  }
	  delete tmpcmd;
	  break;
	} // if(ret != NULL ... 
      } // for(int th = 0 ....          	
    }
    timer.set_etimer(1);
    timer.update_elapsedtime(0, 3);
    if(iterations++ % 1000 == 0){
      //      timer.print_elapsedtime(0, 3);     
    }
    retobjhp->task_cnt = progress; // Don't forget this.       
    strads_msg(INF, "progress(%ld) == temptaskcnt %ld  \n", progress, temptaskcnt); 
    //      if(msgtype == USER_UPDATE){
    if(progress != temptaskcnt){
      strads_msg(ERR, "Fatal : progress(%ld) != temptaskcnt - expected results : %ld \n", 		  
		 progress, temptaskcnt);
      assert(0);
    }
    //      assert(progress == temptaskcnt);   
    strads_msg(INF, "[worker] rank(%d) worker_mid(%d) finish current phase. \n", 
	       shctx->rank, shctx->m_worker_mid);       
    result->cmdid = cmdid;
    result->src_rank = shctx->rank;
    shctx->m_tloghandler->write_cmdevent_end_log(cmdid, 2*10, timenow());
    if(cmdid % shctx->m_params->m_up->m_logfreq == 0){
      shctx->m_tloghandler->flush_hdd();
    }
    while(send_ctx->push_entry_outq(result, sizeof(mbuffer)));
    //    delete(cmd);
  }
  return NULL;
}




