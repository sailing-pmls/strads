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
#include "cd-train.hpp"

using namespace std;


//#define SCHEDULER_DEBUG (1)
// for logging update frequency 
double *gweights;
double *gbetadiff;

pthread_barrier_t  cbarrier; // barrier synchronization object  
void _scheduler_start_remote(sharedctx *ctx, double *weights, uint64_t wsize, bool rflag, cdtask_assignment &tmap);
void get_object_first_half(sharedctx *ctx, int64_t cmdid);
double get_object_second_half(sharedctx *ctx);
void save_beta(double *beta, long coeff, string &fn);
// valid for star topology
static void _send_to_scheduler(sharedctx *ctx, mbuffer *tmpbuf, int len, int schedmid){
  while(ctx->scheduler_sendportmap[schedmid]->ctx->push_entry_outq((void *)tmpbuf, len)); 
  //  free(tmpbuf);
}
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




void  get_object_first_half(sharedctx *ctx, int64_t cmdid){
  mbuffer *task = (mbuffer *)calloc(1, sizeof(mbuffer));
  task->msg_type = USER_PROGRESS_CHECK;
  workhead *workhp =  (workhead *)task->data;
  workhp->type = WORK_OBJECT;
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

#if 1
// serial scheduling 
void *scheduling_emulator(int64_t *taskid){

  mbuffer *mbuf = (mbuffer *)calloc(sizeof(mbuffer), 1);
  mbuf->msg_type = SYSTEM_SCHEDULING;
  schedhead *schedhp = (schedhead *)mbuf->data;	

  //  schedhp->entrycnt = 1;
  schedhp->entrycnt = FLAGS_schedule_size;

  schedhp->sched_thrdgid = 0;
  schedhp->sched_mid = 0;
  schedhp->type = SCHED_PHASE;

  strads_msg(INF, "New Phase: schemid(%d) schedthrdgid(%d) entrycnt(%ld)\n", 
	     schedhp->sched_mid, schedhp->sched_thrdgid, schedhp->entrycnt);
  int64_t *amo_gtaskids =  (int64_t *)((uintptr_t)schedhp + sizeof(schedhead));
  //  amo_gtaskids[0] = taskid;

  for(auto i=0; i<FLAGS_schedule_size; i++){
    amo_gtaskids[i] = *taskid;
    (*taskid) = (*taskid) + 1; 
    if((*taskid) % FLAGS_columns == 0){
      (*taskid) = 0;
    }
  }
  return (void *)mbuf;
}
#endif 

#if 0 
// random scheduling 
void *scheduling_emulator(int64_t *taskid){
  mbuffer *mbuf = (mbuffer *)calloc(sizeof(mbuffer), 1);
  mbuf->msg_type = SYSTEM_SCHEDULING;
  schedhead *schedhp = (schedhead *)mbuf->data;	
  //  schedhp->entrycnt = 1;
  schedhp->entrycnt = FLAGS_schedule_size;
  schedhp->sched_thrdgid = 0;
  schedhp->sched_mid = 0;
  schedhp->type = SCHED_PHASE;
  strads_msg(INF, "New Phase: schemid(%d) schedthrdgid(%d) entrycnt(%ld)\n", 
	     schedhp->sched_mid, schedhp->sched_thrdgid, schedhp->entrycnt);
  int64_t *amo_gtaskids =  (int64_t *)((uintptr_t)schedhp + sizeof(schedhead));
  //  amo_gtaskids[0] = taskid;
  std::map<long, long> tmpmap;
  while(1){
    long randid = _unif01(_statrng)*FLAGS_columns;
    randid = randid % FLAGS_columns;
    tmpmap.insert(std::pair<long,long>(randid, randid));
    if(tmpmap.size() == FLAGS_schedule_size)
      break;
  }  
  int progress=0;
  //  for(auto i=0; i<FLAGS_schedule_size; i++){
  for(auto it = tmpmap.begin(); it != tmpmap.end(); it++){
    amo_gtaskids[progress] = it->first ;
    progress++;
  }
  assert(progress == FLAGS_schedule_size);
  return (void *)mbuf;
}
#endif 

void scheduling_release_buffer(void *buf){
  free(buf);
}

void *coordinator_mach(void *arg){
  sharedctx *ctx = (sharedctx *)arg;
  strads_msg(ERR, "[coordinator-machine] rank(%d) boot up coordinator-mach \n", ctx->rank);

  //  int thrds = ctx->m_params->m_sp->m_thrds_per_coordinator;
  int thrds = 1;

  pthread_barrier_init (&cbarrier, NULL, thrds+1); // +1 for the main thread

  strads_msg(ERR, "[coordinator-mach] pthread-barrier with %d threads", thrds+1);

  //new  coordinator_threadctx **sthrds = (coordinator_threadctx **)calloc(MAX_SCHEDULER_THREAD, sizeof(coordinator_threadctx *));
  coordinator_threadctx **sthrds = (coordinator_threadctx **)calloc(1, sizeof(coordinator_threadctx *));
  assert(thrds > 0);
  for(int i=0; i<thrds; i++){    
    sthrds[i] = new coordinator_threadctx(ctx->rank, ctx->rank, i, ctx); // coordinator_thread 
  }

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

  //new  double *weights = (double *)calloc(ctx->m_params->m_sp->m_modelsize, sizeof(double));
  gweights = (double *)calloc(FLAGS_columns, sizeof(double));
  gbetadiff = (double *)calloc(FLAGS_columns, sizeof(double));
  //new  _scheduler_start_remote(ctx, weights, ctx->m_params->m_sp->m_modelsize, false);     

  cdtask_assignment taskmap;
  make_scheduling_taskpartition(taskmap, FLAGS_columns, FLAGS_scheduler, FLAGS_threads_per_scheduler);
  // based on cont_range class's partitioning         

  _scheduler_start_remote(ctx, gweights, FLAGS_columns, false, taskmap);     // START : just see if it's successful in memory allocation
  strads_msg(OUT, "[coordinator] Sends scheduler sanity checking call to all schedulers and activate them -- done\n");

  long cmdmgt = 0;
  double *beta = gweights;
  pthread_barrier_wait (&cbarrier);

  for(int64_t i=0; i < FLAGS_columns; i++){
    assert(beta[i] == 0.0); 
  }

  int64_t modelsize = FLAGS_columns;
  double lambda = FLAGS_lambda;

  _scheduler_start_remote(ctx, beta, FLAGS_columns, true, taskmap);     
  strads_msg(OUT, "[coordinator] Send initial weight information to all scheduler and make them ready for service -- done\n");

  int rclock=0; // for round robin for scheduler 
  unordered_map<int64_t, idmvals_pair *>*retmap; 
  int64_t iteration=0;
  uint64_t stime = timenow();
  int64_t pending_iteration = 0;
  //  int64_t staleness = ctx->m_params->m_sp->m_pipelinedepth;
  int64_t staleness = FLAGS_pipeline; // pipeline depth   

  //  int64_t switch_iter=(ctx->m_params->m_sp->m_modelsize/ctx->m_params->m_sp->m_maxset)*3 ;
  int64_t switch_iter=(FLAGS_columns/FLAGS_schedule_size)*3 ;

  strads_msg(OUT, "Start iterative update with lambda %lf pipeline depth(%ld) switch point(%ld) \n", lambda, staleness, switch_iter);
  
#if defined(SCHEDULER_DEBUG)
  std::vector<long>upcnt(FLAGS_columns, 0); // columns, intni to 0
#endif 

  int64_t debug_task_clock=0;

  while(1){ // grand while loop 
    while(1){ // inner intinifite loop 

      void *buf = ctx->scheduler_recvportmap[rclock]->ctx->pull_entry_inq();       
      // DEBUGGING with emulator : replace above with the following line.
      //void *buf = scheduling_emulator(&debug_task_clock);

      if(buf != NULL){	
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

	//#if !defined(NO_WEIGHT_SAMPLING) // WEIGHT SAMPLING ON
	if(FLAGS_weight_sampling == true){
	  if(iteration == switch_iter){
	    int toflush = 0;
	    strads_msg(ERR, "I will retry SCHEDULERS \n");
	    //	  int schedmachs = ctx->m_params->m_sp->m_schedulers;
	    int schedmachs = FLAGS_scheduler;
	    //	  int thrds_per_sched = ctx->m_params->m_sp->m_thrds_per_scheduler;
	    int thrds_per_sched = FLAGS_threads_per_scheduler;
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
	    //	  strads_msg(ERR, "I could Restart SCHEDULERS \n");

	    //	  double *weights = (double *)calloc(ctx->m_params->m_sp->m_modelsize, sizeof(double));
	    //	  double *weights = (double *)calloc(FLAGS_columns, sizeof(double));
	    //	  for(int64_t mi=0; mi < modelsize; mi++){
	    //	    weights[mi] = beta[mi]; // beta == gweights 
	    //	  }

	    //	  _scheduler_start_remote(ctx, weights, ctx->m_params->m_sp->m_modelsize, true);     
	    // net TODO _scheduler_start_remote(ctx, weights, FLAGS_columns, true);     
	    _scheduler_start_remote(ctx, beta, FLAGS_columns, true, taskmap);     
	    //	  free(weights);

	  } // if(iteration == 12000 ) ... reset scheduler 
	}
	//#endif 
	/* this buf contains SCHED_PHASE message */
	mbuffer *mbuf = (mbuffer *)buf;
	assert(mbuf->msg_type == SYSTEM_SCHEDULING);
	schedhead *schedhp = (schedhead *)mbuf->data;	
	int entrycnt = schedhp->entrycnt;

	int gthrdid = schedhp->sched_thrdgid;
	int schedmid = schedhp->sched_mid;

	strads_msg(ERR, "\t\t\t @@@ iteration[%ld] from  gthrdid:schedmid (%d : %d)\n", 
		   iteration, gthrdid, schedmid);




	assert(schedhp->type == SCHED_PHASE);

	//	strads_msg(OUT, "New Phase: schemid(%d) schedthrdgid(%d) entrycnt(%ld)\n", 
	//	   schedhp->sched_mid, schedhp->sched_thrdgid, schedhp->entrycnt);

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
#if 1
	for(auto p : *retmap){
	  free(p.second);
	}
	retmap->erase(retmap->begin(), retmap->end());       
	assert(retmap->size() == 0);	
	delete retmap;
#endif 

	//	coord_cmd *gccmd = new coord_cmd(m_coord_gc, retmap, iteration);
	//	sthrds[gcthrd]->put_entry_inq((void *)gccmd);
	//	retmap=NULL;
	/////////////////////////////////////////////////////////////////////////////
	// TODO : change idvalp, and entry cnt as well when you replace above simulation code 
	// CAVEAT : dont forget that.

	idval_pair *tasks = (TASK_ENTRY_TYPE *)calloc(entrycnt, sizeof(TASK_ENTRY_TYPE));
	for(int i=0; i < entrycnt; i++){
	  int64_t id = amo_gtaskids[i];
	  tasks[i].id = amo_gtaskids[i];
#if defined(SCHEDULER_DEBUG)
	  upcnt[id]++;
#endif

	  //	  tasks[i].value = util_get_double_random(0, 1.0); // beta[tasks[i].id] TODO from beta list
	  tasks[i].value = beta[id] ; // beta[tasks[i].id] TODO from beta list 
	}

	//	strads_msg(OUT, "[COORDINATOR MAIN]: iteration(%ld)  size (%dd)\n", iteration, entrycnt);

      	user_func_make_dispatch_msg((void *)uobjhp, WORK_STATUPDATE, tasks, entrycnt, idvalp, prevbetacnt, 
				    USER_MSG_SIZE - sizeof(workhead));  

	//	int64_t scmdid = cmdmgt->get_cmdclock();       
	int64_t scmdid = cmdmgt++;       

	_mcopy_broadcast_to_workers(ctx, task, sizeof(mbuffer), scmdid);

	// memory leak remedy
	free(tasks);
	if( prevbetacnt > 0)
	  free(idvalp);
	// memory leak remedy

	coord_cmd *cmd = new coord_cmd(m_coord_paramupdate, gthrdid, schedmid, entrycnt, amo_gtaskids, iteration, scmdid);

	sthrds[0]->put_entry_inq((void *)cmd);
	iteration++;

	// DEBUGGING with emulator 
	ctx->scheduler_recvportmap[rclock]->ctx->release_buffer((void *)buf);
	//	scheduling_release_buffer((void *)buf);

	rclock++;
	rclock = rclock % ctx->m_sched_machines;

	if(iteration % FLAGS_logfreq == 0){ // default 1000 
	  int64_t nz=0;
	  for(int64_t i =0; i < modelsize; i++){
	    if(beta[i] != 0)
	      nz++;
	  }
	  //	  int64_t scmdid = cmdmgt->get_cmdclock();
	  int64_t scmdid = cmdmgt++;
	  get_object_first_half(ctx, scmdid);
	  coord_cmd *cmd = new coord_cmd(m_coord_object, iteration, scmdid);
	  sthrds[0]->put_entry_inq((void *)cmd);
	  uint64_t etime = timenow();
	  strads_msg(OUT,"%ld processed elapsedtime : %lf second, nz-coeff : %ld \n", 
		     iteration, (etime - stime)/1000000.0, nz);
	}

	if(iteration == FLAGS_max_iter){ // default 100 
	  strads_msg(OUT, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Congratulation ! Finishing task. output file  %s\n", FLAGS_outputfile_coeff.c_str());	
	  save_beta(beta, FLAGS_columns, FLAGS_outputfile_coeff);

#if defined(SCHEDULER_DEBUG)
	  for(auto k=0; k<FLAGS_columns; k++){
	    strads_msg(OUT, "@@@@@ upcnt[%d] = [%d] \n", k, upcnt[k]); 
	  }
#endif
	  sleep(1);
	  exit(0);
	}	  	
      } // buf != NULL
    } // while (1) -- inner infinite loop 
  } // while (1) end of outer grand while loop
  return NULL;
}
// end of coordinator_mach function 

// created by scheduler_mach thread
// a thread in charge of one partition of whole task set 
// - run specified scheduling(weight sampling/dependency checking 
// - send results to the scheduler machine thread 
void *coordinator_thread(void *arg){ 

  coordinator_threadctx *ctx = (coordinator_threadctx *)arg; // this pointer of scheduler_threadctx class  
  strads_msg(OUT, "[Coordinator-thread] rank(%d) coordinatormach(%d) threadid(%d)\n", 
	     ctx->get_rank(), ctx->get_coordinator_mid(), ctx->get_coordinator_thrdid());


  sharedctx *shctx = ctx->m_shctx;

  col_vspmat col_dummy;
  cas_array<double> cas_dummy;
  thread_barrier barrier_dummy(1);

  //  lasso lasso_handler(1, col_dummy, cas_dummy, barrier_dummy);
  lasso lasso_handler(FLAGS_lambda, FLAGS_samples, FLAGS_columns, 0, shctx->m_worker_machines, 0, FLAGS_threads, col_dummy, cas_dummy, barrier_dummy);

  logistic logistic_handler(FLAGS_lambda, FLAGS_samples, FLAGS_columns, 0, shctx->m_worker_machines, 0, FLAGS_threads, col_dummy, cas_dummy, barrier_dummy);

  cd_train *phandler;  
  if(FLAGS_algorithm.compare("lasso") == 0){
    phandler = &lasso_handler;
  }else if(FLAGS_algorithm.compare("logistic") == 0){
    phandler = &logistic_handler;
  }else{
    assert(0);
  }
  cd_train &handler = *phandler;   

  int64_t iteration = 0;
  int64_t updatedsofar=0;
  int64_t validcnt = 0;
  double totaldelta = 0.0;

  idval_pair *m_idvalp_buf = (idval_pair *)calloc(FLAGS_schedule_size*10, sizeof(idval_pair));
  strads_msg(ERR, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ START Filling unordered map.....    \n\n");
  unordered_map<int64_t, int64_t>second_chance;
  int64_t modelsize = FLAGS_columns;

  for(int64_t i=0; i < modelsize; i++){
    second_chance[i] =i;
  }
  strads_msg(ERR, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ END Filling unordered map.....    \n\n");

  int64_t switch_iter=(FLAGS_columns/FLAGS_schedule_size)*3 ;

  pthread_barrier_wait (&cbarrier);
  strads_msg(ERR, "@@@@ [coordinator-thread] Barrier pass !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  \n");

  double *beta = gweights;
  assert(beta);
  int counter = 0;
  while(1){
    void *rcmd = ctx->get_entry_inq_blocking(); // if inq is empty, this thread will be blocked until inq become non-empty 
    coord_cmd *cmd = (coord_cmd *)rcmd;

    // Lasso / LR 
    if(cmd->m_type == m_coord_object){
      double psum = get_object_second_half(shctx);      

      //    double objectvalue = user_get_object_server(beta, psum);
      double objectvalue = handler.get_object_server(beta, psum, FLAGS_lambda, FLAGS_columns, FLAGS_samples);
      strads_msg(INF, "\t\t Objective Value : %lf \n", objectvalue);
      continue;
    }

    double *betadiff = gbetadiff;
    stmwork *mwork = (stmwork *)cmd->m_work;

    // DEBUGGING
    int gthrdid = mwork->gthrdid;
    int schedmid = mwork->schedmid;

    int64_t entrycnt = mwork->entrycnt;
    int64_t *amo_gtaskids = mwork->amo_gtaskids;
    //  pass amo_gtaskids to the second half thread
    unordered_map<int64_t, idmvals_pair *> *retmap = new unordered_map<int64_t, idmvals_pair *>; 

    for(int i=0; i < entrycnt; i++){
      int64_t id = amo_gtaskids[i];
      idmvals_pair *tmp = (idmvals_pair *)calloc(1, sizeof(idmvals_pair)); 
      // it should be calloc since all entry should be zero  
      retmap->insert(std::pair<int64_t, idmvals_pair*>(id, tmp));
    }

    //    user_aggregator(beta, betadiff, *retmap, shctx, FLAGS_lambda, FLAGS_columns);
    handler.aggregator(beta, betadiff, *retmap, shctx, FLAGS_lambda, FLAGS_columns, FLAGS_samples);

    uint64_t currentbetacnt = retmap->size();
    assert((int64_t)currentbetacnt == entrycnt);
    updatedsofar += currentbetacnt;
    if(retmap->size() != 0){
      int progress=0;
      for(auto p : *retmap){
	double absdelta = fabs(betadiff[p.second->id]);
	if(absdelta != 0.0){
	  validcnt ++;
	  totaldelta += absdelta;
	}
#if 0 
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
#endif
	} // second chance 
#endif  // if 0 

	//	    idvalp[progress].id = p.second->id;
	//	    idvalp[progress].value  = betadiff[p.second->id];
	m_idvalp_buf[progress].id = p.second->id;

	//#if !defined(NO_WEIGHT_SAMPLING) // weight sampling ON
	if(FLAGS_weight_sampling == true){

	  if(iteration > switch_iter){
	    m_idvalp_buf[progress].value = betadiff[p.second->id];	
	    if(iteration % 1000 == 0){ // for debugging purpose, every 100 iterations, try to delta 
	      strads_msg(OUT, " @@@ iteration[%ld] ID [%ld]   BETA[%1.15lf] Beta-DIFF [%1.15lf] updatesofar: %ld  \n", 
			 iteration,
			 p.second->id, 
			 beta[p.second->id],
			 betadiff[p.second->id], updatedsofar);
	    }

	  }else{
	    m_idvalp_buf[progress].value = 0.0;
	  }
	}else{
	  //#else
	  m_idvalp_buf[progress].value = 0.0;
	}
	//#endif
	progress++;
      }
    }// retmap->size() != 0 

    if(iteration % FLAGS_logfreq == 0){
      strads_msg(INF, "@@@@@@@@@ iteration [%ld] update parameter so far : %ld \n", 
		 iteration, updatedsofar);
    }

    // CAVEAT : 
    // m_idvalp_buf is used to update schedulers weight information. 
    //DEBUGGING with emulator
    _send_weight_update(shctx, schedmid,  gthrdid, m_idvalp_buf,  currentbetacnt);       		             

    iteration++;
    //    ctx->scheduler_recvportmap[rclock]->ctx->release_buffer((void *)buf);
    /******************************************************************************************
	  Do not use assert .. here. Think about pending q mechanism. 
	  only when there is no pending mach across all cluster for a given cmd, cmd is moved from 
	  pending q to done queue. Here we sent one cmd id to two machines , 	
    *******************************************************************************************/
    int64_t scmdid = cmd->m_cmdid;
    delete cmd;
    coord_cmd *scmd = new coord_cmd(m_coord_paramupdate, retmap, iteration, scmdid);
    //    strads_msg(OUT, "@@@@@@@ coord thread pass update cmd back to main coord thread. \n");
    ctx->put_entry_outq((void *)scmd);
    counter++;

  }
  return NULL;
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

/* Caller: coordinator only
   coordiantor will call this function to let remote scheduler machines be ready for receiving 
   the following initial weight information and start their service
*/
void _scheduler_start_remote(sharedctx *ctx, double *weights, uint64_t wsize, bool rflag, cdtask_assignment &tmap){

  int machines = _get_tosend_machinecnt(ctx, m_scheduler);

  strads_msg(OUT, "Machines : Scheduler %d \n", machines);
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

    //    auto p = ctx->m_tmap.schmach_tmap.find(i);
    //    assert(p != ctx->m_tmap.schmach_tmap.end());
    auto p = tmap.schmach_tmap.find(i);
    assert(p != tmap.schmach_tmap.end());

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
    strads_msg(OUT, "[Coordinator start remote] for %d schedmach start(%ld) end(%ld) chunks(%ld)\n", 
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
      strads_msg(OUT, "@@@@@@ Coordinator entrycnt  : %ld  chunk: %ld  to scheduler mid (%d) \n", entrycnt, ci, i);     

      

      _send_to_scheduler(ctx, mbuf, sizeof(mbuffer), i); // once sent through com stack, it will be realeased
      //      while(1);
      usleep(100);
    }

    strads_msg(OUT, "@@@@@@ Coordinator waiting for ACK from scheduler mid (%d) \n", i);     
    // waiting for ACK from the scheduler. 
    void *buf;
    while(1){
      buf = ctx->scheduler_recvportmap[i]->ctx->pull_entry_inq();       
      if(buf != NULL){
	break;
      }
    }
    strads_msg(OUT, "@@@@@@ Coordinator got ACK from scheduler mid (%d) \n", i);     

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
    strads_msg(OUT, "\tinit weight to sched_mid(%d) -- done \n", i);
  }  
}

void save_beta(double *beta, long coeff, string &fn){
  FILE *fp = (FILE *)fopen(fn.c_str(), "wt");
  assert(fp); 
  long nz=0;
  for(auto i=0; i<coeff; i++){
    if(beta[i] != 0.0)
      nz++;
  }
  fprintf(fp, "%%MatrixMarket matrix coordinate real general\n");
  fprintf(fp, "%d %ld %ld\n", 1, coeff, nz);
  for(auto i=0; i<coeff; i++){
    if(beta[i] != 0.0){
      fprintf(fp, "1 %d %2.5lf\n", i+1, beta[i]); // since MMT array start from 1  
    }
  }
  fclose(fp);
}
