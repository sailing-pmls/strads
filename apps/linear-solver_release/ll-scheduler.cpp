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

#include "cd-util.hpp"
#include "cd-train.hpp"

using namespace std;

// caller : must be scheduler machine agent or worker machine agent 
void *process_sched_system_cmd_start_scheduler(sharedctx *ctx, mbuffer *mbuf, scheduler_threadctx **sthrds, context *recv_ctx, context *send_ctx, cdtask_assignment &taskmap){
  if(mbuf == NULL)
    assert(0);

  if(mbuf->msg_type == SYSTEM_SCHEDULING){  
    schedhead *schedhp = (schedhead *)mbuf->data; // mbuf passed from caller, should be freed in the caller                                                   // do not call free here


    if(schedhp->type == SCHED_START){
      sched_start_p *amo = (sched_start_p *)((uintptr_t)schedhp + sizeof(schedhead));
      int64_t taskcnt = amo->taskcnt;
      int64_t start = amo->start;
      int64_t end = amo->end;
      int64_t chunks = amo->chunks;
      assert(end != 0);
      // check sanity. See if their range match with the machine range
      auto p = taskmap.schmach_tmap.find(ctx->m_scheduler_mid);
      assert(p != taskmap.schmach_tmap.end());

      strads_msg(OUT, "Rank(%d) start(%ld) end(%ld) psecond->start (%ld) psecond->end(%ld) chunks(%ld) \n", 
		 ctx->rank, start, end, p->second->start, p->second->end, chunks);

      assert(p->second->start == start);
      assert(p->second->end == end);
      
      idval_pair *idvalp = (idval_pair *)calloc(end - start + 1, sizeof(idval_pair));
      int64_t progress=0;
      for(int64_t i=0; i < chunks; i++){

	void *msg = NULL;
	while(!msg){
	  msg = recv_ctx->pull_entry_inq(); // block until I got a new message      
	}
	mbuffer *inbuf = (mbuffer *)msg;
	assert(inbuf->msg_type == SYSTEM_SCHEDULING);
	schedhead *schedhp = (schedhead *)inbuf->data;
	assert(schedhp->type == SCHED_INITVAL);
	assert(schedhp->sched_mid == ctx->m_scheduler_mid);
	int64_t entrycnt = schedhp->entrycnt;
	idval_pair *pairs = (idval_pair *)((uintptr_t)schedhp + sizeof(schedhead));

	for(int64_t j=0; j < entrycnt; j++){
	  idvalp[progress].id = pairs[j].id;
	  idvalp[progress].value = pairs[j].value;
	  progress++;
	}	

	strads_msg(OUT, " [rank(%d) %ld th chunk entry(%ld) progress(%ld)] \n", 
		   ctx->rank, i, entrycnt, progress);
	recv_ctx->release_buffer((void *)msg); // don't forget this 
      }// for(int i...     

      strads_msg(OUT, "Rank(%d) chunks (%ld)  progress(%ld) taskcnt(%ld)\n", 
		 ctx->rank, chunks, progress, taskcnt);
      

      // SANITY CHECKING : DON'T REMOVE - this is one time overhead. 
      assert(progress == taskcnt); // count should be equal to progress
      for(int64_t i=0; i < taskcnt; i++){
	assert(idvalp[i].id == start + i); // ids should be contiguous from start and end 
      }

      // distribut weight information to all child threads 
      //   scatter and make them ready for first scheduling request. 

      //int thrds = ctx->m_params->m_sp->m_thrds_per_scheduler;
      int thrds = FLAGS_threads_per_scheduler;

      int basepartno = ctx->m_scheduler_mid * thrds;     
      int64_t idx=0;
      assert(ctx->m_scheduler_mid >=0);


      for(int i=0; i<thrds; i++){    
	int64_t task_start = taskmap.schthrd_tmap[basepartno + i]->start;
	int64_t task_end = taskmap.schthrd_tmap[basepartno + i]->end;
	//	sampling_cmd *cmd = (sampling_cmd *)calloc(1, sizeof(sampling_cmd));		
	sampling_cmd *cmd = new sampling_cmd(m_init_weight_update, task_end-task_start+1, basepartno+i);	
	for(int k=0; k<(task_end - task_start + 1); k++){	
	  cmd->m_winfo->gidxlist[k] = idvalp[idx].id;
	  cmd->m_winfo->new_weights[k] = idvalp[idx].value;	 

	  //#if defined(NO_WEIGHT_SAMPLING) // weight sampling off 
	  if(FLAGS_weight_sampling == false){
	    assert(cmd->m_winfo->new_weights[k] == 0.0);
	  }
	  //#endif 
	  idx++;
	}
	sthrds[i]->put_entry_inq((void*)cmd);			
      }

      // SANITY CHECK
      assert(idx == progress);

      free(idvalp);
      // send ack for SCHED_START to the coordinator 
      mbuffer *ack = (mbuffer *)calloc(1, sizeof(mbuffer));
      ack->msg_type = SYSTEM_SCHEDULING;
      schedhead *schedhp = (schedhead *)ack->data;
      schedhp->type = SCHED_START_ACK;
      amo = (sched_start_p *)((uintptr_t)schedhp + sizeof(schedhead));
      amo->taskcnt = taskcnt;

      while(send_ctx->push_entry_outq(ack, sizeof(mbuffer)));
      // send ACK

      // schedhp->type == SCHED_START
      recv_ctx->release_buffer((void *)mbuf); // don't forget this           
      return (void*)0x100;


    }else{
      LOG(FATAL) << "[scheduler] SYSTEM_SCHEDULING msg contains non-support sched-type " << endl;
      assert(0);
    }
    recv_ctx->release_buffer((void *)mbuf); // don't forget this           
    return NULL;
  } // msg_type == SYSTEM_SCHEDULING 
  assert(0);
  return mbuf ;
}

// caller : must be scheduler machine agent or worker machine agent 
void *process_sched_system_cmd(sharedctx *ctx, mbuffer *mbuf, scheduler_threadctx **sthrds, context *recv_ctx, context *send_ctx, cdtask_assignment &taskmap){

  if(mbuf == NULL)
    return NULL; // since previous handler processes that and set mbuf to NULL
  
  if(mbuf->msg_type == SYSTEM_SCHEDULING){  
    schedhead *schedhp = (schedhead *)mbuf->data; // mbuf passed from caller, should be freed in the caller 
                                                  // do not call free here
    if(schedhp->type == SCHED_START){
      assert(0); // never happen
      // process_sched_system_cmd_start_scheduler already execute this command 

    }else if(schedhp->type == SCHED_RESTART){

      strads_msg(OUT, "@@@@@@@@@@@@@@@ [Scheduler %d] process..cmd  RESTART \n", ctx->rank);
      sched_start_p *amo = (sched_start_p *)((uintptr_t)schedhp + sizeof(schedhead));
      int64_t taskcnt = amo->taskcnt;
      int64_t start = amo->start;
      int64_t end = amo->end;
      int64_t chunks = amo->chunks;
      assert(end != 0);
      // check sanity. See if their range match with the machine range
      auto p = taskmap.schmach_tmap.find(ctx->m_scheduler_mid);
      assert(p != taskmap.schmach_tmap.end());
      strads_msg(ERR, "Rank(%d) RESTART(%ld) end(%ld) psecond->start (%ld) psecond->end(%ld) chunks(%ld) \n", 
		 ctx->rank, start, end, p->second->start, p->second->end, chunks);
      assert(p->second->start == start);
      assert(p->second->end == end);
      idval_pair *idvalp = (idval_pair *)calloc(end - start + 1, sizeof(idval_pair));
      int64_t progress=0;
      for(int64_t i=0; i < chunks; i++){
	void *msg = NULL;
	while(!msg){
	  msg = recv_ctx->pull_entry_inq(); // block until I got a new message      
	}
	mbuffer *inbuf = (mbuffer *)msg;
	assert(inbuf->msg_type == SYSTEM_SCHEDULING);
	schedhead *schedhp = (schedhead *)inbuf->data;
	assert(schedhp->type == SCHED_INITVAL);
	assert(schedhp->sched_mid == ctx->m_scheduler_mid);
	int64_t entrycnt = schedhp->entrycnt;
	idval_pair *pairs = (idval_pair *)((uintptr_t)schedhp + sizeof(schedhead));
	for(int64_t j=0; j < entrycnt; j++){
	  idvalp[progress].id = pairs[j].id;
	  idvalp[progress].value = pairs[j].value;
	  progress++;
	}	
	strads_msg(INF, " [rank(%d) %ld th chunk entry(%ld) progress(%ld)] \n", 
		   ctx->rank, i, entrycnt, progress);
	recv_ctx->release_buffer((void *)msg); // don't forget this 
      }// for(int i...     
      strads_msg(ERR, "Rank(%d) chunks (%ld)  progress(%ld) taskcnt(%ld)\n", 
		 ctx->rank, chunks, progress, taskcnt);     
      // SANITY CHECKING : DON'T REMOVE - this is one time overhead. 
      assert(progress == taskcnt); // count should be equal to progress
      for(int64_t i=0; i < taskcnt; i++){
	assert(idvalp[i].id == start + i); // ids should be contiguous from start and end 
      }
      // distribut weight information to all child threads 
      //   scatter and make them ready for first scheduling request. 
      //      int thrds = ctx->m_params->m_sp->m_thrds_per_scheduler;
      int thrds = FLAGS_threads_per_scheduler;
      int basepartno = ctx->m_scheduler_mid * thrds;     
      int64_t idx=0;
      assert(ctx->m_scheduler_mid >=0);
      for(int i=0; i<thrds; i++){    
	int64_t task_start = taskmap.schthrd_tmap[basepartno + i]->start;
	int64_t task_end = taskmap.schthrd_tmap[basepartno + i]->end;
	//	sampling_cmd *cmd = (sampling_cmd *)calloc(1, sizeof(sampling_cmd));		
	sampling_cmd *cmd = new sampling_cmd(m_restart_weight_update, task_end-task_start+1, basepartno+i);	
	for(int k=0; k<(task_end - task_start + 1); k++){	
	  cmd->m_winfo->gidxlist[k] = idvalp[idx].id;
	  cmd->m_winfo->new_weights[k] = idvalp[idx].value;	 

	  //#if defined(NO_WEIGHT_SAMPLING) // weight sampling off 
	  if(FLAGS_weight_sampling == false){
	    assert(cmd->m_winfo->new_weights[k] == 0.0);
	  }
	  //#endif 
	  idx++;
	}
	sthrds[i]->put_entry_inq((void*)cmd);			
      }
      // SANITY CHECK
      assert(idx == progress);
      free(idvalp);
      // send ack for SCHED_START to the coordinator 
      mbuffer *ack = (mbuffer *)calloc(1, sizeof(mbuffer));
      ack->msg_type = SYSTEM_SCHEDULING;
      schedhead *schedhp = (schedhead *)ack->data;
      schedhp->type = SCHED_START_ACK;
      amo = (sched_start_p *)((uintptr_t)schedhp + sizeof(schedhead));
      amo->taskcnt = taskcnt;
      while(send_ctx->push_entry_outq(ack, sizeof(mbuffer)));
      // send ACK
      // schedhp->type == SCHED_START

    } else if(schedhp->type == SCHED_UW){ // weight update   

      //      strads_msg(OUT, "[Scheduler-mach] create scheduler weight update command \n");

      schedhead *schedhp = (schedhead *)mbuf->data;
      //      int thrds = ctx->m_params->m_sp->m_thrds_per_scheduler;
      int thrds = FLAGS_threads_per_scheduler;
      int basepartno = ctx->m_scheduler_mid * thrds;           
      int gthrdid = schedhp->sched_thrdgid;
      int lthrdid = gthrdid - basepartno; // only for thread context array indexing 
      int64_t entrycnt = schedhp->entrycnt;
      idval_pair *amo_idvalp = (idval_pair *)((uintptr_t)schedhp + sizeof(schedhead));
      sampling_cmd *cmd = new sampling_cmd(m_weight_update, entrycnt, gthrdid);	
      for(int64_t k=0; k < entrycnt; k++){
	cmd->m_winfo->gidxlist[k] = amo_idvalp[k].id ;
	cmd->m_winfo->new_weights[k] = amo_idvalp[k].value;	 	

	//#if defined(NO_WEIGHT_SAMPLING) // weight sampling off 
	if(FLAGS_weight_sampling == false){
	  assert(cmd->m_winfo->new_weights[k] == 0.0);
	}
	//#endif 
      }     
      sthrds[lthrdid]->put_entry_inq((void*)cmd);			
    }else{
      LOG(FATAL) << "[scheduler] SYSTEM_SCHEDULING msg contains non-support sched-type " << endl;
    }
    recv_ctx->release_buffer((void *)mbuf); // don't forget this           
    return NULL;
  } // msg_type == SYSTEM_SCHEDULING 
  assert(0);
  return mbuf ;
}

// main thread of scheduler machine 
// - aggregator/collector of multiple scheduling threads in a machine 
// - fork scheduling threads, communication with dispatcher  
void *scheduler_mach(void *arg){

  int rclock = 0;
  sharedctx *ctx = (sharedctx *)arg;
  //  int thrds = ctx->m_params->m_sp->m_thrds_per_scheduler;
  int thrds = FLAGS_threads_per_scheduler;
  int basepartno = ctx->m_scheduler_mid * thrds;
  assert(ctx->m_scheduler_mid >=0);
  int sched_mid = ctx->m_scheduler_mid;
  strads_msg(OUT, "******* [scheduler-machine] rank(%d) boot up scheduler-mach (%d). create %d threads with baseline partno(%d)\n", 
	     ctx->rank, ctx->m_scheduler_mid, thrds, basepartno);

  scheduler_threadctx **sthrds = (scheduler_threadctx **)calloc(MAX_SCHEDULER_THREAD, sizeof(scheduler_threadctx *));

  // make model partitioning scheme  
  cdtask_assignment taskmap;
  make_scheduling_taskpartition(taskmap, FLAGS_columns, FLAGS_scheduler, FLAGS_threads_per_scheduler);
  // based on cont_range class's partitioning 

  strads_msg(OUT, "[scheduler-mach rank(%d)mid(%d)] scheduler machines(%ld) thread per scheduler(%ld) \n",
	     ctx->rank, ctx->m_scheduler_mid, FLAGS_scheduler, FLAGS_threads_per_scheduler);	     

  if(ctx->rank == 3){
    strads_msg(OUT, "[scheduler-mach rank(%d)mid(%d)] taskmap.schthrd_tmap.size() : %ld \n",
	       ctx->rank, ctx->m_scheduler_mid, taskmap.schthrd_tmap.size());

  }

  col_vspmat input_matrix(FLAGS_samples, FLAGS_columns);


  // read input data : partition by column, physically stored in column major
  //  cd_train::read_col_partition(FLAGS_data_xfile, input_matrix, ctx->m_worker_machines, ctx->rank);
  // col partition by cont_range's partitioning scheme -- match with make_scheduling_taskpartition's scheme 

  for(int i=0; i<thrds; i++){    
    int64_t task_start = taskmap.schthrd_tmap[basepartno + i]->start;
    int64_t task_end = taskmap.schthrd_tmap[basepartno + i]->end;

    strads_msg(OUT, "[scheduler-mach rank(%d)mid(%d)] task_start(%ld) task_end(%ld)\n",
	       ctx->rank, ctx->m_scheduler_mid, task_start, task_end);    

    //    sthrds[i] = new scheduler_threadctx(ctx->rank, ctx->m_scheduler_mid, i, basepartno + i, 
    //					task_start, task_end, ctx->m_params->m_sp->m_bw, ctx->m_params->m_sp->m_maxset,
    //					ctx->m_params->m_sp->m_infthreshold, ctx);
    sthrds[i] = new scheduler_threadctx(ctx->rank, ctx->m_scheduler_mid, i, basepartno + i, 
					task_start, task_end, FLAGS_bw, FLAGS_schedule_size,
					FLAGS_infthreshold, ctx, input_matrix);
  }

  
  sleep(5);

  // TODO: modify the following if you do not user pure star topology
  assert(ctx->star_recvportmap.size() == 1);
  assert(ctx->star_sendportmap.size() == 1);
  auto pr = ctx->star_recvportmap.begin();
  _ringport *rport = pr->second;
  context *recv_ctx = rport->ctx;

  auto ps = ctx->star_sendportmap.begin();
  _ringport *sport = ps->second;
  context *send_ctx = sport->ctx;

  while(1){
    void *msg = recv_ctx->pull_entry_inq();
    if(msg != NULL){
      mbuffer *mbuf = (mbuffer *)msg;
      process_sched_system_cmd_start_scheduler(ctx, mbuf, sthrds, recv_ctx, send_ctx, taskmap);
      break;
    }
  }

  // read input data : partition by column, physically stored in column major
  cd_train::read_col_partition(FLAGS_data_xfile, input_matrix, FLAGS_scheduler, ctx->m_scheduler_mid);
  //  cd_train::read_col_partition(FLAGS_data_xfile, input_matrix, ctx->m_worker_machines, ctx->rank);
  // col partition by cont_range's partitioning scheme -- match with make_scheduling_taskpartition's scheme 
  strads_msg(OUT, "[Scheduler mach] allocated entry : %ld \n", input_matrix.allocatedentry());

  while (1){
    void *msg = recv_ctx->pull_entry_inq();
    if(msg != NULL){
      mbuffer *mbuf = (mbuffer *)msg;
      // new TODO 
      //      mbuf = (mbuffer *)process_common_system_cmd_scheduleronly(ctx, mbuf, recv_ctx, send_ctx);
      mbuf = (mbuffer *)process_sched_system_cmd(ctx, mbuf, sthrds, recv_ctx, send_ctx, taskmap); // process restart and scheduling request 

      if(mbuf != NULL)
	continue;
#if 0 
      recv_ctx->release_buffer((void *)msg); // don't forget this           
      // send ACK message 
      mbuffer *mtmp = (mbuffer *)calloc(1, sizeof(mbuffer));
      testpkt *stpkt = (testpkt *)mtmp->data;
      mtmp->cmdid = cmdid;
      mtmp->src_rank = ctx->rank;
      stpkt->seqno = 0;
      while(send_ctx->push_entry_outq(mtmp, sizeof(mbuffer)));
#endif 
    }
    // rclock is state-clock to do round robin 

    while(1){
      sampling_cmd *scmd = (sampling_cmd *)sthrds[rclock]->get_entry_outq();			
      if(scmd != NULL){
	// send scmd to the coordinator 	
	mbuffer *mbuf = (mbuffer *)calloc(1, sizeof(mbuffer));
	mbuf->msg_type = SYSTEM_SCHEDULING;
	schedhead *schedhp = (schedhead *)mbuf->data; 
	schedhp->type = SCHED_PHASE;
	schedhp->entrycnt = scmd->m_winfo->size;
	schedhp->sched_mid = sched_mid;
	schedhp->sched_thrdgid = scmd->m_samplergid;

	int64_t *amo = (int64_t *)((uintptr_t)schedhp + sizeof(schedhead));
	assert((sizeof(double)*scmd->m_winfo->size) <= (USER_MSG_SIZE - sizeof(schedhead)));
	for(int k=0; k < scmd->m_winfo->size; k++){
	  amo[k] = scmd->m_winfo->gidxlist[k];
	}
	mbuf->src_rank = ctx->rank;
	while(send_ctx->push_entry_outq(mbuf, sizeof(mbuffer)));
	delete scmd;
	rclock++;
	rclock = rclock % thrds;
	break;
      }      
      rclock++;      
      if(rclock % thrds == 0){ 
	rclock = 0;
	break;
      }
    }
    // wait for any update message from a dispatcher 
    // check destination (schedid) and for ward it to it. 
    //  sthrds[i]->put_entry_inq(void....);
    // check next schedid with token 
    // if any, pull out a safe set 
    // Send it to the dispatcher. 
    // token ++ % thrds per mach 
  } // end of while(1)  
  return NULL;
}

struct gwpair{
  int64_t gid;
  double weight;
};

bool mycomp(struct gwpair *a, struct gwpair *b){
  return(a->weight > b->weight);
}


#if 0 
void _sort_selected(int64_t remain, int64_t *m_samples, int64_t start, int64_t end, double *weights){
  // _sort_selected(remain, wsampler->m_samples, wsampler->m_start, wsampler->m_end);
  // remain : size of m_samples -- contains global parameter id 
  // in order to access wsampler's context, local id is necessary 
  // local id = gid - start ; refer to update_weight routine 

  //wsamplers's m_weight[localid] is each sample's weight now. 
  assert(remain < 1024);
  int64_t nzsample[1024];
  int64_t zerosample[1024];
  int nzcnt=0;
  int zerocnt =0;
  vector<struct gwpair *>nzpairs;
  for(int i=0; i<remain; i++){
    int64_t gid = m_samples[i];
    int64_t lid = gid - start;
    if(weights[lid] > 0){
      nzsample[nzcnt] = gid;
      nzcnt++;      
      struct gwpair *pair = (struct gwpair *)calloc(sizeof(struct gwpair), 1);
      pair->gid = gid;
      pair->weight = weights[lid];
      nzpairs.push_back(pair);
    }else{
      zerosample[zerocnt] = gid;
      zerocnt++;
    }
  }
  // sorting nzsample only based on weight
  // since there are enough number of scheduler threads, just do implement sorting routine, 
  // tentatively, not looking for fastest sorting implementation. 
  // main bottleneck is not scheduling... 
  std::sort(nzpairs.begin(), nzpairs.end(), mycomp);
  int progress = 0;
  for(auto p : nzpairs){
    strads_msg(INF, "Pairs gid [ncnt : %d remain : %ld] : %ld   weight %lf \n", 
	       nzcnt, remain, p->gid, p->weight);
    m_samples[progress] = p->gid;
    progress++;

    free(p);

  }
  for(int i=0; i<zerocnt; i++){
    m_samples[progress] = zerosample[i];
    progress++;
  }
  assert(progress == remain);
  strads_msg(INF, " -------------------------\n");
}
#endif 


// created by scheduler_mach thread
// a thread in charge of one partition of whole task set 
// - run specified scheduling(weight sampling/dependency checking 
// - send results to the scheduler machine thread 
void *scheduler_thread(void *arg){ 
  scheduler_threadctx *ctx = (scheduler_threadctx *)arg; // this pointer of scheduler_threadctx class  

  int64_t selected;
  strads_msg(OUT, "[scheduler-thread] rank(%d) scheduermach(%d) schedthrd gid(%d)\n", 
	     ctx->get_rank(), ctx->get_scheduler_mid(), ctx->get_scheduler_thrdgid());

  // scheduler mid : 0 - (N-1) where N is the number of scheduler machines. 
  cont_range col_range(0, FLAGS_columns-1, FLAGS_scheduler, ctx->get_scheduler_mid());
  // initialized sampler 
  // TODO : check if input A correspond to me is ready 

  wsamplerctx *wsampler = ctx->m_wsampler;

  strads_msg(OUT, "[scheduler-thread] rank(%d) scheduermach(%d) schedthrd gid(%d) start scheduling service\n", 
	     ctx->get_rank(), ctx->get_scheduler_mid(), ctx->get_scheduler_thrdgid());
  

  while(1){
    void *recv_cmd = ctx->get_entry_inq_blocking(); 
    assert(recv_cmd != NULL);
    // if inq is empty, this thread will be blocked(non busy waiting) until inq become non-empty.
    sampling_cmd *scmd = (sampling_cmd *)recv_cmd;

    assert(scmd != NULL);

    if(scmd->m_type == m_init_weight_update){ // start scheduler remote start case (to see if mem alloc in scheduler side crash or not 
      strads_msg(OUT, "[scheduler-thread] rank(%d) scheduermach(%d) schedthrd gid(%d) got init_weight_update\n", 
		 ctx->get_rank(), ctx->get_scheduler_mid(), ctx->get_scheduler_thrdgid());
  
      assert(scmd->m_winfo->gidxlist[0] == wsampler->m_start);
      assert(scmd->m_winfo->gidxlist[scmd->m_winfo->size-1] == wsampler->m_end);
      strads_msg(OUT, "M_INIT_WEIGHT_UPDATE___ Call update init weight .... for (%ld) Rank(%d) mid(%d) thrgid(%d) \n", 
		 scmd->m_winfo->size, 
		 ctx->get_rank(), 
		 ctx->get_scheduler_mid(), 
		 ctx->get_scheduler_thrdgid());

      strads_msg(OUT, "[scheduler-thread] rank(%d) scheduermach(%d) schedthrd gid(%d) sampler->update_weight ++ \n", 
		 ctx->get_rank(), ctx->get_scheduler_mid(), ctx->get_scheduler_thrdgid());

      wsampler->update_weight(scmd->m_winfo);      
      // wsampler->m_samples : internal buffer for temporary usage. 
      // you can freely replace it with int64 array .. its size is m_maxset
      // do not free wsampler->m_samples

      strads_msg(OUT, "[scheduler-thread] rank(%d) scheduermach(%d) schedthrd gid(%d) sampler->update_weight -- \n", 
		 ctx->get_rank(), ctx->get_scheduler_mid(), ctx->get_scheduler_thrdgid());

      strads_msg(OUT, "[scheduler-thread] rank(%d) scheduermach(%d) schedthrd gid(%d) sampler->do_sampling ++ \n", 
		 ctx->get_rank(), ctx->get_scheduler_mid(), ctx->get_scheduler_thrdgid());

      selected = wsampler->do_sampling(wsampler->m_maxset, wsampler->m_samples);      

      strads_msg(OUT, "[scheduler-thread] rank(%d) scheduermach(%d) schedthrd gid(%d) sampler->do_sampling -- \n", 
		 ctx->get_rank(), ctx->get_scheduler_mid(), ctx->get_scheduler_thrdgid());

      // TODO : REPLACE strAcol with real one from user definition
      assert(selected <= wsampler->m_maxset);

      int64_t remain = selected; 
      // in start case, it is assumed that input data to the scheduler is not loaded yet. 
      //	remain = wsampler->check_interference(wsampler->m_samples, selected, ctx->m_input_matrix, col_range);
      strads_msg(OUT, "maxset(%ld) Select (%ld) and (%ld) survived after inference checking\n", 
		 wsampler->m_maxset, selected, remain);

      assert(remain <= wsampler->m_maxset);
      // put your m_samples into scheduler_machine 
      // sort the task ids (global ids...) on weight of them. 

      // remove since mcord is disabled.
      //      _sort_selected(remain, wsampler->m_samples, wsampler->m_start, wsampler->m_end, wsampler->m_weights);

      sampling_cmd *cmd = new sampling_cmd(m_make_phase, remain, wsampler->m_gid);	
      for(int64_t k=0; k<remain; k++){	
	cmd->m_winfo->gidxlist[k] = wsampler->m_samples[k];
	cmd->m_winfo->new_weights[k] = INVALID_DOUBLE;	 
      }      
      strads_msg(ERR, "[SCHEDULER THREAD] FINISH init weight .... for (%ld) Rank(%d) mid(%d) thrgid(%d) \n", 
		 scmd->m_winfo->size, 
		 ctx->get_rank(), 
		 ctx->get_scheduler_mid(), 
		 ctx->get_scheduler_thrdgid());
      //      ctx->put_entry_outq((void*)cmd);  // because this is not intended for use. 

    }else if(scmd->m_type == m_restart_weight_update){

      assert(scmd->m_winfo->gidxlist[0] == wsampler->m_start);
      assert(scmd->m_winfo->gidxlist[scmd->m_winfo->size-1] == wsampler->m_end);
      strads_msg(ERR, "M_RESTART_WEIGHT_UPDATE Call update init weight .... for (%ld) Rank(%d) mid(%d) thrgid(%d) \n", 
		 scmd->m_winfo->size, 
		 ctx->get_rank(), 
		 ctx->get_scheduler_mid(), 
		 ctx->get_scheduler_thrdgid());
      wsampler->update_weight(scmd->m_winfo);      
      // wsampler->m_samples : internal buffer for temporary usage. 
      // you can freely replace it with int64 array .. its size is m_maxset
      // do not free wsampler->m_samples
      wsampler->m_restart_cnt ++;
      if(wsampler->m_restart_cnt == 2){
	assert(wsampler->m_restart_flag == false);
	wsampler->m_restart_flag = true;
	strads_msg(ERR, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Restart Flag for Optimization is On\n"); 
      }
      assert(wsampler->m_restart_cnt < 3);
      selected = wsampler->do_sampling(wsampler->m_maxset, wsampler->m_samples);      

      assert(selected <= wsampler->m_maxset);

      //#if defined(NO_INTERFERENCE_CHECK)
      //      int64_t remain = selected;
      //#else
      // at this point, data should be available. 
      //      int64_t remain = wsampler->check_interference(wsampler->m_samples, selected, ctx->m_input_matrix, col_range);
      //#endif

      int64_t remain = selected;
      if(FLAGS_check_interference == true)
       remain = wsampler->check_interference(wsampler->m_samples, selected, ctx->m_input_matrix, col_range);
      // at this point, data should be available. 

      strads_msg(INF, "maxset(%ld) Select (%ld) and (%ld) survive after  inference checking--B\n", 
		 wsampler->m_maxset, selected, remain);
      assert(remain <= wsampler->m_maxset);
      // put your m_samples into scheduler_machine 
      sampling_cmd *cmd = new sampling_cmd(m_make_phase, remain, wsampler->m_gid);	
      for(int64_t k=0; k<remain; k++){	
	cmd->m_winfo->gidxlist[k] = wsampler->m_samples[k];
	cmd->m_winfo->new_weights[k] = INVALID_DOUBLE;	 
      }      
      strads_msg(ERR, "[SCHEDULER THREAD] FINISH init weight .... for (%ld) Rank(%d) mid(%d) thrgid(%d) \n", 
		 scmd->m_winfo->size, 
		 ctx->get_rank(), 
		 ctx->get_scheduler_mid(), 
		 ctx->get_scheduler_thrdgid());
      ctx->put_entry_outq((void*)cmd);	// this is valid one 

    }else if(scmd->m_type == m_weight_update){
      strads_msg(INF, "Call update weight .... for (%ld) \n", scmd->m_winfo->size);
      wsampler->update_weight(scmd->m_winfo);      
      // wsampler->m_samples : internal buffer for temporary usage. 
      // you can freely replace it with int64 array .. its size is m_maxset
      // do not free wsampler->m_samples
      selected = wsampler->do_sampling(wsampler->m_maxset, wsampler->m_samples);      
#if 0 
      // debugging 
      int64_t firstz=-1;
      for(int k=0; k<selected; k++){	
	int64_t lid = wsampler->m_samples[k] - wsampler->m_base;
	double weight = wsampler->m_weights[lid];	
	if(weight == 0.0 and firstz == -1){
	  firstz = k;
	}
	if(firstz != -1){
	  if(weight != 0.0){
	    for(int j=0; j < selected; j++){
	      int64_t slid = wsampler->m_samples[j] - wsampler->m_base;
	      double sweight = wsampler->m_weights[slid];
	      strads_msg(ERR, " SCHED THRD(%d):   %dth element ( %ld -- %2.20lf )\n", 
			 ctx->get_scheduler_thrdgid(), j, slid, sweight);	      
	    }
	  }
	  assert(weight == 0.0);	  
	}       
      }
#endif

      int64_t remain = selected;
      if(FLAGS_check_interference == true)
	remain = wsampler->check_interference(wsampler->m_samples, selected, ctx->m_input_matrix, col_range);
      // at this point, data should be available. 
      
      strads_msg(INF, "\t[] maxset : %ld  selected : %ld  survived  remain : %ld --> goto sorting \n", 
		 wsampler->m_maxset, selected, remain);

      // removed since mcord is disabled. 
      //      _sort_selected(remain, wsampler->m_samples, wsampler->m_start, wsampler->m_end, wsampler->m_weights);

      if(remain < FLAGS_schedule_size*0.6)
	strads_msg(INF, "Abnormal : less than 60 percent of  maxset(%ld) Select (%ld) and (%ld) go through inference checking\n", 
		   wsampler->m_maxset, selected, remain);
      // put your m_samples into scheduler_machine 
      sampling_cmd *cmd = new sampling_cmd(m_make_phase, remain, wsampler->m_gid);	
      for(int64_t k=0; k<remain; k++){	
	cmd->m_winfo->gidxlist[k] = wsampler->m_samples[k];
	cmd->m_winfo->new_weights[k] = INVALID_DOUBLE;	 
      }      
      ctx->put_entry_outq((void*)cmd);			      

    }else if(scmd->m_type == m_bw_change){
      // TODO : 
    }else{
      LOG(FATAL) << "Not Supported Command detected in Scheduler Thread" << endl;
    }
    delete scmd;
  }

  return NULL;
}
