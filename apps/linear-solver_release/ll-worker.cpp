#include <atomic>
#include <assert.h>

#include "ll-worker.hpp"
#include "cd-train.hpp"
#include "strads/ds/spmat.hpp"
#include "cd-util.hpp"
#include <strads/netdriver/comm.hpp>
#include <strads/include/indepds.hpp>
#include <mpi.h>

using namespace std;

void *process_worker_cmd(sharedctx *ctx, mbuffer *mbuf, std::vector<cd_train *> &handlers, context *recv_ctx);
void *process_obj_cmd(sharedctx *ctx, mbuffer *mbuf, context *rctx, void **rbf, cas_array<double> &res, cont_range &row_r, double *Y);

/* 
 * @brief worker threads for updating parameters 
 *   implemented in c++11 thread STL 
 *
 */
void worker_thread(int threadid, cd_train &train, sharedctx *shctx, double *Y){
  LOG(INFO) << "    [worker-thread] RANK : " << train.get_mid() << " thread-local-id: " <<  train.get_threadid() << endl;
  int64_t iterations=0;

  while(1){
    void *rcmd = train.get_entry_inq_blocking(); // if inq is empty, this thread will be blocked until inq become non-empty 
    work_cmd *cmd = (work_cmd *)rcmd;
    uobjhead *uobjhp = (uobjhead *)cmd->m_work;

    uobjhp->start = train.m_row_range.get_min(); // coarse grained sample range at machine level since we use atomic array for residual   
    uobjhp->end  = train.m_row_range.get_max();  // coarse grained sample range at machine level since we use atomic array for residual    

    //    user_update_status(train.m_mat, train.residual, uobjhp, shctx);
    train.update_res(train.m_mat, train.residual, uobjhp, shctx);


    // TODO barrier  
    //    train.m_barrier.wait();

    //    void *ret = user_update_parameter(train.m_mat, train.residual, uobjhp, shctx); 
    void *ret = train.update_feat(train.m_mat, train.residual, uobjhp, shctx, Y); 

    delete cmd; // not free, it should be delete 
    work_cmd *bcmd = new work_cmd(m_work_paramupdate);
    bcmd->m_result = ret; // hook partial result to returning command 
    train.put_entry_outq((void *)bcmd);
  }
  return;
}

void *worker_mach(void *arg){

  sharedctx *ctx = (sharedctx *)arg;
  LOG(INFO) << "[worker " << ctx->rank << "]" << " boot up out of " << ctx->m_worker_machines << " workers " << endl; 

  MPI_Group worker_group, sched_group, orig_group;
  MPI_Comm worker_comm, sched_comm; // to enforce barrier for the coordinators and workers  
  int workerc_ranks[ctx->m_worker_machines+1];
  int schedc_ranks[ctx->m_sched_machines+1];

  for(int i(0); i<ctx->m_worker_machines; ++i){
	  workerc_ranks[i] = i;
  }

  for(int i(ctx->m_worker_machines); i<ctx->m_worker_machines + ctx->m_sched_machines; ++i){
	  schedc_ranks[i - ctx->m_worker_machines] = i;
  }

  workerc_ranks[ctx->m_worker_machines] = ctx->m_sched_machines +  ctx->m_worker_machines;
  schedc_ranks[ctx->m_sched_machines] = ctx->m_sched_machines +  ctx->m_worker_machines;

  MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
  MPI_Group_incl(orig_group, ctx->m_worker_machines+1, workerc_ranks, &worker_group);
  MPI_Comm_create(MPI_COMM_WORLD, worker_group, &worker_comm);
 
  MPI_Group_incl(orig_group, ctx->m_sched_machines+1, schedc_ranks, &sched_group);
  MPI_Comm_create(MPI_COMM_WORLD, sched_group, &sched_comm);

  // create sparse matrix instance and fill it 
  col_vspmat input_matrix(FLAGS_samples, FLAGS_columns);
  //  cd_train::read_partition(FLAGS_data_xfile, input_matrix, ctx->m_worker_machines, ctx->rank); 

  strads_msg(OUT, "[ WORKER  %d ] start READ PARTITION from ring ", ctx->rank);
  cd_train::read_partition_ring(ctx, FLAGS_data_xfile, input_matrix, ctx->m_worker_machines, ctx->rank); 
  cont_range row_range(0, FLAGS_samples-1, ctx->m_worker_machines, ctx->rank);
  strads_msg(OUT,"[Worker Machine]  rank(%d) mid(%d) allocated entry (%ld) \n", ctx->rank, ctx->rank, input_matrix.allocatedentry());
  MPI_Barrier(worker_comm);  

  // create atomic array for keeping residual 
  cas_array<double> residual(FLAGS_samples); // atomic double typed array: (c++11 STL-atomic does not support double/float type) 
  // *************************************************************************************** TODO : read Y data and fill out residual********** 
  //cd_train::read_partition(FLAGS_data_yfile, residual, FLAGS_samples, 1, ctx->m_worker_machines, ctx->rank);
  cd_train::read_partition_ring(ctx, FLAGS_data_yfile, residual, FLAGS_samples, 1, ctx->m_worker_machines, ctx->rank);
  MPI_Barrier(worker_comm);  

  // create barrier instance for the worker threads only, note that worker_mach and second half thread are not included. 
  thread_barrier barrier(FLAGS_threads);

  double *Y = (double *)calloc(sizeof(double), FLAGS_samples);
  for(auto i=0; i<FLAGS_samples; i++){
    Y[i] = residual[i];
  }
  
  // create child threads and keep handlers for inter-thread communication
  std::vector<std::thread> childs;
  std::vector<cd_train *> handlers;

  if(FLAGS_algorithm.compare("lasso") == 0){
    for(auto i=0; i<FLAGS_threads; i++){
      cd_train *handler = new lasso(FLAGS_lambda, FLAGS_samples, FLAGS_columns, ctx->rank, ctx->m_worker_machines, i, 
				    FLAGS_threads, input_matrix, residual, barrier);
      handlers.push_back(handler);
      childs.push_back(std::thread(worker_thread, i, std::ref(*handler), ctx, Y));
    }
  }else if(FLAGS_algorithm.compare("logistic") == 0){
    for(auto i=0; i<FLAGS_threads; i++){
      cd_train *handler = new logistic(FLAGS_lambda, FLAGS_samples, FLAGS_columns, ctx->rank, ctx->m_worker_machines, i, 
				    FLAGS_threads, input_matrix, residual, barrier);
      handlers.push_back(handler);
      childs.push_back(std::thread(worker_thread, i, std::ref(*handler), ctx, Y));
    }    
  }else{
    assert(0);
  }

  whalf_threadctx *whalfthrds = new whalf_threadctx(ctx, handlers); // create a dedicated thread for the second half worker_mach job 
  assert(ctx->star_recvportmap.size() == 1 and ctx->star_sendportmap.size() == 1);
  auto pr = ctx->star_recvportmap.begin();
  _ringport *rport = pr->second;
  context *recv_ctx = rport->ctx;

  while(1){
    void *msg = recv_ctx->pull_entry_inq();
    if(msg != NULL){
      //      strads_msg(OUT, "\t\t[worker %d] main thread get one task from the coordinator \n", ctx->rank); 
      mbuffer *mbuf = (mbuffer *)msg;     
      int64_t cmdid = mbuf->cmdid ;
      message_type msgtype = mbuf->msg_type;
      void *retbuffer = NULL;

      mbuf = (mbuffer *)process_obj_cmd(ctx, mbuf, recv_ctx, &retbuffer, residual, row_range, Y); 
      // done by worker_mach thread  

      if(mbuf == NULL){
	assert(retbuffer != NULL);
	// pass mbuffer that holds partial object value in this local machine to the second half thread and let it send that to the coordinator
	int64_t temptaskcnt = 0;
	work_cmd *cmd = new work_cmd(m_work_object, temptaskcnt, cmdid, msgtype); 
	cmd->m_work = retbuffer;
	whalfthrds->put_entry_inq((void*)cmd);            
	continue;
      }

      msgtype = mbuf->msg_type;
      workhead *workhp = (workhead *)mbuf->data;
      uobjhead *temphp = (uobjhead *)((uintptr_t)workhp + sizeof(workhead));
      int64_t temptaskcnt = temphp->task_cnt;
      // split the task and scatter sub set of ids to update over worker threads. 
      mbuf = (mbuffer *)process_worker_cmd(ctx, mbuf, handlers, recv_ctx);  // done by helper threads (worker threads)
      assert(mbuf == NULL);
      /* collect the results from the worker threads and */
      work_cmd *cmd = new work_cmd(m_work_paramupdate, temptaskcnt, cmdid, msgtype); 
      whalfthrds->put_entry_inq((void*)cmd);            
    }// if (msg != NULL) ..
  } // end of while(1)

  for(auto i=0; i<FLAGS_threads; i++)
    childs[i].join();
  strads_msg(OUT, "[worker %d] terminate job lambda(%lf) \n", ctx->rank, FLAGS_lambda);
  LOG(INFO) << "[worker " << ctx->rank << "] terminate job" << std::endl;
  return NULL;
}

// for lasso / logistic regression 
void *process_obj_cmd(sharedctx *ctx, mbuffer *mbuf, context *recv_ctx, void **retbuf, cas_array<double> &residual, cont_range &row_range, double *Y){

  if(mbuf == NULL)
    return NULL; // since previous handler processes that and set mbuf to NULL

  if(mbuf->msg_type != USER_PROGRESS_CHECK)
    return mbuf;

  if(mbuf->msg_type == USER_PROGRESS_CHECK){  
    workhead *workhp = (workhead *)mbuf->data;
    assert(workhp->type == WORK_OBJECT);  

    double pobjsum=0;
    if(FLAGS_algorithm.compare("lasso") == 0){
      pobjsum = lasso::object_calc(row_range, residual); 
    }else if(FLAGS_algorithm.compare("logistic") == 0){
      pobjsum = logistic::object_calc(row_range, residual, Y);       
    }else{
      assert(0);
    }

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


void *process_worker_cmd(sharedctx *ctx, mbuffer *mbuf, std::vector<cd_train *> &handlers, context *recv_ctx){
  if(mbuf == NULL)
    return NULL; // since previous handler processes that and set mbuf to NULL
  assert(mbuf->msg_type != USER_PROGRESS_CHECK); 
  int maxthrds = FLAGS_threads;
  int64_t cmdid = mbuf->cmdid;  
  if(mbuf->msg_type == USER_UPDATE){  
    workhead *workhp = (workhead *)mbuf->data;
    if(workhp->type == WORK_STATUPDATE){  // in cd case, mostly, residual update 
      // skip this. I merge residual update with parameter update 
    } else if(workhp->type == WORK_PARAMUPDATE){ // weight update   
      strads_msg(INF, "Rank(%d) got param update cmd from coordinator \n", ctx->rank);      
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
	handlers[th]->put_entry_inq((void*)cmd[th]);       
      }
      // scatter the task ........ 
    } else if(workhp->type == WORK_OBJECT){ // weight update   
      assert(0); // this should be handled in the previous process obj cmd handler       
    }else{
      LOG(FATAL) << "[worker] USER_WORKER msg contains non-support sched-type " << endl;
    }
    recv_ctx->release_buffer((void *)mbuf); // don't forget this           
    return NULL;
  } 
  return mbuf ;
}

/* 
 * @brief worker threads for helping worker_mach thread in charge of post processing 
 *   implemented in linux POSIX-Pthread 
 */
void *whalf_thread(void *arg){ 
  whalf_threadctx  *ctx = (whalf_threadctx *)arg; // this pointer of scheduler_threadctx class  
  LOG(INFO) << " [WORKER_MACH SECOND_HALF THREADS] RANK : " << ctx->get_rank() << endl;  
  sharedctx *shctx = ctx->get_shctx();
  auto ps = shctx->star_sendportmap.begin();
  _ringport *sport = ps->second;
  context *send_ctx = sport->ctx;
  int64_t iterations=0;
  std::vector<cd_train *> &sthrds = ctx->get_handlers();
  while(1){

    void *rcmd = ctx->get_entry_inq_blocking(); 
    work_cmd *cmd = (work_cmd *)rcmd;

    if(cmd->m_type == m_work_object){
      assert(cmd->m_work != NULL);
      mbuffer *result = (mbuffer *)cmd->m_work;
      while(send_ctx->push_entry_outq(result, sizeof(mbuffer)));
      //      delete(cmd);
      strads_msg(ERR, "\t\t[worker %d] helper thread object calc \n", shctx->rank); 
      continue;
    }

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
    int maxthrds = FLAGS_threads;
    int donethrdcnt=0;
    int64_t progress = 0;

    for(int th=0; th<maxthrds; th++){
      while(1){
	void *ret = sthrds[th]->get_entry_outq();
	// TODO URGENT TODO : DO SOME MACHINE WISE AGGREGATION WORK HERE
	if(ret != NULL){
	  donethrdcnt++;
	  work_cmd *tmpcmd = (work_cmd *)ret;
	  if(tmpcmd->m_type == m_work_paramupdate){	     
	    uobjhead *rpobjhp = (uobjhead *)tmpcmd->m_result;
	    int64_t rptaskcnt = rpobjhp->task_cnt;
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
    retobjhp->task_cnt = progress; // Don't forget this.       
    if(progress != temptaskcnt){
      strads_msg(OUT, "Fatal : progress(%ld) != temptaskcnt - expected results : %ld \n", 		  
		 progress, temptaskcnt);
      assert(0);
    }
    result->cmdid = cmdid;
    result->src_rank = shctx->rank;
    while(send_ctx->push_entry_outq(result, sizeof(mbuffer)));   
  }
  return NULL;
}

