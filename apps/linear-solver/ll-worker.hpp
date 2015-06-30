#ifndef _LL_LDAWORKER_HPP_
#define _LL_LDAWORKER_HPP_

#include <stdio.h>
#include <strads/include/common.hpp>
#include <strads/include/child-thread.hpp>
#include <strads/util/utility.hpp>
#include <iostream>     // std::cout
#include <algorithm>    // std::for_each
#include <vector>       // std::vector
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <mpi.h>
#include <assert.h>
#include <mutex>
#include <thread>

#include <pthread.h>

#include "lasso.pb.hpp"
#include "lassoll.hpp"

#include "cd-train.hpp"

enum workcmd_type{ m_work_statupdate, m_work_paramupdate, m_work_object };

class work_cmd{
public:
  work_cmd(workcmd_type type): m_type(type), m_work(NULL), m_result(NULL), m_temptaskcnt(-1), m_cmdid(-1){
    m_type = type;
  }
  work_cmd(workcmd_type type, int64_t temptaskcnt, int64_t cmdid, message_type msgtype): m_type(type), m_work(NULL), m_result(NULL), m_temptaskcnt(temptaskcnt), m_cmdid(cmdid), m_msgtype(msgtype){
    m_type = type;
  }
  ~work_cmd(){ 
    if(m_work != NULL){
      free(m_work);     
    }
    if(m_result != NULL){
      free(m_result);
    }
  }
  int m_workergid;
  workcmd_type m_type;
  void *m_work;    // when mach to thread, no NULL,    when thread to mach, should be NULL
  void *m_result; // when mach to thread, should be NULL,  when thread to mach, no NULL
  int64_t m_temptaskcnt;
  int64_t m_cmdid;
  message_type m_msgtype;
  double m_svm_obj_tmp;
};

void *whalf_thread(void *arg);

class whalf_threadctx{

public:
  whalf_threadctx(sharedctx *shctx, std::vector<cd_train *> &handlers):m_created(false), m_mutex(PTHREAD_MUTEX_INITIALIZER), m_upsignal(PTHREAD_COND_INITIALIZER), m_inq_lock(PTHREAD_MUTEX_INITIALIZER), m_outq_lock(PTHREAD_MUTEX_INITIALIZER), m_shctx(shctx), m_handlers(handlers) {

    int rc = pthread_attr_init(&m_attr);
    checkResults("pthread attr init m_attr failed", rc);
    rc = pthread_create(&m_thid, &m_attr, whalf_thread, (void *)this);
    checkResults("pthread create failed in scheduler_threadctx", rc);
    m_created = true;  
    //    m_thrds_per_worker = mparam->m_sp->m_thrds_per_worker;
    //    m_worker_thrdgid = workermid * m_thrds_per_worker + threadid ; // local thread id. 
  }

  // caveat: precondition: cmd should be allocated structued eligible for free().
  void put_entry_inq(void *cmd){
    int rc = pthread_mutex_lock(&m_inq_lock);
    checkResults("pthread mutex lock m inq lock failed ", rc);
    if(m_inq.empty()){     
      //      strads_msg(OUT, "\t\t\t[whalf wthread] put_entry_inq : call cond signal \n"); 

      rc = pthread_cond_signal(&m_upsignal);
      checkResults("pthread cond signal failed ", rc);
    }
    
    m_inq.push_back(cmd);    

    rc = pthread_mutex_unlock(&m_inq_lock);
    checkResults("pthread mutex lock m inq unlock failed ", rc);
  }

  // caveat: if nz returned to a caller, the caller should free nz structure 
  void *get_entry_inq_blocking(void){
    int rc = pthread_mutex_lock(&m_inq_lock);
    void *ret = NULL;
    checkResults("pthread mutex lock m_inq_lock failed ", rc);

    if(!m_inq.empty()){
      ret = m_inq.front();
      m_inq.pop_front();
    }else{
      //      strads_msg(OUT, "\t\t [worker  %d  helper thread ] goes to waiting mode. \n", m_shctx->rank); 
      pthread_cond_wait(&m_upsignal, &m_inq_lock); // when waken up, it will hold the lock. 
      //      strads_msg(OUT, "\t\t [worker  %d  helper thread ] wake up from cv waiting mode. \n", m_shctx->rank); 
      ret = m_inq.front();
      m_inq.pop_front();
    }


    //      strads_msg(OUT, "\t\t [worker  %d  helper thread ] wake up and unlock m_inq_lock haha \n", m_shctx->rank); 
    rc = pthread_mutex_unlock(&m_inq_lock);
    checkResults("pthread mutex lock m_outq_unlock failed ", rc);   
    return ret;
  }

  // caveat: precondition: cmd should be allocated structued eligible for free().
  void put_entry_outq(void *cmd){
    int rc = pthread_mutex_lock(&m_outq_lock);
    checkResults("pthread mutex lock m inq lock failed ", rc);    
    m_outq.push_back(cmd);    
    rc = pthread_mutex_unlock(&m_outq_lock);
    checkResults("pthread mutex lock m inq unlock failed ", rc);
  }

  // caveat: if nz returned to a caller, the caller should free nz structure 
  void *get_entry_outq(void){
    int rc = pthread_mutex_lock(&m_outq_lock);
    void *ret = NULL;
    checkResults("pthread mutex lock m_outq_lock failed ", rc);

    if(!m_outq.empty()){
      ret = m_outq.front();
      m_outq.pop_front();
    }    
    rc = pthread_mutex_unlock(&m_outq_lock);
    checkResults("pthread mutex lock m_outq_unlock failed ", rc);   
    return ret;
  }

  int get_rank(void){ return m_shctx->rank; }
  //  int get_worker_mid(void){ return m_worker_mid; }
  //  int get_worker_thrdlid(void){ return m_worker_thrdlid; }
  //  int get_worker_thrdgid(void){ return m_worker_thrdgid; }
  sharedctx *get_shctx(void) { return m_shctx; }
  //  worker_threadctx **get_sthrds(void) { return m_sthrds; }

  std::vector<cd_train *> &get_handlers(void) { return m_handlers; }

private:

  bool m_created;
  pthread_mutex_t m_mutex;
  pthread_cond_t m_upsignal;

  inter_threadq m_inq;
  inter_threadq m_outq;

  pthread_mutex_t m_inq_lock;
  pthread_mutex_t m_outq_lock;

  pthread_t m_thid;
  pthread_attr_t m_attr; 

  sharedctx *m_shctx;

  //  worker_threadctx **m_sthrds;
  std::vector<cd_train *>&m_handlers;
};

#endif 
