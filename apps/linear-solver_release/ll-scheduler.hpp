#ifndef _CDSCHEDULER_HPP_
#define _CDSCHEDULER_HPP_

/********************************************************
   @author: Jin Kyu Kim (jinkyuk@cs.cmu.edu)

   Distributed scheduler framework 
********************************************************/
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

#include <deque>
#include <vector>
#include <algorithm>


#include "getcorr.hpp"

#include "strads/include/common.hpp"
//#include "ds/dshard.hpp"

#include "strads/util/utility.hpp"
#include "strads/netdriver/comm.hpp"

//#include "com/rdma/rdma-common.hpp"
#if defined(INFINIBAND_SUPPORT)

#include "strads/netdriver/rdma/rdma-common.hpp"

#else

#include "strads/netdriver/zmq/zmq-common.hpp"

#endif

#include <glog/logging.h>

#include "lasso.pb.hpp"
#include "lassoll.hpp"

#include "cd-util.hpp"



#define EXPERIMENT_BW

//#define BW_TUNE_FACTOR (0.00000001) // smaller factor, smaller base line weight

#define BW_TUNE_FACTOR (0.01) // smaller factor, smaller base line weight

// use this baseline number for AD experiment, 

//#define BW_TUNE_FACTOR (0.1) // smaller factor, smaller base line weight
// Lasso used 0.1 for experiments

#define CORRELATION_OPTIMIZATION

void *scheduler_thread(void *);
void *scheduler_mach(void *);

typedef std::deque<void *>inter_threadq;

typedef struct{
  int test;
}update_stat;

typedef struct{
  int64_t size;
  int64_t *vidxlist;
}phase_set;

typedef struct{
  int64_t size; // number of entries in the gidxlist in each update 
  int64_t *gidxlist; // global task ids - global coeffieicnt ids
  double *new_weights;      // new weight information corresponding for the idx in gidxlist
  int64_t maxlistsize; // specify max list size, usually set to length of curreht thread's space
}weight_change;

enum samplingcmd_type{ m_weight_update, m_init_weight_update, m_make_phase, m_bw_change, m_restart_weight_update };

class sampling_cmd{

public:
  sampling_cmd(samplingcmd_type type, int64_t wsize, int samplergid){
    m_type = type;
    m_winfo = (weight_change *)calloc(1, sizeof(weight_change));
    m_winfo->size = wsize;
    m_winfo->gidxlist = (int64_t *)calloc(wsize, sizeof(int64_t));
    m_winfo->new_weights = (double *)calloc(wsize, sizeof(double));
    m_winfo->maxlistsize = wsize;
    m_samplergid = samplergid;
  }

  ~sampling_cmd(){   
    assert(m_winfo);
    assert(m_winfo->gidxlist);
    assert(m_winfo->new_weights);
    free(m_winfo->gidxlist);
    free(m_winfo->new_weights);
    free(m_winfo);
    //    strads_msg(ERR, "Sampling CMD destructor is called \n");
  }

  int m_samplergid;
  samplingcmd_type m_type;
  weight_change *m_winfo; // when mach to thread, no NULL,    when thread to mach, should be NULL
  phase_set *m_phase; // when mach to thread, should be NULL,  when thread to mach, no NULL
};


// CAVEAT : big assumption: scheduler_thread / wsamplerctx are not thread safe. 
//          Only one thread should access these classes 
//          Since each scheduler threads are independent, this is reasonable for current design
class wsamplerctx{

public:
  wsamplerctx(int gid, int lid, int64_t start, int64_t end, double bw, int64_t maxset, double infthreshold) // gid: scheduler thrd's global id.
    : m_gid(gid), m_lid(lid), m_start(start), m_base(start), m_end(end), m_len(end - start +1), m_bw(bw), m_totalw(0), m_maxset(maxset), m_infthreshold(infthreshold) {
    assert(end > start);
    m_weights = (double *)calloc(m_len, sizeof(double));   
    assert(m_weights);
    m_rndpts = (double *)calloc(m_maxset*10, sizeof(int64_t));
    m_tmpsamples = (int64_t *)calloc(m_maxset*10, sizeof(int64_t));
    m_sortspace = (int64_t *)calloc(m_maxset*10, sizeof(int64_t));
    m_sortspaceweight = (double *)calloc(m_maxset*10, sizeof(double));

    assert(m_rndpts);
    srand(SCHEDULING_RANDOM_SEED);
    // ONE CAVEAT : task start and end are global index while weight array starts from 0 to m_len -1
    // Definitely, it needs to do remapping between global and local task id whenever it exchanges with rest of the system
    strads_msg(ERR, " creates wsampler glid(%d:%d) start(%ld)b(%ld) end(%ld) len(%ld) bw(%lf) maxset(%ld) totalw(%lf)\n", 
	       m_gid, m_lid, m_start, m_base, m_end, m_len, m_bw, m_maxset, m_totalw);

    //    m_samples = (int64_t *)calloc(m_maxset*2, sizeof(int64_t));
    m_samples = (int64_t *)calloc(m_maxset*100, sizeof(int64_t));

    m_wchange.size = 0;
    m_wchange.gidxlist = (int64_t *)calloc(m_len, sizeof(int64_t));
    m_wchange.new_weights = (double *)calloc(m_len, sizeof(double));
    m_wchange.maxlistsize = m_len;

    m_nzelement=0;

    m_restart_flag = false;
    m_restart_cnt = 0; 

  };


  /////////////////////////////////////////////////////
  // this for mcord testing with AD ..................
  /////////////////////////////////////////////////////
  int64_t do_sampling_tmp(int64_t setsizegiven, int64_t *samples){

#if defined(EXPERIMENT_BW)
    if(m_nzelement > 10 and m_totalw != 0.0){
      double nzshare = (m_totalw/(1.0*m_nzelement));
      m_bw = nzshare*BW_TUNE_FACTOR;
      if(m_bw == 0){
        m_bw = 0.1;
	strads_msg(INF, "IF  BW set to 0.1 \n");	
      }
      strads_msg(INF, "nzshare : %lf  m_bw : %lf   m_totalw: %lf \n", nzshare, m_bw, m_totalw);
    }else{
      m_bw = 0.1;
      strads_msg(INF, " ELSE BW set to 0.1 \n");
    }
    double maxrange = m_totalw + m_bw*m_len;
#else
    //    m_bw = 0.000000001;
    double maxrange = m_totalw + m_bw*m_len;
#endif

    // for mcor testing 
    int64_t setsize = setsizegiven*4;

    assert(setsize < m_len);
    /* Step 1:  pick up random n points from 0 to fullrangesize   */
    for(int64_t i=0; i<setsize; i++){
      double rndpts = util_get_double_random(0, maxrange);
      m_rndpts[i] = rndpts;
    }

    // sort the random points from 0 to maxrange 
    sort_rndpoints(m_rndpts, setsize);

    /* Step 2: searching for index corresponds to the n random points */
    double acc =0;
    int64_t rndptsidx=0, dupcnt=0, cancelled=0, sampleidx=0, samplecnt;
    for(int64_t i=0; i < m_len; i++){ // move forward on poolindex
      if((acc + m_weights[i] + (i + 1)*m_bw) >= m_rndpts[rndptsidx]){     
	assert(sampleidx <= m_len);
	m_tmpsamples[sampleidx] = i;
	strads_msg(INF, "%ldth task is chosen as %ldth sample by %ldth rndpt %lf \n", 
		   i, sampleidx, rndptsidx, m_rndpts[rndptsidx]);
	sampleidx++; // move forward on sample idx
	dupcnt=0;
	while(acc + m_weights[i] + (i + 1)*m_bw >= m_rndpts[rndptsidx]){
	  /* if more than one random points fall into a task, assign only one chance */
	  dupcnt++;	
	  rndptsidx++; // move forward on rndptsidx 
	  if(rndptsidx >= setsize)
	    break;	
	}      
	if(dupcnt > 1){
	  strads_msg(INF, "  more than one random pts fall into the [%ld]th task's range, move %ld slots\n", i, dupcnt);
	} 
	if(rndptsidx >= setsize){
	  break;
	}
	cancelled += (dupcnt -1);
      }
      acc += m_weights[i];
    }    

    if(sampleidx > setsizegiven){
      sampleidx = setsizegiven; // drop the selected ones overflown 
    }

    _sort(m_tmpsamples, sampleidx);

    for(int64_t i=0; i<sampleidx; i++){
      samples[i] = m_tmpsamples[i] + m_base;  // CAVEAT : m_base to remap local task id into global task id
    }
    samplecnt = sampleidx;
    strads_msg(INF, "Desired %ld but cancelled %ld due to duplication\n",
	       setsize, cancelled);
    return samplecnt;
  }


  /////////////////////////////////////////////////////
  // this is original sampler I used until Feb 3 2015
  /////////////////////////////////////////////////////
  // do sampling and return lists of global task ids selected.
  // setsize: desired sample size, 
  // return: the number of samples to be selected, * samples holds sampled ids
  int64_t do_sampling(int64_t setsize, int64_t *samples){
    assert(setsize < m_len);
    //    assert(setsize <= m_maxset);

    //    double maxrange = m_totalw + m_bw*m_len;
#if defined(EXPERIMENT_BW)
    if(m_nzelement > 10 and m_totalw != 0.0){
      double nzshare = (m_totalw/(1.0*m_nzelement));

      //      m_bw = nzshare*0.1;
      m_bw = nzshare*BW_TUNE_FACTOR;

      if(m_bw == 0){
        m_bw = 0.1;
      }
      strads_msg(INF, "nzshare : %lf  m_bw : %lf   m_totalw: %lf \n", nzshare, m_bw, m_totalw);
    }else{
      m_bw = 0.1;
    }
    double maxrange = m_totalw + m_bw*m_len;
#else
    double maxrange = m_totalw + m_bw*m_len;
#endif

    /* Step 1:  pick up random n points from 0 to fullrangesize   */
    for(int64_t i=0; i<setsize; i++){
      double rndpts = util_get_double_random(0, maxrange);
      m_rndpts[i] = rndpts;
    }



    strads_msg(ERR, "\t\t @@@@@@ sort_rndpoints ++ \n");
    // sort the random points from 0 to maxrange 
    sort_rndpoints(m_rndpts, setsize);
    strads_msg(ERR, "\t\t @@@@@@ sort_rndpoints -- \n");

    /* Step 2: searching for index corresponds to the n random points */
    double acc =0;
    int64_t rndptsidx=0, dupcnt=0, cancelled=0, sampleidx=0, samplecnt;
    for(int64_t i=0; i < m_len; i++){ // move forward on poolindex
      if((acc + m_weights[i] + (i + 1)*m_bw) >= m_rndpts[rndptsidx]){     
	assert(sampleidx <= m_len);
	m_tmpsamples[sampleidx] = i;
	strads_msg(INF, "%ldth task is chosen as %ldth sample by %ldth rndpt %lf \n", 
		   i, sampleidx, rndptsidx, m_rndpts[rndptsidx]);
	sampleidx++; // move forward on sample idx
	dupcnt=0;
	while(acc + m_weights[i] + (i + 1)*m_bw >= m_rndpts[rndptsidx]){
	  /* if more than one random points fall into a task, assign only one chance */
	  dupcnt++;	
	  rndptsidx++; // move forward on rndptsidx 
	  if(rndptsidx >= setsize)
	    break;	
	}      
	if(dupcnt > 1){
	  strads_msg(INF, "  more than one random pts fall into the [%ld]th task's range, move %ld slots\n", i, dupcnt);
	} 
	if(rndptsidx >= setsize){
	  break;
	}
	cancelled += (dupcnt -1);
      }
      acc += m_weights[i];
    }

    strads_msg(ERR, "\t\t @@@@@@ _sort sampled results ++ \n");    
    // sorting samples with their weight. 
    _sort(m_tmpsamples, sampleidx);
    strads_msg(ERR, "\t\t @@@@@@ _sort sampled results -- \n");    

    for(int64_t i=0; i<sampleidx; i++){
      samples[i] = m_tmpsamples[i] + m_base;  // CAVEAT : m_base to remap local task id into global task id
    }
    samplecnt = sampleidx;
    strads_msg(INF, "Desired %ld but cancelled %ld due to duplication\n",
	       setsize, cancelled);

    return samplecnt;
  }
  

  void _sort(int64_t *samples, int64_t cnt){
    //    strads_msg(INF, "Before Sorting \n");
    //    for(int i=0; i < cnt; i++){
    //      strads_msg(INF, " LID (%ld)W(%lf) \n", samples[i], m_weights[samples[i]]);      
    //    }
    //    strads_msg(ERR, "\n\n\n");
    std::vector<uint64_t>zerolist;
    int nzcnt=0;
    int zerocnt=0;
    for(int64_t i=0; i < cnt; i++){
      int64_t lid = samples[i];
      //      if(m_weights[i] == 0.0){
      if(m_weights[lid] == 0.0){
	zerolist.push_back(lid);
	zerocnt++;
      }else{
	m_sortspace[nzcnt] = lid;
	m_sortspaceweight[nzcnt] = m_weights[lid];
	nzcnt++;
      }
    }    
    //    strads_msg(INF, "INTERMEDIAT : Zero : %d   NZ : %d   out of %ld \n", 
    //	       zerocnt, nzcnt, cnt);
    //    if(nzcnt != 0){
    //      sleep(5);
    //    }
    for(int i=0; i<nzcnt; i++){
      double max = -100000000.0;
      int maxpos = -1;
      for(int i=0; i<nzcnt; i++){
	if(m_sortspaceweight[i] > max){
	  max = m_sortspaceweight[i];
	  maxpos = i;
	}	       
      }
      if(maxpos == -1) // for sanity checking. this can not happen 
	assert(0);      
      samples[i] = m_sortspace[maxpos];      
      m_sortspaceweight[maxpos] = -100000000.0; 
    }
    for(int i=0; i<zerocnt; i++){
      samples[i + nzcnt] = zerolist[i];
    }
    //    strads_msg(ERR, "AFTER Sorting \n");
    //    for(int i=0; i < cnt; i++){
    //      strads_msg(ERR, " LID (%ld)W(%lf) \n", samples[i], m_weights[samples[i]]);      
    //    }
    //    strads_msg(ERR, "\n\n\n");
  }


  // assume that wchange holds global task ids
  int64_t update_weight(weight_change *wchange){
    int64_t size = wchange->size;
    int64_t *gidlist = wchange->gidxlist; 
    double *new_weights = wchange->new_weights;
    assert(wchange->size <= wchange->maxlistsize);
    assert(wchange->size <= m_len);
   
    for(int64_t i=0; i < size; i++){
      int64_t changedid = gidlist[i];
      double signed_w = new_weights[i];
      assert(changedid >= m_start); // thread's space range in global address space 
      assert(changedid <= m_end);   

      // TODO : URGENT TODO : TRY square form as well. 
      double new_w = fabs(signed_w); // make unsigned weight
      int64_t localidx = changedid - m_start;

      assert(localidx >= 0);
      assert(localidx <= m_len);

      double old_w = fabs(m_weights[localidx]);
      double diff = new_w - old_w; // get difference between new_w and old_w
      m_weights[localidx] = new_w; // replace old w with new w        
      m_totalw = m_totalw + diff;      
      if(m_totalw < 0.0){
	strads_msg(ERR, "fatal: errors in totalw bookkeeping information : %lf should be GTE 0.0 \n", m_totalw);
      }
      //assert(m_totalw >= 0.0);
    }

    m_nzelement=0;
    // REPLACE above m_totalw = .... assert(m_totalw >= 0.0);.....                                                                   
    m_totalw = 0;
    for(int64_t i=0; i<m_len; i++){
      m_totalw += m_weights[i];
      if(m_weights[i] != 0.0){
        m_nzelement++;
      }
    }

    // verification of fullrnage size and aggregate of weight in the pool
#if 0 // debugging purpose only
    double rangesum=0;
    for(long i=0; i < m_len; i++){
      rangesum += m_weights[i];
    }
    // TODO : Think about more smaller errors ...
    if(fabs(rangesum - stpriority->fullrangesize) > 0.1){    
      // due to the precision error, small error happens 
      strads_msg(ERR, "[*******ERROR] stpriority->fullrange(%20.20lf) != sum(%20.20lf)  %lf\n", 
		 m_totalw, rangesum, m_totalw);
      assert(0);
    }else{
      strads_msg(ERR, "[*******GOOD] stpriority->fullrange(%lf) == sum(%lf)\n", 
		 m_totalw, rangesum);
    }
#endif
    return size;
  }








  // TODO : Add wrapper to start user dependency checking code 
  //gidx samples holds sampled idx in global space 
  //  int64_t check_interference(int64_t *gidxsamples, int64_t size, dshardctx *dshard){    
  // col_range : machine level column range 
  int64_t check_interference(int64_t *gidxsamples, int64_t size, col_vspmat &cm_vmat, cont_range &col_range){    

    //    strads_msg(INF, "Check Interference: m_start(%ld) m_end(%ld) dshard range c_s(%ld) c_end(%ld)\n", 
    //	       m_start, m_end, dshard->m_range.c_start, dshard->m_range.c_end);

    if( !((uint64_t)m_start >= col_range.get_min())){
     
      strads_msg(OUT, "@@@@@@@@ m_start : %ld   col_range.min(start)(%ld)\n", m_start, col_range.get_min()); 
    }

    if(!((uint64_t)m_end <= col_range.get_max())){
      strads_msg(OUT, "@@@@@@@@ m_end : %ld   col_range.max(start)(%ld)\n", m_end, col_range.get_max()); 
    }

    assert((uint64_t)m_start >= col_range.get_min()); 
    assert((uint64_t)m_end <= col_range.get_max()); 

#if !defined(COL_SORT_INPUT) 
    //    assert(0); // only for sanity checking 
#endif

    int64_t gvidsample=0, spos=0;
    int64_t gvidret=0, chkpos=0, rvidxsize=0;
    int64_t *rvidx=(int64_t *)calloc(size, sizeof(int64_t));
    double infdegree=0;
    bool conflict=false;

    rvidx[0] = gidxsamples[0];
    rvidxsize=1;

    for(spos=1; spos < size; spos++){

      gvidsample = gidxsamples[spos];  
      assert(gvidsample <= m_end); // m_end/start scheduler thread's range in global space 
      assert(gvidsample >= m_start);

      conflict=false;
      // see if this lvidsample does cause conflict with existing ones in rvidx list.  
      for(chkpos = 0; chkpos < rvidxsize; chkpos++){
	assert(rvidxsize < size);
	gvidret = rvidx[chkpos];
	assert(gvidret >= m_start);
	assert(gvidret <= m_end);

#if defined(CORRELATION_OPTIMIZATION)
	// apply further optimization 
        int64_t lid1 = gvidsample - m_base;  // CAVEAT : m_base to remap local task id into global task id
        int64_t lid2 = gvidret - m_base;  // CAVEAT : m_base to remap local task id into global task id
	double weight1 = m_weights[lid1];
	double weight2 = m_weights[lid2];

	if(m_restart_flag == true){
	  if(weight1 != 0.0 and weight2 != 0.0){
	    infdegree = getcorr_pair_sparse_vmat(cm_vmat.col(gvidsample), cm_vmat.col(gvidret), FLAGS_samples, 
						 0, 0, 0, 0, 0, gvidsample, gvidret); // lastcorr.cpp
	  }else{
	    // skip and set correlation to zero.. because zero coeff rarely becomes non zero 
	    // and zeroed-coeff does not cause conflict with other nz or zero 
	    // if zero coeff become nz .. it will cause conflicts.. 
	    // but very very rarely. 
	    infdegree = 0.0;
	  }
	}else{

	  // for the scheduling on case with dependency checking

	  //#if !defined(NO_WEIGHT_SAMPLING) // WEIGHT SAMPLING ON. 
	  if(FLAGS_weight_sampling == true){
	    infdegree = getcorr_pair_sparse_vmat(cm_vmat.col(gvidsample), cm_vmat.col(gvidret), FLAGS_samples, 
						 0, 0, 0, 0, 0, gvidsample, gvidret); // lastcorr.cpp	    
	    //#else
	  }else{
	    // for the scheduling off case with dependency checking
	    if(weight1 != 0.0 and weight2 != 0.0){
	      infdegree = getcorr_pair_sparse_vmat(cm_vmat.col(gvidsample), cm_vmat.col(gvidret), FLAGS_samples, 
						   0, 0, 0, 0, 0, gvidsample, gvidret); // lastcorr.cpp
	    }
	  }
	  //#endif // NO_WEIGHT_SAMPLING

	}
#else  // CORRELATION_OPTIMIZATION
	infdegree = getcorr_pair_sparse_vmat(cm_vmat.col(gvidsample), cm_vmat.col(gvidret), FLAGS_samples, 
					     0, 0, 0, 0, 0, gvidsample, gvidret); // lastcorr.cp	
#endif // end CORRELATION_OPTIMIZATION

	strads_msg(INF, "interference between (%ld) (%ld)  == %lf \n", 
		   gvidsample, gvidret, infdegree);

	if(infdegree >= m_infthreshold){
	  strads_msg(INF, "@@ 2@@ CORR between (%ld)(%ld): %lf is too high \n", gvidsample, gvidret, infdegree);
	  conflict = true;
	  break;
	}
	// TODO : give second chance to the filtered out coeff.
      }

      if(!conflict){ // if false - does not cause conflicts
	rvidx[rvidxsize] = gvidsample;
	rvidxsize++;
      }
      if(rvidxsize >= size){
	break;
      }		   
    }
    for(int64_t i=0; i < rvidxsize; i++){
      gidxsamples[i] = rvidx[i];
    }

    free(rvidx);
    strads_msg(INF, "full interference checking target(%ld) survive %ld \n", size, rvidxsize); 
    return rvidxsize;
  }  

  void sort_rndpoints(double *rndlist, long size){
    std::vector<double> array(size, 0); // reserve 
    for(auto i=0; i<size; i++){
      array[i] = rndlist[i];
    }
    std::sort(array.begin(), array.end());
    for(auto i=0; i<size; i++){
      rndlist[i] = array[i];
    }
  }
	  
#if 0 
  void sort_rndpoints(double *rndlist, long size){
    dataset_t sortmap;
    value_index_t &sortmap_valueidx = sortmap.get<value_tag>();
    long idx=0;
    for(long i=0; i < size; i++){
      sortmap.insert(element_t(i, fabs(rndlist[i])));
    }
    for(value_index_t::iterator it = sortmap_valueidx.begin(); it != sortmap_valueidx.end(); it++){
      rndlist[idx] = it->second;
      idx++;
    }
    sortmap.clear();
    // TODO : URGENT TODO : CHECK MEMORY LEAK WITH CLEAR 
  }
#endif 

  int m_gid; // threads global id.
  int m_lid; // local thread id in a scheduler machine 
  int64_t m_start; // local thread's start and end 
  int64_t m_base; // same as m_start 
  int64_t m_end;
  int64_t m_len;
  double m_bw; // base weight  
  double m_totalw; // total sum of wehgits over all my scheduling space from start to end. 
  int64_t m_maxset; // limits on the number of set to be seleted from one sampling 

  double *m_weights;
  double *m_rndpts; // temporary space for keeping randomly chosen points.

  int64_t *m_tmpsamples; // for temporary buffer for sampling
  int64_t *m_samples;  //  buffer structure for sampling. scheduler thread can use this buffer one

  weight_change m_wchange;
  double m_infthreshold; // threshold on the degree of interference between a pair of tasks

  int64_t m_nzelement;
  int64_t *m_sortspace; // for temporary buffer for sampling
  double *m_sortspaceweight; // for temporary buffer for sampling

  bool m_restart_flag;
  int  m_restart_cnt; 


};



// CAVEAT : big assumption: scheduler_thread / wsamplerctx are not thread safe. 
//          Only one thread should access these classes 
//          Since each scheduler threads are independent, this is reasonable for current desig
class scheduler_threadctx{

public: 
  scheduler_threadctx(int rank, int schedulermid, int lid, int threadid, int64_t task_start, int64_t task_end, double bw, int64_t maxset, double infthreshold, sharedctx *parenctx, col_vspmat &input_matrix): m_created(false), m_input_matrix(input_matrix), m_mutex(PTHREAD_MUTEX_INITIALIZER), m_upsignal(PTHREAD_COND_INITIALIZER), m_rank(rank), m_scheduler_mid(schedulermid), m_scheduler_thrdgid(threadid), m_task_start(task_start), m_task_end(task_end), m_inq_lock(PTHREAD_MUTEX_INITIALIZER), m_outq_lock(PTHREAD_MUTEX_INITIALIZER) {
    // threadid == gid 

    m_wsampler = new wsamplerctx(threadid, lid, task_start, task_end, bw, maxset, infthreshold);

    // fill out my data structure for sampling for start - end tasks


    int rc = pthread_attr_init(&m_attr);
    checkResults("pthread attr init m_attr failed", rc);
    rc = pthread_create(&m_thid, &m_attr, scheduler_thread, (void *)this);
    checkResults("pthread create failed in scheduler_threadctx", rc);
    m_created = true;  
    m_parenctx = parenctx;

  };

  // caveat: precondition: cmd should be allocated structued eligible for free().
  void put_entry_inq(void *cmd){
    int rc = pthread_mutex_lock(&m_inq_lock);
    checkResults("pthread mutex lock m inq lock failed ", rc);
    if(m_inq.empty()){
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
    checkResults("pthread mutex lock m_outq_lock failed ", rc);

    if(!m_inq.empty()){
      ret = m_inq.front();
      m_inq.pop_front();
    }else{
      pthread_cond_wait(&m_upsignal, &m_inq_lock); // when waken up, it will hold the lock. 
      ret = m_inq.front();
      m_inq.pop_front();
    }

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

  int get_rank(void){ return m_rank; }
  int get_scheduler_mid(void){ return m_scheduler_mid; }
  int get_scheduler_thrdgid(void){ return m_scheduler_thrdgid; }

  int64_t get_task_start(void){ return m_task_start; }
  int64_t get_task_end(void){ return m_task_end; }

  wsamplerctx *m_wsampler; // weight based sampling context initialized in the constructure of scheduler thread ctx 
  sharedctx *m_parenctx;

  bool m_created;

  col_vspmat &m_input_matrix;

private:


  pthread_mutex_t m_mutex;
  pthread_cond_t m_upsignal;

  int m_rank;
  int m_scheduler_mid;
  int m_scheduler_thrdgid; // global thread id. 

  int64_t m_task_start; // local thread's start / end 
  int64_t m_task_end;

  inter_threadq m_inq;
  inter_threadq m_outq;

  pthread_mutex_t m_inq_lock;
  pthread_mutex_t m_outq_lock;

  pthread_t m_thid;
  pthread_attr_t m_attr; 

};











#endif 
