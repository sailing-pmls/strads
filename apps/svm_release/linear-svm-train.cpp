#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <math.h>
#include <chrono>
#include <random>
#include <numeric>      // std::iota
#include <algorithm>
#include <ctime>
#include <assert.h>
#include <glog/logging.h>

#include "linear-svm.hpp"
#include "param.hpp"
#include "lasso.pb.hpp"

using std::cout;
using std::endl;
using std::string;
using std::ifstream;
using std::istringstream;
using std::vector;
using std::stol;
using std::stod;
using std::min;
using std::max;

struct dpair{
  long vid;
  double PG;
};

#define MAX_PARALLEL 2048
void dualcd_svm::train_coordinator(sharedctx *ctx){

  train_init(); // reserve m_alpha and m_w and initialize them to all zeros  
  double Dii, U;
  if(!FLAGS_loss.compare("l1")){
    Dii = 0.0;
    U = FLAGS_C;
  }else if(!FLAGS_loss.compare("l2)")){
    Dii = 1/(2*FLAGS_C);
    U = std::numeric_limits<double>::max();
  }else{
    LOG(FATAL) << " loss function option should be either l1 or l2 " << endl;
  }

  std::mt19937 randgen(0x122);
  std::vector<long> active_set(m_l);
  std::iota(active_set.begin(), active_set.end(), 0);
  std::vector<long> randidx;
  double M_bar = INFINITY;
  double m_bar = -INFINITY;

  std::clock_t begin=std::clock();   
  double *G = (double *)calloc(sizeof(double), MAX_PARALLEL);
  double *spdot = (double *)calloc(sizeof(double), MAX_PARALLEL);
  double *mwsp =  (double *)calloc(sizeof(double), m_l);

  // Implement  the algorithm 3 in the reference paper for shrinkage 
  for(auto iter=0; iter < FLAGS_max_iter; iter++){

    active_set.reserve(m_l);
    std::iota(active_set.begin(), active_set.end(), 0);
    std::shuffle(active_set.begin(), active_set.end(), randgen);   

    // 1. 
    double M = -INFINITY;
    double m = INFINITY;

    assert(randidx.size() == 0);
    for(auto i=0; i<active_set.size(); i++)
      randidx.push_back(active_set[i]);

    active_set.erase(active_set.begin(), active_set.end());
    assert(active_set.size() == 0);
    //    LOG(INFO) << "     Iteration " << iter << " RandIDX Set Size " << randidx.size() << endl;
    vector<dpair> prev_vids;
    vector<long> current_vids;

    // 2. 
    int subiter=0;
    while(!randidx.empty()){

      current_vids.erase(current_vids.begin(), current_vids.end());
      for(int cnt=0; cnt<FLAGS_parallels; cnt++){
	long ii = randidx.front();
	randidx.erase(randidx.begin());
	active_set.push_back(ii);
	current_vids.push_back(ii);            
	if(randidx.empty() == true)
	  break;
      }

      stradsvm::bcwmsg msg;
      if(!prev_vids.empty()){
	for(int j=0; j<prev_vids.size(); j++){
	  stradsvm::pairt *oldpair = msg.add_oldpairs(); // current 
	  long oldvid = prev_vids[j].vid;	  
	  oldpair->set_vid(oldvid);
	  oldpair->set_value(m_alpha[oldvid]);	
	  oldpair->set_pg(prev_vids[j].PG);
	}
      }

      prev_vids.erase(prev_vids.begin(), prev_vids.end());

      for(int j=0; j<current_vids.size(); j++){
	stradsvm::pairt *newpair = msg.add_newpairs(); // current 
	long newvid = current_vids[j];
	newpair->set_vid(newvid);
	newpair->set_value(m_alpha[newvid]);	
	//	prev_vids.push_back(i); // keeping for previous 
      }
      msg.set_type(0);
      msg.set_src(ctx->rank);
      string *buffer = new string;

      msg.SerializeToString(buffer);
      for(int i=0; i<ctx->m_worker_machines; i++){
	ctx->send((char *)buffer->c_str(), buffer->size(), dst_worker, i); 
      }
      delete buffer;

      for(int k=0; k<MAX_PARALLEL;k++){
	G[k]=0.0;
	spdot[k]=0.0;
      }

      vector<long> vidarray;
      for(int i=0; i<ctx->m_worker_machines; i++){
	int length=-1;
	void *buf = ctx->sync_recv(src_worker, i, &length);
	assert(length > 0);
	string stringbuffer((char *)buf, length);
	stradsvm::bcwmsg msgs;      
	msgs.ParseFromString(stringbuffer);	
	for(int j=0; j<msgs.result_size(); j++){
	  long vid = msgs.result(j).vid();	 
	  G[j] += msgs.result(j).wspdot();
	  spdot[j] += msgs.result(j).spspdot();
	  if(i==0){
	    vidarray.push_back(vid);
	  }
	}
      }

      for(int j=0; j<vidarray.size(); j++){
	long vid = vidarray[j];	 
	G[j] = G[j] - 1 + Dii * m_alpha[vid];

	double PG = 0.0;
	double Gval = G[j];

	if( m_alpha[vid] == 0){
	  if(Gval > M_bar){
	    active_set.pop_back();
	  }
	  if(Gval < 0.0){
	    PG = Gval;
	  }	
	}else if( m_alpha[vid] == U){
	  if(Gval < m_bar){
	    active_set.pop_back();
	  }
	  if(Gval > 0.0){
	    PG = Gval;
	  }
	}else{
	  PG = Gval;
	}
	// (c)
	M = std::max(M, PG);
	m = std::min(m, PG);      
	// (d)
	if(fabs(PG) > 1e-12){
	  //if(fabs(PG) != 0.0){
	  m_alpha[vid] = std::min(std::max(m_alpha[vid] - Gval/spdot[j], 0.0), U); // vector       	
	}    
	dpair tmp;
	tmp.vid = vid;
	tmp.PG = PG;
	prev_vids.push_back(tmp);
      }

      subiter++;
    }// end of while(!randidx.empty()) 

    // 3. 
    if( (M - m) < FLAGS_epsilon){
      if(active_set.size() == m_l){
	break; // termination condition 
      }else{
	active_set.erase(active_set.begin(), active_set.end());
	active_set.reserve(m_l);
	for(auto j=0; j<m_l; j++){
	  active_set.push_back(j);
	}	  
	M_bar = std::numeric_limits<double>::max();
	m_bar = std::numeric_limits<double>::min();	  
      }
    }

    // 4. 
    if(M <= 0.0){
      M_bar = std::numeric_limits<double>::max(); 
    }else{
      M_bar = M;
    }

    // 5. 
    if(m >= 0){
      m_bar = std::numeric_limits<double>::min();	  
    }else{
      m_bar = m;
    }

    assert(randidx.size() == 0);

    LOG(INFO) << "*** Iteration ***   " << iter <<  endl;
    // calc objective value
    stradsvm::bcwmsg msg;
    msg.set_type(1);
    msg.set_src(ctx->rank);
    if(!prev_vids.empty()){
      for(int j=0; j<prev_vids.size(); j++){
	stradsvm::pairt *oldpair = msg.add_oldpairs(); // current 
	long oldvid = prev_vids[j].vid;	  
	oldpair->set_vid(oldvid);
	oldpair->set_value(m_alpha[oldvid]);	
	oldpair->set_pg(prev_vids[j].PG);
      }
    }
    prev_vids.erase(prev_vids.begin(), prev_vids.end());

    string *buffer = new string;
    msg.SerializeToString(buffer);
    for(int i=0; i<ctx->m_worker_machines; i++){
      ctx->send((char *)buffer->c_str(), buffer->size(), dst_worker, i); 
    }
    delete buffer;

    double wterm(.0);
    for(int k=0; k<m_l; k++)
      mwsp[k] = 0.0;

    for(int i=0; i<ctx->m_worker_machines; i++){
      int length=-1;
      void *buf = ctx->sync_recv(src_worker, i, &length);
      assert(length > 0);
      string stringbuffer((char *)buf, length);
      stradsvm::bcwmsg msgs;      
      msgs.ParseFromString(stringbuffer);	

      wterm += msgs.wterm();
      assert(msgs.mwsp_size() == m_l);

      for(int k=0; k<m_l; k++)
	mwsp[k] += msgs.mwsp(k);
    }

    double sum(.0), sum_alpha(.0);    
    for(int k=0; k<m_l; k++){
      sum += std::max(1-mwsp[k], 0.0); 
      sum_alpha += m_alpha[k];      
    }    
    sum = FLAGS_C*sum;
    double primal_obj=sum+wterm;
    double dual_obj=sum_alpha - wterm; 
    LOG(INFO) << "Primal obj:" << primal_obj << " Dual obj: " << dual_obj << " Gap: " << primal_obj - dual_obj << endl;  
    //    LOG(INFO) << " PGMax " << M << " PGmin " << m << " PG difference " << M - m << endl;
  } // for iter = 0 .... 

  LOG(INFO) << "Elapsed time: " << double(std::clock() - begin)/CLOCKS_PER_SEC;

  // send termination command 
  stradsvm::bcwmsg msg;
  msg.set_type(2);
  msg.set_src(ctx->rank);
  string *buffer = new string;
  msg.SerializeToString(buffer);
  for(int i=0; i<ctx->m_worker_machines; i++){
    ctx->send((char *)buffer->c_str(), buffer->size(), dst_worker, i); 
  }
  delete buffer;
} // coordinator 
