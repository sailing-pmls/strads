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

void dualcd_svm::train_worker(sharedctx *ctx){

  train_init(); // reserve m_alpha and m_w and initialize them to all zeros  

  vector <long> cvids;
  while(1){

    int len=-1;
    void *recv = ctx->async_recv(&len);

    if(recv != NULL){
      string buffer((char*)recv, len);
      stradsvm::bcwmsg msg;
      msg.ParseFromString(buffer);

      if(msg.type() == 0){

	// process previous pairs 
	for(int i=0; i<msg.oldpairs_size(); i++){
	  long vid = msg.oldpairs(i).vid();
	  double value = msg.oldpairs(i).value();
	  double PG = msg.oldpairs(i).pg();
	  double alpha_bar = m_alpha[vid];
	  m_alpha[vid] = value; // in server side: std::min(std::max(m_alpha[i] - G/vector_dot(sp, sp), 0.0), U); // vector
	  vector<_pair> &sp = (*m_x[vid]); // x_i

	  if(PG != 0.0){
	    for(auto k=0; k<sp.size(); k++){
	      long id = sp[k].id;
	      m_w[id] = m_w[id] + sp[k].val*((m_alpha[vid]-alpha_bar)*m_y[vid]);
	    }
	  }
	}
	stradsvm::bcwmsg sendmsg;
	sendmsg.set_type(0);
	sendmsg.set_src(ctx->rank);
	// process current pairs 
	for(int i=0; i<msg.newpairs_size(); i++){
	  long vid = msg.newpairs(i).vid();       

	  double value = msg.newpairs(i).value();
	  m_alpha[vid] = value;
	  vector<_pair> &sp = (*m_x[vid]); // x_i
	  //	double G = m_y[vid]*vector_dot(m_w, sp) - 1.0 + Dii*m_alpha[vid]; 
	  // make partial G without -1 + Dii*m_alpha[vid]
	  double p_G = m_y[vid]*vector_dot(m_w, sp) ; // -1.0 + Dii*m_alpha[vid] after aggregation in server side 
	  //	strads_msg(OUT, " rank(%d) vid(%ld) m_y:%lf , m_w_dot_sp:%lf \n", 
	  //		   ctx->rank, vid, m_y[vid], vector_dot(m_w, sp));       
	  // make partial sp dot sp 
	  double p_spspdot = vector_dot(sp, sp);	
	  stradsvm::triplepair *entry = sendmsg.add_result();
	  entry->set_vid(vid);
	  entry->set_wspdot(p_G);
	  entry->set_spspdot(p_spspdot);
	}
	string *sbuffer = new string;
	sendmsg.SerializeToString(sbuffer);
	ctx->send((char *)sbuffer->c_str(), sbuffer->size());
	delete sbuffer;
      }else if(msg.type() == 1){ // if object calc routine  

	// process previous pairs 
	for(int i=0; i<msg.oldpairs_size(); i++){
	  long vid = msg.oldpairs(i).vid();
	  double value = msg.oldpairs(i).value();
	  double PG = msg.oldpairs(i).pg();
	  double alpha_bar = m_alpha[vid];
	  m_alpha[vid] = value; // in server side: std::min(std::max(m_alpha[i] - G/vector_dot(sp, sp), 0.0), U); // vector
	  vector<_pair> &sp = (*m_x[vid]); // x_i

	  if(fabs(PG) > 1e-12){                                                       
	    //if(PG != 0.0){
	    for(auto k=0; k<sp.size(); k++){
	      long id = sp[k].id;
	      m_w[id] = m_w[id] + sp[k].val*((m_alpha[vid]-alpha_bar)*m_y[vid]);
	    }
	  }
	}

	stradsvm::bcwmsg sendmsg;
	sendmsg.set_src(ctx->rank);
	sendmsg.set_type(1);
	sendmsg.set_wterm(vector_dot(m_w, m_w)*0.5);
	for(auto k=0; k<m_l; k++){
	  vector<_pair> &sp = (*m_x[k]);
	  sendmsg.add_mwsp(m_y[k]*vector_dot(m_w, sp));
	}

	string *sbuffer = new string;
	sendmsg.SerializeToString(sbuffer);
	ctx->send((char *)sbuffer->c_str(), sbuffer->size());
	delete sbuffer;
     
      }else if(msg.type() == 2){ // if object calc routine  
	// write log file with machine rank with input file prefix       
	char *fn = (char *)calloc(sizeof(char), 1000);
	sprintf(fn, "./output/%sout.mach-%d", FLAGS_svmout.c_str(), m_workerid); 
	FILE *fp = fopen(fn, "wt");
	assert(fp);
	for(long k=0; k<m_m; k++){
	  if(k % m_workers == m_workerid and m_w[k] != 0){
	    fprintf(fp, "%ld %lf \n", k+1, m_w[k]); // index start from 1 ,, not 0 
	  }
	}
	fclose(fp);
	string outdir(fn);
	LOG(ERROR) << "Worker "<< m_workerid << " store weight partition file " << outdir << endl;
	break;
      }else{
	assert(0); // undefined command 
      }   
    } // if not NULL
  }// for while(1)
}
