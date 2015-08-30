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


double vector_dot(vector<double> &a, vector<double> &b){
  assert(a.size() == b.size());
  double sum(0);
  for(auto i=0; i<a.size(); i++)
    sum += a[i]*b[i];
  return sum;
} 

double vector_dot(vector<double> &fulla, vector<_pair> &b){
  assert(fulla.size() >= b.size());
  double sum(0);
  for(auto i=0; i<b.size(); i++){
    sum += fulla[b[i].id]*b[i].val;
  }
  return sum;
} 

double vector_dot(vector<_pair> &partiala, vector<_pair> &partialb){
  assert(partiala.size() == partialb.size());
  double sum(0);
  for(auto i=0; i<partiala.size(); i++)
    sum += partiala[i].val*partialb[i].val;
  return sum;
} 

double get_PG(double G, double alpha, double U){
  double ret(.0);
  if(alpha == 0){
    ret = std::min(G, 0.0);
  }else if(alpha == U){
    ret = std::max(G, 0.0);
  }else if(alpha > 0 and alpha < U){
    ret = G;
  }else{
    LOG(ERROR) << "Abnormal : alpha value : "<< alpha << endl;  
  }
  return ret;
}

void vector_const(vector<double> &fulldv, vector<_pair> &partialv, double c){
  assert(fulldv.size() >= partialv.size());
  for(auto i=0; i<partialv.size(); i++){
    long id = partialv[i].id;
    fulldv[id] = partialv[i].val;
  }
}



void dualcd_svm::dist_read_data(string &fn, long verifysample){
  string linebuf;
  ifstream ifs(fn, ifstream::in);
  long linecnt(0);
  long maxfeat=-1;
  
  if(ifs.is_open()){
    while(getline(ifs, linebuf)){
      istringstream istring(linebuf);          
      bool first(true);
      string token;
      vector<_pair> &vpair = *new vector<_pair>;
      while(getline(istring, token, ' ')){	       
	if(token.size() == 0) continue; // skip extra space char 
	// read response 
	if(first){	 
	  m_y.push_back(stod(token));
	  first=false;
	}else{
	  istringstream istring_inner(token);	  
	  string idx, val;
	  getline(istring_inner, idx, ':');
	  getline(istring_inner, val, ':');	         

	  long lidx = stol(idx)-1; // lidx : c array aligned starting from 0 
	  if(lidx % m_workers == m_workerid){ 	 
	    vpair.emplace_back(stol(idx)-1, stod(val)); 
	    // stol(idx)-1: to be aligned with c array (libsvm array start from 1)
	  }
	  maxfeat = std::max(stol(idx), maxfeat);
	  // even if no nz entry belining to a worker, maxfeat count is valid 
	}
      }
      // even if there is no nz entry belonging to a worker, the worker should keep sample entry for it. 
      // it's not much harmful. 
      m_x.push_back(&vpair);
      linecnt++;
    } 
  }else{
    cout<< "File open failed for " << fn << endl;
    assert(0);
  }

  LOG(ERROR) << "dist-read total samples from the file :" << m_x.size() <<  "worker mid: " << m_workerid << endl; 

  for(auto it=m_x.rbegin(); it != (m_x.rbegin()+verifysample) and it != m_x.rend(); it++){
    m_xv.push_back(*it);
  }
  m_x.erase(m_x.end()-m_xv.size(), m_x.end());       

  for(auto it=m_y.rbegin(); it != (m_y.rbegin()+verifysample) and it != m_y.rend(); it++){
    m_yv.push_back(*it);
  }
  m_y.erase(m_y.end()-m_yv.size(), m_y.end());       

  assert(m_y.size() == m_x.size());
  assert(m_yv.size() == m_xv.size());
  LOG(ERROR) << "dist-read For training, samples : " << m_x.size() << endl;
  LOG(ERROR) << "dist-read For verification, samples : " << m_xv.size() << endl; 

  m_l = m_x.size();
  m_m = maxfeat; // libsvm array start from 1 
  m_l4v = m_xv.size();

  // for verification: count all nz entries in a local partition 
  //
  long nzcnt=0; 
  for(int sid=0; sid<m_x.size(); sid++){
    vector<_pair> &sp = (*m_x[sid]);
    nzcnt += sp.size();
  }
  LOG(ERROR) << "@@@@@ dist-read total nzentrie :  :" << nzcnt <<  "  at worker mid: " << m_workerid << endl;
}


void dualcd_svm::dist_read_res(string &fn, long verifysample){

  string linebuf;
  ifstream ifs(fn, ifstream::in);
  long linecnt(0);
  
  if(ifs.is_open()){
    while(getline(ifs, linebuf)){
      istringstream istring(linebuf);          
      bool first(true);
      string token;
      while(getline(istring, token, ' ')){	       
	if(token.size() == 0) continue; // skip extra space char 
	// read response 
	if(first){	 
	  m_y.push_back(stod(token));
	  first=false;
	}
      }
      linecnt++;
    } 
  }else{
    cout<< "File open failed for " << fn << endl;
    assert(0);
  }

  LOG(ERROR) << "dist-read-response total samples-res from the file :" << m_y.size() <<  "worker mid: " << m_workerid << endl; 

  for(auto it=m_y.rbegin(); it != (m_y.rbegin()+verifysample) and it != m_y.rend(); it++){
    m_yv.push_back(*it);
  }
  m_y.erase(m_y.end()-m_yv.size(), m_y.end());       

  //  assert(m_y.size() == m_x.size());
  //  assert(m_yv.size() == m_xv.size());
  LOG(ERROR) << "dist-read-res For training, samples : " << m_y.size() << endl;
  LOG(ERROR) << "dist-read-res For verification, samples : " << m_yv.size() << endl; 

  m_l = m_y.size();
  m_l4v = m_yv.size();

  // collect maxfeat from all workers and verify if all workers see the same max feat 
  // and fill out it 
  //m_m = maxfeat; // libsvm array start from 1 
}



void dualcd_svm::train_init(void){
  // init alpha and w according to the algorithm 1 
  m_alpha.reserve(m_l);
  for(auto i=0; i<m_l; i++)
    m_alpha.push_back(0);
  
  m_w.reserve(m_m);
  for(auto i=0; i<m_m; i++)
    m_w.push_back(0);
}

double dualcd_svm::primal_obj(void){
  double wterm = vector_dot(m_w, m_w)*0.5;
  double sum(.0), sum_alpha(.0);
  for(auto i=0; i<m_l; i++){
    vector<_pair> &sp = (*m_x[i]);
    sum += std::max(1-m_y[i]*vector_dot(m_w, sp), 0.0);  
    sum_alpha += m_alpha[i];
  }
  sum = FLAGS_C*sum;
  double primal_obj=sum+wterm;
  double dual_obj=sum_alpha - wterm; 
  LOG(INFO) << "Primal obj:" << primal_obj << " Dual obj: " << dual_obj << " Gap: " << primal_obj - dual_obj << endl;  

#if 0
  // verification only. In ref, calc dual obj by using equation 4
  // result equals sum_alpha-wterm from the above. 
  // Question : How ?? 
  vector<double> tmp_alpha(m_l, .0);
  for(auto i=0; i<m_l; i++){ 
    double sum=0;
    for(auto j=0; j<m_l; j++){
      sum += m_alpha[j]*get_Qij(j,i);
    }
    tmp_alpha[i] = sum;
  }
  double full_dual = 0;
  for(auto i=0; i<m_l; i++){ 
    full_dual += (0.5*tmp_alpha[i]*m_alpha[i] - m_alpha[i]);
  }
  LOG(INFO) << "FULL DUAL OBJECTIVE VALUE : " << full_dual << endl;
#endif 
  return (primal_obj);
}





double dualcd_svm::get_Qij(long i, long j){
  vector<_pair> &xi = (*m_x[i]);
  vector<_pair> &xj = (*m_x[j]);  
  vector<double> tmp(m_m, 0.0);
  for(auto i=0; i<xi.size(); i++){
    long id = xi[i].id;
    tmp[id] = xi[i].val;
  }
  double sum(.0);
  for(auto i=0; i<xj.size(); i++){
    long id = xj[i].id;
    sum += tmp[id]*xj[i].val;
  }
  return sum*(m_y[i]*m_y[j]);
}

void dualcd_svm::update_dualcoeff(long i, double Dii, double U){ // id of duall coeff

  vector<_pair> &sp = (*m_x[i]);
  double G = m_y[i]*vector_dot(m_w, sp) - 1 + Dii*m_alpha[i];      
  double PG = get_PG(G, m_alpha[i], U);      

  if(fabs(PG) != 0){
    double alpha_bar = m_alpha[i];
    m_alpha[i] = std::min(std::max(m_alpha[i] - G/vector_dot(sp, sp), 0.0), U); // vector       	
    for(auto k=0; k<sp.size(); k++){
      long id = sp[k].id;
      m_w[id] = m_w[id] + sp[k].val*((m_alpha[i]-alpha_bar)*m_y[i]);
    }
  }    
}





