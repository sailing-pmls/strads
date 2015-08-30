#ifndef _LINEAR_SVM_HPP_
#define _LINEAR_SVM_HPP_

#include <vector>
#include <string>
#include <strads/include/common.hpp>

using std::vector;

class _pair{
public: _pair(long mid, double mval): id(mid), val(mval){}
  long id;
  double val;
};

class dualcd_svm{

public: 

  dualcd_svm(int workerid, int workers):m_workerid(workerid), m_workers(workers){};
  void train_coordinator(sharedctx *ctx);
  void train_worker(sharedctx *ctx);
  //  void train(void); // obsolete in distributed version 
  void verify(void);
  double primal_obj(void);
  double dual_obj(void);
  double get_Qij(long i, long j);
  void update_dualcoeff(long did, double Dii, double U);
  void dist_read_data(std::string &fn, long verifysample);
  void dist_read_res(std::string &fn, long verifysample);
  long get_m_l(void){ return m_l; } // sample count 
  long get_m_m(void){ return m_m; } // feat fount 

  void set_m_l(long samples){
    m_l = samples;
  }
  void set_m_m(long feats){
    m_m = feats;
  }


  std::vector<std::vector<_pair> *>m_x;
  std::vector<std::vector<_pair> *>m_xv;

  std::vector<double> m_y;
  std::vector<double> m_yv; // for verification

private: 
  void train_init(void);
  void grad_pi(double, double);

  long m_l; // trainsamples
  long m_m; // feat
  long m_l4v; // for verification

  std::vector<double> m_alpha;
  std::vector<double> m_w;
  int m_workerid;
  int m_workers;
};





double vector_dot(vector<double> &a, vector<double> &b);
double vector_dot(vector<double> &fulla, vector<_pair> &b);
double vector_dot(vector<_pair> &partiala, vector<_pair> &partialb);
double get_PG(double G, double alpha, double U);
void vector_const(vector<double> &fulldv, vector<_pair> &partialv, double c);




#endif 
