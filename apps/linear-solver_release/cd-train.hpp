
#ifndef _CDTRAIN_HPP_
#define _CDTRAIN_HPP_

#include "lassoll.hpp"
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

#include "strads/ds/spmat.hpp"  // include strads sparse matrix
#include "strads/include/cas_array.hpp"

#include "strads/include/common.hpp"
#include "strads/include/indepds.hpp"
#include "strads/netdriver/comm.hpp"

#include "cd-util.hpp"

// trainer for coordinate descent family algorithms  
// workers create one instance for each machine
class cd_train {

public:

  cd_train() = delete;
  cd_train(double lambda, long sample, long feature, long mid, long machines, long threads, col_vspmat &matrix, cas_array<double> &atom, thread_barrier &barrier)
    : m_mat(matrix), residual(atom), 
      m_row_range(0, sample-1, machines, mid), 
      m_col_range(0, feature-1, 1, 0), 
      m_barrier(barrier),
      m_lambda(lambda), m_sample(sample), m_feat(feature), m_mid(mid), m_machines(machines), m_threads(threads){} 

  virtual void *update_feat(col_vspmat &cm_vmap, cas_array<double>&res, void *userdata, void *ctx, double *Y) = 0; // worker-thread  
  virtual void *update_res(col_vspmat &cm_vmat, cas_array<double> &residual, void *userdata, void *ctx) = 0; // worker-thread

  virtual double get_object_server(double *beta, double psum, double lambda, long modelsize, long samples)=0; // coordinator-thread
  virtual void *aggregator(double *beta, double *betadiff, std::unordered_map<int64_t, idmvals_pair *> &retmap, sharedctx *ctx, 
			   double lambda, long modelsize, long samples)=0;

	virtual long get_threadid(void)=0;
	static int read_partition(const std::string &fn, col_vspmat &matrix, long machines, long mid);
	static int read_partition_ring(sharedctx *ctx, const std::string &fn, col_vspmat &matrix, long machines, long mid);   
	// partition by row, but store them in column-major physical format

  static int read_col_partition(const std::string &fn, col_vspmat &matrix, long machines, long mid);
	static int read_col_partition_ring(sharedctx *ctx, const std::string &fn, col_vspmat &matrix, long machines, long mid);   
  // partition by col and store them in column-major physical format

  static int read_partition(const std::string &fn, cas_array<double> &residual, long samples, long columns, long machines, long mid);
	static int read_partition_ring(sharedctx *ctx, const std::string &fn, cas_array<double> &residual, long samples, long columns, long machines, long mid);

  void put_entry_inq(void *cmd);
  void *get_entry_inq_blocking();
  void put_entry_outq(void *cmd);
  void *get_entry_outq(void);

  long get_mid(void) { return m_mid; }
  ~cd_train(){}
  col_vspmat &m_mat; // column-major matrix  // when algorithm specific code is migrated to class, move m_mat to protected mode and change the order of m_mat initialization on the constructor. 
  cas_array<double> &residual;
  const cont_range m_row_range; // coarse grained row range - machine level range 
  const cont_range m_col_range; // coarse grained col range - machine level range
  thread_barrier &m_barrier;

protected:
  const double m_lambda;
  const long m_sample;
  const long m_feat;
  const long m_mid;
  const long m_machines;
  const long m_threads;

  //  col_vspmat &m_mat; // column-major matrix 

  inter_threadq m_inq; // for thread communication 
  inter_threadq m_outq;

  std::condition_variable m_cv; // associated with inqlock 
  std::mutex m_inqlock;
  std::mutex m_outqlock;
  int m_r;
};

// create lasso instance per thread 
class lasso : public cd_train { //

public:

  lasso(double lambda, long sample, long feature, long mid, long machines, long threadid, long threads, col_vspmat &matrix, cas_array<double> &atom, thread_barrier &barrier)
    :cd_train(lambda, sample, feature, mid, machines, threads, matrix, atom, barrier), 
     m_thread_row_range(m_row_range.get_min(), m_row_range.get_max(), machines, threadid),
     m_thread_col_range(m_col_range.get_min(), m_col_range.get_max(), 1, 0),  
     m_threadid(threadid){}

  ~lasso(){}

  //  static int read_partition(const std::string &fn, col_vspmat &matrix, long machines, long mid);
  static double object_calc(const cont_range &row_range, cas_array<double> &atom);

  //  int train_feat(const long feat);
  //  int update_res(const long feat);
  void *update_feat(col_vspmat &cm_vmap, cas_array<double>&res, void *userd, void *ctx, double *Y); // worker-thread  
  void *update_res(col_vspmat &cm_vmat, cas_array<double> &residual, void *userdata, void *ctx); // worker-thread
  double get_object_server(double *beta, double psum, double lambda, long modelsize, long samples); // coordinator-thread
  void * aggregator(double *beta, double *betadiff, std::unordered_map<int64_t, idmvals_pair *> &retmap, sharedctx *ctx, 
		    double lambda, long modelsize, long samples); // coordinator-thread

  long get_threadid(void){ return m_threadid; }
  void print_info(void){ 
    std::cout << "printinfo : " << m_mat.row_size(); 
  }
  const cont_range m_thread_row_range;
  const cont_range m_thread_col_range;

private:

  double soft_threshold(double sum, double lambda);
  double _partialsum_update(col_vspmat &cm_vmat, cas_array<double> &residual, int64_t vid, double vidbeta, double *xsqmsum, int64_t sample_s, int64_t sample_e);
  void _update_residual_by_sample(col_vspmat &cm_vmat, cas_array<double> &residual, double delta, int64_t vid, int64_t row_s, int64_t row_e);
  const long m_threadid;
};


// create logistic instance per thread 
class logistic: public cd_train { //

public:

  logistic(double lambda, long sample, long feature, long mid, long machines, long threadid, long threads, col_vspmat &matrix, cas_array<double> &atom, thread_barrier &barrier)
    :cd_train(lambda, sample, feature, mid, machines, threads, matrix, atom, barrier), 
     m_thread_row_range(m_row_range.get_min(), m_row_range.get_max(), machines, threadid),
     m_thread_col_range(m_col_range.get_min(), m_col_range.get_max(), 1, 0),  
     m_threadid(threadid){}

  ~logistic(){}

  //  static int read_partition(const std::string &fn, col_vspmat &matrix, long machines, long mid);
  static double object_calc(const cont_range &row_range, cas_array<double> &atom, double *Y);

  //  int train_feat(const long feat);
  //  int update_res(const long feat);
  void *update_feat(col_vspmat &cm_vmap, cas_array<double>&res, void *userd, void *ctx, double *Y); // worker-thread  
  void *update_res(col_vspmat &cm_vmat, cas_array<double> &residual, void *userdata, void *ctx); // worker-thread
  double get_object_server(double *beta, double psum, double lambda, long modelsize, long samples); // coordinator-thread
  void * aggregator(double *beta, double *betadiff, std::unordered_map<int64_t, idmvals_pair *> &retmap, sharedctx *ctx, 
		    double lambda, long modelsize, long samples); // coordinator-thread

  long get_threadid(void){ return m_threadid; }
  void print_info(void){ 
    std::cout << "printinfo : " << m_mat.row_size(); 
  }
  const cont_range m_thread_row_range;
  const cont_range m_thread_col_range;

private:

  double soft_threshold(double sum, double lambda);
  double _partialsum_update(col_vspmat &cm_vmat, cas_array<double> &residual, int64_t vid, double *Y);
  void _update_residual_by_sample(col_vspmat &cm_vmat, cas_array<double> &residual, double delta, int64_t vid, int64_t row_s, int64_t row_e);
  const long m_threadid;
};

#endif 
