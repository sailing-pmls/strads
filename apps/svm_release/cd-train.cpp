#include "cd-train.hpp"
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <stdio.h>
#include <list>

#include "strads/util/utility.hpp"
#include "strads/ds/spmat.hpp"
#include "cd-util.hpp"


using namespace std;

/* 
 *  @brief  Read a partition of data from mmt file data partition is partitioned by sample, 
 *    but physically stored in column major sparse matrix foramt. 
 *    This is because of CD specific algorithmic requirement and update access parttern 
 *
 *  @param fn  The file name 
 *  @param matrix The empty sparse matrix class (column major) 
 *  @param machines The number of machines ( equal to the number of partitions) 
 *  @param mid The machine id ( partition id which is used to partition the data by samples.) 
 *  @return 0:success -1 failure
 */
int cd_train::read_partition(const std::string &fn, col_vspmat &matrix, long machines, long mid){

  LOG(INFO) << "  Input file name : " << fn << endl;
  LOG(INFO) << "  matrix.row_size () : " << matrix.row_size() << "matrix.col_size(): " << matrix.col_size() << endl;   

  matrix.set_range(false, 0, matrix.col_size()-1); // column matrix's column range (start - end) Inclusive 
  const cont_range row_range(0, matrix.row_size()-1, machines, mid); // use contiguous partitioning i.e. P0: 0 - 99 P1:100 - 199 P3: 200 - 299 
  const cont_range col_range(0, matrix.col_size()-1, 1, 0);          // 1 : one chunk : so all ranges. 0 : dummy 

  mmt_partial_read<col_vspmat>(matrix, fn, mid, row_range, col_range); 
  // read data chunk from mmt file correspond to row_range and col range  
  LOG(INFO) << "  [worker mid : " << mid << " ] read nz entries :  " << matrix.allocatedentry() << endl; 
  return 0;
}


int cd_train::read_col_partition(const std::string &fn, col_vspmat &matrix, long machines, long mid){

  LOG(INFO) << "  Input file name : " << fn << endl;
  LOG(INFO) << "  matrix.row_size () : " << matrix.row_size() << "matrix.col_size(): " << matrix.col_size() << endl;   

  matrix.set_range(false, 0, matrix.col_size()-1); // column matrix's column range (start - end) Inclusive 
  const cont_range row_range(0, matrix.row_size()-1, 1,0);         // 1 : one chunk : so all ranges. 0 : dummy 
  const cont_range col_range(0, matrix.col_size()-1, machines, mid); // use contiguous partitioning i.e. P0: 0 - 99 P1:100 - 199 P3: 200 - 299 

  mmt_partial_read<col_vspmat>(matrix, fn, mid, row_range, col_range); 
  // read data chunk from mmt file correspond to row_range and col range  
  LOG(INFO) << "  [scheduler mid : " << mid << " ] read nz entries :  " << matrix.allocatedentry() << endl; 
  return 0;
}


int cd_train::read_partition(const std::string &fn, cas_array<double> &residual, long samples, long columns, long machines, long mid){
  LOG(INFO) << "  Input file name Y : " << fn << endl;
  const cont_range row_range(0, samples-1, machines, mid); // use contiguous partitioning i.e. P0: 0 - 99 P1:100 - 199 P3: 200 - 299 
  const cont_range col_range(0, columns-1, 1, 0);          // 1 : one chunk : so all ranges. 0 : dummy 
  mmt_partial_read_vector(residual, fn, mid, row_range, col_range); 
  return 0;
}

void cd_train::put_entry_inq(void *cmd){
  std::unique_lock<std::mutex>lk(m_inqlock);
  if(m_inq.empty()) 
    m_cv.notify_one();
  m_inq.push_back(cmd);
}

void *cd_train::get_entry_inq_blocking(void){
  std::unique_lock<std::mutex>lk(m_inqlock);
  void *ret = NULL;
  if(!m_inq.empty()){
    ret = m_inq.front();
    m_inq.pop_front();
  }else{
    m_cv.wait(lk);
    ret = m_inq.front();
    m_inq.pop_front();
  }
  return ret;
}

// caveat: precondition: cmd should be allocated structued eligible for free().                                                                   
void cd_train::put_entry_outq(void *cmd){

  std::lock_guard<std::mutex>lk(m_outqlock);
  m_outq.push_back(cmd);
}

// caveat: if nz returned to a caller, the caller should free nz structure                                                                        
void *cd_train::get_entry_outq(void){
  std::lock_guard<std::mutex>lk(m_outqlock);
  void *ret = NULL;
  if(!m_outq.empty()){
    ret = m_outq.front();
    m_outq.pop_front();
  }
  return ret;
}
