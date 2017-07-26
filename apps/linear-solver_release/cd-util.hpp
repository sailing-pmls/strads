#ifndef _CDUTIL_HPP_
#define _CDUTIL_HPP_

#include <string>
#include <vector>
#include <set>
#include <list>
#include <stdexcept>
#include <iostream>
#include <typeinfo>
#include <mutex>
#include <condition_variable>

#include <chrono>
#include <random>


#include <glog/logging.h>
#include <assert.h>

#include "strads/util/utility.hpp"
#include "strads/ds/spmat.hpp"
#include "strads/ds/mm_io.hpp"
#include "strads/include/cas_array.hpp"
#include "strads/include/indepds.hpp"

#include "strads/include/ringio.hpp"
#include "./lassoll.hpp"

#define CLOCK (std::chrono::system_clock::now().time_since_epoch().count())
static std::mt19937 _rng(CLOCK);
static std::uniform_real_distribution<double> _unif01;
#define STATCLOCK (0x3751)
static std::mt19937 _statrng(STATCLOCK);




class cdtask_assignment{
public:
  std::map<int, range *>schmach_tmap;
  std::map<int, range *>schthrd_tmap;
};



// contiguous type only 
class cont_range{
public:
  cont_range(long start, long end):m_min(start), m_max(end) {} 
  // minval: start  maxval:end 
  cont_range(long min_value, long maxval, long parts, long id);

  // get start and end of id-th partition 
  long get_min(void)const;
  long get_max(void)const;

  // minval: start  maxval:end 
  virtual int make_part(long minval, long maxval, long parts, long id);
  
  // minval, maxval are inclusive 
  // parts : the number of partitions  
private:
  long m_min;
  long m_max;
};

bool negative_check(int64_t value);
bool negative_check(int32_t value);

long mmt_partial_read_vector(cas_array<double> &residual , const std::string &fn, long mid, const cont_range &row_range, const cont_range &col_range);

long mmt_partial_read_vector_ring(sharedctx *ctx, cas_array<double> &residual , const std::string &fn, long mid, const cont_range &row_range, const cont_range &col_range);


class thread_barrier{
public:
  explicit thread_barrier(long count): m_count(count){}
  void wait(void);
private:
  std::mutex m_mutex;
  std::condition_variable m_cv;
  long m_count;
};


void make_scheduling_taskpartition(cdtask_assignment &m_tmap, int64_t modelsize, int schedmach, int thrd_permach);




template <typename T1>
long mmt_partial_read(T1 &mat, const std::string &fn, long mid, const cont_range &row_range, const cont_range &col_range){

  //  std::cout << std::endl << " ****** mmt_partial_read : row size : " << mat.row_size() << " col size : " << mat.col_size() << std::endl;
  LOG(INFO) << "[mmt_partial_read] target matrix type : " << typeid(mat).name() << std::endl;

  long ret_code;
  MM_typecode matcode;
  FILE *f;
  long unsigned int maxrow, maxcol, nzcnt;   
  long unsigned int tmprow, tmpcol;
  double tmpval;

  f = fopen(fn.c_str(), "r");
  assert(f);

  if(mm_read_banner(f, &matcode) != 0){
    strads_msg(OUT, "mmio fatal error. Could not process Matrix Market banner.\n");
    //    std::terminate();
    assert(0);
  }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */
  if(mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode)){
    strads_msg(ERR, "mmio fatal error. this application does not support ");
    strads_msg(ERR, "\tMarket Market type: [%s]\n", mm_typecode_to_str(matcode));
    assert(0);
  }

  /* find out size of sparse matrix .... */
  if((ret_code = mm_read_mtx_crd_size(f, &maxrow, &maxcol, &nzcnt)) !=0){
    strads_msg(ERR, "mmio fatal error. Sorry, this application does not support");
    //    terminate();
    assert(0);
  }

  //  long min=row_range.get_min();

  strads_msg(OUT, "** FILE %ld nzcnt to read, mmt(row,col)(%ld, %ld): mat-portion for mid(%ld) range: row(%ld - %ld) col(%ld - %ld) \n", 
	     nzcnt, maxrow, maxcol, mid, row_range.get_min(), row_range.get_max(), col_range.get_min(), col_range.get_max());

  char *chbuffer = (char *)calloc(1024*1024*2, sizeof(char));
  uint64_t nzprogress=0;

  while(fgets(chbuffer, 1024*1024, f) != NULL){
    sscanf(chbuffer, "%lu %lu %lf", &tmprow, &tmpcol, &tmpval);   
    if(0){
      strads_msg(ERR, "buffer line (%s)\n", chbuffer);
      strads_msg(ERR, " tmprow(%lu)  tmpcol(%lu) tmpval(%lf)\n", 
		 tmprow, tmpcol, tmpval);
    }

    // mmt format: origin coordinate is at (1,1)
    assert(tmprow >= 1);
    assert(tmpcol >= 1);

    tmprow--;  /* adjust from 1-based to 0-based */
    tmpcol--;    

    if(!(tmprow < maxrow)){
      strads_msg(OUT, "\n tmprow (%lu) maxrow(%lu)\n", 
		 tmprow, maxrow);
      assert(0);
    }
    if(!(tmpcol < maxcol)){
      strads_msg(ERR, "\n tmpcol (%lu) maxcol(%lu)\n", 
		 tmpcol, maxcol);
      assert(0);
    }

    if(   (tmprow >= (long unsigned int)row_range.get_min()) 
       && (tmprow <= (long unsigned int)row_range.get_max())
       && (tmpcol >= (long unsigned int)col_range.get_min())
       && (tmpcol <= (long unsigned int)col_range.get_max())) {

      //      mat(tmprow, tmpcol)=tmpval;     
      mat.add(tmprow, tmpcol, tmpval);     

      nzprogress++;
      if(nzprogress %10000 == 0){
	strads_msg(INF, " %ld th nzelement (%s) at mid (%ld)", nzprogress, chbuffer, mid);
      }
    }
  }

  fclose(f);
  return 0;
}




template <typename T1>
long mmt_partial_read_ring(sharedctx *ctx, T1 &mat, const std::string &fn, long mid, const cont_range &row_range, const cont_range &col_range){

  //  std::cout << std::endl << " ****** mmt_partial_read : row size : " << mat.row_size() << " col size : " << mat.col_size() << std::endl;
  LOG(INFO) << "[mmt_partial_read] target matrix type : " << typeid(mat).name() << std::endl;

  long ret_code;
  MM_typecode matcode;
  //  FILE *f;
  long unsigned int maxrow, maxcol, nzcnt;   
  long unsigned int tmprow, tmpcol;
  double tmpval;

#if 0 
  //  f = fopen(fn.c_str(), "r");
  //  assert(f);  
  if(mm_read_banner(f, &matcode) != 0){
    strads_msg(OUT, "mmio fatal error. Could not process Matrix Market banner.\n");
    //    std::terminate();
    assert(0);
  }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */
  if(mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode)){
    strads_msg(ERR, "mmio fatal error. this application does not support ");
    strads_msg(ERR, "\tMarket Market type: [%s]\n", mm_typecode_to_str(matcode));
    assert(0);
  }
  
  /* find out size of sparse matrix .... */
  if((ret_code = mm_read_mtx_crd_size(f, &maxrow, &maxcol, &nzcnt)) !=0){
    strads_msg(ERR, "mmio fatal error. Sorry, this application does not support");
    //    terminate();
    assert(0);
  }
  //  long min=row_range.get_min();
  #endif 

  nzcnt = FLAGS_nzcount;
  maxrow = FLAGS_samples;
  maxcol = FLAGS_columns;

  
  strads_msg(OUT, "** FILE %ld nzcnt to read, mmt(row,col)(%ld, %ld): mat-portion for mid(%ld) range: row(%ld - %ld) col(%ld - %ld) \n", 
	     nzcnt, maxrow, maxcol, mid, row_range.get_min(), row_range.get_max(), col_range.get_min(), col_range.get_max());

  char *chbuffer = (char *)calloc(1024*1024*2, sizeof(char));
  uint64_t nzprogress=0;

  ringio ringfile(ctx, fn);

  //  while(1);
  

  size_t seqno(0);
  //while(fgets(chbuffer, 1024*1024, f) != NULL){
  while(ringfile.getline(chbuffer, 1024*1024) != NULL){
    sscanf(chbuffer, "%lu %lu %lf", &tmprow, &tmpcol, &tmpval);   

    if(0){
      strads_msg(ERR, "buffer line (%s)\n", chbuffer);
      strads_msg(ERR, " tmprow(%lu)  tmpcol(%lu) tmpval(%lf)\n", 
		 tmprow, tmpcol, tmpval);
    }


    seqno++;
    // mmt format: origin coordinate is at (1,1)
    assert(tmprow >= 1);
    assert(tmpcol >= 1);

    tmprow--;  /* adjust from 1-based to 0-based */
    tmpcol--;    

    if(!(tmprow < maxrow)){
      strads_msg(OUT, "\n tmprow (%lu) maxrow(%lu)  seqno(%lu)\n", 
		 tmprow, maxrow, seqno);
      assert(0);
    }
    if(!(tmpcol < maxcol)){
      strads_msg(ERR, "\n tmpcol (%lu) maxcol(%lu)\n", 
		 tmpcol, maxcol);
      assert(0);
    }

    if(   (tmprow >= (long unsigned int)row_range.get_min()) 
       && (tmprow <= (long unsigned int)row_range.get_max())
       && (tmpcol >= (long unsigned int)col_range.get_min())
       && (tmpcol <= (long unsigned int)col_range.get_max())) {

      //      mat(tmprow, tmpcol)=tmpval;     
      mat.add(tmprow, tmpcol, tmpval);     

      nzprogress++;
      if(nzprogress %10000 == 0){
	strads_msg(INF, " %ld th nzelement (%s) at mid (%ld)", nzprogress, chbuffer, mid);
      }
    }
  }

  //  fclose(f);
  return 0;
}




template <typename T1>
long mmt_partial_read_ring_scheduler(sharedctx *ctx, T1 &mat, const std::string &fn, long mid, const cont_range &row_range, const cont_range &col_range){

  //  std::cout << std::endl << " ****** mmt_partial_read : row size : " << mat.row_size() << " col size : " << mat.col_size() << std::endl;
  LOG(INFO) << "[mmt_partial_read] target matrix type : " << typeid(mat).name() << std::endl;

  long ret_code;
  MM_typecode matcode;
  long unsigned int maxrow, maxcol, nzcnt;   
  long unsigned int tmprow, tmpcol;
  double tmpval;


#if 0
  //  FILE *f;
  f = fopen(fn.c_str(), "r");
  assert(f);
  if(mm_read_banner(f, &matcode) != 0){
    strads_msg(OUT, "mmio fatal error. Could not process Matrix Market banner.\n");
    //    std::terminate();
    assert(0);
  }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */
  if(mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode)){
    strads_msg(ERR, "mmio fatal error. this application does not support ");
    strads_msg(ERR, "\tMarket Market type: [%s]\n", mm_typecode_to_str(matcode));
    assert(0);
  }
  
  /* find out size of sparse matrix .... */
  if((ret_code = mm_read_mtx_crd_size(f, &maxrow, &maxcol, &nzcnt)) !=0){
    strads_msg(ERR, "mmio fatal error. Sorry, this application does not support");
    //    terminate();
    assert(0);
  }
  //  long min=row_range.get_min();
  #endif 

  nzcnt = FLAGS_nzcount;
  maxrow = FLAGS_samples;
  maxcol = FLAGS_columns;

  
  strads_msg(OUT, "** FILE %ld nzcnt to read, mmt(row,col)(%ld, %ld): mat-portion for mid(%ld) range: row(%ld - %ld) col(%ld - %ld) \n", 
	     nzcnt, maxrow, maxcol, mid, row_range.get_min(), row_range.get_max(), col_range.get_min(), col_range.get_max());

  char *chbuffer = (char *)calloc(1024*1024*2, sizeof(char));
  uint64_t nzprogress=0;

  ringio4scheduler ringfile(ctx, fn);

  //  while(1);
  

  size_t seqno(0);
  //while(fgets(chbuffer, 1024*1024, f) != NULL){
  while(ringfile.getline(chbuffer, 1024*1024) != NULL){
    sscanf(chbuffer, "%lu %lu %lf", &tmprow, &tmpcol, &tmpval);   

    if(0){
      strads_msg(ERR, "buffer line (%s)\n", chbuffer);
      strads_msg(ERR, " tmprow(%lu)  tmpcol(%lu) tmpval(%lf)\n", 
		 tmprow, tmpcol, tmpval);
    }


    seqno++;
    // mmt format: origin coordinate is at (1,1)
    assert(tmprow >= 1);
    assert(tmpcol >= 1);

    tmprow--;  /* adjust from 1-based to 0-based */
    tmpcol--;    

    if(!(tmprow < maxrow)){
      strads_msg(OUT, "\n tmprow (%lu) maxrow(%lu)  seqno(%lu)\n", 
		 tmprow, maxrow, seqno);
      assert(0);
    }
    if(!(tmpcol < maxcol)){
      strads_msg(ERR, "\n tmpcol (%lu) maxcol(%lu)\n", 
		 tmpcol, maxcol);
      assert(0);
    }

    if(   (tmprow >= (long unsigned int)row_range.get_min()) 
       && (tmprow <= (long unsigned int)row_range.get_max())
       && (tmpcol >= (long unsigned int)col_range.get_min())
       && (tmpcol <= (long unsigned int)col_range.get_max())) {

      //      mat(tmprow, tmpcol)=tmpval;     
      mat.add(tmprow, tmpcol, tmpval);     

      nzprogress++;
      if(nzprogress %10000 == 0){
	strads_msg(INF, " %ld th nzelement (%s) at mid (%ld)", nzprogress, chbuffer, mid);
      }
    }
  }

  //  fclose(f);
  return 0;
}










#endif 
