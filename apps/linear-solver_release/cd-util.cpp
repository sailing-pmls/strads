#include "cd-util.hpp"
#include <glog/logging.h>
#include "strads/include/cas_array.hpp"

using namespace std;

cont_range::cont_range(long min_value, long max_value, long parts, long id){
  make_part(min_value, max_value, parts, id);
}

long cont_range::get_min(void)const{ 
  return m_min; 
}

long cont_range::get_max(void)const{
  return m_max; 
}

int cont_range::make_part(long minval, long maxval, long parts, long id){

  assert(id < parts); // because id : 0 , .. , parts-1 

  if(minval < 0){
    throw out_of_range("cont_range::make_part: negative minvalue");
  }
  if(maxval < 0){
    throw out_of_range("cont_range::make_part: negative maxvalue");
  }

  long share = (maxval - minval + 1)/parts;
  long remain = (maxval - minval + 1)%parts;
  long local_min=0;
  long local_max=minval-1;

  for(auto i=0; i<parts; i++){
    local_min = local_max + 1;
    if(i < remain){
      local_max = local_min + share; 
    }else{
      local_max = local_min + share -1; 
    }
    if(i == id){
      m_max = local_max;
      m_min = local_min;
      break;
    }
  }
  LOG(INFO) << "min: " << minval << " max: " << maxval << " parts: " << parts << " id: " << id << " m_min: " << m_min << " m_max: " << m_max << endl;     
  return 0; // normal 
}

bool negative_check(int64_t value){
  if(value > 0){
    return true;
  }
  if(value <= 0){
    return false;
  }
  return true;
}

bool negative_check(int32_t value){
  if(value > 0){
    return true;
  }
  if(value <= 0){
    return false;
  }
  return true;
}

void thread_barrier::wait(void){
  std::unique_lock<std::mutex>lk(m_mutex);
  if(--m_count == 0){                                                                        
    m_cv.notify_all();                                                                       
    m_count = 0; // for reuse                                                                               
  }else{                                                                                   
    m_cv.wait(lk, [this] { return m_count == 0; });                                                             
  }                                                                                                       
}

long mmt_partial_read_vector(cas_array<double> &residual , const std::string &fn, long mid, const cont_range &row_range, const cont_range &col_range){

  //  std::cout << std::endl << " ****** mmt_partial_read : row size : " << mat.row_size() << " col size : " << mat.col_size() << std::endl;
  LOG(INFO) << "[mmt_partial_read] target matrix type : " << typeid(residual).name() << std::endl;

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

      //      mat.add(tmprow, tmpcol, tmpval);     
      residual[tmprow] = tmpval;

      nzprogress++;
      if(nzprogress %10000 == 0){
	strads_msg(INF, " %ld th nzelement (%s) at mid (%ld)", nzprogress, chbuffer, mid);
      }
    }
  }

  fclose(f);
  return 0;

}

void make_scheduling_taskpartition(cdtask_assignment &m_tmap, int64_t modelsize, int schedmach, int thrd_permach){

  for(auto m=0; m<schedmach; m++){
    cont_range mrange(0, modelsize-1, schedmach, m);
    range *atmp = new range;
    atmp->start = mrange.get_min();
    atmp->end = mrange.get_max();
    m_tmap.schmach_tmap.insert(std::pair<int, range *>(m, atmp));

    for(auto t=0; t<thrd_permach; t++){
      cont_range trange(mrange.get_min(), mrange.get_max(), thrd_permach, t);
      range *tmp = new range;
      tmp->start = trange.get_min();
      tmp->end = trange.get_max();
      m_tmap.schthrd_tmap.insert(std::pair<int, range*>((m*thrd_permach+t), tmp));
    }
  }
  
  for(auto p : m_tmap.schthrd_tmap){
    strads_msg(ERR, "[scheduling task partitioning] scheduler thread(%d) start(%ld) end(%ld)\n",
	       p.first, p.second->start, p.second->end);
  }

  for(auto p : m_tmap.schmach_tmap){
    strads_msg(ERR, "[scheduling task partitioning] scheduler mach(%d) start(%ld) end(%ld)\n",
	       p.first, p.second->start, p.second->end);
  }
}

#if 0 
// old one from the old repo. 
void make_scheduling_taskpartition(cdtask_assignment &m_tmap, int64_t modelsize, int schedmach, int thrd_permach){

  int parts = schedmach*thrd_permach;
  int64_t share = modelsize/parts;
  int64_t remain = modelsize % parts;
  int64_t start, end, myshare;

  //  give global assignment scheme per thread                                                                                                    
  for(int i=0; i < parts; i++){
    if(i==0){
      start = 0;
    }else{
      start = m_tmap.schthrd_tmap[i-1]->end + 1;
    }
    if(i < remain){
      myshare = share+1;
    }else{
      myshare = share;
    }
    end = start + myshare -1;
    if(end >= modelsize){
      end = modelsize -1;
    }
    range *tmp = new range;
    tmp->start = start;
    tmp->end = end;
    m_tmap.schthrd_tmap.insert(std::pair<int, range *>(i, tmp));
  }

  // give global assignment per mach                                                                                                              
  for(int i=0; i < schedmach; i++){
    int start_thrd = i*thrd_permach;
    int end_thrd = start_thrd + thrd_permach - 1;
    range *tmp = new range;
    tmp->start = m_tmap.schthrd_tmap[start_thrd]->start;
    tmp->end = m_tmap.schthrd_tmap[end_thrd]->end;
    m_tmap.schmach_tmap.insert(std::pair<int, range *>(i, tmp));
  }

  if(rank == 0){
    for(auto p : m_tmap.schthrd_tmap){
      strads_msg(ERR, "[scheduling task partitioning] scheduler thread(%d) start(%ld) end(%ld)\n",
		 p.first, p.second->start, p.second->end);
    }

    for(auto p : m_tmap.schmach_tmap){
      strads_msg(ERR, "[scheduling task partitioning] scheduler mach(%d) start(%ld) end(%ld)\n",
		 p.first, p.second->start, p.second->end);
    }
  }
}
#endif 



long mmt_partial_read_vector_ring(sharedctx *ctx, cas_array<double> &residual , const std::string &fn, long mid, const cont_range &row_range, const cont_range &col_range){

  //  std::cout << std::endl << " ****** mmt_partial_read : row size : " << mat.row_size() << " col size : " << mat.col_size() << std::endl;
  LOG(INFO) << "[mmt_partial_read] target matrix type : " << typeid(residual).name() << std::endl;

  long ret_code;
  MM_typecode matcode;
  FILE *f;
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
#endif
  
  maxrow = FLAGS_samples;
  maxcol = 1;
  nzcnt = FLAGS_samples;
  
  //  long min=row_range.get_min();

  strads_msg(OUT, "** FILE %ld nzcnt to read, mmt(row,col)(%ld, %ld): mat-portion for mid(%ld) range: row(%ld - %ld) col(%ld - %ld) \n", 
	     nzcnt, maxrow, maxcol, mid, row_range.get_min(), row_range.get_max(), col_range.get_min(), col_range.get_max());

  char *chbuffer = (char *)calloc(1024*1024*2, sizeof(char));
  uint64_t nzprogress=0;
  
  ringio ringfile(ctx, fn);
  
  //  while(fgets(chbuffer, 1024*1024, f) != NULL){
  while(ringfile.getline(chbuffer, 1024*1024) != NULL){
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

      //      mat.add(tmprow, tmpcol, tmpval);     
      residual[tmprow] = tmpval;

      nzprogress++;
      if(nzprogress %10000 == 0){
	strads_msg(INF, " %ld th nzelement (%s) at mid (%ld)", nzprogress, chbuffer, mid);
      }
    }
  }

  //  fclose(f);
  return 0;

}
