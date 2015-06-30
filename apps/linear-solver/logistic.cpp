#include <atomic>
#include <assert.h>

#include "ll-worker.hpp"
#include "cd-train.hpp"
#include "strads/ds/spmat.hpp"
#include "cd-util.hpp"
#include <strads/netdriver/comm.hpp>
#include <strads/include/indepds.hpp>
#include "lassoll.hpp" 
#include "strads/include/indepds.hpp"

#include "strads/include/common.hpp"
#include "strads/netdriver/comm.hpp"
#include "strads/netdriver/zmq/zmq-common.hpp"

#include <math.h>
//#include <gsl/gsl_sf.h>
#include <strads/util/utility.hpp>

using namespace std;

double logistic::_partialsum_update(col_vspmat &cm_vmat, cas_array<double> &residual, int64_t vid, double *Y){
  double thsum = 0;
  uint64_t tmprow;
  spmat_vector *ccol = &cm_vmat.col((uint64_t)vid);
  for(uint64_t idx=0; idx<ccol->size(); idx++){
    uint64_t tmprow = ccol->idx[idx];
    double tmpval = ccol->val[idx];
    double tmp = residual[tmprow]*Y[tmprow];
    //    double ch =  gsl_sf_exp(-tmp) + 1;
    double ch =  exp(-tmp) + 1;
    double pi = 1.0/ch;
    thsum += Y[tmprow]*tmpval*(pi-1.0);
  }
  return thsum;
}

void * logistic::update_feat(col_vspmat &cm_vmat, cas_array<double> &residual, void *userdata, void *ctx, double *Y){

  uobjhead *uobjhp = (uobjhead *)userdata;
  int64_t taskcnt = uobjhp->task_cnt;

  TASK_ENTRY_TYPE *ids = (TASK_ENTRY_TYPE *)((uintptr_t)uobjhp + sizeof(uobjhead));
  void *tmp = (void *)calloc(1, sizeof(uobjhead) + sizeof(TASK_MVAL_TYPE)*taskcnt);
  uobjhead *retobjhp = (uobjhead *)tmp;
  TASK_MVAL_TYPE *midvalp = (TASK_MVAL_TYPE *)((uintptr_t)retobjhp + sizeof(uobjhead));
  retobjhp->task_cnt = taskcnt;
  retobjhp->stat_cnt = 0;

  for(int64_t i=0; i < taskcnt; i++){
    int64_t id = ids[i].id;
    double sum = _partialsum_update(cm_vmat, residual, id, Y);
    midvalp[i].id = id;
    midvalp[i].psum = sum;
  }
  return (void *)retobjhp;
}

// DONE 
// static function 
double logistic::object_calc(const cont_range &row_range, cas_array<double> &residual, double *Y){
  double term1=0;
  double xib;
  uint64_t sample_start = row_range.get_min();
  uint64_t sample_end = row_range.get_max();
  double pobjsum=0;
  for(uint64_t i=sample_start; i<= sample_end; i++){
    xib = residual[i];
    //    term1 += (log(1 + gsl_sf_exp(-1*Y[i]*xib)));
    term1 += (log(1 + exp(-1*Y[i]*xib)));

  }
  pobjsum = term1;
  return pobjsum;

}

// DONE
void * logistic::update_res(col_vspmat &cm_vmat, cas_array<double> &residual, void *userdata, void *ctx){
  uobjhead *uobjhp = (uobjhead *)userdata;
  int64_t sample_s = uobjhp->start;
  int64_t sample_e = uobjhp->end;
  int64_t taskcnt = uobjhp->task_cnt;
  int64_t statcnt = uobjhp->stat_cnt;  
  TASK_ENTRY_TYPE *ids = (TASK_ENTRY_TYPE *)((uintptr_t)uobjhp + sizeof(uobjhead));
  idval_pair *idvalp = (idval_pair *)((uintptr_t)ids + sizeof(TASK_ENTRY_TYPE)*taskcnt);
  for(int64_t i=0; i < statcnt; i++){
    int64_t id = idvalp[i].id;
    double delta = idvalp[i].value;
    if(delta != 0.0)
      _update_residual_by_sample(cm_vmat, residual, delta, id, sample_s, sample_e);
  }
  return NULL;
}


// DONE
void logistic::_update_residual_by_sample(col_vspmat &cm_vmat, cas_array<double> &residual, double delta, int64_t vid, int64_t row_s, int64_t row_e){
  assert(row_s >= 0);
  assert(row_e >= 0);
  assert(vid >= 0);
  double rdelta=0;
  spmat_vector *ccol = &cm_vmat.col((uint64_t)vid);
  for(long unsigned int idx=0; idx< ccol->size(); idx++){
    uint64_t tmprow = ccol->idx[idx];
    double tmpval = ccol->val[idx];
    if(tmprow >= (uint64_t)row_s && tmprow <= (uint64_t)row_e){ // row_s , row_e is thread level fine grained. 
      rdelta = tmpval*delta;
      residual.add(tmprow, rdelta);
    }
  }
  return;
}



// DONE
/* soft_thrd, multiresidual  for workers */
double logistic::soft_threshold(double sum, double lambda){
  double res;
  if(sum >=0){
    if(sum > lambda){
      res = sum - lambda;
    }else{
      res = 0;
    }
  }else{
    if(sum < -lambda){
      res = sum + lambda;
    }else{
      res = 0;
    }
  }
  return res;
}

// DONE
// psum is sum of square error 
double logistic::get_object_server(double *beta, double psum, double lambda, long modelsize, long samples){

  double betasum=0;
  int64_t nzcnt=0;

  for(int64_t i=0; i<modelsize; i++){
    betasum += fabs(beta[i]);
    if(beta[i] != 0){
      nzcnt++;
    }
  }
  double objvalue = psum/samples + lambda*betasum;
  //  double objvalue = psum + lambda*betasum;        
  strads_msg(OUT, "\t\t Object %lf nz(%ld)\n",
             objvalue, nzcnt);

  return objvalue;
}


// DONE
void *logistic::aggregator(double *beta, double *betadiff, unordered_map<int64_t, idmvals_pair *> &retmap, sharedctx *ctx, double lambda, long modelsize, long samples){
  int64_t entrycnt = retmap.size();
  //  double lambda = FLAGS_lambda;
  //  int64_t modelsize = FLAGS_columns;  
  for(int i=0; i < ctx->m_worker_machines; i++){
    while(1){
      void *buf = ctx->worker_recvportmap[i]->ctx->pull_entry_inq();
      if(buf != NULL){
        mbuffer *tmpbuf = (mbuffer *)buf;
        workhead *retworkhp = (workhead *)tmpbuf->data;
        assert(retworkhp->type == WORK_PARAMUPDATE);
        uobjhead *retuobjhp = (uobjhead *)((uintptr_t)retworkhp + sizeof(workhead));
        idmvals_pair *retidmvalp = (idmvals_pair *)((uintptr_t)retuobjhp + sizeof(uobjhead));
        int64_t ptaskcnt = retuobjhp->task_cnt;
        if(ptaskcnt != entrycnt){
          strads_msg(ERR, "[user aggregator function] PTASKCNT (%ld)  != ENTRYCNT (%ld) \n",
                     ptaskcnt, entrycnt);
        }
        assert(ptaskcnt == entrycnt);
        for(int64_t i =0; i < ptaskcnt; i++){
          int64_t id = retidmvalp[i].id;
          auto p = retmap.find(id);
          //            assert(retmap.find(id) != retmap.end());                                                                                     
          assert(p != retmap.end());
          assert(p->second->id == 0 || p->second->id == id);
          p->second->id = id;
          p->second->psum =  p->second->psum + retidmvalp[i].psum;
	  //          p->second->sqpsum =  p->second->sqpsum + retidmvalp[i].sqpsum;
        }
        ctx->worker_recvportmap[i]->ctx->release_buffer((void *)buf);
        break;
      }
    }
  }

  assert(retmap.size() == (uint64_t)entrycnt);

  assert(retmap.size() == (uint64_t)entrycnt);
  for(auto p : retmap){
    assert(p.first == p.second->id);
    int64_t id = p.second->id;
    double aj = p.second->psum / samples;
    double newcoeff = soft_threshold(beta[id]-4*aj, 4*lambda);
    assert(id >= 0);
    assert(id < modelsize);
    //    betadiff[id] = beta[id] - newcoeff;                                                            
    betadiff[id] = newcoeff - beta[id]; // in logistic regression, we keep Ax instead of residual: Y-Ax      
    beta[id] = newcoeff;
    strads_msg(INF, " p.first(%ld) p->id(%ld)  psum(%lf)  sqpsum(%lf) Beta[%ld] = [%lf] -- diff(%lf)\n",
	       p.first, p.second->id, p.second->psum, p.second->sqpsum, id, beta[id], betadiff[id]);
  }

  // TODO : add reuse psum or xqpsum as beta diff storage....            
  // and use it in the first half thread ....... beta diff reference.                                                          
  for(auto p : retmap){
    assert(p.first == p.second->id);
    int64_t id = p.second->id;
    p.second->sqpsum = betadiff[id];
  }

  return NULL;
}
