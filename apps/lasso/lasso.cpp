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

using namespace std;

// in lasso Y is not used 
void * lasso::update_feat(col_vspmat &cm_vmat, cas_array<double> &residual, void *userdata, void *ctx, double *Y){
  uobjhead *uobjhp = (uobjhead *)userdata;
  int64_t sample_s = uobjhp->start;
  int64_t sample_e = uobjhp->end;
  int64_t taskcnt = uobjhp->task_cnt;
  TASK_ENTRY_TYPE *ids = (TASK_ENTRY_TYPE *)((uintptr_t)uobjhp + sizeof(uobjhead));
  void *tmp = (void *)calloc(1, sizeof(uobjhead) + sizeof(TASK_MVAL_TYPE)*taskcnt);
  uobjhead *retobjhp = (uobjhead *)tmp;
  TASK_MVAL_TYPE *midvalp = (TASK_MVAL_TYPE *)((uintptr_t)retobjhp + sizeof(uobjhead));
  retobjhp->task_cnt = taskcnt;
  retobjhp->stat_cnt = 0;
  for(int64_t i=0; i < taskcnt; i++){
    int64_t id = ids[i].id;
    double coeff = ids[i].value;
    double xsqmsum=0;
    double sum = _partialsum_update(cm_vmat, residual, id, coeff, &xsqmsum, sample_s, sample_e);    
    midvalp[i].id = id;
    midvalp[i].psum = sum;
    midvalp[i].sqpsum = xsqmsum;
  }
  return (void *)retobjhp;
}

// static member function of lasso classl                                                                                        
double lasso::object_calc(const cont_range &row_range, cas_array<double> &residual){
  double pobjsum=0;
  for(auto i=row_range.get_min(); i <= row_range.get_max(); i++){
    pobjsum += (residual[i]*residual[i]);
  }
  return pobjsum;
}

void * lasso::update_res(col_vspmat &cm_vmat, cas_array<double> &residual, void *userdata, void *ctx){
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



double lasso::_partialsum_update(col_vspmat &cm_vmat, cas_array<double> &residual, int64_t vid, double vidbeta, double *xsqmsum, int64_t sample_s, int64_t sample_e){
  double thsum = 0;
  double xsq=0;
  assert((uint64_t)vid >= 0);
  spmat_vector *ccol = &cm_vmat.col((uint64_t)vid);
  for(uint64_t idx=0; idx<ccol->size(); idx++){
    uint64_t tmprow = ccol->idx[idx];
    double tmpval = ccol->val[idx];
    thsum += ((vidbeta*tmpval*tmpval ) + tmpval*residual[tmprow]);
    xsq += (tmpval * tmpval);    
  }
  *xsqmsum = xsq;
  return thsum;
}
void lasso::_update_residual_by_sample(col_vspmat &cm_vmat, cas_array<double> &residual, double delta, int64_t vid, int64_t row_s, int64_t row_e){
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




/* soft_thrd, multiresidual  for workers */
double lasso::soft_threshold(double sum, double lambda){
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

// psum is sum of square error 
double lasso::get_object_server(double *beta, double psum, double lambda, long modelsize, long samples){
  double betasum=0;
  int64_t nzcnt=0;
  //  int64_t modelsize = FLAGS_columns;
  //  double lambda = FLAGS_lambda;
  for(int64_t i=0; i<modelsize; i++){
    betasum += fabs(beta[i]);
    if(beta[i] != 0){
      nzcnt++;
    }
  }
  double objvalue = psum + lambda*betasum;
  if(std::isnan(objvalue)){
    strads_msg(ERR, "NAN ERROR psum : %lf, betasum : %lf\n", psum, betasum);
    assert(0);
    exit(-1);
  }
  strads_msg(OUT, "\t\t Object %lf nz(%ld)\n",
             objvalue/2.0, nzcnt);
  return objvalue/2.0;
}

//void *user_aggregator(dshardctx *dshardcoeff, unordered_map<int64_t, idmvals_pair *> &retmap, void *cctx){
void *lasso::aggregator(double *beta, double *betadiff, unordered_map<int64_t, idmvals_pair *> &retmap, sharedctx *ctx, double lambda, long modelsize, long samples){
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
          p->second->sqpsum =  p->second->sqpsum + retidmvalp[i].sqpsum;
        }
        ctx->worker_recvportmap[i]->ctx->release_buffer((void *)buf);
        break;
      }
    }
  }

  strads_msg(INF, "COORDINATOR GOT PARTIAL RESULTS FROM ALL WORKERS \n");
  assert(retmap.size() == (uint64_t)entrycnt);
  for(auto p : retmap){
    //    strads_msg(ERR, " P first : %ld \n", p.first);          
    assert(p.first == p.second->id);
    int64_t id = p.second->id;
    double tmp = soft_threshold(p.second->psum, lambda/2.0);
    double newcoeff = tmp / p.second->sqpsum;
    assert(id >= 0);
    assert(id < modelsize);
    betadiff[id] = beta[id] - newcoeff;
    double oldbeta = beta[id];
    beta[id] = newcoeff;
    strads_msg(INF, " p.first(%ld) p->id(%ld)  psum(%lf)  sqpsum(%lf) Beta[%ld] = [%lf]fromold[%lf] -- diff(%lf)\n",
               p.first, p.second->id, p.second->psum, p.second->sqpsum, id, beta[id], oldbeta, betadiff[id]);
    // keep the old beta value for revoking.          
    // let the coordinator_thread do revoke if necessary                                       
    p.second->psum = oldbeta;
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
