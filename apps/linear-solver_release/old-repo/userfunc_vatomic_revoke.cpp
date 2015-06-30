#include <iostream>
#include <stdio.h>

#include <cmath>

#include <assert.h>
#include <unordered_map>
#include "common.hpp"
#include "ds/dshard.hpp"
#include "indepds.hpp"

using namespace std;

#if 0 
/* soft_thrd, multiresidual  for workers */
static double _soft_threshold(double sum, double lambda){
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


double user_get_object_server(dshardctx *dshardcoeff, double psum, void *cctx){
  // call update multi residual function for a given sample range (uobj->start, uobj->end ) for all columns 

  sharedctx *ctx = (sharedctx *)cctx;
  assert(dshardcoeff != NULL);
  double **betatmp = dshardcoeff->m_dmat.m_mem;
  double *beta = betatmp[0];

  double betasum=0;
  int64_t nzcnt=0;
  int64_t modelsize = ctx->m_params->m_sp->m_modelsize;
  double lambda = ctx->m_params->m_up->m_beta;
  for(int64_t i=0; i<modelsize; i++){
    betasum += fabs(beta[i]);
    if(beta[i] != 0){
      nzcnt++;
    }
  }	 
  //  double objvalue = psum/2.0 + lambda*betasum;
  double objvalue = psum + lambda*betasum;

  if(std::isnan(objvalue)){
    strads_msg(ERR, "NAN ERROR psum : %lf, betasum : %lf\n", psum, betasum);
    assert(0);
    exit(-1);
  }

  strads_msg(ERR, "\t\t Object %lf nz(%ld)\n",
	     objvalue/2.0, nzcnt);	 

  return objvalue/2.0;
}



double _partialsum_update(dshardctx *dshardA, dshardctx *dshardRes, int64_t vid, double vidbeta, double *xsqmsum, int64_t s, int64_t e);
void _update_residual_by_sample(dshardctx *dshardA, dshardctx *dshardRes, double delta, int64_t id, int64_t row_s, int64_t row_end);

//void *user_aggregator(dshardctx *dshardcoeff, void *userdata, void *ctx){
void *user_aggregator(dshardctx *dshardcoeff, dshardctx *dshardYcord, unordered_map<int64_t, idmvals_pair *> &retmap, void *cctx){
  
  // not for lasso 
  assert(0);

}

void *user_aggregator(dshardctx *dshardcoeff, unordered_map<int64_t, idmvals_pair *> &retmap, void *cctx){

  sharedctx *ctx = (sharedctx *)cctx;
  double **betatmp = dshardcoeff->m_dmat.m_mem;
  
  double *beta = betatmp[0];
  int64_t entrycnt = retmap.size();
  double lambda = ctx->m_params->m_up->m_beta;  
  int64_t modelsize = ctx->m_params->m_sp->m_modelsize;
  
  double *betadiff = ctx->m_weights;
  assert(modelsize == ctx->m_weights_size);

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
	  //		assert(retmap.find(id) != retmap.end());
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
    //	  strads_msg(ERR, " P first : %ld \n", p.first);
    assert(p.first == p.second->id);
    int64_t id = p.second->id;
    double tmp = _soft_threshold(p.second->psum, lambda/2.0);
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


  // TODO : add reuse psum or xqpsum as beta diff storage.... haha 
  // and use it in the first half thread ....... beta diff reference. 
  for(auto p : retmap){
    assert(p.first == p.second->id);
    int64_t id = p.second->id;
    p.second->sqpsum = betadiff[id];
  }



  return NULL;
}



void *user_update_status(dshardctx *dshardA, dshardctx *dshardRes, void *userdata, void *ctx){
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
      _update_residual_by_sample(dshardA, dshardRes, delta, id, sample_s, sample_e);
  }
  return NULL;
}

/*
  dshardA : col major , row partition 
  dshardRes : row major, row partition 
  delta : difference between old, new value of id-coefficient
  CAVEAT Each thread is in charge of finegrained sample range of machine level sample range
  row_s and row_e is such thread level fine-grained sample range.  
*/
void _update_residual_by_sample(dshardctx *dshardA, dshardctx *dshardRes, double delta, int64_t vid, int64_t row_s, int64_t row_e){
  assert(dshardA->m_type == cvspt);
  assert(row_s >= 0);
  assert(row_e >= 0);
  assert(vid >= 0);
  strads_msg(INF, "_update residual by sample : vid (%ld) diff(%lf) sam_start(%ld) sam_end(%ld)\n", 
	     vid, delta, row_s, row_e);
  //  strads_msg(ERR,  "__update Residual by saple shard A's name %s \n", dshardA->m_alias); 
  assert((uint64_t)vid >= dshardA->m_range.c_start); // range checking. 
  assert((uint64_t)vid <= dshardA->m_range.c_end);
  double rdelta=0;
  //  double **residual = dshardRes->m_dmat.m_mem;
  spmat_vector *ccol = &dshardA->m_cm_vmat.col((uint64_t)vid);
  for(long unsigned int idx=0; idx< ccol->size(); idx++){
    uint64_t tmprow = ccol->idx[idx];
    double tmpval = ccol->val[idx];
    if(tmprow >= (uint64_t)row_s && tmprow <= (uint64_t)row_e){ // row_s , row_e is thread level fine grained. 
      rdelta = tmpval*delta;
      dshardRes->m_atomic.add(tmprow, rdelta);
    }
  }
  return;
}

#endif 


void *user_update_parameter(dshardctx *dshardA, dshardctx *dshardRes, void *userdata, void *ctx){
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
    double sum = _partialsum_update(dshardA, dshardRes, id, coeff, &xsqmsum, sample_s, sample_e);    
    midvalp[i].id = id;
    midvalp[i].psum = sum;
    midvalp[i].sqpsum = xsqmsum;
  }
  return (void *)retobjhp;
}


double _partialsum_update(dshardctx *dshardA, dshardctx *dshardRes, int64_t vid, double vidbeta, double *xsqmsum, int64_t sample_s, int64_t sample_e){
  double thsum = 0;
  double xsq=0;

  assert((uint64_t)vid >= 0);
  assert((uint64_t)vid >= dshardA->m_range.c_start); // range checking. 
  assert((uint64_t)vid <= dshardA->m_range.c_end);

  assert(dshardA->m_type == cvspt);

  assert(dshardA->m_range.r_start == dshardRes->m_range.r_start); // shard A and Res should cover the same samples 
  assert(dshardA->m_range.r_end == dshardRes->m_range.r_end);
  assert(sample_s >= 0);
  assert(sample_e >= 0);
  assert(dshardA->m_range.r_start == (uint64_t)sample_s);  
  assert(dshardA->m_range.r_end == (uint64_t)sample_e);

  spmat_vector *ccol = &dshardA->m_cm_vmat.col((uint64_t)vid);
  for(uint64_t idx=0; idx<ccol->size(); idx++){
    uint64_t tmprow = ccol->idx[idx];
    double tmpval = ccol->val[idx];
    if(tmprow < dshardA->m_range.r_start){
      strads_msg(ERR, " out of range in partial sum update \n");
      exit(0);
    }
    if(tmprow > dshardA->m_range.r_end){
      strads_msg(ERR, " out of range in partial sum update \n");
      exit(0);
    }

    thsum += ((vidbeta*tmpval*tmpval ) + tmpval*dshardRes->m_atomic[tmprow]);
    //    thsum += ((vidbeta*tmpval*tmpval ) + tmpval*residual[tmprow][0]);
    xsq += (tmpval * tmpval);    
  }

  *xsqmsum = xsq;
  return thsum;
}
