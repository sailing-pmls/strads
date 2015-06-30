#pragma once

#include <assert.h>
#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <string>
#include <vector>
#include <list>
#include <map>

#include "indepds.hpp"

class userparam;

userparam *user_initparams(void);

class userparam{
 public:
  userparam(double alpha, double beta, int topic, int64_t iter, int64_t samples, int64_t logfreq, std::string &logdir, std::string &tlogprefix)
    :m_alpha(alpha), m_beta(beta), m_topics(topic), m_iter(iter), m_samples(samples), m_logfreq(logfreq), m_logdir(logdir.c_str()), m_tlogprefix(tlogprefix.c_str()), m_empty5_size(0), m_empty7_size(0), m_cost(1.0) {      
    m_machstring.insert(std::pair<machtype, std::string>(m_coordinator, "coordinator"));
    m_machstring.insert(std::pair<machtype, std::string>(m_scheduler, "scheduler"));
    m_machstring.insert(std::pair<machtype, std::string>(m_worker, "worker"));
    // WHEN 
  } 


  userparam(double alpha, double beta, int topic, int64_t iter, int64_t samples, int64_t logfreq, std::string &logdir, std::string &tlogprefix, int64_t empty5_size, int64_t empty7_size, double cost)
    :m_alpha(alpha), m_beta(beta), m_topics(topic), m_iter(iter), m_samples(samples), m_logfreq(logfreq), m_logdir(logdir.c_str()), m_tlogprefix(tlogprefix.c_str()), m_empty5_size(empty5_size), m_empty7_size(empty7_size), m_cost(cost) {      
    m_machstring.insert(std::pair<machtype, std::string>(m_coordinator, "coordinator"));
    m_machstring.insert(std::pair<machtype, std::string>(m_scheduler, "scheduler"));
    m_machstring.insert(std::pair<machtype, std::string>(m_worker, "worker"));
    // WHEN 
  } 

  userparam(void)
    :m_alpha(0), m_beta(0), m_topics(0), m_iter(0) {  
    std::cout << "User Param is called with VOID " << std::endl;
  }

  ~userparam(void){
    std::cout<< "@@@@@@@@@@ user parameter destroyer " << std::endl;
  }

  void print(void){
    std::cout << "[user parameters] alpha     : " <<  m_alpha << std::endl;
    std::cout << "[user parameters] beta      : " << m_beta << std::endl;
    std::cout << "[user parameters] topics    : " << m_topics << std::endl;
    std::cout << "[user parameters] iterations: " << m_iter << std::endl;
    std::cout << "[user parameters] Samples   : " << m_samples << std::endl;
    
    for(auto p: m_fnmap){
      userfn_entity *entry = p.second;
      assert(p.first == entry->m_fileid);     
      std::cout << "[user parameters] FN: " << entry->m_strfn << "Type: "<< entry->m_strtype << " machtype: " << entry->m_strmachtype << "enummachtype: " << entry->m_mtype << std::endl;
    }
    //    auto p = m_fnlist.begin();
    //    std::cout << "[user paramerts] filename : " << *p << std::endl;
  }

  void insert_fn(std::string &fn, std::string &type, std::string &machtype, std::string &alias, std::string &pscheme){
    int id = m_fnmap.size();
    userfn_entity *entry = new userfn_entity(id, fn, type, machtype, alias, pscheme); 
    bool found=false;
    for(auto p: m_machstring){
      //LOG(INFO) << "COMPARE :" << p.second << " v.s " << machtype <<std::endl;
      if(p.second.compare(machtype) == 0){
	if(found){
	  LOG(FATAL) << "Something Wrong, duplicated match :" << p.second << " v.s " << machtype <<std::endl;
	}
	found = true;
	entry->m_mtype = p.first;
      }
    }
    if(!found){
      LOG(FATAL) << "User Machine("<< machtype <<")type not match scheduler/coordinator/worker" << std::endl;
    }
    m_fnmap.insert(std::pair<int, userfn_entity *>(id, entry));
  }

  void insert_func(std::string &func, int shardcnt, const char **shalias){
    std::cout << "##### Insert Function " << func << " shardcnt " << shardcnt  << std::endl;
    for(int i=0; i< shardcnt; i++){
      //      std::cout << "\t Shard Idx " << i << "shalias \n" << shalias[i] << std::endl;
      assert(shalias[i]);
      //      printf("\t shared cnt(%d) shalias %p  %s \n", shalias], shalias[i]); 
    }
    userfunc_entity *entry = new userfunc_entity(func, shardcnt, shalias);
    int id = m_funcmap.size();
    m_funcmap.insert(std::pair<int, userfunc_entity *>(id, entry));            
  }

  void bind_func_param(const char *alias, void *dshard){
    for(auto p : m_funcmap){
      userfunc_entity *ent = p.second;           
      std::cout << "Function Name : " << ent->m_strfuncname << std::endl;
      for(auto pp : ent->m_func_paramap){
	func_params *fe = pp.second;
	if(strcmp(fe->alias, alias) == 0){       
	  fe->dshard = dshard;
	  std::cout << "Alias " << alias << " is bound to this function " << ent->m_strfuncname << std::endl;
	}
      }
    }
  }

  double m_alpha;
  double m_beta;
  int m_topics;
  int64_t m_iter;
  int64_t m_samples;
  int64_t m_logfreq;
  const char *m_logdir;
  const char *m_tlogprefix;
  int64_t m_empty5_size;
  int64_t m_empty7_size;

  double m_cost;

  std::map<int, userfn_entity *>m_fnmap;
  std::map<machtype, std::string>m_machstring;
  std::map<int, userfunc_entity *>m_funcmap;
  

  private:
};
