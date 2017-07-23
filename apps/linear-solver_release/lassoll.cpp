
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <mpi.h>
#include <assert.h>
#include <strads/include/common.hpp>
#include "lassoll.hpp"
#include "cd-util.hpp"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////
// WARNING: If you make changes to GFLAGS DEFINITION, you should synchronize changes witht those of lassoll.hpp file 
//////////////////////////////////////////////////////////////////////////////////////////////////
DEFINE_bool(output_hdfs_use, false, "on/off for using HDFS. All output files should be stored in one type of FS");
DEFINE_string(outputfile_coeff, "./output/coeff.out", "output file name to store non-zero coefficient");
DEFINE_string(logfile, "./output/output.log", "log file to record objective value per freq");

DEFINE_int64(logfreq, 1000, "objective value logging frequency");
DEFINE_int64(threads, 1, "The number of threads per machine");  // different meaning from multi thread one 

// non-mandotory but highly recommend to tune 
DEFINE_int64(scheduler, 1, "the number of scheduler process instance by default 1");
DEFINE_int64(threads_per_scheduler, 1, "the number of scheduler process instance by default 1");

DEFINE_int64(schedule_size, 32, "the number of CD coefficients to update in parallel");
DEFINE_bool(weight_sample, true, "on/off for dynamic priority sampling");
DEFINE_bool(dep_check, true, "on/off for dependency checking");

// mandatory to set by user since it has problem specific meaning 
DEFINE_double(lambda, 0.01, "lambda to control sparsity on the output");
DEFINE_int64(max_iter, 100, "Number of maximum training iteration");
DEFINE_bool(input_hdfs_use, false, "on/off for using HDFS. All input files should be stored in one type of FS");
DEFINE_string(data_xfile, "", "design matrix denoted as X : M by N matrix  "); 
DEFINE_string(data_yfile, "", "observation vector denoted as Y : M by 1 matrix  "); 
DEFINE_int64(columns, 0, "the number of columns of x files .. denoted as N "); 
DEFINE_int64(samples, 0, "the number of rows of x files, rows of Y file .. denoted as M ");
DEFINE_int64(nzcount, 0, "the number of non zero elements in the input matrix X  ");
DEFINE_int64(pipeline, 0, "pipeline depth : staleness ");

DEFINE_double(bw, 0.0001, " base width ");
DEFINE_double(infthreshold, 0.1, " threshold on correlation  ");

DEFINE_bool(weight_sampling, false, " Weight Sampling on/off");
DEFINE_bool(check_interference, false, " Interference Checking on/off");

DEFINE_string(algorithm, "lasso", " algoritm : lasso or logistic");

static bool ValidateSamples(const char *flagname, int64_t value){ return negative_check(value); }
static const bool samples_dummy = google::RegisterFlagValidator(&FLAGS_samples, &ValidateSamples);
static bool ValidateColumns(const char *flagname, int64_t value){ return negative_check(value); }
static const bool columns_dummy = google::RegisterFlagValidator(&FLAGS_columns, &ValidateColumns);

void print_arg(int argc, char **argv){
  cout << "     Argc : " << argc;
  for(auto i=0; i<argc; i++){
    std::cout << "  argv[" << i << "] : " << argv[i] << endl;;
  }
}

int main(int argc, char **argv){
  
  google::ParseCommandLineFlags(&argc, &argv, false);
  GOOGLE_PROTOBUF_VERIFY_VERSION; 

  string string_scheduler = std::to_string(FLAGS_scheduler);
  string change = google::SetCommandLineOption("schedulers", string_scheduler.c_str()); 
  assert(FLAGS_scheduler == FLAGS_schedulers);

  sharedctx *ctx = strads_init(argc, argv); // src/components/strads_init.cpp
  LOG(INFO) <<  " in main function MPI rank :  " <<  ctx->rank << "mach_role: " << ctx->m_mrole << std::endl;  

  ctx->m_sp->reset_schedule(FLAGS_scheduler, true);
  ctx->prepare_machine(ctx->m_mpi_size);

  if(ctx->m_mrole == mrole_worker){
    worker_mach(ctx);
  }else if(ctx->m_mrole == mrole_coordinator){
    coordinator_mach(ctx);
  }else if(ctx->m_mrole == mrole_scheduler){
    scheduler_mach(ctx);
  }else{
    strads_msg(ERR, "Fatal: Unknown machine identity\n");    
    assert(0);
  }

  MPI_Finalize();
  strads_msg(OUT, "in main : done exit program normally logout \n");
  return 0;
}
