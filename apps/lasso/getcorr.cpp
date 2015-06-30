#include "getcorr.hpp"
#include "strads/util/utility.hpp"
#include <unordered_map>

//#include "spmat.hpp"

using namespace std;


// assume that rows in column vector is fully sorted on idx and has sorted vecotr of value corresponding to the sorted idx list
double getcorr_pair_sparse_vmat(spmat_vector &xcol, spmat_vector &ycol, uint64_t nsamples, double pxsum, double pxsqsum, double pysum, double pysqsum, int avail, uint64_t xcol_id, uint64_t ycol_id){

  //  strads_msg(OUT, "sparse vmat with vector cals, xcol.id(%ld) size (%ld) ycol.id(%ld) size(%ld)\n", 
  //	     xcol_id, xcol.idx.size(), ycol_id, ycol.idx.size());
#if 0 
  for(uint64_t i=0; i < xcol.size(); i++ ){
    strads_msg(ERR, " col (%ld) row(%ld) value (%lf)\n", 
	       xcol_id, xcol.idx[i], xcol.val[i]);
  }
  for(uint64_t i=0; i < ycol.size(); i++ ){
    strads_msg(ERR, " col (%ld) row(%ld) value (%lf)\n", 
	       ycol_id, ycol.idx[i], ycol.val[i]);
  }
#endif 

  double xysum=0, xsum=0, xsqsum=0, ysum=0, ysqsum=0;
  double corrcoef=0;
  //  unordered_map<long unsigned int, double>::iterator  xiter = xcol.begin();
  //  unordered_map<long unsigned int, double>::iterator  yiter = ycol.begin();
  assert(xcol.idx.size() != 0); // there should be no column that has zero entries in rows 
  assert(ycol.idx.size() != 0); // If this is triggers, there might be error in 
  assert(xcol.val.size() == xcol.idx.size());
  assert(ycol.val.size() == ycol.idx.size());


  for(uint64_t id=0; id < xcol.idx.size(); id++){
    xsum += xcol.val[id];
    xsqsum += (xcol.val[id]*xcol.val[id]);
  }

  for(uint64_t id=0; id < ycol.idx.size(); id++){
    ysum += ycol.val[id];
    ysqsum += (ycol.val[id]*ycol.val[id]);
  }

  uint64_t xsize = xcol.idx.size();
  uint64_t ysize = ycol.idx.size();
  uint64_t xprogress = 0;
  uint64_t yprogress = 0;

  while(xprogress != xsize and yprogress != ysize){
    if(xcol.idx[xprogress] == ycol.idx[yprogress]){
      xysum += xcol.val[xprogress]*ycol.val[yprogress];
      xprogress++;
      yprogress++;

      if(xprogress == xsize)
	break;

      if(yprogress == ysize)
	break;
    }else{
      if(xcol.idx[xprogress] > ycol.idx[yprogress]){
	yprogress++;
	if(yprogress == ysize)
	  break;
      }else{
	xprogress++;
	if(xprogress == xsize)
	  break;
      }
    }
  }

#if 0 
  if(xysum != 0){
    strads_msg(ERR, " col(%ld) col(%ld) hits \n", xcol_id, ycol_id);
  }
#endif 

  double divide = (sqrt(nsamples*xsqsum - (xsum)*(xsum))*(sqrt(nsamples*ysqsum - (ysum)*(ysum))));
  if(divide == 0.0){
    strads_msg(ERR, "##### nsamples(%ld) xsqsum(%2.20lf) xsum(%2.20lf) ysqsum(%2.20lf) ysum(%2.20lf)\n", 
	       nsamples, xsqsum, xsum, ysqsum, ysum);
    strads_msg(ERR, "##### nsamples*xsqsum(%2.20lf) xsum*xsum(%2.20lf) nsamples*ysqsum(%2.20lf) ysum*ysum(%2.20lf)\n", 
	       nsamples*xsqsum, xsum*xsum, nsamples*ysqsum, ysum*ysum);
    strads_msg(ERR, "##### sqrt(..) %2.20lf sqrt(..) %2.20lf\n", 
	       sqrt(nsamples*xsqsum - (xsum)*(xsum)),  
	       sqrt(nsamples*ysqsum - (ysum)*(ysum)));	           
    for(uint64_t i = 0; i < xcol.idx.size(); i++){
      strads_msg(ERR, " xcolid (%ld) Xrow[%ld] = %lf\n", xcol_id, xcol.idx[i], xcol.val[i]);
    }
    for(uint64_t i = 0; i < ycol.idx.size(); i++){
      strads_msg(ERR, " ycolid (%ld) Yrow[%ld] = %lf\n", ycol_id, ycol.idx[i], ycol.val[i]);
    }
  }

  assert(divide != 0.0);
  corrcoef = 1.0*(nsamples*xysum-(xsum*ysum))/divide;
  return fabs(corrcoef);
}







//double getcorr_pair_sparsem(unordered_map<long unsigned int, double> &xcol, unordered_map<long unsigned int, double> &ycol, uint64_t nsamples, double pxsum, double pxsqsum, double pysum, double pysqsum, int avail)
double getcorr_pair_sparsem(unordered_map<long unsigned int, double> &xcol, unordered_map<long unsigned int, double> &ycol, uint64_t nsamples, double pxsum, double pxsqsum, double pysum, double pysqsum, int avail){

  double xysum=0, xsum=0, xsqsum=0, ysum=0, ysqsum=0;
  double corrcoef=0;
  //unordered_map<long unsigned int, double>::iterator  xiter = xcol.begin();
  //unordered_map<long unsigned int, double>::iterator  yiter = ycol.begin();
  unordered_map<long unsigned int, double>::iterator  xiter = xcol.begin();
  unordered_map<long unsigned int, double>::iterator  yiter = ycol.begin();

  assert(xcol.size() != 0); // there should be no column that has zero entries in rows 
  assert(ycol.size() != 0); // If this is triggers, there might be error in 

#if 1
  for(; xiter != xcol.end(); xiter++){
    strads_msg(INF, "X[%ld] = %lf\n", xiter->first, xiter->second);
    xsum += xiter->second;
    xsqsum += ((xiter->second)*(xiter->second));
  }

  for(; yiter != ycol.end(); yiter++){
    strads_msg(INF, "Y[%ld] = %lf\n", yiter->first, yiter->second);
    ysum += yiter->second;
    ysqsum += ((yiter->second)*(yiter->second));
  }

  xiter = xcol.begin();
  yiter = ycol.begin();
  //  while(1){

  //  bool hitflag = false;
  while(xiter != xcol.end() && yiter != ycol.end()){
    if(xiter->first == yiter->first){
      xysum += ((xiter->second)*(yiter->second));
      //      strads_msg(ERR, "hit at %ld th row .... \n", xiter->first);
      //      hitflag = true;

      xiter++;
      yiter++;
      if(xiter == xcol.end())
	break;
      if(yiter == ycol.end())
	break;
    }else{
      if(xiter->first > yiter->first){
	yiter++;
	if(yiter == ycol.end())
	  break;
      }else{
	xiter++;
	if(xiter == xcol.end())
	  break;
      }
    }
  }
#endif 

  double divide = (sqrt(nsamples*xsqsum - (xsum)*(xsum))*(sqrt(nsamples*ysqsum - (ysum)*(ysum))));
  if(divide == 0.0){
    strads_msg(ERR, "##### nsamples(%ld) xsqsum(%2.20lf) xsum(%2.20lf) ysqsum(%2.20lf) ysum(%2.20lf)\n", 
	       nsamples, xsqsum, xsum, ysqsum, ysum);

    strads_msg(ERR, "##### nsamples*xsqsum(%2.20lf) xsum*xsum(%2.20lf) nsamples*ysqsum(%2.20lf) ysum*ysum(%2.20lf)\n", 
	       nsamples*xsqsum, xsum*xsum, nsamples*ysqsum, ysum*ysum);

    strads_msg(ERR, "##### sqrt(..) %2.20lf sqrt(..) %2.20lf\n", 
	       sqrt(nsamples*xsqsum - (xsum)*(xsum)),  
	       sqrt(nsamples*ysqsum - (ysum)*(ysum)));	          

    xiter = xcol.begin();
    yiter = ycol.begin();
  
    for(; xiter != xcol.end(); xiter++){
      strads_msg(ERR, "X[%ld] = %lf\n", xiter->first, xiter->second);
    }

    for(; yiter != ycol.end(); yiter++){
      strads_msg(ERR, "Y[%ld] = %lf\n", yiter->first, yiter->second);
    }

  }

  assert(divide != 0.0);
  corrcoef = 1.0*(nsamples*xysum-(xsum*ysum))/divide;

  //  if(hitflag){    
  //    strads_msg(ERR, "Hit Case: nsamples(%ld), xysum(%lf) xsym*ysum(%lf) divide(%lf)\n", 
  //	       nsamples, xysum, xsum*ysum, divide);
  //  }

  return fabs(corrcoef);
}







































































#if 0 // TEST CODE with OPTIMIZATION 
double getcorr_pair_sparsem(map<long unsigned int, double> &xcol, map<long unsigned int, double> &ycol, uint64_t nsamples, double pxsum, double pxsqsum, double pysum, double pysqsum, int avail){

  double xysum=0, xsum=0, xsqsum=0, ysum=0, ysqsum=0;
  double corrcoef=0;
  //unordered_map<long unsigned int, double>::iterator  xiter = xcol.begin();
  //unordered_map<long unsigned int, double>::iterator  yiter = ycol.begin();
  map<long unsigned int, double>::iterator  xiter = xcol.begin();
  map<long unsigned int, double>::iterator  yiter = ycol.begin();

  assert(xcol.size() != 0); // there should be no column that has zero entries in rows 
  assert(ycol.size() != 0); // If this is triggers, there might be error in 
  // input data or data partitioning plan in scheduler side. 
  // 

#if 0 // dummy code to measure x's column traverse overhead... 
  for(; xiter != xcol.end(); xiter++){
    strads_msg(INF, "X[%ld] = %lf\n", xiter->first, xiter->second);
    xsum += xiter->second;
  }
  for(; yiter != ycol.end(); yiter++){
    strads_msg(INF, "Y[%ld] = %lf\n", yiter->first, yiter->second);
    ysum += yiter->second;
  }
  //assert(xcol.size() == 100);
  //assert(ycol.size() == 100);
  return 0.0;
#endif 

  // turn on this 
#if 1
  for(; xiter != xcol.end(); xiter++){
    strads_msg(INF, "X[%ld] = %lf\n", xiter->first, xiter->second);
    xsum += xiter->second;
    xsqsum += ((xiter->second)*(xiter->second));
  }

  for(; yiter != ycol.end(); yiter++){
    strads_msg(INF, "Y[%ld] = %lf\n", yiter->first, yiter->second);
    ysum += yiter->second;
    ysqsum += ((yiter->second)*(yiter->second));
  }

  xiter = xcol.begin();
  yiter = ycol.begin();
  //  while(1){
  while(xiter != xcol.end() && yiter != ycol.end()){
    if(xiter->first == yiter->first){
      xysum += ((xiter->second)*(yiter->second));
      strads_msg(INF, "hit at %ld th row .... \n", xiter->first);
      
      xiter++;
      yiter++;
      if(xiter == xcol.end())
	break;
      if(yiter == ycol.end())
	break;

    }else{
      if(xiter->first > yiter->first){
	yiter++;
	if(yiter == ycol.end())
	  break;
      }else{
	xiter++;
	if(xiter == xcol.end())
	  break;
      }
    }
  }

#endif 


  // OPTIMIZATION
#if 0

#if 0 
  for(; xiter != xcol.end(); xiter++){
    strads_msg(INF, "X[%ld] = %lf\n", xiter->first, xiter->second);
    xsum += xiter->second;
    xsqsum += ((xiter->second)*(xiter->second));
  }

  for(; yiter != ycol.end(); yiter++){
    strads_msg(INF, "Y[%ld] = %lf\n", yiter->first, yiter->second);
    ysum += yiter->second;
    ysqsum += ((yiter->second)*(yiter->second));
  }
#endif 

  xiter = xcol.begin();
  yiter = ycol.begin();  

  while(xiter != xcol.end() && yiter != ycol.end()){
    if(xiter->first == yiter->first){
      xysum += ((xiter->second)*(yiter->second));
      strads_msg(INF, "hit at %ld th row .... \n", xiter->first);

      xsum += xiter->second;
      xsqsum += ((xiter->second)*(xiter->second));

      ysum += yiter->second;
      ysqsum += ((yiter->second)*(yiter->second));
      
      xiter++;
      yiter++;
      if(xiter == xcol.end())
	break;
      if(yiter == ycol.end())
	break;

    }else{
      if(xiter->first > yiter->first){
	ysum += yiter->second;
	ysqsum += ((yiter->second)*(yiter->second));
	yiter++;
	if(yiter == ycol.end()){
	  while(xiter != xcol.end()){
	    xsum += xiter->second;
	    xsqsum += ((xiter->second)*(xiter->second));
	    xiter++;
	  }
	  break;
	}

      }else{

	xsum += xiter->second;
	xsqsum += ((xiter->second)*(xiter->second));

	xiter++;
	if(xiter == xcol.end()){
	  while(yiter != ycol.end()){
	    ysum += yiter->second;
	    ysqsum += ((yiter->second)*(yiter->second));
	    yiter++;
	  }
	  break;
	}
      }
    }
  }

#endif 


  double divide = (sqrt(nsamples*xsqsum - (xsum)*(xsum))*(sqrt(nsamples*ysqsum - (ysum)*(ysum))));
  if(divide == 0.0){
    strads_msg(ERR, "##### nsamples(%2.20lf) xsqsum(%2.20lf) xsum(%2.20lf) ysqsum(%2.20lf) ysum(%2.20lf)\n", 
	       nsamples, xsqsum, xsum, ysqsum, ysum);

    strads_msg(ERR, "##### nsamples*xsqsum(%2.20lf) xsum*xsum(%2.20lf) nsamples*ysqsum(%2.20lf) ysum*ysum(%2.20lf)\n", 
	       nsamples*xsqsum, xsum*xsum, nsamples*ysqsum, ysum*ysum);

    strads_msg(ERR, "##### sqrt(..) %2.20lf sqrt(..) %2.20lf\n", 
	       sqrt(nsamples*xsqsum - (xsum)*(xsum)),  
	       sqrt(nsamples*ysqsum - (ysum)*(ysum)));	          

    xiter = xcol.begin();
    yiter = ycol.begin();
  
    for(; xiter != xcol.end(); xiter++){
      strads_msg(ERR, "X[%ld] = %lf\n", xiter->first, xiter->second);
    }

    for(; yiter != ycol.end(); yiter++){
      strads_msg(ERR, "Y[%ld] = %lf\n", yiter->first, yiter->second);
    }

  }

  assert(divide != 0.0);
  corrcoef = 1.0*(nsamples*xysum-(xsum*ysum))/divide;
  return fabs(corrcoef);
}


#endif





























































#if 0 


#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
//#include <unordered_map>


/* return a square matrix features by features */
double getpstar(int *featurelist, int nfeatures, int nsamples, problemctx *cfg){
  double max;
  int row, col, i;


  //printf("Feature List \n");
  //for(int k=0; k<nfeatures; k++){
  //  printf(" %d ", featurelist[k]);
  //}

  gsl_matrix *A = gsl_matrix_alloc(nsamples, nfeatures);
  gsl_matrix *AT = gsl_matrix_alloc(nfeatures, nsamples);
  gsl_matrix *ATA = gsl_matrix_alloc(nfeatures, nfeatures);

  /* re structure features into one matrix A for GSL */
  for(col=0; col<nfeatures; col++){
    for(row=0; row<nsamples; row++){
      gsl_matrix_set(A, row, col, cfg->X[row][featurelist[col]]);
    }
  }

  /* get AT*A */
  gsl_matrix_transpose_memcpy(AT, A);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, AT, A, 0, ATA);

  gsl_vector *eval = gsl_vector_alloc(nfeatures);
  gsl_matrix *evec = gsl_matrix_alloc(nfeatures, nfeatures);
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(nfeatures);
  gsl_eigen_symmv(ATA, eval, evec, w);
  gsl_eigen_symmv_free(w);
  gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_DESC);

  max = gsl_vector_get(eval, 0);
  for(i=0; i<nfeatures; i++){
    double eval_i = gsl_vector_get(eval, i);
    strads_msg(INF, "Eigen Value [%g]\n", eval_i);
  }

  gsl_vector_free(eval);
  gsl_matrix_free(AT);
  gsl_matrix_free(ATA);
  gsl_matrix_free(A);
  /* get the maximum magnitude of (AT*A) */
  return max;
}



/* calculate Pearson correlation coeff of x, y coordinate of n samples. 
 * If avail flag == 1, Parameter xsum, xsqsum, double ysum, double ysqsum should contain 
 * right values
 * If avail flage != 1, the function calculate params on the fly 
 */
double getcorr_pair(double *x, double *y, int nsamples, double pxsum, double pxsqsum, double pysum, double pysqsum, int avail){
  int i;
  double xysum=0, xsum=0, xsqsum=0, ysum=0, ysqsum=0;
  double corrcoef=0;
  
  if(avail == 1){
    for(i=0; i<nsamples; i++){
      xysum  += x[i]*y[i];
    }    
    xsum = pxsum;
    ysum = pysum;
    xsqsum = pxsqsum;
    ysqsum = pysqsum;

  }else{    
    for(i=0; i<nsamples; i++){
      xysum  += x[i]*y[i];      
      xsqsum += x[i]*x[i];
      xsum   += x[i];
      ysqsum += y[i]*y[i];
      ysum   += y[i];
    }
  }

  corrcoef = 1.0*(nsamples*xysum-(xsum*ysum))/(sqrt(nsamples*xsqsum - (xsum)*(xsum))*(sqrt(nsamples*ysqsum - (ysum)*(ysum))));
  //printf("correlation coef: [%lf]\n", corrcoef);
  return fabs(corrcoef);
}

#endif 
