#ifndef _GETCORR_HPP_
#define _GETCORR_HPP_


#include <stdio.h>
#include <unordered_map>
#include "strads/ds/spmat.hpp"

/* calculate Pearson correlation coeff of x, y coordinate of n samples. 
 * If avail flag == 1, Parameter xsum, xsqsum, double ysum, double ysqsum should contain 
 * right values
 * If avail flage != 1, the function calculate params on the fly 
 */
//double getcorr_pair(double *, double *, int, double, double, double, double, int );
//double getpstar(int *featurelist, int nfeatures, int nsamples, problemctx *cfg);

double getcorr_pair_sparsem(std::unordered_map<long unsigned int, double> &xcol, std::unordered_map<long unsigned int, double> &ycol, long unsigned int nsamples, double pxsum, double pxsqsum, double pysum, double pysqsum, int avail);


double getcorr_pair_sparse_vmat(spmat_vector &xcol, spmat_vector &ycol, uint64_t nsamples, double pxsum, double pxsqsum, double pysum, double pysqsum, int avail, uint64_t xcol_id, uint64_t ycol_id);

#endif 
