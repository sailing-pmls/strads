#pragma once 

#include <unordered_map>
#include "common.hpp"
#include "ds/dshard.hpp"

//void *user_aggregator(dshardctx *dshard, void *userdata, void *ctx);
void *user_aggregator(dshardctx *dshardcoeff, std::unordered_map<int64_t, idmvals_pair *>&map, void *ctx);

void *user_aggregator(dshardctx *dshardcoeff,dshardctx *dshardYcord, std::unordered_map<int64_t, idmvals_pair *>&map, void *ctx);

void *user_update_parameter(dshardctx *dshardA, dshardctx *dshardRes, void *userdata, void *ctx);
void *user_update_parameter(dshardctx *dshardA, dshardctx *dshardRes, dshardctx *dummy, void *userdata, void *ctx);
void *user_update_status(dshardctx *dshardA, dshardctx *dshardB, void *userdata, void *ctx);

double user_get_object(dshardctx *dshardA, dshardctx *dshardB, void *ctx);

double user_get_object(dshardctx *dshardA, dshardctx *dshardWeight, void *ctx, idval_pair *idvalp, int64_t len);

double user_get_object_server(dshardctx *dshardcoeff, double psum, void *ctx);
