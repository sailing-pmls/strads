#pragma once 

#include "common.hpp"
#include <strads/netdriver/zmq/zmq-common.hpp>

class channels{
public: 
  channels(sharedctx *ctx){    
    auto pr = ctx->star_recvportmap.begin();
    _ringport *rport = pr->second;
    context *recv_ctx = rport->ctx;
    auto ps = ctx->star_sendportmap.begin();
    _ringport *sport = ps->second;
    context *send_ctx = sport->ctx;
    m_sendctx = send_ctx;
    m_recvctx = recv_ctx;
  }

  int push_entry_outq(void *data, unsigned long int len){ m_sendctx->push_entry_outq(data, len); }


private:
  context *m_sendctx;
  context *m_recvctx;
};

