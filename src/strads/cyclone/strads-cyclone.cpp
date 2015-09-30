#include <strads/include/common.hpp>
#include <strads/netdriver/zmq/zmq-common.hpp>
#include <strads/netdriver/comm.hpp>
#include <strads/ps/strads-ps.hpp>
#include <mpi.h>
#include <zmq.hpp>
#include <string>
#include <vector>
#include <map>
#include <strads/cyclone/strads-cyclone.hpp>

using namespace std;

eCycloneRole Cyclone::role_ = cycloneUnknown;
int Cyclone::serverCount_ = -1;
int Cyclone::rank_ = -1;

map<int, _ringport *>*Cyclone::sendPortMap_ = NULL;
map<int, _ringport *>*Cyclone::recvPortMap_ = NULL;
int Cyclone::maxasync_ = -1;
int Cyclone::clientCount_ = -1;

void Cyclone::cycloneClientBGThread(Cyclone *Obj){
}

void Cyclone::printCycloneConf(void){
  if(role == cycloneClient){
    strads_msg(ERR, "[ Client rank %d ] -- sendmap.size(%ld) recvmap.size(%ld) \n", rank, sendPortMap.size(), recvPortMap.size()); 
  }else if(role == cycloneServer){
    strads_msg(ERR, "[ Server rank %d ] -- sendmap.size(%ld) recvmap.size(%ld) \n", rank, sendPortMap.size(), recvPortMap.size()); 
  }
}

void Cyclone::cycloneServerBGThread(Cyclone *Obj){
}

void *Cyclone::_makeCyclonePacket(void *usrPacket, int usrLen, CyclonePacket *pkt, int sendLen, int srcRank){
  return NULL;
}
void *Cyclone::serialize(std::string &key, std::string &value, int *pLen, int *pServerId){
  return NULL;
}

void Cyclone::deserialize(void *bytes, std::string &key, std::string &value){
}

void Cyclone::cycloneAsyncPutGet(Strads &ctx, std::string &key, std::string &value){

  int len=-1;
  int serverid=-1;
  void *buf = serialize(key, value, &len, &serverid); // convert user bytes into CyclonePacket with meta information 
  CyclonePacket *packet = (CyclonePacket *)calloc(sizeof(CyclonePacket) + len, 1);
  void *ubuf = (void *)((uintptr_t)(packet) + sizeof(CyclonePacket));
  assert((uintptr_t)ubuf % sizeof(long) == 0);
  packet->ubuf = ubuf;
  memcpy(ubuf, buf, len);
  packet->len = sizeof(CyclonePacket) + len;
  packet->src = rank;
  packet->cbtype = cycmd_putgetasync;  // don't forget this 

  auto ps = sendPortMap[serverid];
  _ringport *sport = ps;
}

void Cyclone::cycloneSyncPut(Strads &ctx, std::string &key, std::string &value){

}

void Cyclone::cycloneSyncGet(Strads &ctx, std::string &key, std::string &value){

}
