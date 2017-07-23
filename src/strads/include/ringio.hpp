# ifndef RING_IO_H
# define RING_IO_H

#include <strads/include/common.hpp>
#include <strads/include/indepds.hpp>
#include <strads/util/utility.hpp>
#include <queue>
#include <glog/logging.h>
#include <thread>
#include <assert.h>
#include <unistd.h>

#define FBUFSIZE (4*1024*1024)
#define EXIT_RELAY (0x157)
#define MAX_PEND_MSG (1000)

typedef struct{
	unsigned long seqno;
	unsigned long  buflen;  // real size of valid data
	//	char buf[FBUFSIZE];
}fspacket;

typedef struct{
	void *ptr;
	int len;
}pendjob;

// ring IO class with the coordinator and worker nodes 
class ringio{
public:
	ringio(sharedctx *pctx, std::string ifn):
		ctx(pctx), fn(ifn), stoprelay(false), seqno(0),rcvmsg(0), rline(0)
	{
		strads_msg(INF, " RINGIO CONSTRUCTOR  at rank (%d) with %s fn ", pctx->rank, fn.c_str());
		if(ctx->m_mrole  == mrole_worker){
			relayer = std::thread(&ringio::relayer_async, this);		
		} else if (ctx->m_mrole  == mrole_coordinator){
			relayer = std::thread(&ringio::cdthread, this);					
		} else if (ctx->m_mrole  == mrole_scheduler){
			// do nothing temporarily 
		}		
		// if master: create two threads: one for reading file line and flow it along the ring
		//                                one for receiving and dropping it from the ring 	   		
		// if worker : create one thread: one that reads ring port, copies it into a new packet, puch the new packet into queue,
		//                                         and forwards // received packet to next neighbor 
		// worker thread will consume the packet in the queue through this rigio.gets();
		starttime = timenow();
	}
	
	char *getline(long unsigned int &linenum){
		char *pt(nullptr);
		while(1){
			que_mutex.lock();
			if(que.empty()){				
			} else {
				pt = que.front();
				que.pop();
				que_mutex.unlock();
				break;				
			}
			que_mutex.unlock();			
		}
		return pt;		
	}
	
	char *getline(char *buffer, unsigned long buflen){
		char *pt(nullptr);		

		while(1){
			que_mutex.lock();
			if(que.empty()){				
				que_mutex.unlock();			
				// do nothing waiting for input line from the ring 
			} else {
				pt = que.front();
				fspacket *pkt = (fspacket *)pt;
				rline++;
				if(pkt->buflen > 0){
					size_t payloadsize = pkt->buflen ; // including +1 for \0 
					assert(buflen > payloadsize);
					char *msg = &pt[sizeof(fspacket)];				       
					memcpy((void *)buffer, (void *)msg, pkt->buflen);
					free(pt);
					pt = buffer;					
				} else {
					pt = NULL;
					assert(pkt->buflen == 0); // end of line signal
					flag_mutex.lock();
					stoprelay = true; // if end of line arrives, set stoprelay to true and quit relay thread      	       
					flag_mutex.unlock();
					strads_msg(INF, " WORKER  receive ringio stopping message rline(%lu) \n", rline);
				}					
				que.pop();
				if(pkt->buflen == 0){
					assert(que.empty());
				}
				que_mutex.unlock();
				break;				
			}
		}	       		
		return pt;		
	}

	
     	void relayer_async(void){		
		while(1){
			void *recv = NULL;
			int len=-1;
			recv = ctx->ring_asyncrecv_aux(&len);
			if(recv != NULL){
				fspacket *pkt = (fspacket *)recv;
				//strads_msg(OUT, " @@@@@ Rank (%d) got message seq no (%ld)  pkt->seqno (%lu)\n", ctx->rank, seqno, pkt->seqno);
				assert(seqno == pkt->seqno);								
				if(pkt->buflen != 0){ // pkt->buflen == 0 --> end of ring file io 
					char *crecv = (char *)recv;
					char *line = (char *)(&crecv[sizeof(fspacket)]);
					#if 0
					if((strlen(line)+1) != pkt->buflen){
						strads_msg(OUT, " (strlen(line)+1) : %lu , pkt->buflen: %lu msg total len %d seq(%lu)",
							   (strlen(line)+1), pkt->buflen, len, seqno);
						strads_msg(OUT, "  : %lu", (strlen(line)+1)); 
					}
					#endif					
					assert((strlen(line)+1) == pkt->buflen); 
				}				
				rcvmsg++;
				seqno++;			       
				while(1){				
					void *ret = ctx->ring_async_send_aux(recv, len);			     
					if(ret != NULL){
						break;
					}
				}
				que_mutex.lock();
				que.push((char *)pkt);				
				que_mutex.unlock();				
			}
			if(stoprelay == true) // if worker finish reading all input, stop relay thread 
				break;
		} // while(1)		
		return; // join the main thread
	}

     	void cdthread(void){		
		while(1){
			void *recv = NULL;
			int len=-1;
			recv = ctx->ring_asyncrecv_aux(&len);
			if(recv != NULL){
				fspacket *pkt = (fspacket *)recv;
				if(seqno % 1000000 == 0)
					strads_msg(OUT, " Coordinator Rank (%d) process message seq no (%ld) pkt-seqno(%lu) pkt-buflen(%lu) elaptime sec : %lf\n",
						   ctx->rank, seqno, pkt->seqno, pkt->buflen, (timenow() - starttime)/1000000.0);

				assert(seqno == pkt->seqno);				
				rcvmsg++;
				seqno++;				
				free((char *)recv);
				if(pkt->buflen == 0){ // if all lines are circulated, stop set stoprelay to true and quit cd thread 
					strads_msg(INF, " ring CD thread at Coordinator rank(%d) finish ringio seqno(%ld) pkt-seqno(%lu) pkt-buflen(%lu)\n",
						   ctx->rank, seqno, pkt->seqno, pkt->buflen);
					flag_mutex.lock();
					stoprelay = true;
					flag_mutex.unlock();
				}
			}
			if(stoprelay == true){
				break;
			}
		}		
		return; // join the main thread
	}


	void reader(void){
	
		FILE *fp;
		fp = fopen(fn.c_str(), "r");
		assert(fp != NULL);

		char *chbuffer = (char *)calloc(1024*1024*2, sizeof(char));
		uint64_t nzprogress=0;

		fgets(chbuffer, 1024*1024*2-1, fp); // for skipping the first two lines (header of MMT file) 
		fgets(chbuffer, 1024*1024*2-1, fp);
		
		while(fgets(chbuffer, 1024*1024*2-1, fp)){
			size_t payloadsize = strlen(chbuffer)+1; 
			assert(payloadsize > 1); // strlen of input string should be greater than 0
			size_t blen = payloadsize+sizeof(fspacket); // fspacket : header part		       
			char *buf = (char *)calloc(blen, 1);
			fspacket *pkt = (fspacket *)buf;
			pkt->seqno = nzprogress;
			pkt->buflen  = payloadsize ;
			memcpy((void *)&buf[sizeof(fspacket)], chbuffer, payloadsize);
			assert(pkt->seqno == nzprogress);
			while(1){				
			      void *ret = ctx->ring_async_send_aux(buf, blen);			     
			      if(ret != NULL){
				      if(nzprogress < 10 or nzprogress > 19990)
					      strads_msg(INF, "[io4worker] Coordinator (%d) read seqno %lu  pkt->buflen(%lu)  blen(%lu), pkt->seqno (%lu)\n",
							 ctx->rank, nzprogress, pkt->buflen, blen, pkt->seqno); 
				      free(buf);
				      break;
			      }
			}			
			nzprogress++; // sequence number 
		}
		strads_msg(OUT, " @@@@@@@@@@@@@@@@@ FINISH READING at coordinator total msg including terminating :  %lu  \n", nzprogress);
		// send terminating message along the ring 		
		size_t payloadsize = 0; 
		size_t blen = payloadsize+sizeof(fspacket); // fspacket : header part		       
		char *buf = (char *)calloc(blen, 1);
		fspacket *pkt = (fspacket *)buf;
		pkt->seqno = nzprogress;
		pkt->buflen  = payloadsize ;		
		while(1){				
			void *ret = ctx->ring_async_send_aux(buf, blen);
			if(ret != NULL){
				free(buf);
				break;
			}
		}				       
	}

       	~ringio()
	{		
		while(1){ // if destructor is called before finishing input lines, wait for completion 
			flag_mutex.lock();
			if(stoprelay == true){
				flag_mutex.unlock();
				break;
			}
			flag_mutex.unlock();			
		}		
		relayer.join(); // stop relayer, no more messages 
		while(1){			
			que_mutex.lock();
			if(que.empty()){
				que_mutex.unlock();
				break;
			}
			que_mutex.unlock();
		}		
		strads_msg(OUT, "[RING I/O] RANK (%d) finishes ring io and calls destructore (queue.size() : %lu) IO time: %lf \n",
			   ctx->rank, que.size(), (timenow() - starttime)/1000000.0);		
		assert(que.size() == 0);
	}

	sharedctx *ctx;
	std::queue<char *> que; 
	std::mutex que_mutex;
	std::string fn;
	std::thread relayer;
	bool stoprelay;
	std::mutex flag_mutex;
	unsigned long seqno;
	unsigned long rcvmsg;
	unsigned long rline;       		

	unsigned long starttime; // when constructore is creted, set this 
};

// ring IO class with the coordinator and schedulers nodes
class ringio4scheduler{
public:
	ringio4scheduler(sharedctx *pctx, std::string ifn):
		ctx(pctx), fn(ifn), stoprelay(false), seqno(0),rcvmsg(0), rline(0)
	{

		if(ctx->m_mrole  == mrole_worker){
			assert(0);
		} else if (ctx->m_mrole  == mrole_coordinator){
			relayer = std::thread(&ringio4scheduler::sinkthread, this);					
		} else if (ctx->m_mrole  == mrole_scheduler){
			relayer = std::thread(&ringio4scheduler::relayer_async, this);		
		}		
		// if master: create two threads: one for reading file line and flow it along the ring
		//                                one for receiving and dropping it from the ring 	   		
		// if worker : create one thread: one that reads ring port, copies it into a new packet, puch the new packet into queue,
		//                                         and forwards // received packet to next neighbor 
		// worker thread will consume the packet in the queue through this rigio.gets();
	}
	
	char *getline(long unsigned int &linenum){
		char *pt(nullptr);
		while(1){
			que_mutex.lock();
			if(que.empty()){				
			} else {
				pt = que.front();
				que.pop();
				que_mutex.unlock();
				break;				
			}
			que_mutex.unlock();			
		}
		return pt;		
	}

	char *getline(char *buffer, unsigned long buflen){
		char *pt(nullptr);		

		while(1){
			que_mutex.lock();
			if(que.empty()){				
				que_mutex.unlock();			
				// do nothing waiting for input line from the ring 
			} else {
				pt = que.front();
				fspacket *pkt = (fspacket *)pt;
				rline++;
				if(pkt->buflen > 0){
					size_t payloadsize = pkt->buflen ; // including +1 for \0 
					assert(buflen > payloadsize);
					char *msg = &pt[sizeof(fspacket)];				       
					memcpy((void *)buffer, (void *)msg, pkt->buflen);
					free(pt);
					pt = buffer;					
				} else {
					pt = NULL;
					assert(pkt->buflen == 0); // end of line signal
					stoprelay = true; // if end of line arrives, set stoprelay to true and quit relay thread      	       
					strads_msg(INF, " WORKER  receive ringio stopping message rline(%lu) \n", rline);
				}					

				que.pop();
				if(pkt->buflen == 0){
					assert(que.empty());
				}

				que_mutex.unlock();
				break;				
			}
		}	       		
		return pt;		
	}

	void relayer_async(void){		
		while(1){
			void *recv = NULL;
			int len=-1;
			recv = ctx->ring_asyncrecv_aux(&len, 2);
			if(recv != NULL){
				fspacket *pkt = (fspacket *)recv;
				strads_msg(INF, " @@@@@ Rank (%d) got message seq no (%ld)  pkt->seqno (%lu)\n", ctx->rank, seqno, pkt->seqno);
				assert(seqno == pkt->seqno);								
				if(pkt->buflen != 0){ // pkt->buflen == 0 --> end of ring file io 
					char *crecv = (char *)recv;
					char *line = (char *)(&crecv[sizeof(fspacket)]);
					#if 0
					if((strlen(line)+1) != pkt->buflen){
						strads_msg(OUT, " (strlen(line)+1) : %lu , pkt->buflen: %lu msg total len %d seq(%lu)",
							   (strlen(line)+1), pkt->buflen, len, seqno);
						strads_msg(OUT, "  : %lu", (strlen(line)+1)); 
					}
					#endif					
					assert((strlen(line)+1) == pkt->buflen); 
				}				
				//long etime = timenow();
				// check if exit message,
				rcvmsg++;
				seqno++;			       
				while(1){				
					void *ret = ctx->ring_async_send_aux(recv, len, 2);			     
					if(ret != NULL){
						break;
					}
				}

				que_mutex.lock();
				que.push((char *)pkt);				
				que_mutex.unlock();
				// don't free here. packet buffer will be release by user routine
			}
			if(stoprelay == true) // if worker finish reading all input, stop relay thread 
				break;
		} // end of while(1)		
		return; // join the main thread
	}

     	void sinkthread(void){		
		while(1){
			void *recv = NULL;
			int len=-1;
			recv = ctx->ring_asyncrecv_aux(&len, 2);
			if(recv != NULL){
				if(seqno % 1000000 == 0)
					strads_msg(OUT, " Coordinator Rank (%d) sink message seq no (%ld) \n", ctx->rank, seqno);
				fspacket *pkt = (fspacket *)recv;
				assert(seqno == pkt->seqno);				
				rcvmsg++;
				seqno++;				
				free((char *)recv);
				if(pkt->buflen == 0){ // if all lines are circulated, stop set stoprelay to true and quit sink thread 
					stoprelay = true;
				}
			}
			if(stoprelay == true)
				break;
		}		
		return; // join the main thread
	}

	void reader(void){	
		FILE *fp;
		fp = fopen(fn.c_str(), "r");
		assert(fp != NULL);
		char *chbuffer = (char *)calloc(1024*1024*2, sizeof(char));
		uint64_t nzprogress=0;
		fgets(chbuffer, 1024*1024*2-1, fp); // for skipping the first two lines (header of MMT file) 
		fgets(chbuffer, 1024*1024*2-1, fp);
		
		while(fgets(chbuffer, 1024*1024*2-1, fp)){
			size_t payloadsize = strlen(chbuffer)+1; 
			assert(payloadsize > 1); // strlen of input string should be greater than 0
			size_t blen = payloadsize+sizeof(fspacket); // fspacket : header part		       
			char *buf = (char *)calloc(blen, 1);
			fspacket *pkt = (fspacket *)buf;
			pkt->seqno = nzprogress;
			pkt->buflen  = payloadsize ;
			memcpy((void *)&buf[sizeof(fspacket)], chbuffer, payloadsize);
			assert(pkt->seqno == nzprogress);
			while(1){				
				void *ret = ctx->ring_async_send_aux(buf, blen, 2);			     
			      if(ret != NULL){
				      if(nzprogress < 100)
					      strads_msg(INF, " [io4sched]Coordinator (%d) read seqno %lu  pkt->buflen(%lu)  blen(%lu), pkt->seqno (%lu)\n",
							 ctx->rank, nzprogress, pkt->buflen, blen, pkt->seqno); 
				      free(buf);
				      break;
			      }
			}			
			nzprogress++; // sequence number 
		}
		strads_msg(OUT, " Coordinator finish file reading. sent msg count : %lu  \n", nzprogress);
		// send terminating message along the ring 		
		size_t payloadsize = 0; 
		size_t blen = payloadsize+sizeof(fspacket); // fspacket : header part		       
		char *buf = (char *)calloc(blen, 1);
		fspacket *pkt = (fspacket *)buf;
		pkt->seqno = nzprogress;
		pkt->buflen  = payloadsize ;		
		while(1){				
			void *ret = ctx->ring_async_send_aux(buf, blen, 2);
			if(ret != NULL){
				free(buf);
				break;
			}
		}				       
	}

       	~ringio4scheduler()
	{		
		while(stoprelay != true); // if destructor is called before finishing input lines, wait for completion 
		relayer.join(); // stop relayer, no more messages 
		while(1){			
			que_mutex.lock();
			if(que.empty()){
				que_mutex.unlock();
				break;
			}
			que_mutex.unlock();
		}		
		strads_msg(INF, " at Destructor RANK (%d)  queue.size() : %lu \n",
			   ctx->rank, que.size());		
		assert(que.size() == 0);
	}
	sharedctx *ctx;
	std::queue<char *> que; 
	std::mutex que_mutex;
	std::string fn;
	std::thread relayer;
	bool stoprelay;
	unsigned long seqno;
	unsigned long rcvmsg;
	unsigned long rline;       		
};












#endif
