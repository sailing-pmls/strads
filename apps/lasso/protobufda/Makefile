runda: ./lasso.proto
	protoc -I./ --cpp_out=./ ./lasso.proto
	cp lasso.pb.cc ../lasso.pb.cpp
	cp lasso.pb.h ../lasso.pb.hpp


clean:
	rm -rf $(LASSO_BIN)/lassoll


.PHONY: lassoll clean
