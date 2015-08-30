LASSO_DIR := $(shell readlink $(dir $(lastword $(MAKEFILE_LIST))) -f)
STRADS_ROOT = $(LASSO_DIR)/../../

include $(STRADS_ROOT)/defns.mk

LASSO_LDFLAGS = -Wl,-rpath \
          -pthread \
	-lpthread \
          -lglog \
          -lgflags \
	-lm \
          -lprotobuf

LASSO_SRC = $(wildcard $(LASSO_DIR)/*.cpp)
LASSO_HDR = $(wildcard $(LASSO_DIR)/*.hpp)

LASSO_BIN = $(LASSO_DIR)/bin
LASSO = $(LASSO_BIN)/svm-dual

svm-dist: $(LASSO)

$(LASSO): $(LASSO_SRC) $(STRADS_STRADS_LIB) $(LASSO_HDR)
	mkdir -p $(LASSO_BIN)
	$(STRADS_CXX) $(STRADS_CXXFLAGS) $(STRADS_INCFLAGS) -DNO_WEIGHT_SAMPLING -DNOREVOKING -DNO_INTERFERENCE_CHECK $^ \
	$(STRADS_LDFLAGS) $(LASSO_LDFLAGS) -I./ -o $@

da: ./protobufda/lasso.proto
	protoc -I./protobufda/ --cpp_out=./protobufda ./protobufda/lasso.proto

clean:
	rm -rf $(LASSO_BIN)/cdsolver

.PHONY: cdsolver clean
