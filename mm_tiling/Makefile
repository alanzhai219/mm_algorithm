# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= ${CUDA_HOME}
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

# Common binaries
NVCC            ?= $(shell which nvcc > /dev/null && echo "nvcc" || echo "$(CUDA_BIN_PATH)/nvcc")

# Extra user flags
EXTRA_NVCCFLAGS ?=
EXTRA_LDFLAGS   ?=
#CFLAGS = -O3 -Wall std=gnu99
CFLAGS=-O3 -Wall

# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10 -Wno-deprecated-gpu-targets
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20 -Wno-deprecated-gpu-targets
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70 -Wno-deprecated-gpu-targets
GENCODE_FLAGS   := $(GENCODE_SM70)


# OS-specific build flags
LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
CCFLAGS   := -m64

# OS-architecture specific flags
NVCCFLAGS := -m64 -g

# Common includes and paths for CUDA
INCLUDES      := -I$(CUDA_INC_PATH)

SRCS=$(wildcard *.c)
CPP_SRCS=$(wildcard *.cpp)
CU_SRCS=$(wildcard *.cu)
OBJS=$(SRCS:.c=.o) $(CPP_SRCS:.cpp=.o)  $(CU_SRCS:.cu=.o)
EXE=matrixmul

#Target Rules

$(EXE):$(CU_SRCS)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) $(CU_SRCS) -o $@

clean:
	rm -rf $(OBJS) $(CU_OBJS) $(EXE)
