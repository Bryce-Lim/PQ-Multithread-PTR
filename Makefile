# AMX Inner Product Project Makefile with HNSWLIB
# ===============================================

# Compiler and flags
CXX = g++
BASE_FLAGS = -std=c++17 -Wall -Wextra -pthread
ARCH_FLAGS = -mavx512f -mavx512bf16 -mavx512bw -mavx512vl -mamx-tile -mamx-int8 -mamx-bf16
OPT_FLAGS = -O3 -fopenmp -flax-vector-conversions
SUPPRESS = -Wno-missing-field-initializers -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable

CXXFLAGS = $(BASE_FLAGS) $(ARCH_FLAGS) $(OPT_FLAGS) $(SUPPRESS)
DEBUG_FLAGS = -g -O0 -DDEBUG
RELEASE_FLAGS = -O3 -DNDEBUG -march=native

# Arrow/Parquet support
ARROW_CFLAGS = $(shell pkg-config --cflags arrow parquet 2>/dev/null)
ARROW_LIBS = $(shell pkg-config --libs arrow parquet 2>/dev/null || echo "-larrow -lparquet")

# HNSWLIB support
HNSWLIB_CFLAGS = -Ihnswlib
HNSWLIB_LIBS = 

# Core sources
SCALAR_SRC = ScalarInnerProduct.cpp
AMX_SRC = AMXInnerProductBF16Ptr.cpp  
AMX_MT_SRC = AMXInnerProductBF16PtrMT.cpp
HNSWLIB_SRC = HNSWLIBInnerProductPtr.cpp
CORE_OBJS = $(SCALAR_SRC:.cpp=.o) $(AMX_SRC:.cpp=.o) $(AMX_MT_SRC:.cpp=.o) $(HNSWLIB_SRC:.cpp=.o)

# Programs
ARROW_PROGS = comprehensive_testing comprehensive_testing_with_hnswlib
ALL_PROGS = $(ARROW_PROGS)

# Default target
.PHONY: all clean debug release test help check-deps
.DEFAULT_GOAL = all

all: check-deps $(ALL_PROGS)

# Check dependencies
check-deps:
	@echo "=== Checking Dependencies ==="
	@pkg-config --exists arrow parquet || (echo "ERROR: Install Arrow/Parquet: sudo apt-get install libarrow-dev libparquet-dev" && exit 1)
#	@test -d hnswlib || (echo "ERROR: HNSWLIB not found. Please clone: git clone https://github.com/nmslib/hnswlib.git" && exit 1)
	@echo "✅ All dependencies found"

# Arrow-dependent programs
$(ARROW_PROGS): %: %.o $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) $(ARROW_CFLAGS) $(HNSWLIB_CFLAGS) $^ $(ARROW_LIBS) $(HNSWLIB_LIBS) -o $@

# Object file rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(HNSWLIB_CFLAGS) -c $< -o $@

# Arrow-dependent object files
$(patsubst %.cpp,%.o,$(wildcard *testing*.cpp)): %.o: %.cpp
	$(CXX) $(CXXFLAGS) $(SUPPRESS) $(ARROW_CFLAGS) $(HNSWLIB_CFLAGS) -c $< -o $@

# AMX object files (suppress warnings)
$(AMX_SRC:.cpp=.o) $(AMX_MT_SRC:.cpp=.o): %.o: %.cpp
	$(CXX) $(CXXFLAGS) $(SUPPRESS) -c $< -o $@

# HNSWLIB object files
$(HNSWLIB_SRC:.cpp=.o): %.o: %.cpp
	$(CXX) $(CXXFLAGS) $(SUPPRESS) $(HNSWLIB_CFLAGS) -c $< -o $@

# Build variants
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean all

release: CXXFLAGS += $(RELEASE_FLAGS) 
release: clean all

# Testing targets
test-large: $(ARROW_PROGS)
	@echo "=== Running Large-Scale Tests ==="
	@echo "Note: Requires dataset at /mnt/ceph/district9/dataset/openai/openai_large_5m/"
	LD_LIBRARY_PATH=/usr/local/lib:$$LD_LIBRARY_PATH ./comprehensive_testing

test-hnswlib: comprehensive_testing_with_hnswlib
	@echo "=== Running HNSWLIB Comparison Tests ==="
	@echo "Note: Requires dataset at /mnt/ceph/district9/dataset/openai/openai_large_5m/"
	LD_LIBRARY_PATH=/usr/local/lib:$$LD_LIBRARY_PATH ./comprehensive_testing_with_hnswlib

test-all: test test-large test-hnswlib

# System checks
check:
	@echo "=== System Check ==="
	@echo "Hardware threads: $$(nproc)"
	@echo -n "AMX support: "; grep -q amx /proc/cpuinfo && echo "✅" || echo "❌"
	@echo -n "Arrow/Parquet: "; pkg-config --exists arrow parquet && echo "✅" || echo "❌"
	@echo -n "HNSWLIB: "; test -d hnswlib && echo "✅" || echo "❌"
	@echo -n "OpenMP: "; echo '#include <omp.h>' | $(CXX) -x c++ -fopenmp -E - >/dev/null 2>&1 && echo "✅" || echo "❌"
	@echo -n "Compiler: "; $(CXX) --version | head -1

# Setup dependencies
setup-hnswlib:
	@echo "=== Setting up HNSWLIB ==="
	@if [ ! -d hnswlib ]; then \
		echo "Cloning HNSWLIB..."; \
		git clone https://github.com/nmslib/hnswlib.git; \
		echo "✅ HNSWLIB cloned successfully"; \
	else \
		echo "✅ HNSWLIB already exists"; \
	fi

# Utilities
clean:
	rm -f *.o $(ALL_PROGS)

distclean: clean
	rm -f *~ *.bak core

install-deps:
	sudo apt-get update && sudo apt-get install -y build-essential libomp-dev libarrow-dev libparquet-dev pkg-config git

setup: install-deps setup-hnswlib
	@echo "=== Setup Complete ==="
	@echo "✅ All dependencies installed and configured"

help:
	@echo "AMX Inner Product Project with HNSWLIB"
	@echo "====================================="
	@echo ""
	@echo "Setup targets:"
	@echo "  setup            Install all dependencies and setup HNSWLIB"
	@echo "  setup-hnswlib    Clone HNSWLIB repository"
	@echo "  install-deps     Install system dependencies (Ubuntu/Debian)"
	@echo ""
	@echo "Build targets:"
	@echo "  all              Build all programs (default)"
	@echo "  debug            Build with debug flags"
	@echo "  release          Build optimized release"
	@echo ""
	@echo "Test targets:"
	@echo "  test             Run basic tests (no dataset required)"
	@echo "  test-large       Run large-scale tests (requires dataset)"
	@echo "  test-hnswlib     Run HNSWLIB comparison tests"
	@echo "  test-all         Run all tests"
	@echo ""
	@echo "Utilities:"
	@echo "  check            Check system capabilities"
	@echo "  check-deps       Check if all dependencies are available"
	@echo "  clean            Remove build artifacts"
	@echo "  help             Show this message"
	@echo ""
	@echo "Individual programs:"
	@echo "  comprehensive_testing            Original AMX comparison"
	@echo "  comprehensive_testing_with_hnswlib   AMX + HNSWLIB comparison"
	@echo ""
	@echo "Dependencies:"
	@echo "  - Arrow/Parquet libraries"
	@echo "  - HNSWLIB (header-only, auto-downloaded)"
	@echo "  - OpenMP for multithreading"
	@echo "  - AMX-capable CPU for specialized functions"

# Special targets
.PHONY: setup setup-hnswlib
