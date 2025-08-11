# AMX Inner Product Project Makefile with HNSWLIB and Enhanced AMX
# ================================================================

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
AMX_MT_ENHANCED_SRC = AMXInnerProductBF16PtrMTEnhanced.cpp
HNSWLIB_SRC = HNSWLIBInnerProductPtr.cpp

# Object files for different configurations
CORE_OBJS = $(SCALAR_SRC:.cpp=.o) $(AMX_SRC:.cpp=.o) $(AMX_MT_SRC:.cpp=.o) $(AMX_MT_ENHANCED_SRC:.cpp=.o)
HNSWLIB_OBJS = $(CORE_OBJS) $(HNSWLIB_SRC:.cpp=.o)

# Programs
ENHANCED_PROGS = comprehensive_testing
HNSWLIB_PROGS = comprehensive_testing_with_hnswlib
ALL_PROGS = $(ENHANCED_PROGS) $(HNSWLIB_PROGS)

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

# Enhanced AMX programs (without HNSWLIB)
$(ENHANCED_PROGS): %: %.o $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) $(ARROW_CFLAGS) $^ $(ARROW_LIBS) -o $@

# HNSWLIB programs (with HNSWLIB)
$(HNSWLIB_PROGS): %: %.o $(HNSWLIB_OBJS)
	$(CXX) $(CXXFLAGS) $(ARROW_CFLAGS) $(HNSWLIB_CFLAGS) $^ $(ARROW_LIBS) $(HNSWLIB_LIBS) -o $@

# Object file rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Arrow-dependent object files (enhanced version without HNSWLIB)
comprehensive_testing.o: comprehensive_testing.cpp
	$(CXX) $(CXXFLAGS) $(SUPPRESS) $(ARROW_CFLAGS) -c $< -o $@

# Arrow + HNSWLIB dependent object files  
comprehensive_testing_with_hnswlib.o: comprehensive_testing_with_hnswlib.cpp
	$(CXX) $(CXXFLAGS) $(SUPPRESS) $(ARROW_CFLAGS) $(HNSWLIB_CFLAGS) -c $< -o $@

# AMX object files (suppress warnings) - includes enhanced version
$(AMX_SRC:.cpp=.o) $(AMX_MT_SRC:.cpp=.o) $(AMX_MT_ENHANCED_SRC:.cpp=.o): %.o: %.cpp
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
test-large: comprehensive_testing
	@echo "=== Running Enhanced Large-Scale Tests ==="
	@echo "Note: Requires dataset at /mnt/ceph/district9/dataset/openai/openai_large_5m/"
	@echo "Features: Enhanced AMX multithreading with detailed timing analysis"
	LD_LIBRARY_PATH=/usr/local/lib:$$LD_LIBRARY_PATH ./comprehensive_testing

test-hnswlib: comprehensive_testing_with_hnswlib
	@echo "=== Running HNSWLIB Comparison Tests ==="
	@echo "Note: Requires dataset at /mnt/ceph/district9/dataset/openai/openai_large_5m/"
	@echo "Features: AMX vs HNSWLIB performance comparison"
	LD_LIBRARY_PATH=/usr/local/lib:$$LD_LIBRARY_PATH ./comprehensive_testing_with_hnswlib

test-enhanced: comprehensive_testing
	@echo "=== Running Enhanced AMX Analysis ==="
	@echo "Note: Focuses on detailed timing breakdown of multithreaded AMX"
	LD_LIBRARY_PATH=/usr/local/lib:$$LD_LIBRARY_PATH ./comprehensive_testing

test-all: test test-large test-hnswlib test-enhanced

# System checks
check:
	@echo "=== System Check ==="
	@echo "Hardware threads: $$(nproc)"
	@echo -n "AMX support: "; grep -q amx /proc/cpuinfo && echo "✅" || echo "❌"
	@echo -n "Arrow/Parquet: "; pkg-config --exists arrow parquet && echo "✅" || echo "❌"
	@echo -n "HNSWLIB: "; test -d hnswlib && echo "✅" || echo "❌"
	@echo -n "OpenMP: "; echo '#include <omp.h>' | $(CXX) -x c++ -fopenmp -E - >/dev/null 2>&1 && echo "✅" || echo "❌"
	@echo -n "Enhanced AMX: "; test -f AMXInnerProductBF16PtrMTEnhanced.h && echo "✅" || echo "❌"
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

setup-enhanced:
	@echo "=== Checking Enhanced AMX Files ==="
	@if [ ! -f AMXInnerProductBF16PtrMTEnhanced.h ]; then \
		echo "❌ AMXInnerProductBF16PtrMTEnhanced.h missing"; \
		echo "   Please create this file with the enhanced AMX implementation"; \
		exit 1; \
	fi
	@if [ ! -f AMXInnerProductBF16PtrMTEnhanced.cpp ]; then \
		echo "❌ AMXInnerProductBF16PtrMTEnhanced.cpp missing"; \
		echo "   Please create this file with the enhanced AMX implementation"; \
		exit 1; \
	fi
	@echo "✅ Enhanced AMX files found"

# Utilities
clean:
	rm -f *.o $(ALL_PROGS)

distclean: clean
	rm -f *~ *.bak core

install-deps:
	sudo apt-get update && sudo apt-get install -y build-essential libomp-dev libarrow-dev libparquet-dev pkg-config git

setup: install-deps setup-hnswlib setup-enhanced
	@echo "=== Setup Complete ==="
	@echo "✅ All dependencies installed and configured"

help:
	@echo "AMX Inner Product Project with Enhanced AMX and HNSWLIB"
	@echo "======================================================="
	@echo ""
	@echo "Setup targets:"
	@echo "  setup            Install all dependencies and setup everything"
	@echo "  setup-hnswlib    Clone HNSWLIB repository"
	@echo "  setup-enhanced   Check enhanced AMX implementation files"
	@echo "  install-deps     Install system dependencies (Ubuntu/Debian)"
	@echo ""
	@echo "Build targets:"
	@echo "  all              Build all programs (default)"
	@echo "  debug            Build with debug flags"
	@echo "  release          Build optimized release"
	@echo ""
	@echo "Test targets:"
	@echo "  test             Run basic tests (no dataset required)"
	@echo "  test-large       Run enhanced large-scale tests (with detailed timing)"
	@echo "  test-hnswlib     Run HNSWLIB comparison tests"
	@echo "  test-enhanced    Run enhanced AMX analysis (detailed breakdown)"
	@echo "  test-all         Run all tests"
	@echo ""
	@echo "Utilities:"
	@echo "  check            Check system capabilities and dependencies"
	@echo "  check-deps       Check if all dependencies are available"
	@echo "  clean            Remove build artifacts"
	@echo "  help             Show this message"
	@echo ""
	@echo "Individual programs:"
	@echo "  comprehensive_testing                AMX with Enhanced Timing Analysis"
	@echo "  comprehensive_testing_with_hnswlib   AMX + HNSWLIB Comparison"
	@echo ""
	@echo "Key Features:"
	@echo "  Enhanced AMX Implementation:"
	@echo "    - Detailed per-thread timing breakdown"
	@echo "    - Load balancing analysis"
	@echo "    - Memory efficiency metrics"
	@echo "    - Threading overhead analysis"
	@echo "    - Performance bottleneck identification"
	@echo ""
	@echo "  HNSWLIB Integration:"
	@echo "    - Head-to-head performance comparison"
	@echo "    - Memory usage analysis"
	@echo "    - Accuracy vs speed trade-offs"
	@echo ""
	@echo "Dependencies:"
	@echo "  - Arrow/Parquet libraries"
	@echo "  - HNSWLIB (header-only, auto-downloaded)"
	@echo "  - Enhanced AMX implementation files"
	@echo "  - OpenMP for multithreading"
	@echo "  - AMX-capable CPU for specialized functions"

# Special targets
.PHONY: setup setup-hnswlib setup-enhanced test-enhanced

# File dependencies for better incremental builds
comprehensive_testing: comprehensive_testing.o $(CORE_OBJS)
comprehensive_testing_with_hnswlib: comprehensive_testing_with_hnswlib.o $(HNSWLIB_OBJS)

# Dependency tracking
comprehensive_testing.o: AMXInnerProductBF16PtrMTEnhanced.h
comprehensive_testing_with_hnswlib.o: HNSWLIBInnerProductPtr.h
$(AMX_MT_ENHANCED_SRC:.cpp=.o): AMXInnerProductBF16PtrMTEnhanced.h AMXCommon.h
