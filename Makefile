# ==========================================
# Object File Rules
# ==========================================

# Standard object file compilation
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Arrow-dependent object files (enhanced version)
comprehensive_testing.o: comprehensive_testing.cpp
	@echo "Compiling $< with Arrow support..."
	$(CXX) $(CXXFLAGS) $(SUPPRESS) $(ARROW_CFLAGS) -c $< -o $@

# Arrow + HNSWLIB dependent object files  
comprehensive_testing_with_hnswlib.o: comprehensive_testing_with_hnswlib.cpp
	@echo "Compiling $< with Arrow and HNSWLIB support..."
	$(CXX) $(CXXFLAGS) $(SUPPRESS) $(ARROW_CFLAGS) $(HNSWLIB_CFLAGS) -c $< -o $@

# AMX object files with warning suppression
$(AMX_SRC:.cpp=.o) $(AMX_MT_ENHANCED_SRC:.cpp=.o): %.o: %.cpp
	@echo "Compiling AMX implementation $<..."
	$(CXX) $(CXXFLAGS) $(SUPPRESS) -c $< -o $@

# HNSWLIB object files
$(HNSWLIB_SRC:.cpp=.o): %.o: %.cpp
	@echo "Compiling HNSWLIB wrapper $<..."
	$(CXX) $(CXXFLAGS) $(SUPPRESS) $(HNSWLIB_CFLAGS) -c $< -o $@

# ==========================================
# Build Variants
# ==========================================

debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean all
	@echo "Debug build completed"

release: CXXFLAGS += $(RELEASE_FLAGS)
release: clean all
	@echo "Release build completed"

# ==========================================
# Testing Targets
# ==========================================

test-enhanced: comprehensive_testing
	@echo "=== Running Enhanced AMX Analysis ==="
	@echo "Features: Detailed timing breakdown of multithreaded AMX"
	LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./comprehensive_testing

test-large: comprehensive_testing
	@echo "=== Running Large-Scale Performance Tests ==="
	@echo "Note: Requires dataset at $(dataroot)"
	@echo "Features: Enhanced AMX multithreading with comprehensive analysis"
	LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./comprehensive_testing

test-hnswlib: comprehensive_testing_with_hnswlib
	@echo "=== Running AMX vs HNSWLIB Comparison ==="
	@echo "Features: Head-to-head performance and accuracy comparison"
	LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./comprehensive_testing_with_hnswlib

test-all: test-enhanced test-large test-hnswlib
	@echo "All tests completed"

# ==========================================
# System Checks and Dependencies
# ==========================================

check-deps:
	@echo "=== Checking Dependencies ==="
	@pkg-config --exists arrow parquet || (echo "ERROR: Install Arrow/Parquet: sudo apt-get install libarrow-dev libparquet-dev" && exit 1)
	@echo "✅ All dependencies found"

check:
	@echo "=== System Capability Check ==="
	@echo "Hardware threads: $(nproc)"
	@echo -n "AMX support: "; grep -q amx /proc/cpuinfo && echo "✅" || echo "❌"
	@echo -n "Arrow/Parquet: "; pkg-config --exists arrow parquet && echo "✅" || echo "❌"
	@echo -n "HNSWLIB: "; test -d hnswlib && echo "✅" || echo "❌"
	@echo -n "OpenMP: "; echo '#include <omp.h>' | $(CXX) -x c++ -fopenmp -E - >/dev/null 2>&1 && echo "✅" || echo "❌"
	@echo -n "Enhanced AMX: "; test -f AMXInnerProductBF16PtrMTEnhanced.h && echo "✅" || echo "❌"
	@echo "Compiler: $($(CXX) --version | head -1)"

# ==========================================
# Setup and Installation
# ==========================================

setup-hnswlib:
	@echo "=== Setting up HNSWLIB ==="
	@if [ ! -d hnswlib ]; then \
		echo "Cloning HNSWLIB repository..."; \
		git clone https://github.com/nmslib/hnswlib.git; \
		echo "✅ HNSWLIB cloned successfully"; \
	else \
		echo "✅ HNSWLIB already exists"; \
	fi

setup-enhanced:
	@echo "=== Checking Enhanced AMX Implementation ==="
	@if [ ! -f AMXInnerProductBF16PtrMTEnhanced.h ]; then \
		echo "❌ AMXInnerProductBF16PtrMTEnhanced.h missing"; \
		echo "   Please ensure enhanced AMX implementation files are present"; \
		exit 1; \
	fi
	@if [ ! -f AMXInnerProductBF16PtrMTEnhanced.cpp ]; then \
		echo "❌ AMXInnerProductBF16PtrMTEnhanced.cpp missing"; \
		echo "   Please ensure enhanced AMX implementation files are present"; \
		exit 1; \
	fi
	@echo "✅ Enhanced AMX implementation files found"

install-deps:
	@echo "=== Installing System Dependencies ==="
	sudo apt-get update && sudo apt-get install -y \
		build-essential libomp-dev libarrow-dev libparquet-dev pkg-config git
	@echo "✅ System dependencies installed"

setup: install-deps setup-hnswlib setup-enhanced
	@echo "=== Setup Complete ==="
	@echo "✅ All dependencies installed and configured"
	@echo "Ready to build with: make all"

# ==========================================
# Cleanup Targets
# ==========================================

clean:
	@echo "Cleaning build artifacts..."
	rm -f *.o $(ALL_PROGS)

distclean: clean
	@echo "Deep cleaning..."
	rm -f *~ *.bak core

# ==========================================
# Dependency Tracking
# ==========================================

# Program dependencies
comprehensive_testing: comprehensive_testing.o $(CORE_OBJS)
comprehensive_testing_with_hnswlib: comprehensive_testing_with_hnswlib.o $(HNSWLIB_OBJS)

# Header dependencies
comprehensive_testing.o: AMXInnerProductBF16PtrMTEnhanced.h AMXCommon.h ScalarInnerProduct.h
comprehensive_testing_with_hnswlib.o: HNSWLIBInnerProductPtr.h AMXInnerProductBF16PtrMTEnhanced.h AMXCommon.h ScalarInnerProduct.h
$(AMX_MT_ENHANCED_SRC:.cpp=.o): AMXInnerProductBF16PtrMTEnhanced.h AMXCommon.h
$(AMX_SRC:.cpp=.o): AMXInnerProductBF16Ptr.h AMXCommon.h
$(HNSWLIB_SRC:.cpp=.o): HNSWLIBInnerProductPtr.h
$(SCALAR_SRC:.cpp=.o): ScalarInnerProduct.h

# ==========================================
# Help Documentation
# ==========================================

help:
	@echo "AMX Inner Product Project - High-Performance Vector Computation"
	@echo "==============================================================="
	@echo ""
	@echo "SETUP TARGETS:"
	@echo "  setup            Install all dependencies and setup environment"
	@echo "  setup-hnswlib    Clone HNSWLIB repository for comparisons"
	@echo "  setup-enhanced   Verify enhanced AMX implementation files"
	@echo "  install-deps     Install system dependencies (Ubuntu/Debian)"
	@echo ""
	@echo "BUILD TARGETS:"
	@echo "  all              Build all programs (default target)"
	@echo "  debug            Build with debug flags and symbols"
	@echo "  release          Build optimized release version"
	@echo ""
	@echo "TEST TARGETS:"
	@echo "  test-enhanced    Run enhanced AMX timing analysis"
	@echo "  test-large       Run large-scale performance tests"
	@echo "  test-hnswlib     Run AMX vs HNSWLIB comparison"
	@echo "  test-all         Run comprehensive test suite"
	@echo ""
	@echo "UTILITY TARGETS:"
	@echo "  check            Check system capabilities and dependencies"
	@echo "  check-deps       Verify build dependencies are available"
	@echo "  clean            Remove build artifacts"
	@echo "  distclean        Deep clean including temporary files"
	@echo "  help             Show this help message"
	@echo ""
	@echo "INDIVIDUAL PROGRAMS:"
	@echo "  comprehensive_testing                Enhanced AMX with detailed timing"
	@echo "  comprehensive_testing_with_hnswlib   AMX + HNSWLIB comparison"
	@echo ""
	@echo "KEY FEATURES:"
	@echo "  ✅ Intel AMX (Advanced Matrix Extensions) acceleration"
	@echo "  ✅ Multi-threaded processing with load balancing analysis"
	@echo "  ✅ Detailed per-thread timing breakdown"
	@echo "  ✅ Memory efficiency and bottleneck identification"
	@echo "  ✅ Performance comparison with HNSWLIB"
	@echo "  ✅ Comprehensive accuracy validation"
	@echo "  ✅ BF16 precision for optimal AMX performance"
	@echo ""
	@echo "REQUIREMENTS:"
	@echo "  - Intel CPU with AMX support (4th Gen Xeon Scalable or newer)"
	@echo "  - Apache Arrow/Parquet libraries for data loading"
	@echo "  - OpenMP for multi-threading support"
	@echo "  - Large embedding dataset for comprehensive testing"
	@echo ""
	@echo "EXAMPLE USAGE:"
	@echo "  make setup       # First-time setup"
	@echo "  make all         # Build all programs"
	@echo "  make test-large  # Run comprehensive performance analysis"
	@echo ""
	@echo "For detailed documentation, see README.md"

# ==========================================
# Special Targets
# ==========================================

.PHONY: setup setup-hnswlib setup-enhanced test-enhanced test-large test-hnswlib AMX Inner Product Project Makefile
# High-performance inner product computation with Intel AMX
# ==========================================

# Compiler configuration
CXX = g++
BASE_FLAGS = -std=c++17 -Wall -Wextra -pthread
ARCH_FLAGS = -mavx512f -mavx512bf16 -mavx512bw -mavx512vl -mamx-tile -mamx-int8 -mamx-bf16
OPT_FLAGS = -O3 -fopenmp -flax-vector-conversions
SUPPRESS = -Wno-missing-field-initializers -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable

CXXFLAGS = $(BASE_FLAGS) $(ARCH_FLAGS) $(OPT_FLAGS) $(SUPPRESS)
DEBUG_FLAGS = -g -O0 -DDEBUG
RELEASE_FLAGS = -O3 -DNDEBUG -march=native

# External library support
ARROW_CFLAGS = $(shell pkg-config --cflags arrow parquet 2>/dev/null)
ARROW_LIBS = $(shell pkg-config --libs arrow parquet 2>/dev/null || echo "-larrow -lparquet")
HNSWLIB_CFLAGS = -Ihnswlib
HNSWLIB_LIBS =

# Source files
SCALAR_SRC = ScalarInnerProduct.cpp
AMX_SRC = AMXInnerProductBF16Ptr.cpp
AMX_MT_ENHANCED_SRC = AMXInnerProductBF16PtrMTEnhanced.cpp
HNSWLIB_SRC = HNSWLIBInnerProductPtr.cpp

# Object files for different configurations
CORE_OBJS = $(SCALAR_SRC:.cpp=.o) $(AMX_SRC:.cpp=.o) $(AMX_MT_ENHANCED_SRC:.cpp=.o)
HNSWLIB_OBJS = $(CORE_OBJS) $(HNSWLIB_SRC:.cpp=.o)

# Executable programs
ENHANCED_PROGS = comprehensive_testing
HNSWLIB_PROGS = comprehensive_testing_with_hnswlib
ALL_PROGS = $(ENHANCED_PROGS) $(HNSWLIB_PROGS)

# Default configuration
.PHONY: all clean debug release test help check-deps
.DEFAULT_GOAL = all

# ==========================================
# Main Targets
# ==========================================

all: check-deps $(ALL_PROGS)

# Enhanced AMX programs (core functionality)
$(ENHANCED_PROGS): %: %.o $(CORE_OBJS)
	@echo "Linking $@..."
	$(CXX) $(CXXFLAGS) $(ARROW_CFLAGS) $^ $(ARROW_LIBS) -o $@

# HNSWLIB comparison programs
$(HNSWLIB_PROGS): %: %.o $(HNSWLIB_OBJS)
	@echo "Linking $@ with HNSWLIB support..."
	$(CXX) $(CXXFLAGS) $(ARROW_CFLAGS) $(HNSWLIB_CFLAGS) $^ $(ARROW_LIBS) $(HNSWLIB_LIBS) -o $@

# ==========================================
#
