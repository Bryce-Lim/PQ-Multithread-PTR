# Enhanced Makefile for AMX Inner Product Project - Updated with Comprehensive Testing
# ===================================================================================

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -mavx512f -mavx512bf16 -mavx512bw -mavx512vl -mamx-tile -mamx-int8 -mamx-bf16 -fopenmp -flax-vector-conversions -pthread
SUPPRESS_WARNINGS = -Wno-missing-field-initializers -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable
DEBUG_FLAGS = -g -O0 -DDEBUG
RELEASE_FLAGS = -O3 -DNDEBUG -march=native

# Include directories
INCLUDES = -I.

# Library settings for Arrow/Parquet (only needed for large_testing and comprehensive_testing)
ARROW_LIBS = -larrow -lparquet -larrow_dataset -larrow_acero -larrow_compute -larrow_csv -larrow_filesystem -larrow_flight -larrow_flight_sql -larrow_json
ARROW_INCLUDES = $(shell pkg-config --cflags arrow parquet 2>/dev/null || echo "")
ARROW_LDFLAGS = $(shell pkg-config --libs arrow parquet 2>/dev/null || echo "$(ARROW_LIBS)")

# Source files
SCALAR_SOURCES = ScalarInnerProduct.cpp
AMX_SOURCES = AMXInnerProductBF16Ptr.cpp
AMX_MT_SOURCES = AMXInnerProductBF16PtrMT.cpp
COMMON_SOURCES = $(SCALAR_SOURCES) $(AMX_SOURCES) $(AMX_MT_SOURCES)

# Object files
SCALAR_OBJECTS = $(SCALAR_SOURCES:.cpp=.o)
AMX_OBJECTS = $(AMX_SOURCES:.cpp=.o)
AMX_MT_OBJECTS = $(AMX_MT_SOURCES:.cpp=.o)
COMMON_OBJECTS = $(COMMON_SOURCES:.cpp=.o)

# Executables (updated with comprehensive_testing)
EXECUTABLES = small_test simple_amx_test large_testing large_testing_mt synthetic_large_test verify_multithread multithread_test comprehensive_testing

# Default target
.PHONY: all clean debug release help test

all: $(EXECUTABLES)

# Individual program targets
small_test: small_test.o $(SCALAR_OBJECTS) $(AMX_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

simple_amx_test: simple_amx_test.o $(AMX_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

synthetic_large_test: synthetic_large_test.o $(SCALAR_OBJECTS) $(AMX_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

verify_multithread: verify_multithread.o $(SCALAR_OBJECTS) $(AMX_OBJECTS) $(AMX_MT_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

multithread_test: multithread_test.o $(SCALAR_OBJECTS) $(AMX_OBJECTS) $(AMX_MT_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# New comprehensive testing target
comprehensive_testing: comprehensive_testing.o $(SCALAR_OBJECTS) $(AMX_OBJECTS) $(AMX_MT_OBJECTS)
	@echo "Checking for Arrow/Parquet libraries..."
	@pkg-config --exists arrow parquet || (echo "WARNING: Arrow/Parquet not found. Install with: sudo apt-get install libarrow-dev libparquet-dev" && false)
	$(CXX) $(CXXFLAGS) $(ARROW_INCLUDES) $^ $(ARROW_LDFLAGS) -o $@

large_testing: large_testing.o $(SCALAR_OBJECTS) $(AMX_OBJECTS)
	@echo "Checking for Arrow/Parquet libraries..."
	@pkg-config --exists arrow parquet || (echo "WARNING: Arrow/Parquet not found. Install with: sudo apt-get install libarrow-dev libparquet-dev" && false)
	$(CXX) $(CXXFLAGS) $(ARROW_INCLUDES) $^ $(ARROW_LDFLAGS) -o $@

large_testing_mt: large_testing_mt.o $(SCALAR_OBJECTS) $(AMX_OBJECTS) $(AMX_MT_OBJECTS)
	@echo "Checking for Arrow/Parquet libraries..."
	@pkg-config --exists arrow parquet || (echo "WARNING: Arrow/Parquet not found. Install with: sudo apt-get install libarrow-dev libparquet-dev" && false)
	$(CXX) $(CXXFLAGS) $(ARROW_INCLUDES) $^ $(ARROW_LDFLAGS) -o $@

# Special compilation for Arrow-dependent files
large_testing.o: large_testing.cpp
	@echo "Compiling large_testing with Arrow support..."
	$(CXX) $(CXXFLAGS) $(SUPPRESS_WARNINGS) $(ARROW_INCLUDES) $(INCLUDES) -c $< -o $@

large_testing_mt.o: large_testing_mt.cpp
	@echo "Compiling large_testing_mt with Arrow support..."
	$(CXX) $(CXXFLAGS) $(SUPPRESS_WARNINGS) $(ARROW_INCLUDES) $(INCLUDES) -c $< -o $@

comprehensive_testing.o: comprehensive_testing.cpp
	@echo "Compiling comprehensive_testing with Arrow support..."
	$(CXX) $(CXXFLAGS) $(SUPPRESS_WARNINGS) $(ARROW_INCLUDES) $(INCLUDES) -c $< -o $@

# AMX object files (need warning suppression)
AMXInnerProductBF16Ptr.o: AMXInnerProductBF16Ptr.cpp
	$(CXX) $(CXXFLAGS) $(SUPPRESS_WARNINGS) $(INCLUDES) -c $< -o $@

AMXInnerProductBF16PtrMT.o: AMXInnerProductBF16PtrMT.cpp
	$(CXX) $(CXXFLAGS) $(SUPPRESS_WARNINGS) $(INCLUDES) -c $< -o $@

# Generic object file compilation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Debug builds
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean $(EXECUTABLES)

# Release builds (optimized)
release: CXXFLAGS += $(RELEASE_FLAGS)
release: clean $(EXECUTABLES)

# Test targets (updated with comprehensive testing)
test: test-small test-simple test-verify

test-small: small_test
	@echo "Running small test..."
	./small_test

test-simple: simple_amx_test
	@echo "Running simple AMX test..."
	./simple_amx_test

test-verify: verify_multithread
	@echo "Running multithreaded verification test..."
	./verify_multithread

test-multithread: multithread_test
	@echo "Running multithreaded performance test..."
	./multithread_test

test-large: large_testing
	@echo "Running large-scale test (requires dataset)..."
	@echo "Checking library dependencies first..."
	@ldd large_testing | grep -E "(arrow|parquet)" || echo "Warning: Arrow/Parquet libraries not found"
	@echo "If you see library errors, run: make fix-libs"
	@echo ""
	LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:$LD_LIBRARY_PATH ./large_testing

test-large-mt: large_testing_mt
	@echo "Running multithreaded large-scale test (requires dataset)..."
	@echo "Checking library dependencies first..."
	@ldd large_testing_mt | grep -E "(arrow|parquet)" || echo "Warning: Arrow/Parquet libraries not found"
	@echo "If you see library errors, run: make fix-libs"
	@echo ""
	LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:$LD_LIBRARY_PATH ./large_testing_mt

# New comprehensive testing target
test-comprehensive: comprehensive_testing
	@echo "Running comprehensive implementation comparison (requires dataset)..."
	@echo "This will test and compare all implementations with detailed analysis"
	@echo "=================================================================="
	@ldd comprehensive_testing | grep -E "(arrow|parquet)" || echo "Warning: Arrow/Parquet libraries not found"
	@echo ""
	LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:$LD_LIBRARY_PATH ./comprehensive_testing

# Quick verification workflow
quick-verify: verify_multithread
	@echo "Running quick multithreaded verification..."
	@echo "This will test if multithreading produces correct results"
	@echo "=================================================="
	./verify_multithread

# Full testing workflow
full-test: test test-multithread test-comprehensive
	@echo "Full testing workflow completed!"

# Check if AMX is available
check-amx:
	@echo "Checking AMX availability..."
	@if grep -q amx /proc/cpuinfo; then \
		echo "✅ AMX support detected in CPU"; \
	else \
		echo "❌ AMX not detected - tests will show initialization failure"; \
	fi
	@echo "CPU flags:"
	@grep flags /proc/cpuinfo | head -1 | tr ' ' '\n' | grep -E "(amx|avx512)" || echo "No AMX/AVX512 flags found"

# Check threading capabilities
check-threading:
	@echo "Checking threading capabilities..."
	@echo "Hardware threads: $$(nproc)"
	@echo "OpenMP threads: $$(echo 'int main(){return 0;}' | $(CXX) -xc++ -fopenmp -E - >/dev/null 2>&1 && echo 'Available' || echo 'Not available')"
	@echo "C++ thread support: $$(echo '#include <thread>' | $(CXX) -xc++ -pthread -E - >/dev/null 2>&1 && echo 'Available' || echo 'Not available')"

# Installation check for dependencies
check-deps:
	@echo "Checking build dependencies..."
	@echo -n "GCC version: "
	@$(CXX) --version | head -1
	@echo -n "OpenMP support: "
	@echo '#include <omp.h>' | $(CXX) -xc++ -fopenmp -E - >/dev/null 2>&1 && echo "✅ Available" || echo "❌ Missing"
	@echo -n "pthread support: "
	@echo '#include <thread>' | $(CXX) -xc++ -pthread -E - >/dev/null 2>&1 && echo "✅ Available" || echo "❌ Missing"
	@echo -n "AVX512 support: "
	@echo 'int main() { return 0; }' | $(CXX) -xc++ -mavx512f -E - >/dev/null 2>&1 && echo "✅ Available" || echo "❌ Missing"
	@echo -n "Arrow/Parquet: "
	@pkg-config --exists arrow parquet && echo "✅ Available" || echo "❌ Missing (needed for large_testing and comprehensive_testing)"

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing dependencies for Ubuntu/Debian..."
	sudo apt-get update
	sudo apt-get install -y build-essential libomp-dev libarrow-dev libparquet-dev pkg-config numactl

# Clean build artifacts
clean:
	rm -f *.o $(EXECUTABLES)

# Deep clean (including backup files)
distclean: clean
	rm -f *~ *.bak core

# Help target (updated)
help:
	@echo "Available targets:"
	@echo "=================="
	@echo "Building:"
	@echo "  all                    - Build all executables (default)"
	@echo "  small_test             - Build the basic functionality test"
	@echo "  simple_amx_test        - Build the simple AMX test"
	@echo "  verify_multithread     - Build the multithreaded verification test"
	@echo "  multithread_test       - Build the multithreaded performance test"
	@echo "  large_testing          - Build the large-scale performance test"
	@echo "  large_testing_mt       - Build the multithreaded large-scale test"
	@echo "  comprehensive_testing  - Build the comprehensive comparison test (NEW)"
	@echo "  debug                  - Build with debug flags"
	@echo "  release                - Build with optimization flags"
	@echo ""
	@echo "Testing:"
	@echo "  test                   - Run small, simple, and verification tests"
	@echo "  test-small             - Run small_test only"
	@echo "  test-simple            - Run simple_amx_test only"
	@echo "  test-verify            - Run multithreaded verification test"
	@echo "  test-multithread       - Run multithreaded performance test"
	@echo "  test-large             - Run large_testing (requires dataset)"
	@echo "  test-large-mt          - Run multithreaded large_testing (requires dataset)"
	@echo "  test-comprehensive     - Run comprehensive comparison test (requires dataset) (NEW)"
	@echo "  full-test              - Run all tests in sequence"
	@echo "  quick-verify           - Quick verification workflow"
	@echo "  check-amx              - Check if AMX is available on this system"
	@echo "  check-threading        - Check threading capabilities"
	@echo ""
	@echo "Dependencies:"
	@echo "  check-deps             - Check if all dependencies are available"
	@echo "  install-deps           - Install dependencies (Ubuntu/Debian)"
	@echo ""
	@echo "Utilities:"
	@echo "  clean                  - Remove build artifacts"
	@echo "  distclean              - Remove all generated files"
	@echo "  help                   - Show this help message"
	@echo ""
	@echo "Quick start for comprehensive testing:"
	@echo "  make comprehensive_testing  # Build comprehensive test"
