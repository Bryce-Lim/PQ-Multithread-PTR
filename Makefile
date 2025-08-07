# Comprehensive Makefile for AMX Inner Product Project
# ==================================================

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -mavx512f -mavx512bf16 -mavx512bw -mavx512vl -mamx-tile -mamx-int8 -mamx-bf16 -fopenmp -flax-vector-conversions
SUPPRESS_WARNINGS = -Wno-missing-field-initializers -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable
DEBUG_FLAGS = -g -O0 -DDEBUG
RELEASE_FLAGS = -O3 -DNDEBUG -march=native

# Include directories
INCLUDES = -I.

# Library settings for Arrow/Parquet (only needed for large_testing)
ARROW_LIBS = -larrow -lparquet -larrow_dataset -larrow_acero -larrow_compute -larrow_csv -larrow_filesystem -larrow_flight -larrow_flight_sql -larrow_json
ARROW_INCLUDES = $(shell pkg-config --cflags arrow parquet 2>/dev/null || echo "")
ARROW_LDFLAGS = $(shell pkg-config --libs arrow parquet 2>/dev/null || echo "$(ARROW_LIBS)")

# Source files
SCALAR_SOURCES = ScalarInnerProduct.cpp
AMX_SOURCES = AMXInnerProductBF16Ptr.cpp
COMMON_SOURCES = $(SCALAR_SOURCES) $(AMX_SOURCES)

# Object files
SCALAR_OBJECTS = $(SCALAR_SOURCES:.cpp=.o)
AMX_OBJECTS = $(AMX_SOURCES:.cpp=.o)
COMMON_OBJECTS = $(COMMON_SOURCES:.cpp=.o)

# Executables
EXECUTABLES = small_test simple_amx_test large_testing synthetic_large_test

# Default target
.PHONY: all clean debug release help test

all: $(EXECUTABLES)

# Individual program targets
small_test: small_test.o $(COMMON_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

simple_amx_test: simple_amx_test.o $(AMX_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

synthetic_large_test: synthetic_large_test.o $(COMMON_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

large_testing: large_testing.o $(COMMON_OBJECTS)
	@echo "Checking for Arrow/Parquet libraries..."
	@pkg-config --exists arrow parquet || (echo "WARNING: Arrow/Parquet not found. Install with: sudo apt-get install libarrow-dev libparquet-dev" && false)
	$(CXX) $(CXXFLAGS) $(ARROW_INCLUDES) $^ $(ARROW_LDFLAGS) -o $@

# Special compilation for large_testing with Arrow dependencies
large_testing.o: large_testing.cpp
	@echo "Compiling large_testing with Arrow support..."
	$(CXX) $(CXXFLAGS) $(SUPPRESS_WARNINGS) $(ARROW_INCLUDES) $(INCLUDES) -c $< -o $@

# AMX object files (need warning suppression)
AMXInnerProductBF16Ptr.o: AMXInnerProductBF16Ptr.cpp
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

# Test targets with library path handling
test: test-small test-simple

test-small: small_test
	@echo "Running small test..."
	./small_test

test-simple: simple_amx_test
	@echo "Running simple AMX test..."
	./simple_amx_test

test-large: large_testing
	@echo "Running large-scale test (requires dataset)..."
	@echo "Checking library dependencies first..."
	@ldd large_testing | grep -E "(arrow|parquet)" || echo "Warning: Arrow/Parquet libraries not found"
	@echo "If you see library errors, run: make fix-libs"
	@echo ""
	LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:$LD_LIBRARY_PATH ./large_testing

# Run with explicit library path
test-large-fixed: large_testing
	@echo "Running large test with library path fixes..."
	LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:/opt/conda/lib:$LD_LIBRARY_PATH ./large_testing

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

# Performance testing
perf-test: simple_amx_test
	@echo "Running performance test with simple AMX test..."
	perf stat -e cycles,instructions,cache-references,cache-misses ./simple_amx_test

# Memory check (requires valgrind)
memcheck: simple_amx_test
	@echo "Running memory check (requires valgrind)..."
	valgrind --leak-check=full --show-leak-kinds=all ./simple_amx_test

# Installation check for dependencies
check-deps:
	@echo "Checking build dependencies..."
	@echo -n "GCC version: "
	@$(CXX) --version | head -1
	@echo -n "OpenMP support: "
	@echo '#include <omp.h>' | $(CXX) -xc++ -fopenmp -E - >/dev/null 2>&1 && echo "✅ Available" || echo "❌ Missing"
	@echo -n "AVX512 support: "
	@echo 'int main() { return 0; }' | $(CXX) -xc++ -mavx512f -E - >/dev/null 2>&1 && echo "✅ Available" || echo "❌ Missing"
	@echo -n "Arrow/Parquet: "
	@pkg-config --exists arrow parquet && echo "✅ Available" || echo "❌ Missing (needed for large_testing)"

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing dependencies for Ubuntu/Debian..."
	sudo apt-get update
	sudo apt-get install -y build-essential libomp-dev libarrow-dev libparquet-dev pkg-config

# Install dependencies (RHEL/CentOS/Fedora)
install-deps-rhel:
	@echo "Installing dependencies for RHEL/CentOS/Fedora..."
	sudo dnf install -y gcc-c++ libomp-devel arrow-devel parquet-devel pkgconfig || \
	sudo yum install -y gcc-c++ libomp-devel arrow-devel parquet-devel pkgconfig

# Clean build artifacts
clean:
	rm -f *.o $(EXECUTABLES)

# Deep clean (including backup files)
distclean: clean
	rm -f *~ *.bak core

# Code formatting (requires clang-format)
format:
	@echo "Formatting code (requires clang-format)..."
	@find . -name "*.cpp" -o -name "*.h" | xargs clang-format -i -style="{BasedOnStyle: Google, IndentWidth: 4}" 2>/dev/null || echo "clang-format not available"

# Static analysis (requires cppcheck)
analyze:
	@echo "Running static analysis (requires cppcheck)..."
	@cppcheck --enable=all --std=c++17 --platform=unix64 --suppress=missingIncludeSystem *.cpp *.h 2>/dev/null || echo "cppcheck not available"

# Show compiler info
compiler-info:
	@echo "Compiler Information:"
	@echo "====================="
	@$(CXX) --version
	@echo ""
	@echo "Supported CPU features:"
	@$(CXX) -march=native -Q --help=target | grep -E "(mavx|mamx)" || echo "No AVX/AMX info available"

# Help target
help:
	@echo "Available targets:"
	@echo "=================="
	@echo "Building:"
	@echo "  all              - Build all executables (default)"
	@echo "  small_test       - Build the basic functionality test"
	@echo "  simple_amx_test  - Build the simple AMX test"
	@echo "  large_testing    - Build the large-scale performance test"
	@echo "  debug            - Build with debug flags"
	@echo "  release          - Build with optimization flags"
	@echo ""
	@echo "Testing:"
	@echo "  test             - Run small and simple tests"
	@echo "  test-small       - Run small_test only"
	@echo "  test-simple      - Run simple_amx_test only"
	@echo "  test-large       - Run large_testing (requires dataset)
	test-large-fixed - Run large_testing with library path fixes"
	@echo "  check-amx        - Check if AMX is available on this system"
	@echo ""
	@echo "Performance & Analysis:"
	@echo "  perf-test        - Run with performance counters"
	@echo "  memcheck         - Run memory leak check (requires valgrind)"
	@echo "  analyze          - Run static code analysis (requires cppcheck)"
	@echo "  format           - Format code (requires clang-format)"
	@echo ""
	@echo "Dependencies:"
	@echo "  check-deps       - Check if all dependencies are available"
	@echo "  install-deps     - Install dependencies (Ubuntu/Debian)"
	@echo "  install-deps-rhel- Install dependencies (RHEL/CentOS/Fedora)"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            - Remove build artifacts"
	@echo "  distclean        - Remove all generated files"
	@echo "  compiler-info    - Show compiler and CPU feature info"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Quick start:"
	@echo "  make check-deps  # Check dependencies"
	@echo "  make all         # Build everything"
	@echo "  make test        # Run basic tests"
	@echo "  make check-amx   # Check AMX support"

# Example usage patterns
examples:
	@echo "Example Usage Patterns:"
	@echo "======================="
	@echo ""
	@echo "1. Quick test on any system:"
	@echo "   make small_test && ./small_test"
	@echo ""
	@echo "2. Check AMX and run simple test:"
	@echo "   make check-amx"
	@echo "   make simple_amx_test && ./simple_amx_test"
	@echo ""
	@echo "3. Full development cycle:"
	@echo "   make check-deps    # Verify dependencies"
	@echo "   make debug         # Build with debug info"
	@echo "   make test          # Run tests"
	@echo "   make analyze       # Static analysis"
	@echo ""
	@echo "4. Performance testing:"
	@echo "   make release       # Optimized build"
	@echo "   make perf-test     # Run with performance counters"
	@echo ""
	@echo "5. Large-scale testing (requires dataset):"
	@echo "   make large_testing"
	@echo "   ./large_testing"

# Force rebuild of specific targets
.PHONY: force-small force-simple force-large
force-small: clean small_test
force-simple: clean simple_amx_test  
force-large: clean large_testing

# Print build configuration
config:
	@echo "Build Configuration:"
	@echo "==================="
	@echo "CXX:      $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "INCLUDES: $(INCLUDES)"
	@echo "ARROW_INCLUDES: $(ARROW_INCLUDES)"
	@echo "ARROW_LDFLAGS: $(ARROW_LDFLAGS)"
