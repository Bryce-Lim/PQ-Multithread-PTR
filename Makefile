# Makefile for AMX Inner Product Testing
# Author: Generated for AMXInnerProductBF16Ptr testing

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp
WARNINGS = -Wall -Wextra -Wno-unused-parameter
DEBUG_FLAGS = -g -DDEBUG

# AMX-specific flags
AMX_FLAGS = -mamx-tile -mamx-int8 -mamx-bf16

# Include directories (adjust paths as needed)
INCLUDES = -I/usr/local/include \
           -I/usr/include/arrow \
           -I/usr/include/parquet

# Library directories and libraries
LIBDIRS = -L/usr/local/lib \
          -L/usr/lib/x86_64-linux-gnu

LIBS = -larrow -lparquet -fopenmp

# Source files
SOURCES = detailed_test.cpp AMXInnerProductBF16Ptr.cpp
HEADERS = AMXInnerProductBF16Ptr.h

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Target executable
TARGET = detailed_test

# Default target
all: $(TARGET)

# Main target
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(AMX_FLAGS) $(OBJECTS) -o $(TARGET) $(LIBDIRS) $(LIBS)
	@echo "Build completed successfully!"

# Object file compilation
%.o: %.cpp $(HEADERS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(AMX_FLAGS) $(WARNINGS) $(INCLUDES) -c $< -o $@

# Debug build
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean $(TARGET)
	@echo "Debug build completed!"

# Performance build (extra optimizations)
performance: CXXFLAGS += -Ofast -flto -funroll-loops -ffast-math
performance: clean $(TARGET)
	@echo "Performance build completed!"

# Clean build files
clean:
	@echo "Cleaning build files..."
	rm -f $(OBJECTS) $(TARGET)
	@echo "Clean completed!"

# Run the test
run: $(TARGET)
	@echo "Running detailed test..."
	./$(TARGET)

# Run with specific number of threads
run-threads: $(TARGET)
	@echo "Running with OMP_NUM_THREADS=16..."
	OMP_NUM_THREADS=16 ./$(TARGET)

# Check system AMX support
check-amx:
	@echo "Checking system AMX support..."
	@lscpu | grep -i amx || echo "AMX not found in lscpu output"
	@cat /proc/cpuinfo | grep -i "amx\|avx512" | head -5 || echo "No AMX/AVX512 found in cpuinfo"

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing dependencies (requires sudo)..."
	sudo apt-get update
	sudo apt-get install -y \
		build-essential \
		libarrow-dev \
		libparquet-dev \
		libomp-dev \
		pkg-config

# Alternative dependency installation using conda
install-deps-conda:
	@echo "Installing dependencies with conda..."
	conda install -c conda-forge arrow-cpp parquet-cpp openmp

# Check if required libraries are available
check-deps:
	@echo "Checking dependencies..."
	@pkg-config --exists arrow && echo "✓ Arrow found" || echo "✗ Arrow not found"
	@pkg-config --exists parquet && echo "✓ Parquet found" || echo "✗ Parquet not found"
	@echo -e '#include <omp.h>\nint main(){return 0;}' | $(CXX) -fopenmp -x c++ - -o /tmp/omp_test 2>/dev/null && echo "✓ OpenMP found" || echo "✗ OpenMP not found"
	@rm -f /tmp/omp_test

# Print build info
info:
	@echo "=== Build Configuration ==="
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS) $(AMX_FLAGS)"
	@echo "Includes: $(INCLUDES)"
	@echo "Libraries: $(LIBS)"
	@echo "Sources: $(SOURCES)"
	@echo "Target: $(TARGET)"
	@echo "=========================="

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the test executable (default)"
	@echo "  debug        - Build with debug flags"
	@echo "  performance  - Build with maximum optimization"
	@echo "  clean        - Remove build files"
	@echo "  run          - Build and run the test"
	@echo "  run-threads  - Run with specific thread count"
	@echo "  check-amx    - Check if system supports AMX"
	@echo "  check-deps   - Check if dependencies are installed"
	@echo "  install-deps - Install dependencies (Ubuntu/Debian)"
	@echo "  install-deps-conda - Install dependencies with conda"
	@echo "  info         - Show build configuration"
	@echo "  help         - Show this help message"

# Prevent make from treating these as file targets
.PHONY: all debug performance clean run run-threads check-amx install-deps install-deps-conda check-deps info help

# Special handling for systems without AMX support (fallback)
fallback: CXXFLAGS += -DSKIP_AMX_INSTRUCTIONS
fallback: AMX_FLAGS =
fallback: $(TARGET)
	@echo "Fallback build completed (AMX instructions disabled)!"
