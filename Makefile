# AMX Inner Product Project Makefile
# ===================================

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

# Core sources
SCALAR_SRC = ScalarInnerProduct.cpp
AMX_SRC = AMXInnerProductBF16Ptr.cpp  
AMX_MT_SRC = AMXInnerProductBF16PtrMT.cpp
CORE_OBJS = $(SCALAR_SRC:.cpp=.o) $(AMX_SRC:.cpp=.o) $(AMX_MT_SRC:.cpp=.o)

# Programs
ARROW_PROGS = comprehensive_testing
ALL_PROGS = $(ARROW_PROGS)

# Default target
.PHONY: all clean debug release test help
.DEFAULT_GOAL = all

all: $(ALL_PROGS)

# Arrow-dependent programs
$(ARROW_PROGS): %: %.o $(CORE_OBJS)
	@pkg-config --exists arrow parquet || (echo "ERROR: Install Arrow/Parquet: sudo apt-get install libarrow-dev libparquet-dev" && exit 1)
	$(CXX) $(CXXFLAGS) $(ARROW_CFLAGS) $^ $(ARROW_LIBS) -o $@

# Object file rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Arrow-dependent object files
$(patsubst %.cpp,%.o,$(wildcard *testing*.cpp)): %.o: %.cpp
	$(CXX) $(CXXFLAGS) $(SUPPRESS) $(ARROW_CFLAGS) -c $< -o $@

# AMX object files (suppress warnings)
$(AMX_SRC:.cpp=.o) $(AMX_MT_SRC:.cpp=.o): %.o: %.cpp
	$(CXX) $(CXXFLAGS) $(SUPPRESS) -c $< -o $@

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

test-all: test test-large

# System checks
check:
	@echo "=== System Check ==="
	@echo "Hardware threads: $$(nproc)"
	@echo -n "AMX support: "; grep -q amx /proc/cpuinfo && echo "✅" || echo "❌"
	@echo -n "Arrow/Parquet: "; pkg-config --exists arrow parquet && echo "✅" || echo "❌"
	@echo -n "Compiler: "; $(CXX) --version | head -1

# Utilities
clean:
	rm -f *.o $(ALL_PROGS)

distclean: clean
	rm -f *~ *.bak core

install-deps:
	sudo apt-get update && sudo apt-get install -y build-essential libomp-dev libarrow-dev libparquet-dev pkg-config

help:
	@echo "AMX Inner Product Project"
	@echo "========================"
	@echo ""
	@echo "Build targets:"
	@echo "  all              Build all programs (default)"
	@echo "  debug            Build with debug flags"
	@echo "  release          Build optimized release"
	@echo ""
	@echo "Test targets:"
	@echo "  test             Run basic tests (no dataset required)"
	@echo "  test-large       Run large-scale tests (requires dataset)"
	@echo "  test-all         Run all tests"
	@echo ""
	@echo "Utilities:"
	@echo "  check            Check system capabilities"
	@echo "  install-deps     Install dependencies (Ubuntu/Debian)"
	@echo "  clean            Remove build artifacts"
	@echo "  help             Show this message"
	@echo ""
	@echo "Individual programs:"
	@$(printf "  %-16s %s\n" "$(foreach prog,$(ALL_PROGS),$(prog))" "")

