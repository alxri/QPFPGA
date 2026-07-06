# Makefile for ZCU104 PYNQ: Python Data Conversion and C++ Native Compilation

# C++ Compiler and Flags
CXX = g++
# Point to the global install location for headers
CXXFLAGS = -O3 -Wall -std=c++17 -I/usr/local/include

# C Compiler and Flags
CC = gcc
CFLAGS = -O3 -Wall -std=c11 -D_POSIX_C_SOURCE=199309L -I/usr/local/include

# Point to the global install location for the library
LDFLAGS_COMMON = -L/usr/local/lib -lcma -lm -lrt -Wl,-rpath=/usr/local/lib
LDFLAGS_CPP = $(LDFLAGS_COMMON) -lpmt
LDFLAGS_C = $(LDFLAGS_COMMON)

# Python Interpreter and Script
PYTHON = python3
PY_SCRIPT = data_export_from_py.py

# Target Executable and Source
TARGET = admm_test
SRC = main.cpp

# Alternate host that mirrors the notebook flow (Ruiz + tiling + run)
TARGET_LIKE_PY = admm_test_like_python
SRC_LIKE_PY = main_like_python.cpp

# Pure-C port of the like-python host
TARGET_LIKE_PY_C = admm_test_like_python_c
SRC_LIKE_PY_C = main_like_python_c.c

# Output directory for the binary data
BIN_DIR = data_bin

all: data $(TARGET)

like_python: data $(TARGET_LIKE_PY)

like_python_c: data $(TARGET_LIKE_PY_C)

data:
	@echo "==> Running Python data conversion..."
	$(PYTHON) $(PY_SCRIPT)

$(TARGET): $(SRC)
	@echo "==> Compiling C++ application..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS_CPP)

$(TARGET_LIKE_PY): $(SRC_LIKE_PY)
	@echo "==> Compiling like-python application..."
	$(CXX) $(CXXFLAGS) -o $(TARGET_LIKE_PY) $(SRC_LIKE_PY) $(LDFLAGS_COMMON)

$(TARGET_LIKE_PY_C): $(SRC_LIKE_PY_C)
	@echo "==> Compiling like-python (pure C) application..."
	$(CC) $(CFLAGS) -o $(TARGET_LIKE_PY_C) $(SRC_LIKE_PY_C) $(LDFLAGS_C)

clean:
	@echo "==> Cleaning up build files and binary data..."
	rm -f $(TARGET)
	rm -f $(TARGET_LIKE_PY)
	rm -f $(TARGET_LIKE_PY_C)
	rm -rf $(BIN_DIR)

.PHONY: all data like_python like_python_c clean