# Makefile for ZCU104 PYNQ native compilation (C with libcma)

CC = gcc
CFLAGS = -O3 -Wall -std=gnu11
LDFLAGS = -lcma -lm -lrt

TARGET1 = admm_test
TARGET2 = admm_test_1024

# By default, running 'make' will compile both targets
all: $(TARGET1) $(TARGET2)

# Build rule for the 1024x1024 testbench
$(TARGET1): $(TARGET1).c
	$(CC) $(CFLAGS) -o $(TARGET1) $(TARGET1).c $(LDFLAGS)

# Build rule for the 3x3 testbench
$(TARGET2): $(TARGET2).c
	$(CC) $(CFLAGS) -o $(TARGET2) $(TARGET2).c $(LDFLAGS)

# Clean up both executables
clean:
	rm -f $(TARGET1) $(TARGET2)