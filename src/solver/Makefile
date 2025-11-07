# Makefile for parallel GMRES solver
# 
# Usage:
#   make              # Build the library and test
#   make test         # Run the test
#   make clean        # Clean build artifacts

# Compiler and flags
MPICC ?= mpicc
CFLAGS = -O2 -Wall -Wextra -std=c99
LDFLAGS = -lm

# Source files
LIB_SRC = gmres.c
LIB_OBJ = $(LIB_SRC:.c=.o)
LIB_TARGET = libgmres.a

TEST_SRC = gmres_test.c
TEST_TARGET = gmres_test

# Default target
all: $(LIB_TARGET) $(TEST_TARGET)

# Build static library
$(LIB_TARGET): $(LIB_OBJ)
	ar rcs $@ $^

# Build test executable
$(TEST_TARGET): $(TEST_SRC) $(LIB_TARGET)
	$(MPICC) $(CFLAGS) -o $@ $< -L. -lgmres $(LDFLAGS)

# Compile object files
%.o: %.c gmres.h
	$(MPICC) $(CFLAGS) -c $< -o $@

# Run test
test: $(TEST_TARGET)
	@echo "Running sequential test..."
	mpirun -np 1 --allow-run-as-root ./$(TEST_TARGET) 100
	@echo ""
	@echo "Running parallel test with 2 processes..."
	mpirun -np 2 --allow-run-as-root --oversubscribe ./$(TEST_TARGET) 100
	@echo ""
	@echo "Running parallel test with 4 processes..."
	mpirun -np 4 --allow-run-as-root --oversubscribe ./$(TEST_TARGET) 100

# Clean
clean:
	rm -f $(LIB_OBJ) $(LIB_TARGET) $(TEST_TARGET)

.PHONY: all test clean
