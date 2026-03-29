CC := cc
CFLAGS := -O2 -Wall -Wextra -std=c11 -Iinclude
LDFLAGS := -lm

SRC := $(wildcard src/*.c)
OBJ := $(patsubst src/%.c,build/%.o,$(SRC))
BIN := bin/train

all: $(BIN)

build/%.o: src/%.c | build
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN): $(OBJ) | bin
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

build:
	mkdir -p build

bin:
	mkdir -p bin

clean:
	rm -rf build bin

run: $(BIN)
	./$(BIN)

.PHONY: all clean run
