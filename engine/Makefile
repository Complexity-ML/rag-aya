CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11 -Isrc
LDFLAGS = -lm
SRC     = main.c gguf.c model.c
TARGET  = aya-server

# Windows-specific
ifeq ($(OS),Windows_NT)
    LDFLAGS += -lws2_32
    TARGET  := aya-server.exe
endif

# macOS / Apple Silicon - use Accelerate framework + NEON
UNAME_S := $(shell uname -s 2>/dev/null)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -framework Accelerate
    UNAME_M := $(shell uname -m)
    ifeq ($(UNAME_M),arm64)
        CFLAGS += -mcpu=apple-m1
    endif
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) ../tiny-aya-global-q4_k_m.gguf 8080

.PHONY: all clean run
