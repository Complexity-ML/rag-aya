CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11 -Isrc
LDFLAGS = -lm
SRC     = src/main.c src/gguf.c src/model.c
TARGET  = aya-server

# Windows-specific
ifeq ($(OS),Windows_NT)
    LDFLAGS += -lws2_32
    TARGET  := aya-server.exe
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) ../tiny-aya-global-q4_k_m.gguf 8080

.PHONY: all clean run
