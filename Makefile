.PHONY: all debug clean

MKDIR_P = mkdir -p
RM_RF = rm -rf
CFLAGS = --std=c11 -O3
LDFLAGS = -lm
DEBUG = -Wall -Wextra -Wpedantic -g
CD = cd

all:
	$(MKDIR_P) target
	$(CD) src && $(CC) $(CFLAGS) utils.c Neural_Network.c MNIST_Labels.c MNIST_Images.c main.c -o ../target/main $(LDFLAGS)

debug:
	$(MKDIR_P) target
	$(CD) src && $(CC) $(CFLAGS) $(DEBUG) utils.c Neural_Network.c MNIST_Labels.c MNIST_Images.c main.c -o ../target/main_debug $(LDFLAGS)

clean:
	$(RM_RF) target/