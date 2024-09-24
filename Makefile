.PHONY: all clean test matrix_tests test_matrix push fast

CC=cc
C_STANDARD=c11
C_OPTIONS=-Wall -Wextra -Wpedantic -g
C_FAST_OPTIONS=-Wall -Wextra -Wpedantic -O3
LD_LIBS=-lm -lpthread

all:
	cd src && ${CC} ${C_OPTIONS} --std=${C_STANDARD} utils.c Neural_Network.c MNIST_Labels.c MNIST_Images.c main.c -o ../exe/main ${LD_LIBS}

clean:
	rm -v exe/*

test:
	valgrind -s --leak-check=full --show-leak-kinds=all --track-origins=yes ./exe/main threaded-predict data/t10k-labels-idx1-ubyte data/t10k-images-idx3-ubyte 100 models/small_100.model

push:
	rsync -av --progress --exclude .git --exclude .vscode . sam@10.1.7.57:~/digits/
	
fast:
	cd src && ${CC} ${C_FAST_OPTIONS} --std=${C_STANDARD} utils.c Neural_Network.c MNIST_Labels.c MNIST_Images.c main.c -o ../exe/main ${LD_LIBS}
	