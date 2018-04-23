CC=g++

main:
	$(CC) main.cpp `pkg-config --cflags --libs opencv`

