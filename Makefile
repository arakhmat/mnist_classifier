CC = g++ -o
CFLAGS  = -g -Wall

SRCDIR =./src
INCDIR = ./inc

SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(SRCS:.c=*.o)


default: all


all:  $(SRCS)
	$(CC) $(CFLAGS) -I$(INCDIR) $(SRCS) -o bin/mnist_classifier

clean: 
	$(RM) all *.o *~