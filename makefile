CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp

ifeq "$(CXX)" "icpc" # conditionals
CXXFLAGS += -qopenmp # for Intel
LIBS = -mkl
else
CXXFLAGS += -fopenmp # for GCC
LIBS = -lblas
endif

TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
