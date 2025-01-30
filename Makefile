CUDA = nvcc
CUDAFLAGS = -O3 -std=c++20

CXX = g++
CXXFLAGS = -O3 -std=gnu++20 -Wall -Wextra -Wshadow -D_GLIBCXX_ASSERTIONS -fmax-errors=2

LDFLAGS = -lpng

ADD =

LOCAL = -DGPU_RENDERING

SOURCES_CPP = $(wildcard *.cpp)
SOURCES_CU = $(wildcard *.cu)

OBJECTS_CPP = $(SOURCES_CPP:.cpp=.o)
OBJECTS_CU = $(SOURCES_CU:.cu=.o)
OBJECTS = $(OBJECTS_CPP) $(OBJECTS_CU)
EXECUTABLE = raytracer

all: $(EXECUTABLE)

$(EXECUTABLE): main.o parse_scene.o image.o primitive.o kdtree.o gpu.o accelerator.o ray_tracer.o analytic_direct.o direct_light.o path_tracer.o
	$(CUDA) $(CUDAFLAGS) $(ADD) $(LDFLAGS) main.o parse_scene.o image.o primitive.o kdtree.o gpu.o accelerator.o ray_tracer.o analytic_direct.o direct_light.o path_tracer.o -o $@

cpu:
	$(CXX) $(CXXFLAGS) $(ADD) -o raytracer-cpu main.cpp parse_scene.cpp image.cpp primitive.cpp kdtree.cpp accelerator.cpp ray_tracer.cpp analytic_direct.cpp direct_light.cpp path_tracer.cpp $(LDFLAGS)

gpu:
	nvcc $(CUDAFLAGS) $(ADD) $(LDFLAGS) -DGPU_RENDERING -o raytracer-gpu main.cpp parse_scene.cpp image.cpp primitive.cpp kdtree.cpp gpu.cu accelerator.cpp ray_tracer.cpp analytic_direct.cpp direct_light.cpp path_tracer.cpp

gdb:
	nvcc -O0 -g $(LDFLAGS) -DGPU_RENDERING -o raytracer-gdb main.cpp parse_scene.cpp image.cpp primitive.cpp kdtree.cpp gpu.cu accelerator.cpp ray_tracer.cpp analytic_direct.cpp direct_light.cpp path_tracer.cpp

main.o: main.cpp parse_scene.o image.o primitive.o kdtree.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LOCAL) -c main.cpp -o $@

image.o: image.hpp image.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LOCAL) -c image.cpp -o $@

parse_scene.o: parse_scene.hpp parse_scene.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LOCAL) -c parse_scene.cpp -o $@

primitive.o: primitive.hpp primitive.cpp parse_scene.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LOCAL) -c primitive.cpp -o $@

gpu.o: gpu.cu gpu.hpp primitive.hpp parse_scene.hpp
	$(CUDA) $(CUDAFLAGS) $(ADD) $(LOCAL) -c gpu.cu -o $@

kdtree.o: kdtree.hpp kdtree.cpp primitive.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LOCAL) -c kdtree.cpp -o $@

accelerator.o: accelerator.hpp accelerator.cpp primitive.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LOCAL) -c accelerator.cpp -o $@

ray_tracer.o: ray_tracer.hpp ray_tracer.cpp image.hpp accelerator.hpp parse_scene.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LOCAL) -c ray_tracer.cpp -o $@

analytic_direct.o: analytic_direct.hpp analytic_direct.cpp image.hpp accelerator.hpp parse_scene.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LOCAL) -c analytic_direct.cpp -o $@

direct_light.o: direct_light.hpp direct_light.cpp image.hpp accelerator.hpp parse_scene.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LOCAL) -c direct_light.cpp -o $@

path_tracer.o: path_tracer.hpp path_tracer.cpp image.hpp accelerator.hpp parse_scene.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LOCAL) -c path_tracer.cpp -o $@

bear:
	make clean
	bear -- make
	jq 'map(if (.arguments[0] | contains("nvcc")) then .arguments = (.arguments[0:1] + ["--cuda-gpu-arch=sm_86"] + .arguments[1:]) else . end)' compile_commands.json > temp_compile_commands.json && mv temp_compile_commands.json compile_commands.json

clean:
	rm -f $(OBJECTS) $(EXECUTABLE) compile_commands.json

.PHONY: all clean cpu gpu

