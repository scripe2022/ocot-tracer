#ifndef ACCELERATOR_HPP
#define ACCELERATOR_HPP

#include "kdtree.hpp"
#ifdef GPU_RENDERING
#include "gpu.hpp"
#endif
#include "parse_scene.hpp"
#include "primitive.hpp"

struct accelerator {
    kdtree tree;
    #ifdef GPU_RENDERING
    gpuacc acc;
    #endif
    bool usegpu = false;

    accelerator();
    void build(Scene &scene, bool gpurender);
    void build_tree(Scene &scene);
    void build_gpu(Scene &scene);
    accelerator(Scene &scene, bool gpurender = false);
    hitobj intersect(const Ray &ray, bool gpurender);
    hitobj intersect(const Ray &ray);
    std::vector<hitobj> intersect_multi(std::vector<Ray> &rays);
};

#endif

