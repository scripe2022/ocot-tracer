#include "accelerator.hpp"

void accelerator::build_tree(Scene &scene) { tree.build(scene); }

void accelerator::build_gpu(Scene &scene) {
    #ifdef GPU_RENDERING
    acc.build(scene);
    #else
    (void)(scene);
    #endif
}

accelerator::accelerator() {}

void accelerator::build(Scene &scene, bool gpurender) {
    usegpu = gpurender;
    build_tree(scene);
    build_gpu(scene);
}

accelerator::accelerator(Scene &scene, bool gpurender) {
    build(scene, gpurender);
}

hitobj accelerator::intersect(const Ray &ray) {
    #ifdef GPU_RENDERING
    return usegpu ? acc.intersect(ray) : tree.intersect(ray);
    #else
    return tree.intersect(ray);
    #endif
}
hitobj accelerator::intersect(const Ray &ray, bool gpurender) {
    #ifdef GPU_RENDERING
    return gpurender ? acc.intersect(ray) : tree.intersect(ray);
    #else
    (void)(gpurender);
    return tree.intersect(ray);
    #endif
}
std::vector<hitobj> accelerator::intersect_multi(std::vector<Ray> &rays) {
    #ifdef GPU_RENDERING
    return acc.intersect_multi(rays);
    #else
    (void)(rays);
    assert(false);
    #endif
}

