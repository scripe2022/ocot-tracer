#ifndef ACC_GPU_HPP
#define ACC_GPU_HPP

#include "parse_scene.hpp"
#include "primitive.hpp"

struct gputri {
    float transform[16];
    float transform_inv[16];
    float normal_transform[9];
    float v0[3];
    float v1[3];
    float v2[3];
};

struct gpusph {
    float transform[16];
    float transform_inv[16];
    float normal_transform[9];
    float center[3];
    float radius;
};

struct gpures {
    float t;
    float normal[3];
    int idx;
};

struct gpuray {
    float origin[3];
    float direction[3];
};

struct gpuacc {
    Scene scene;
    gputri *triangles, *triangles_device;
    gpusph *spheres, *spheres_device;
    gpures *res, *res_device;
    uint *shapes_inv;
    int thread_size = 1024, block_size = 0;
    int ntriangles = 0, nspheres = 0;
    gpuacc();
    void build(Scene &s);
    ~gpuacc();
    hitobj intersect(const Ray &ray);
    std::vector<hitobj> intersect_multi(std::vector<Ray> &rayobjs);
};
#endif
