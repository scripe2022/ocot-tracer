// comp := make
// run  := time ./raytracer scenes/test3/cornellNEE.test && kcat output/cornellNEE.png

#include <cuda_runtime.h>

#include <iostream>
#include <primitive.hpp>
#include <variant>
using namespace std;

#include "gpu.hpp"

__global__ void find_intersection(const gputri *triangles, const gpusph *spheres, int ntriangles, int nspheres, gpuray ray, gpures *res) {
    __shared__ float mem_t[1024];
    __shared__ int mem_idx[1024];
    __shared__ float mem_normal[1024][3];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    mem_t[threadIdx.x] = 10.0f * INF;
    mem_idx[threadIdx.x] = threadIdx.x;
    float *transform, *transform_inv, *normal_transform;
    if (i < ntriangles) { transform = (float *)triangles[i].transform, transform_inv = (float *)triangles[i].transform_inv, normal_transform = (float *)triangles[i].normal_transform; }
    else if (i < ntriangles + nspheres) {
        transform = (float *)spheres[i - ntriangles].transform, transform_inv = (float *)spheres[i - ntriangles].transform_inv, normal_transform = (float *)spheres[i - ntriangles].normal_transform;
    }

    bool hit = false;
    float t_local = 10.0f * INF, t_world = 10.0f * INF;
    float normal_x = 0.0f, normal_y = 0.0f, normal_z = 0.0f;
    float normal_world_x = 0.0f, normal_world_y = 0.0f, normal_world_z = 0.0f;

    if (i < ntriangles + nspheres) {
        float ray_local_ori_x = ray.origin[0] * transform_inv[0] + ray.origin[1] * transform_inv[1] + ray.origin[2] * transform_inv[2] + transform_inv[3];
        float ray_local_ori_y = ray.origin[0] * transform_inv[4] + ray.origin[1] * transform_inv[5] + ray.origin[2] * transform_inv[6] + transform_inv[7];
        float ray_local_ori_z = ray.origin[0] * transform_inv[8] + ray.origin[1] * transform_inv[9] + ray.origin[2] * transform_inv[10] + transform_inv[11];
        float ray_local_dir_x = ray.direction[0] * transform_inv[0] + ray.direction[1] * transform_inv[1] + ray.direction[2] * transform_inv[2];
        float ray_local_dir_y = ray.direction[0] * transform_inv[4] + ray.direction[1] * transform_inv[5] + ray.direction[2] * transform_inv[6];
        float ray_local_dir_z = ray.direction[0] * transform_inv[8] + ray.direction[1] * transform_inv[9] + ray.direction[2] * transform_inv[10];
        float ray_local_dir_len = sqrt(ray_local_dir_x * ray_local_dir_x + ray_local_dir_y * ray_local_dir_y + ray_local_dir_z * ray_local_dir_z);
        ray_local_dir_x /= ray_local_dir_len;
        ray_local_dir_y /= ray_local_dir_len;
        ray_local_dir_z /= ray_local_dir_len;
        t_local = 10.0 * INF;
        normal_x = 0.0f, normal_y = 0.0f, normal_z = 0.0f;
        if (i < ntriangles) {
            float edge1_x = triangles[i].v1[0] - triangles[i].v0[0];
            float edge1_y = triangles[i].v1[1] - triangles[i].v0[1];
            float edge1_z = triangles[i].v1[2] - triangles[i].v0[2];
            float edge2_x = triangles[i].v2[0] - triangles[i].v0[0];
            float edge2_y = triangles[i].v2[1] - triangles[i].v0[1];
            float edge2_z = triangles[i].v2[2] - triangles[i].v0[2];
            float h_x = ray_local_dir_y * edge2_z - ray_local_dir_z * edge2_y;
            float h_y = ray_local_dir_z * edge2_x - ray_local_dir_x * edge2_z;
            float h_z = ray_local_dir_x * edge2_y - ray_local_dir_y * edge2_x;
            float a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z;
            if (a <= -EPS || a >= EPS) {
                float f = 1.0f / a;
                float s_x = ray_local_ori_x - triangles[i].v0[0];
                float s_y = ray_local_ori_y - triangles[i].v0[1];
                float s_z = ray_local_ori_z - triangles[i].v0[2];
                float u = f * (s_x * h_x + s_y * h_y + s_z * h_z);
                if (u >= 0.0f && u <= 1.0f) {
                    float q_x = s_y * edge1_z - s_z * edge1_y;
                    float q_y = s_z * edge1_x - s_x * edge1_z;
                    float q_z = s_x * edge1_y - s_y * edge1_x;
                    float v = f * (ray_local_dir_x * q_x + ray_local_dir_y * q_y + ray_local_dir_z * q_z);
                    if (v >= 0.0f && u + v <= 1.0f) {
                        t_local = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z);
                        if (t_local >= EPS) {
                            hit = true;
                            normal_x = edge1_y * edge2_z - edge1_z * edge2_y;
                            normal_y = edge1_z * edge2_x - edge1_x * edge2_z;
                            normal_z = edge1_x * edge2_y - edge1_y * edge2_x;
                            float normal_len = sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);
                            normal_x /= normal_len;
                            normal_y /= normal_len;
                            normal_z /= normal_len;
                            float dot = normal_x * ray_local_dir_x + normal_y * ray_local_dir_y + normal_z * ray_local_dir_z;
                            if (dot > 0.0f) {
                                normal_x = -normal_x;
                                normal_y = -normal_y;
                                normal_z = -normal_z;
                            }
                        }
                    }
                }
            }
        }
        else if (i < ntriangles + nspheres) {
            float oc_x = ray_local_ori_x - spheres[i - ntriangles].center[0];
            float oc_y = ray_local_ori_y - spheres[i - ntriangles].center[1];
            float oc_z = ray_local_ori_z - spheres[i - ntriangles].center[2];
            float a = ray_local_dir_x * ray_local_dir_x + ray_local_dir_y * ray_local_dir_y + ray_local_dir_z * ray_local_dir_z;
            float b = 2.0f * (oc_x * ray_local_dir_x + oc_y * ray_local_dir_y + oc_z * ray_local_dir_z);
            float c = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z - spheres[i - ntriangles].radius * spheres[i - ntriangles].radius;
            float discriminant = b * b - 4.0f * a * c;
            if (discriminant >= 0.0f) {
                float sqrtd = sqrt(discriminant);
                float t0 = (-b - sqrtd) / (2.0f * a);
                float t1 = (-b + sqrtd) / (2.0f * a);
                t_local = t0 >= 0.0f ? t0 : t1;
                if (t_local >= EPS) {
                    hit = true;
                    float inter_point_local_x = ray_local_ori_x + t_local * ray_local_dir_x;
                    float inter_point_local_y = ray_local_ori_y + t_local * ray_local_dir_y;
                    float inter_point_local_z = ray_local_ori_z + t_local * ray_local_dir_z;
                    normal_x = inter_point_local_x - spheres[i - ntriangles].center[0];
                    normal_y = inter_point_local_y - spheres[i - ntriangles].center[1];
                    normal_z = inter_point_local_z - spheres[i - ntriangles].center[2];
                    float normal_len = sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);
                    normal_x /= normal_len;
                    normal_y /= normal_len;
                    normal_z /= normal_len;
                }
            }
        }

        if (hit) {
            normal_world_x = normal_x * normal_transform[0] + normal_y * normal_transform[1] + normal_z * normal_transform[2];
            normal_world_y = normal_x * normal_transform[3] + normal_y * normal_transform[4] + normal_z * normal_transform[5];
            normal_world_z = normal_x * normal_transform[6] + normal_y * normal_transform[7] + normal_z * normal_transform[8];
            float normal_world_len = sqrt(normal_world_x * normal_world_x + normal_world_y * normal_world_y + normal_world_z * normal_world_z);
            normal_world_x /= normal_world_len;
            normal_world_y /= normal_world_len;
            normal_world_z /= normal_world_len;
            float intersect_local_x = ray_local_ori_x + t_local * ray_local_dir_x;
            float intersect_local_y = ray_local_ori_y + t_local * ray_local_dir_y;
            float intersect_local_z = ray_local_ori_z + t_local * ray_local_dir_z;
            float intersect_world_x = intersect_local_x * transform[0] + intersect_local_y * transform[1] + intersect_local_z * transform[2] + transform[3];
            float intersect_world_y = intersect_local_x * transform[4] + intersect_local_y * transform[5] + intersect_local_z * transform[6] + transform[7];
            float intersect_world_z = intersect_local_x * transform[8] + intersect_local_y * transform[9] + intersect_local_z * transform[10] + transform[11];
            t_world = sqrt((intersect_world_x - ray.origin[0]) * (intersect_world_x - ray.origin[0]) + (intersect_world_y - ray.origin[1]) * (intersect_world_y - ray.origin[1]) +
                           (intersect_world_z - ray.origin[2]) * (intersect_world_z - ray.origin[2]));
            t_world = t_world < 0.0f ? 10.0f * INF : t_world;
        }
    }

    mem_t[threadIdx.x] = t_world;
    mem_normal[threadIdx.x][0] = normal_world_x;
    mem_normal[threadIdx.x][1] = normal_world_y;
    mem_normal[threadIdx.x][2] = normal_world_z;
    __syncthreads();
    int tx = threadIdx.x;

    for (int sz = 512; sz > 0; sz >>= 1) {
        if (tx < sz) {
            if (mem_t[tx] > mem_t[tx + sz]) {
                mem_t[tx] = mem_t[tx + sz];
                mem_idx[tx] = mem_idx[tx + sz];
                mem_normal[tx][0] = mem_normal[tx + sz][0];
                mem_normal[tx][1] = mem_normal[tx + sz][1];
                mem_normal[tx][2] = mem_normal[tx + sz][2];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        res[blockIdx.x].t = mem_t[0];
        res[blockIdx.x].normal[0] = mem_normal[0][0];
        res[blockIdx.x].normal[1] = mem_normal[0][1];
        res[blockIdx.x].normal[2] = mem_normal[0][2];
        res[blockIdx.x].idx = mem_idx[0];
    }
}

gpuacc::gpuacc() {}

void gpuacc::build(Scene &s) {
    scene = s;
    ntriangles = nspheres = 0;
    for (auto &shape: scene.shapes) {
        if (std::holds_alternative<Triangle>(shape)) ++ntriangles;
        else if (std::holds_alternative<Sphere>(shape))
            ++nspheres;
        else
            assert(false);
    }
    shapes_inv = (uint *)malloc((ntriangles + nspheres) * sizeof(uint));
    thread_size = 1024;
    block_size = (ntriangles + nspheres + thread_size - 1) / thread_size;

    triangles = (gputri *)malloc(ntriangles * sizeof(gputri));
    cudaMalloc(&triangles_device, ntriangles * sizeof(gputri));

    spheres = (gpusph *)malloc(nspheres * sizeof(gpusph));
    cudaMalloc(&spheres_device, nspheres * sizeof(gpusph));

    res = (gpures *)malloc(block_size * sizeof(gpures));
    cudaMalloc(&res_device, block_size * sizeof(gpures));

    uint tri_cnt = 0, sph_cnt = 0;
    for (uint si = 0; si < scene.shapes.size(); ++si) {
        auto &shape = scene.shapes[si];
        glm::mat4 transform = std::visit([](auto &&arg) { return arg.transform; }, shape);
        glm::mat4 transform_inv = glm::inverse(transform);
        glm::mat3 normal_transform = glm::mat3(glm::transpose(glm::inverse(transform)));
        if (std::holds_alternative<Triangle>(shape)) {
            Triangle triobj = std::get<Triangle>(shape);
            gputri tri;
            for (int i = 0; i < 16; ++i) tri.transform[i] = transform[i % 4][i / 4];
            for (int i = 0; i < 16; ++i) tri.transform_inv[i] = transform_inv[i % 4][i / 4];
            for (int i = 0; i < 9; ++i) tri.normal_transform[i] = normal_transform[i % 3][i / 3];
            for (int i = 0; i < 3; ++i) tri.v0[i] = triobj.v0[i], tri.v1[i] = triobj.v1[i], tri.v2[i] = triobj.v2[i];
            shapes_inv[tri_cnt] = si;
            triangles[tri_cnt] = tri;
            ++tri_cnt;
        }
        else if (std::holds_alternative<Sphere>(shape)) {
            Sphere sphobj = std::get<Sphere>(shape);
            gpusph sph;
            for (int i = 0; i < 16; ++i) sph.transform[i] = transform[i % 4][i / 4];
            for (int i = 0; i < 16; ++i) sph.transform_inv[i] = transform_inv[i % 4][i / 4];
            for (int i = 0; i < 9; ++i) sph.normal_transform[i] = normal_transform[i % 3][i / 3];
            for (int i = 0; i < 3; ++i) sph.center[i] = sphobj.center[i];
            sph.radius = sphobj.radius;
            shapes_inv[sph_cnt + ntriangles] = si;
            spheres[sph_cnt] = sph;
            ++sph_cnt;
        }
    }
    cudaMemcpy(triangles_device, triangles, ntriangles * sizeof(gputri), cudaMemcpyHostToDevice);
    cudaMemcpy(spheres_device, spheres, nspheres * sizeof(gpusph), cudaMemcpyHostToDevice);
}

gpuacc::~gpuacc() {
    free(triangles);
    free(spheres);
    free(res);
    free(shapes_inv);
    cudaFree(triangles_device);
    cudaFree(spheres_device);
    cudaFree(res_device);
}

hitobj gpuacc::intersect(const Ray &rayobj) {
    gpuray ray;
    ray.origin[0] = rayobj.origin.x;
    ray.origin[1] = rayobj.origin.y;
    ray.origin[2] = rayobj.origin.z;
    ray.direction[0] = rayobj.direction.x;
    ray.direction[1] = rayobj.direction.y;
    ray.direction[2] = rayobj.direction.z;
    find_intersection<<<block_size, thread_size>>>(triangles_device, spheres_device, ntriangles, nspheres, ray, res_device);
    cudaDeviceSynchronize();

    cudaMemcpy(res, res_device, block_size * sizeof(gpures), cudaMemcpyDeviceToHost);

    hitobj hit;
    for (int i = 0; i < block_size; ++i) {
        if (res[i].t < 0 || res[i].t > hit.t) continue;
        hit.t = res[i].t;
        hit.normal = glm::normalize(glm::vec3(res[i].normal[0], res[i].normal[1], res[i].normal[2]));
        hit.shape = scene.shapes[shapes_inv[res[i].idx]];
    }
    return hit;
}

__global__ void find_multi_intersection(const gputri *triangles, const gpusph *spheres, int ntriangles, int nspheres, const gpuray *rays, int nrays, gpures *res) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nrays) return;

    gpuray ray = rays[id];
    float mint = 10.0f * INF;
    uint minidx = 0;
    float minnormal[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < ntriangles; ++i) {
        float *transform = (float *)triangles[i].transform, *transform_inv = (float *)triangles[i].transform_inv, *normal_transform = (float *)triangles[i].normal_transform;

        bool hit = false;
        float t_local = 10.0f * INF, t_world = 10.0f * INF;
        float normal_x = 0.0f, normal_y = 0.0f, normal_z = 0.0f;
        float normal_world_x = 0.0f, normal_world_y = 0.0f, normal_world_z = 0.0f;

        float ray_local_ori_x = ray.origin[0] * transform_inv[0] + ray.origin[1] * transform_inv[1] + ray.origin[2] * transform_inv[2] + transform_inv[3];
        float ray_local_ori_y = ray.origin[0] * transform_inv[4] + ray.origin[1] * transform_inv[5] + ray.origin[2] * transform_inv[6] + transform_inv[7];
        float ray_local_ori_z = ray.origin[0] * transform_inv[8] + ray.origin[1] * transform_inv[9] + ray.origin[2] * transform_inv[10] + transform_inv[11];
        float ray_local_dir_x = ray.direction[0] * transform_inv[0] + ray.direction[1] * transform_inv[1] + ray.direction[2] * transform_inv[2];
        float ray_local_dir_y = ray.direction[0] * transform_inv[4] + ray.direction[1] * transform_inv[5] + ray.direction[2] * transform_inv[6];
        float ray_local_dir_z = ray.direction[0] * transform_inv[8] + ray.direction[1] * transform_inv[9] + ray.direction[2] * transform_inv[10];
        float ray_local_dir_len = sqrt(ray_local_dir_x * ray_local_dir_x + ray_local_dir_y * ray_local_dir_y + ray_local_dir_z * ray_local_dir_z);
        ray_local_dir_x /= ray_local_dir_len;
        ray_local_dir_y /= ray_local_dir_len;
        ray_local_dir_z /= ray_local_dir_len;
        float edge1_x = triangles[i].v1[0] - triangles[i].v0[0];
        float edge1_y = triangles[i].v1[1] - triangles[i].v0[1];
        float edge1_z = triangles[i].v1[2] - triangles[i].v0[2];
        float edge2_x = triangles[i].v2[0] - triangles[i].v0[0];
        float edge2_y = triangles[i].v2[1] - triangles[i].v0[1];
        float edge2_z = triangles[i].v2[2] - triangles[i].v0[2];
        float h_x = ray_local_dir_y * edge2_z - ray_local_dir_z * edge2_y;
        float h_y = ray_local_dir_z * edge2_x - ray_local_dir_x * edge2_z;
        float h_z = ray_local_dir_x * edge2_y - ray_local_dir_y * edge2_x;
        float a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z;
        if (a <= -EPS || a >= EPS) {
            float f = 1.0f / a;
            float s_x = ray_local_ori_x - triangles[i].v0[0];
            float s_y = ray_local_ori_y - triangles[i].v0[1];
            float s_z = ray_local_ori_z - triangles[i].v0[2];
            float u = f * (s_x * h_x + s_y * h_y + s_z * h_z);
            if (u >= 0.0f && u <= 1.0f) {
                float q_x = s_y * edge1_z - s_z * edge1_y;
                float q_y = s_z * edge1_x - s_x * edge1_z;
                float q_z = s_x * edge1_y - s_y * edge1_x;
                float v = f * (ray_local_dir_x * q_x + ray_local_dir_y * q_y + ray_local_dir_z * q_z);
                if (v >= 0.0f && u + v <= 1.0f) {
                    t_local = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z);
                    if (t_local >= EPS) {
                        hit = true;
                        normal_x = edge1_y * edge2_z - edge1_z * edge2_y;
                        normal_y = edge1_z * edge2_x - edge1_x * edge2_z;
                        normal_z = edge1_x * edge2_y - edge1_y * edge2_x;
                        float normal_len = sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);
                        normal_x /= normal_len;
                        normal_y /= normal_len;
                        normal_z /= normal_len;
                        float dot = normal_x * ray_local_dir_x + normal_y * ray_local_dir_y + normal_z * ray_local_dir_z;
                        if (dot > 0.0f) {
                            normal_x = -normal_x;
                            normal_y = -normal_y;
                            normal_z = -normal_z;
                        }
                    }
                }
            }
        }
        if (hit) {
            normal_world_x = normal_x * normal_transform[0] + normal_y * normal_transform[1] + normal_z * normal_transform[2];
            normal_world_y = normal_x * normal_transform[3] + normal_y * normal_transform[4] + normal_z * normal_transform[5];
            normal_world_z = normal_x * normal_transform[6] + normal_y * normal_transform[7] + normal_z * normal_transform[8];
            float normal_world_len = sqrt(normal_world_x * normal_world_x + normal_world_y * normal_world_y + normal_world_z * normal_world_z);
            normal_world_x /= normal_world_len;
            normal_world_y /= normal_world_len;
            normal_world_z /= normal_world_len;
            float intersect_local_x = ray_local_ori_x + t_local * ray_local_dir_x;
            float intersect_local_y = ray_local_ori_y + t_local * ray_local_dir_y;
            float intersect_local_z = ray_local_ori_z + t_local * ray_local_dir_z;
            float intersect_world_x = intersect_local_x * transform[0] + intersect_local_y * transform[1] + intersect_local_z * transform[2] + transform[3];
            float intersect_world_y = intersect_local_x * transform[4] + intersect_local_y * transform[5] + intersect_local_z * transform[6] + transform[7];
            float intersect_world_z = intersect_local_x * transform[8] + intersect_local_y * transform[9] + intersect_local_z * transform[10] + transform[11];
            t_world = sqrt((intersect_world_x - ray.origin[0]) * (intersect_world_x - ray.origin[0]) + (intersect_world_y - ray.origin[1]) * (intersect_world_y - ray.origin[1]) +
                           (intersect_world_z - ray.origin[2]) * (intersect_world_z - ray.origin[2]));
            t_world = t_world < 0.0f ? 10.0f * INF : t_world;
            if (t_world < mint) {
                mint = t_world;
                minidx = i;
                minnormal[0] = normal_world_x;
                minnormal[1] = normal_world_y;
                minnormal[2] = normal_world_z;
            }
        }
    }
    for (int i = 0; i < nspheres; ++i) {
        float *transform, *transform_inv, *normal_transform;
        transform = (float *)spheres[i].transform, transform_inv = (float *)spheres[i].transform_inv, normal_transform = (float *)spheres[i].normal_transform;

        bool hit = false;
        float t_local = 10.0f * INF, t_world = 10.0f * INF;
        float normal_x = 0.0f, normal_y = 0.0f, normal_z = 0.0f;
        float normal_world_x = 0.0f, normal_world_y = 0.0f, normal_world_z = 0.0f;

        float ray_local_ori_x = ray.origin[0] * transform_inv[0] + ray.origin[1] * transform_inv[1] + ray.origin[2] * transform_inv[2] + transform_inv[3];
        float ray_local_ori_y = ray.origin[0] * transform_inv[4] + ray.origin[1] * transform_inv[5] + ray.origin[2] * transform_inv[6] + transform_inv[7];
        float ray_local_ori_z = ray.origin[0] * transform_inv[8] + ray.origin[1] * transform_inv[9] + ray.origin[2] * transform_inv[10] + transform_inv[11];
        float ray_local_dir_x = ray.direction[0] * transform_inv[0] + ray.direction[1] * transform_inv[1] + ray.direction[2] * transform_inv[2];
        float ray_local_dir_y = ray.direction[0] * transform_inv[4] + ray.direction[1] * transform_inv[5] + ray.direction[2] * transform_inv[6];
        float ray_local_dir_z = ray.direction[0] * transform_inv[8] + ray.direction[1] * transform_inv[9] + ray.direction[2] * transform_inv[10];
        float ray_local_dir_len = sqrt(ray_local_dir_x * ray_local_dir_x + ray_local_dir_y * ray_local_dir_y + ray_local_dir_z * ray_local_dir_z);
        ray_local_dir_x /= ray_local_dir_len;
        ray_local_dir_y /= ray_local_dir_len;
        ray_local_dir_z /= ray_local_dir_len;

        float oc_x = ray_local_ori_x - spheres[i].center[0];
        float oc_y = ray_local_ori_y - spheres[i].center[1];
        float oc_z = ray_local_ori_z - spheres[i].center[2];
        float a = ray_local_dir_x * ray_local_dir_x + ray_local_dir_y * ray_local_dir_y + ray_local_dir_z * ray_local_dir_z;
        float b = 2.0f * (oc_x * ray_local_dir_x + oc_y * ray_local_dir_y + oc_z * ray_local_dir_z);
        float c = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z - spheres[i].radius * spheres[i].radius;
        float discriminant = b * b - 4.0f * a * c;
        if (discriminant >= 0.0f) {
            float sqrtd = sqrt(discriminant);
            float t0 = (-b - sqrtd) / (2.0f * a);
            float t1 = (-b + sqrtd) / (2.0f * a);
            t_local = t0 >= 0.0f ? t0 : t1;
            if (t_local >= EPS) {
                hit = true;
                float inter_point_local_x = ray_local_ori_x + t_local * ray_local_dir_x;
                float inter_point_local_y = ray_local_ori_y + t_local * ray_local_dir_y;
                float inter_point_local_z = ray_local_ori_z + t_local * ray_local_dir_z;
                normal_x = inter_point_local_x - spheres[i].center[0];
                normal_y = inter_point_local_y - spheres[i].center[1];
                normal_z = inter_point_local_z - spheres[i].center[2];
                float normal_len = sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z);
                normal_x /= normal_len;
                normal_y /= normal_len;
                normal_z /= normal_len;
            }
        }
        if (hit) {
            normal_world_x = normal_x * normal_transform[0] + normal_y * normal_transform[1] + normal_z * normal_transform[2];
            normal_world_y = normal_x * normal_transform[3] + normal_y * normal_transform[4] + normal_z * normal_transform[5];
            normal_world_z = normal_x * normal_transform[6] + normal_y * normal_transform[7] + normal_z * normal_transform[8];
            float normal_world_len = sqrt(normal_world_x * normal_world_x + normal_world_y * normal_world_y + normal_world_z * normal_world_z);
            normal_world_x /= normal_world_len;
            normal_world_y /= normal_world_len;
            normal_world_z /= normal_world_len;
            float intersect_local_x = ray_local_ori_x + t_local * ray_local_dir_x;
            float intersect_local_y = ray_local_ori_y + t_local * ray_local_dir_y;
            float intersect_local_z = ray_local_ori_z + t_local * ray_local_dir_z;
            float intersect_world_x = intersect_local_x * transform[0] + intersect_local_y * transform[1] + intersect_local_z * transform[2] + transform[3];
            float intersect_world_y = intersect_local_x * transform[4] + intersect_local_y * transform[5] + intersect_local_z * transform[6] + transform[7];
            float intersect_world_z = intersect_local_x * transform[8] + intersect_local_y * transform[9] + intersect_local_z * transform[10] + transform[11];
            t_world = sqrt((intersect_world_x - ray.origin[0]) * (intersect_world_x - ray.origin[0]) + (intersect_world_y - ray.origin[1]) * (intersect_world_y - ray.origin[1]) +
                           (intersect_world_z - ray.origin[2]) * (intersect_world_z - ray.origin[2]));
            t_world = t_world < 0.0f ? 10.0f * INF : t_world;
            if (t_world < mint) {
                mint = t_world;
                minidx = i + ntriangles;
                minnormal[0] = normal_world_x;
                minnormal[1] = normal_world_y;
                minnormal[2] = normal_world_z;
            }
        }
    }
    res[id].t = mint;
    res[id].normal[0] = minnormal[0];
    res[id].normal[1] = minnormal[1];
    res[id].normal[2] = minnormal[2];
    res[id].idx = minidx;
}

std::vector<hitobj> gpuacc::intersect_multi(std::vector<Ray> &rayobjs) {
    gpuray *ray = (gpuray *)malloc(rayobjs.size() * sizeof(gpuray));
    gpuray *ray_device;
    cudaMalloc(&ray_device, rayobjs.size() * sizeof(gpuray));
    gpures *res = (gpures *)malloc(rayobjs.size() * sizeof(gpures));
    gpures *res_device;
    cudaMalloc(&res_device, rayobjs.size() * sizeof(gpures));

    for (uint i = 0; i < rayobjs.size(); ++i) {
        ray[i].origin[0] = rayobjs[i].origin.x;
        ray[i].origin[1] = rayobjs[i].origin.y;
        ray[i].origin[2] = rayobjs[i].origin.z;
        ray[i].direction[0] = rayobjs[i].direction.x;
        ray[i].direction[1] = rayobjs[i].direction.y;
        ray[i].direction[2] = rayobjs[i].direction.z;
    }
    cudaMemcpy(ray_device, ray, rayobjs.size() * sizeof(gpuray), cudaMemcpyHostToDevice);

    // DONE:  kernel
    uint nblocks = (rayobjs.size() + thread_size - 1) / thread_size;
    find_multi_intersection<<<nblocks, thread_size>>>(triangles_device, spheres_device, ntriangles, nspheres, ray_device, (int)rayobjs.size(), res_device);
    cudaDeviceSynchronize();

    cudaMemcpy(res, res_device, rayobjs.size() * sizeof(gpures), cudaMemcpyDeviceToHost);
    std::vector<hitobj> hits(rayobjs.size());

    for (uint i = 0; i < rayobjs.size(); ++i) {
        hits[i].t = res[i].t;
        if (hits[i].t < EPS || hits[i].t > INF) continue;
        hits[i].normal = glm::vec3(res[i].normal[0], res[i].normal[1], res[i].normal[2]);
        hits[i].shape = scene.shapes[shapes_inv[res[i].idx]];
    }

    free(ray);
    cudaFree(ray_device);
    free(res);
    cudaFree(res_device);
    return hits;
}

