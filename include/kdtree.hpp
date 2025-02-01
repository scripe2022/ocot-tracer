#ifndef KDTREE_HPP
#define KDTREE_HPP

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include "primitive.hpp"

#define KD_TREE_MIN_NODE_SIZE 10
#define KD_TREE_MAX_DEPTH 4

struct AABB {
    glm::vec3 min = glm::vec3(-INF), max = glm::vec3(INF);
    Shape shape;
    AABB(const Shape &sp);
};

struct node {
    glm::vec3 min = glm::vec3(-INF), max = glm::vec3(INF);
    node *ch[2] = {nullptr, nullptr};
    std::vector<AABB> aabbs;
    int axis = 0;
    float split = 0;
    bool intersect(const Ray &ray, float &t);
};

struct kdtree {
    Scene scene_orig;
    node *root = nullptr;
    float split_plane(const std::vector<AABB> &aabbs, int axis);
    node *build_tree(std::vector<AABB> &aabbs, int depth);
    kdtree();
    void build(Scene &scene);
    inline bool in_box(const glm::vec3 &origin, const glm::vec3 &min, const glm::vec3 &max);
    bool dfs(node *x, const Ray &ray, hitobj &hit);
    hitobj intersect(const Ray &ray);
};

#endif

