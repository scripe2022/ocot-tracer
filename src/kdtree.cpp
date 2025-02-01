// comp := make
// run  := time ./raytracer scenes/test4/cornellCosine.test && kcat output/cornellCosine.png

#include <variant>
#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <algorithm>

#include "kdtree.hpp"

AABB::AABB(const Shape &sp) {
    this->shape = sp;
    glm::mat4 transform = std::visit([](auto &&arg) { return arg.transform; }, sp);
    if (std::holds_alternative<Triangle>(sp)) {
        Triangle tri = std::get<Triangle>(sp);
        glm::vec3 v0t = glm::vec3(transform * glm::vec4(tri.v0, 1.0f));
        glm::vec3 v1t = glm::vec3(transform * glm::vec4(tri.v1, 1.0f));
        glm::vec3 v2t = glm::vec3(transform * glm::vec4(tri.v2, 1.0f));
        this->min = glm::min(v0t, glm::min(v1t, v2t));
        this->max = glm::max(v0t, glm::max(v1t, v2t));
    }
    else {
        Sphere sphere = std::get<Sphere>(sp);
        glm::vec3 scale = glm::vec3(glm::length(transform[0]), glm::length(transform[1]), glm::length(transform[2]));
        float trans_radius = sphere.radius * glm::max(scale.x, glm::max(scale.y, scale.z));
        glm::vec3 trans_center = glm::vec3(transform * glm::vec4(sphere.center, 1.0f));
        this->min = trans_center - glm::vec3(trans_radius);
        this->max = trans_center + glm::vec3(trans_radius);
    }
}

bool node::intersect(const Ray &ray, float &t) {
    float tmin, tmax, tymin, tymax, tzmin, tzmax;
    if (ray.direction.x != 0.0) {
        tmin = (min.x - ray.origin.x) / ray.direction.x;
        tmax = (max.x - ray.origin.x) / ray.direction.x;
        if (tmin > tmax) std::swap(tmin, tmax);
    }
    else {
        tmin = (ray.origin.x >= min.x && ray.origin.x <= max.x) ? -INF : INF;
        tmax = (ray.origin.x >= min.x && ray.origin.x <= max.x) ? INF : -INF;
    }
    if (ray.direction.y != 0.0) {
        tymin = (min.y - ray.origin.y) / ray.direction.y;
        tymax = (max.y - ray.origin.y) / ray.direction.y;
        if (tymin > tymax) std::swap(tymin, tymax);
    }
    else {
        tymin = (ray.origin.y >= min.y && ray.origin.y <= max.y) ? -INF : INF;
        tymax = (ray.origin.y >= min.y && ray.origin.y <= max.y) ? INF : -INF;
    }
    if (ray.direction.z != 0.0) {
        tzmin = (min.z - ray.origin.z) / ray.direction.z;
        tzmax = (max.z - ray.origin.z) / ray.direction.z;
        if (tzmin > tzmax) std::swap(tzmin, tzmax);
    }
    else {
        tzmin = (ray.origin.z >= min.z && ray.origin.z <= max.z) ? -INF : INF;
        tzmax = (ray.origin.z >= min.z && ray.origin.z <= max.z) ? INF : -INF;
    }
    if ((tmin > tymax) || (tymin > tmax)) return false;
    tmin = std::max(tmin, tymin);
    tmax = std::min(tmax, tymax);
    if ((tmin > tzmax) || (tzmin > tmax)) return false;
    tmin = std::max(tmin, tzmin);
    tmax = std::min(tmax, tzmax);
    t = tmin < 0 ? tmax : tmin;
    return t >= 0;
}

float kdtree::split_plane(const std::vector<AABB> &aabbs, int axis) {
    std::vector<float> edges;
    for (auto aabb: aabbs) {
        edges.emplace_back(aabb.min[axis]);
        edges.emplace_back(aabb.max[axis]);
    }
    std::sort(edges.begin(), edges.end());
    return edges[edges.size() >> 1];
}

node* kdtree::build_tree(std::vector<AABB> &aabbs, int depth) {
    if (aabbs.empty()) return nullptr;
    node *x = new node();
    if (depth == 0) {
        for (auto aabb: aabbs) {
            x->min = glm::min(x->min, aabb.min);
            x->max = glm::max(x->max, aabb.max);
        }
    }
    if (aabbs.size() <= KD_TREE_MIN_NODE_SIZE || depth >= KD_TREE_MAX_DEPTH) {
        x->aabbs = aabbs;
        x->ch[0] = x->ch[1] = nullptr;
        return x;
    }
    glm::vec3 diff = x->max - x->min;
    x->axis = (diff.x > diff.y && diff.x > diff.z) ? 0 : (diff.y > diff.z ? 1 : 2);
    x->split = this->split_plane(aabbs, x->axis);
    std::vector<AABB> l, r;
    for (auto aabb: aabbs) {
        /* if (aabb.max[x->axis] <= x->split) l.push_back(aabb); */
        /* else r.push_back(aabb); */
        if (aabb.min[x->axis] < x->split) l.push_back(aabb);
        if (aabb.max[x->axis] > x->split) r.push_back(aabb);
    }
    x->ch[0] = build_tree(l, depth + 1), x->ch[1] = build_tree(r, depth + 1);
    if (x->ch[0]) x->ch[0]->min = x->min, x->ch[0]->max[x->axis] = x->split;
    if (x->ch[1]) x->ch[1]->max = x->max, x->ch[1]->min[x->axis] = x->split;
    return x;
}

kdtree::kdtree() {

}

void kdtree::build(Scene &scene) {
    scene_orig = scene;
    std::vector<AABB> aabbs;
    for (auto &shape: scene.shapes) aabbs.push_back(AABB(shape));
    root = build_tree(aabbs, 0);
}

inline bool kdtree::in_box(const glm::vec3 &origin, const glm::vec3 &min, const glm::vec3 &max) {
    return origin.x >= min.x && origin.x <= max.x &&
           origin.y >= min.y && origin.y <= max.y &&
           origin.z >= min.z && origin.z <= max.z;
}

bool kdtree::dfs(node *x, const Ray &ray, hitobj &hit) {
    if (!x) return false;
    float boxt = -1; bool boxhit = x->intersect(ray, boxt);
    if (!boxhit || (!in_box(ray.origin, x->min, x->max) && boxt > hit.t)) return false;
    /* if (!boxhit || boxt > hit.t) return false; */
    if (!x->ch[0] && !x->ch[1]) {
        bool shapehit = false;
        for (auto &aabb: x->aabbs) {
            auto [t, n] = shape_intersect(ray, aabb.shape);
            if (t < EPS || t > hit.t) continue;
            hit.t = t;
            hit.normal = n;
            hit.shape = aabb.shape;
            shapehit = true;
        }
        return shapehit;
    }
    else {
        bool lhit = dfs(x->ch[0], ray, hit), rhit = dfs(x->ch[1], ray, hit);
        return lhit || rhit;
    }
}

hitobj kdtree::intersect(const Ray &ray) {
    hitobj hit;
    // bool flag = dfs(root, ray, hit);
    // if (!flag) hit.t = -1;
    // return hit;
    for (auto &shape: scene_orig.shapes) {
        auto [t, n] = shape_intersect(ray, shape);
        if (t < EPS || t > hit.t) continue;
        hit.t = t;
        hit.normal = n;
        hit.shape = shape;
    }
    return hit;
}


