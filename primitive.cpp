// comp := make
// run  := time ./raytracer scenes/test3/cornellNEE.test && kcat output/cornellNEE.png
// run  := time ./raytracer cornellNEE-small.test && kcat output/cornellNEE-small.png

#include <chrono>
#include <cmath>
#include <random>

#include "parse_scene.hpp"
#include "primitive.hpp"

Ray transform_ray(const Ray &ray, const glm::mat4 &transform) {
    glm::mat4 transform_inv = glm::inverse(transform);
    Ray transformed_ray;
    transformed_ray.origin = glm::vec3(transform_inv * glm::vec4(ray.origin, 1.0f));
    transformed_ray.direction = glm::normalize(glm::vec3(transform_inv * glm::vec4(ray.direction, 0.0f)));
    return transformed_ray;
}

Ray ray_thru_pixel(const Scene &scene, const float i, const float j) {
    float width = static_cast<float>(scene.width);
    float height = static_cast<float>(scene.height);
    float aspect = width / height;
    float fovy_rad = glm::radians(static_cast<float>(scene.camera.fovy));
    float fovx_rad = 2 * atan(aspect * tan(fovy_rad / 2));
    float alpha = ((j - (width / 2)) / (width / 2)) * tan(fovx_rad / 2);
    float beta = ((i - (height / 2)) / (height / 2)) * tan(fovy_rad / 2);
    glm::vec3 w = glm::normalize(scene.camera.lookat - scene.camera.pos);
    glm::vec3 u = glm::normalize(glm::cross(scene.camera.up, w));
    glm::vec3 v = glm::cross(w, u);
    glm::vec3 ray_dir = glm::normalize(alpha * u + beta * v - w);
    Ray ray;
    ray.origin = scene.camera.pos;
    ray.direction = -ray_dir;
    return ray;
}

bool triangle_intersect(const Ray &ray, const Triangle &tri, float &t, glm::vec3 &normal) {
    glm::vec3 edge1 = tri.v1 - tri.v0, edge2 = tri.v2 - tri.v0, h, s, q;
    float a, f;
    h = glm::cross(ray.direction, edge2);
    a = glm::dot(edge1, h);
    if (a > -EPS && a < EPS) return false;
    f = 1.0 / a;
    s = ray.origin - tri.v0;
    float u = f * glm::dot(s, h);
    if (u < 0.0 || u > 1.0) return false;
    q = glm::cross(s, edge1);
    float v = f * glm::dot(ray.direction, q);
    if (v < 0.0 || u + v > 1.0) return false;
    t = f * glm::dot(edge2, q);
    if (tri.has_normal) normal = glm::normalize((1 - u - v) * tri.n0 + u * tri.n1 + v * tri.n2);
    else
        normal = glm::normalize(glm::cross(edge1, edge2));
    if (glm::dot(normal, ray.direction) > 0) normal = -normal;
    return t >= EPS;
}

bool sphere_intersect(const Ray &ray, const Sphere &sphere, float &t, glm::vec3 &normal) {
    glm::vec3 oc = ray.origin - sphere.center;
    float a = glm::dot(ray.direction, ray.direction);
    float b = 2.0f * glm::dot(oc, ray.direction);
    float c = glm::dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return false;
    float sqrtd = sqrt(discriminant);
    float t0 = (-b - sqrtd) / (2.0f * a);
    float t1 = (-b + sqrtd) / (2.0f * a);
    t = t0 >= 0 ? t0 : t1;
    if (t < EPS) return false;
    glm::vec3 intersection_point_local = ray.origin + ray.direction * t;
    normal = glm::normalize(intersection_point_local - sphere.center);
    return true;
}

std::pair<float, glm::vec3> shape_intersect(const Ray &ray, const Shape shape) {
    glm::mat4 transform = std::visit([](auto &&arg) { return arg.transform; }, shape);
    glm::mat3 normal_transform = glm::mat3(glm::transpose(glm::inverse(transform)));
    Ray ray_local = transform_ray(ray, transform);
    float t_local;
    bool hit = false;
    glm::vec3 normal_local;
    if (std::holds_alternative<Triangle>(shape)) hit = triangle_intersect(ray_local, std::get<Triangle>(shape), t_local, normal_local);
    else if (std::holds_alternative<Sphere>(shape))
        hit = sphere_intersect(ray_local, std::get<Sphere>(shape), t_local, normal_local);
    if (!hit) return std::make_pair(-1, normal_local);
    glm::vec3 normal_world = glm::normalize(normal_transform * normal_local);
    glm::vec3 intersect_local = ray_local.origin + ray_local.direction * t_local;
    glm::vec3 intersect_world = glm::vec3(transform * glm::vec4(intersect_local, 1.0f));
    float t_world = glm::length(intersect_world - ray.origin) / glm::length(ray.direction);
    return (t_world < EPS) ? std::make_pair(-1.0f, normal_local) : std::make_pair(t_world, normal_world);
}

glm::vec3 brdf_const(const hitobj &hit) {
    glm::vec3 diffuse = std::visit([](auto &&arg) { return arg.diffuse; }, hit.shape);
    return diffuse / (float)M_PI;
}

// NOTE: w_i = hit -> next_hit normalized
// NOTE: w_o = hit -> ray.origin(camera)
float D(float half_angle, float r) {
    return (r * r) / ((float)M_PI * glm::pow(glm::cos(half_angle), 4) * glm::pow(r * r + glm::pow(glm::tan(half_angle), 2), 2));
}

float G(glm::vec3 n, glm::vec3 v, float r) {
    if (glm::dot(n, v) <= 0) return 0;
    float thv = glm::acos(glm::dot(n, v));
    return 2.0f / (1.0f + glm::sqrt(glm::max(0.0f, 1.0f + glm::pow(r, 2.0f) * glm::pow(glm::tan(thv), 2.0f))));
}

glm::vec3 F(glm::vec3 v, glm::vec3 half_vec, glm::vec3 specular) {
    return specular + (glm::vec3(1.0f) - specular) * glm::pow(1.0f - glm::dot(v, half_vec), 5.0f);
}

glm::vec3 brdf(hitobj &hit, glm::vec3 w_i, glm::vec3 w_o) {
    std::string brdf_type = std::visit([](auto &&arg) { return arg.brdf; }, hit.shape);
    float roughness = std::visit([](auto &&arg) { return arg.roughness; }, hit.shape);
    glm::vec3 specular = std::visit([](auto &&arg) { return arg.specular; }, hit.shape);
    glm::vec3 diffuse = std::visit([](auto &&arg) { return arg.diffuse; }, hit.shape);
    float shininess = std::visit([](auto &&arg) { return arg.shininess; }, hit.shape);

    auto get_ggx_brdf = [&]() -> glm::vec3 {
        if (glm::dot(hit.normal, w_i) < 0 || glm::dot(hit.normal, w_o) < 0) return glm::vec3(0);
        glm::vec3 half_vec = glm::normalize(glm::normalize(w_i) + glm::normalize(w_o));
        float half_angle = glm::acos(glm::min(1.0f, glm::dot(half_vec, hit.normal)));
        float d = D(half_angle, roughness);
        float g = G(hit.normal, w_i, roughness) * G(hit.normal, w_o, roughness);
        glm::vec3 f = F(w_i, hit.normal, specular);
        return diffuse / (float)M_PI + d * g * f / (4.0f * glm::dot(w_i, hit.normal) * glm::dot(w_o, hit.normal));
    };

    auto get_phong_brdf = [&]() -> glm::vec3 {
        glm::vec3 reflection = 2.0f * glm::dot(hit.normal, w_o) * hit.normal - w_o;
        glm::vec3 d = diffuse / (float)M_PI;
        glm::vec3 s = specular * (shininess + 2.0f) / (2.0f * (float)M_PI) * glm::pow(glm::dot(reflection, w_i), shininess);
        return d + s;
    };

    return brdf_type == "ggx" ? get_ggx_brdf() : get_phong_brdf();
}

float pdf(hitobj &hit, glm::vec3 w_i, glm::vec3 w_o) {
    glm::vec3 specular = std::visit([](auto &&arg) { return arg.specular; }, hit.shape);
    glm::vec3 diffuse = std::visit([](auto &&arg) { return arg.diffuse; }, hit.shape);
    float roughness = std::visit([](auto &&arg) { return arg.roughness; }, hit.shape);
    float shininess = std::visit([](auto &&arg) { return arg.shininess; }, hit.shape);
    glm::vec3 half_vec = glm::normalize(glm::normalize(w_i) + glm::normalize(w_o));
    float half_angle = glm::acos(glm::min(1.0f, glm::dot(half_vec, hit.normal)));
    std::string brdf_type = std::visit([](auto &&arg) { return arg.brdf; }, hit.shape);
    float ks = (specular.x + specular.y + specular.z) / 3.0f;
    float kd = (diffuse.x + diffuse.y + diffuse.z) / 3.0f;
    float t = ks / (ks + kd);

    auto ggx_pdf = [&]() -> float {
        float t_th = glm::max(0.25f, t);
        float d = (1.0f - t_th) * glm::dot(hit.normal, w_i) / (float)M_PI;
        float g = t_th * D(half_angle, roughness) * glm::dot(hit.normal, half_vec) / (4.0f * glm::dot(half_vec, w_i));
        return d + g;
    };

    auto phong_pdf = [&]() -> float {
        glm::vec3 reflection = (2 * glm::dot(hit.normal, w_o) * hit.normal - w_o);
        float d = (1 - t) * (glm::max(0.0f, glm::dot(w_i, hit.normal))) / (float)M_PI;
        float s = t * (shininess + 1) / (2 * (float)M_PI) * glm::pow(glm::max(0.0f, glm::dot(reflection, w_i)), shininess);
        return d + s;
    };

    return brdf_type == "ggx" ? ggx_pdf() : phong_pdf();
}

static std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
// static std::mt19937 rng(42);
static std::uniform_real_distribution<> dis(0.0f, 1.0f);

std::vector<std::pair<float, float>> gen_sample(uint n, bool stratified) {
    std::vector<std::pair<float, float>> res;
    if (!stratified) {
        for (uint i = 0; i < n; ++i) res.emplace_back(dis(rng), dis(rng));
        return res;
    }
    uint sqrt_n = (uint)(sqrt(n));
    float unit = 1.0f / sqrt_n;
    for (uint i = 0; i < sqrt_n; ++i) {
        for (uint j = 0; j < sqrt_n; ++j) { res.emplace_back(dis(rng) / sqrt_n + unit * i, dis(rng) / sqrt_n + unit * j); }
    }
    return res;
}

float gen_real() { return dis(rng); }

float gterm(glm::vec3 x, glm::vec3 n, glm::vec3 xp, glm::vec3 n_l) {
    glm::vec3 lightvec = glm::normalize(xp - x);
    float cosi = glm::dot(lightvec, n);
    float coso = glm::dot(lightvec, -n_l);
    return glm::max(cosi, 0.0f) * glm::max(coso, 0.0f) / glm::pow(glm::length(xp - x), 2);
}

float quad_intersect(Ray ray, Quad quad) {
    glm::vec3 normal = glm::cross(quad.ab, quad.ac);
    normal = glm::normalize(normal);
    glm::vec3 p0_to_origin = quad.a - ray.origin;

    float denominator = glm::dot(ray.direction, normal);
    if (denominator == 0) return -1;
    float t = glm::dot(p0_to_origin, normal) / denominator;
    if (t < 0) { return -1; }
    glm::vec3 intersection = ray.origin + t * ray.direction;
    glm::vec3 pa = intersection - quad.a;

    float d00 = glm::dot(quad.ac, quad.ac);
    float d01 = glm::dot(quad.ac, quad.ab);
    float d11 = glm::dot(quad.ab, quad.ab);
    float d20 = glm::dot(pa, quad.ac);
    float d21 = glm::dot(pa, quad.ab);
    float denom = d00 * d11 - d01 * d01;
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    if (v >= 0.0f && w >= 0.0f && v <= 1.0f && w <= 1.0f) return t;
    return -1.0f;
}

