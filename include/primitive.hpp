#ifndef PRIMITIVE_HPP
#define PRIMITIVE_HPP

#include "parse_scene.hpp"

#define EPS 5e-5f
// #define EPS 0.0001f
#define INF 1e10f

struct hitobj {
    float t = 10.0f * INF;
    glm::vec3 normal;
    Shape shape;
    bool hit() { return t > EPS && t < INF; }
};

Ray transform_ray(const Ray &ray, const glm::mat4 &transform);
Ray ray_thru_pixel(const Scene &scene, const float i, const float j);
bool triangle_intersect(const Ray &ray, const Triangle &tri, float &t, glm::vec3 &normal);
bool sphere_intersect(const Ray &ray, const Sphere &sphere, float &t, glm::vec3 &normal);
std::pair<float, glm::vec3> shape_intersect(const Ray &ray, const Shape shape);
glm::vec3 brdf_const(const hitobj &hit);
float D(float half_angle, float r);
float G(glm::vec3 n, glm::vec3 v, float r);
glm::vec3 F(glm::vec3 v, glm::vec3 half_vec, glm::vec3 specular);
glm::vec3 brdf(hitobj &hit, glm::vec3 w_i, glm::vec3 w_o);
float pdf(hitobj &hit, glm::vec3 w_i, glm::vec3 w_o);
std::vector<std::pair<float, float>> gen_sample(uint n, bool stratified);
float gterm(glm::vec3 x, glm::vec3 n, glm::vec3 xp, glm::vec3 n_l);
float gen_real();
float quad_intersect(Ray ray, Quad quad);

#endif
