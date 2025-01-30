#ifndef RAY_TRACING_HPP
#define RAY_TRACING_HPP

#include "accelerator.hpp"
#include "image.hpp"
#include "parse_scene.hpp"

namespace raytracer {
bool visible_to_light(accelerator &acc, const Light &light, const glm::vec3 &p, const glm::vec3 &L);
glm::vec3 get_intensity(accelerator &acc, const Scene &scene, const Shape &shape, const Ray &ray, const glm::vec3 &p, glm::vec3 &N);
glm::vec3 ray_tracing(accelerator &acc, Scene &scene, Ray &ray, uint depth, glm::vec3 weight);
void render_segment(accelerator &acc, Image &img, Scene &scene, uint start, uint end, bool show_progress);
void run(Scene &scene, Image &img, bool multi_processes, bool show_progress);
}

#endif
