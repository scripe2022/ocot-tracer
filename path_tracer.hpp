#ifndef PATH_TRACER_HPP 
#define PATH_TRACER_HPP

#include "accelerator.hpp"
#include "image.hpp"
#include "parse_scene.hpp"

namespace path_tracer {
// bool visible_to_point(accelerator &acc, const glm::vec3 &light, const glm::vec3 &p, const glm::vec3 &n);
// glm::vec3 get_irradiance(accelerator &acc, Quad &quad, Scene &scene, Ray &ray, hitobj &hit);
// glm::vec3 tracing(accelerator &acc, Scene &scene, Ray &ray);
// void render_segment(accelerator &acc, Image &img, Scene &scene, uint start, uint end, bool show_progress);
Image run(Scene &s, int mode, bool show_progress);
// glm::vec3 trace(Ray &ray, uint depth);
// glm::vec3 trace(accelerator &acc, Scene &scene, Ray &ray, uint depth);
}

#endif
