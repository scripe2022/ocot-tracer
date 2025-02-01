#ifndef DIRECT_LIGHT_HPP
#define DIRECT_LIGHT_HPP

#include "accelerator.hpp"
#include "image.hpp"
#include "parse_scene.hpp"

namespace direct {
bool visible_to_point(accelerator &acc, glm::vec3 &light, glm::vec3 &p, glm::vec3 &n);
glm::vec3 get_irradiance(accelerator &acc, Quad &quad, Scene &scene, Ray &ray, hitobj &hit);
glm::vec3 tracing(accelerator &acc, Scene &scene, Ray &ray);
void render_segment(accelerator &acc, Image &img, Scene &scene, uint start, uint end, bool show_progress);
void run(Scene &scene, Image &img, int mode, bool show_progress);
}

#endif
