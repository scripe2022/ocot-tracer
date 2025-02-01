#ifndef ANALYTIC_DIRECT_HPP
#define ANALYTIC_DIRECT_HPP

#include "accelerator.hpp"
#include "image.hpp"
#include "parse_scene.hpp"

namespace analytic_direct {
glm::vec3 get_irradiance(Quad &quad, Ray &ray, hitobj &hit);
glm::vec3 tracing(accelerator &acc, Scene &scene, Ray &ray);
void render_segment(accelerator &acc, Image &img, Scene &scene, uint start, uint end);
void run(Scene &scene, Image &img);
}

#endif
