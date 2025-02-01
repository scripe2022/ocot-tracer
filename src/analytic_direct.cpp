#include "analytic_direct.hpp"
#include <thread>
#include "accelerator.hpp"

namespace analytic_direct {
glm::vec3 get_irradiance(Quad &quad, Ray &ray, hitobj &hit) {
    auto get_theta = [](glm::vec3 &v, glm::vec3 &vn, glm::vec3 &r) -> float {
        glm::vec3 a = glm::normalize(v - r);
        glm::vec3 b = glm::normalize(vn - r);
        return glm::acos(glm::dot(a, b));
    };
    auto get_gamma = [](glm::vec3 &v, glm::vec3 &vn, glm::vec3 &r) -> glm::vec3 { return glm::normalize(glm::cross(v - r, vn - r)); };

    glm::vec3 p = ray.origin + ray.direction * hit.t;
    glm::vec3 a = quad.a;
    glm::vec3 b = quad.a + quad.ab;
    glm::vec3 d = quad.a + quad.ac;
    glm::vec3 c = quad.a + quad.ab + quad.ac;
    glm::vec3 n = glm::normalize(glm::cross(quad.ab, quad.ac));
    glm::vec3 phi = get_theta(a, b, p) * get_gamma(a, b, p);
    phi += get_theta(b, c, p) * get_gamma(b, c, p);
    phi += get_theta(c, d, p) * get_gamma(c, d, p);
    phi += get_theta(d, a, p) * get_gamma(d, a, p);
    float irradiance = glm::dot(phi * 0.5f, n);
    return irradiance * quad.intensity * brdf_const(hit);
}

glm::vec3 tracing(accelerator &acc, Scene &scene, Ray &ray) {
    hitobj hit = acc.intersect(ray);
    if (hit.t < EPS) return glm::vec3(0.0f);
    glm::vec3 emission = std::visit([](auto &&arg) { return arg.emission; }, hit.shape);
    glm::vec3 color = emission;
    for (auto &light: scene.lights) {
        if (std::holds_alternative<Quad>(light)) { color += get_irradiance(std::get<Quad>(light), ray, hit); }
    }
    return glm::clamp(color, 0.0f, 1.0f);
}

void render_segment(accelerator &acc, Image &img, Scene &scene, uint start, uint end) {
    for (uint i = start; i < end; ++i) {
        for (uint j = 0; j < scene.width; ++j) {
            Ray ray = ray_thru_pixel(scene, i + 0.5f, j + 0.5f);
            glm::vec3 color = tracing(acc, scene, ray);
            img.set(i, j, color);
        }
    }
}

void run(Scene &scene, Image &img) {
    accelerator acc(scene, false);
    uint num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);
    uint rows_per_thread = scene.height / num_threads;

    for (uint i = 0; i < num_threads; ++i) {
        int start = i * rows_per_thread;
        int end = (i == num_threads - 1) ? scene.height : (i + 1) * rows_per_thread;
        threads[i] = std::thread(render_segment, std::ref(acc), std::ref(img), std::ref(scene), start, end);
    }
    for (auto &thread: threads) thread.join();
}
}

