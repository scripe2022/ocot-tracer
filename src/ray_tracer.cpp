// comp := make
// run  := time ./raytracer scenes1/scene1.test && kcat raytrace.png

#include <iostream>
#include <thread>
#include <variant>

#include "accelerator.hpp"
#include "image.hpp"
#include "parse_scene.hpp"
#include "primitive.hpp"
#include "ray_tracer.hpp"

namespace raytracer {
bool visible_to_light(accelerator &acc, const Light &light, const glm::vec3 &p, const glm::vec3 &L) {
    Ray ray;
    ray.origin = p + EPS * L;
    ray.direction = L;
    float dis = std::holds_alternative<Point>(light) ? glm::length(std::get<Point>(light).position - p) : INF;
    hitobj hit = acc.intersect(ray, true);
    if (hit.t > 2 * EPS && hit.t < dis - 2 * EPS) return false;
    return true;
}

glm::vec3 get_intensity(accelerator &acc, const Scene &scene, const Shape &shape, const Ray &ray, const glm::vec3 &p, glm::vec3 &N) {
    auto get_point_intensity = [](const Point &point, const glm::vec3 &orig, const glm::vec3 &attenuation) {
        float r = glm::length(point.position - orig);
        glm::vec3 Li = point.color;
        return Li / (attenuation.x + attenuation.y * r + attenuation.z * r * r);
    };
    glm::vec3 A = std::visit([](auto &&arg) { return arg.ambient; }, shape);
    glm::vec3 E = std::visit([](auto &&arg) { return arg.emission; }, shape);
    glm::vec3 D = std::visit([](auto &&arg) { return arg.diffuse; }, shape);
    glm::vec3 color = A + E;
    for (auto &light: scene.lights) {
        glm::vec3 L = std::holds_alternative<Point>(light) ? glm::normalize(std::get<Point>(light).position - p) : glm::normalize(std::get<Directional>(light).direction);
        if (!visible_to_light(acc, light, p, L)) continue;
        glm::vec3 light_color = (std::holds_alternative<Point>(light)) ? get_point_intensity(std::get<Point>(light), p, scene.attenuation) : std::get<Directional>(light).color;
        glm::vec3 sub_color = D * glm::max(glm::dot(N, L), 0.0f);
        glm::vec3 S = std::visit([](auto &&arg) { return arg.specular; }, shape);
        glm::vec3 H = glm::normalize(L - ray.direction);
        float s = std::visit([](auto &&arg) { return arg.shininess; }, shape);
        sub_color += S * glm::pow(glm::max(glm::dot(N, H), 0.0f), s);
        color += light_color * sub_color;
    }
    return color;
}

glm::vec3 ray_tracing(accelerator &acc, Scene &scene, Ray &ray, uint depth, glm::vec3 weight) {
    // if (depth >= scene.maxdepth) return glm::vec3(0.0f);
    if (depth >= 1) return glm::vec3(0.0f);
    hitobj hit = acc.intersect(ray, true);
    if (hit.t < EPS || hit.t > INF) return glm::vec3(0.0f);
    glm::vec3 primary = get_intensity(acc, scene, hit.shape, ray, ray.origin + ray.direction * hit.t, hit.normal);
    glm::vec3 ks = std::visit([](auto &&arg) { return arg.specular; }, hit.shape);
    Ray new_ray;
    new_ray.direction = glm::normalize(glm::reflect(ray.direction, hit.normal));
    new_ray.origin = ray.origin + ray.direction * hit.t + EPS * new_ray.direction;
    if (weight.x > 1 || weight.y > 1 || weight.z > 1) { std::cout << glm::to_string(weight) << std::endl; }
    return weight * primary + ray_tracing(acc, scene, new_ray, depth + 1, weight * ks);
}

uint cnt = 0;
void render_segment(accelerator &acc, Image &img, Scene &scene, uint start, uint end, bool show_progress) {
    for (uint i = start; i < end; ++i) {
        for (uint j = 0; j < scene.width; ++j) {
            if (show_progress) std::cout << ++cnt << std::endl;
            Ray ray = ray_thru_pixel(scene, i + 0.5f, j + 0.5f);
            glm::vec3 color = ray_tracing(acc, scene, ray, 0, glm::vec3(1.0f, 1.0f, 1.0f));
            img.set(i, j, color);
        }
    }
}

void run(Scene &scene, Image &img, bool multi_processes, bool show_progress) {
    accelerator acc(scene);
    if (multi_processes) {
        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads(num_threads);
        int rows_per_thread = scene.height / num_threads;
        for (int i = 0; i < num_threads; ++i) {
            int start = i * rows_per_thread;
            int end = (i == num_threads - 1) ? scene.height : (i + 1) * rows_per_thread;
            threads[i] = std::thread(render_segment, std::ref(acc), std::ref(img), std::ref(scene), start, end, show_progress);
        }
        for (auto &thread: threads) thread.join();
    }
    else {
        std::thread render_thread(render_segment, std::ref(acc), std::ref(img), std::ref(scene), 0, scene.height, show_progress);
        render_thread.join();
    }
}
}
