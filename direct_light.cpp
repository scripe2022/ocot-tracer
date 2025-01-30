// comp := make
// run  := time ./raytracer scenes/test2/dragon.test && kcat output/dragon.png
// run  := time ./raytracer scenes/test2/sphere.test && kcat output/sphere.png

#include <iostream>
#include <thread>
#include <variant>

#include "accelerator.hpp"
#include "direct_light.hpp"
#include "image.hpp"
#include "parse_scene.hpp"

namespace direct {

bool visible_to_point(accelerator &acc, glm::vec3 &light, glm::vec3 &p, glm::vec3 &n) {
    Ray ray(p + EPS * n, glm::normalize(light - p));
    float dis = glm::length(light - p);
    hitobj hit = acc.intersect(ray);
    if (hit.t > EPS && hit.t < dis - 10 * EPS) return false;
    return true;
}

glm::vec3 get_irradiance(accelerator &acc, Quad &quad, Scene &scene, Ray &ray, hitobj &hit) {
    glm::vec3 xpp = scene.camera.pos;
    glm::vec3 x = ray.origin + ray.direction * hit.t;
    std::vector<std::pair<float, float>> samples = gen_sample(scene.samples, scene.stratified);
    glm::vec3 n_l = glm::normalize(glm::cross(quad.ac, quad.ab));

    glm::vec3 color = glm::vec3(0.0f);
    for (auto [da, db]: samples) {
        // NOTE: xp: light position
        glm::vec3 xp = quad.a + da * quad.ab + db * quad.ac;
        glm::vec3 w_i = glm::normalize(xp - x);
        glm::vec3 w_o = glm::normalize(xpp - x);

        if (!visible_to_point(acc, xp, x, hit.normal)) continue;

        glm::vec3 f = brdf(hit, w_i, w_o);
        float G = gterm(x, hit.normal, xp, n_l);
        color += f * G;
    }
    color *= quad.intensity;
    color *= glm::length(glm::cross(quad.ab, quad.ac)) / (float)scene.samples;
    return color;
}

glm::vec3 tracing(accelerator &acc, Scene &scene, Ray &ray) {
    hitobj hit = acc.intersect(ray);
    if (hit.t < EPS) return glm::vec3(0.0f);
    glm::vec3 emission = std::visit([](auto &&arg) { return arg.emission; }, hit.shape);
    glm::vec3 color = emission;
    for (auto &light: scene.lights) {
        if (std::holds_alternative<Quad>(light)) {
            glm::vec3 irr = get_irradiance(acc, std::get<Quad>(light), scene, ray, hit);
            color += irr;
        }
    }
    return glm::clamp(color, 0.0f, 1.0f);
}

int cnt = 0;
void render_segment(accelerator &acc, Image &img, Scene &scene, uint start, uint end, bool show_progress) {
    for (uint i = start; i < end; ++i) {
        for (uint j = 0; j < scene.width; ++j) {
            if (show_progress) std::cout << ++cnt << std::endl;
            Ray ray = ray_thru_pixel(scene, i + 0.5f, j + 0.5f);
            glm::vec3 color = tracing(acc, scene, ray);
            img.set(i, j, color);
        }
    }
}

std::vector<glm::vec3> get_irradiance_gpu(accelerator &acc, Quad &quad, Scene &scene, std::vector<Ray> &rays, std::vector<hitobj> &hits, bool show_progress) {
    glm::vec3 xpp = scene.camera.pos;
    std::vector<std::vector<std::pair<float, float>>> samples(scene.height * scene.width);
    for (uint i = 0; i < scene.height * scene.width; ++i) samples[i] = gen_sample(scene.samples, scene.stratified);
    glm::vec3 n_l = glm::normalize(glm::cross(quad.ac, quad.ab));

    std::vector<glm::vec3> colors(scene.height * scene.width, glm::vec3(0.0f));
    uint N = scene.samples;
    // NOTE: j == samples
    for (uint j = 0; j < N; ++j) {
        if (show_progress) std::cout << "    sample " << j << "/" << N << std::endl;
        // NOTE: new_ray = (x + EPS*hit.normal, glm::normalize(xp - x))
        std::vector<Ray> new_rays(scene.height * scene.width);
        for (uint i = 0; i < scene.height * scene.width; ++i) {
            glm::vec3 x = rays[i].origin + rays[i].direction * hits[i].t;
            glm::vec3 xp = quad.a + samples[i][j].first * quad.ab + samples[i][j].second * quad.ac;
            new_rays[i].origin = x + EPS * hits[i].normal;
            new_rays[i].direction = glm::normalize(xp - x);
        }
        std::vector<hitobj> new_hits = acc.intersect_multi(new_rays);
        for (uint i = 0; i < scene.height * scene.width; ++i) {
            glm::vec3 x = rays[i].origin + rays[i].direction * hits[i].t;
            glm::vec3 xp = quad.a + samples[i][j].first * quad.ab + samples[i][j].second * quad.ac;
            glm::vec3 w_i = glm::normalize(xp - x);
            glm::vec3 w_o = glm::normalize(xpp - x);
            float dis = glm::length(xp - x);
            if (new_hits[i].t > EPS && new_hits[i].t < dis - 10 * EPS) continue;
            glm::vec3 f = brdf(hits[i], w_i, w_o);
            float G = gterm(x, hits[i].normal, xp, n_l);
            colors[i] += f * G;
        }
    }
    for (uint i = 0; i < scene.height * scene.width; ++i) {
        colors[i] *= quad.intensity;
        colors[i] *= glm::length(glm::cross(quad.ab, quad.ac)) / (float)scene.samples;
    }
    return colors;
}

std::vector<glm::vec3> tracing_gpu(accelerator &acc, Scene &scene, std::vector<Ray> &rays, bool show_progress) {
    std::vector<glm::vec3> colors(scene.height * scene.width, glm::vec3(0.0f));
    std::vector<hitobj> hits = acc.intersect_multi(rays);
    for (uint i = 0; i < scene.height; ++i) {
        for (uint j = 0; j < scene.width; ++j) {
            hitobj &hit = hits[i * scene.width + j];
            if (hit.t < EPS || hit.t > INF) continue;
            glm::vec3 emission = std::visit([](auto &&arg) { return arg.emission; }, hit.shape);
            colors[i * scene.width + j] = emission;
        }
    }
    uint light_cnt = 0;
    for (auto &light: scene.lights) {
        if (show_progress) std::cout << "light " << light_cnt++ << "/" << scene.lights.size() << std::endl;
        if (std::holds_alternative<Quad>(light)) {
            std::vector<glm::vec3> irr = get_irradiance_gpu(acc, std::get<Quad>(light), scene, rays, hits, show_progress);
            for (uint i = 0; i < scene.height * scene.width; ++i) colors[i] += irr[i];
        }
    }
    for (uint i = 0; i < scene.height * scene.width; ++i) colors[i] = glm::clamp(colors[i], 0.0f, 1.0f);
    return colors;
}

void render_gpu(accelerator &acc, Image &img, Scene &scene, bool show_progress) {
    std::vector<Ray> rays(scene.height * scene.width);
    for (uint i = 0; i < scene.height; ++i) {
        for (uint j = 0; j < scene.width; ++j) { rays[i * scene.width + j] = ray_thru_pixel(scene, i + 0.5f, j + 0.5f); }
    }
    std::vector<glm::vec3> colors = tracing_gpu(acc, scene, rays, show_progress);
    for (uint i = 0; i < scene.height; ++i) {
        for (uint j = 0; j < scene.width; ++j) { img.set(i, j, colors[i * scene.width + j]); }
    }
}

// NOTE: mode 0: cpu single
// NOTE: mode 1: cpu multi-thread
// NOTE: mode 2: gpu single
// NOTE: mode 3: gpu multi-thread
void run(Scene &scene, Image &img, int mode, bool show_progress) {
    accelerator acc(scene);
    if (mode == 0) {
        acc.usegpu = false;
        std::thread render_thread(render_segment, std::ref(acc), std::ref(img), std::ref(scene), 0, scene.height, show_progress);
        render_thread.join();
    }
    else if (mode == 1) {
        acc.usegpu = false;
        uint num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads(num_threads);
        uint rows_per_thread = scene.height / num_threads;

        for (uint i = 0; i < num_threads; ++i) {
            int start = i * rows_per_thread;
            int end = (i == num_threads - 1) ? scene.height : (i + 1) * rows_per_thread;
            threads[i] = std::thread(render_segment, std::ref(acc), std::ref(img), std::ref(scene), start, end, show_progress);
        }
        for (auto &thread: threads) thread.join();
    }
    else if (mode == 2) {
        acc.usegpu = true;
        std::thread render_thread(render_segment, std::ref(acc), std::ref(img), std::ref(scene), 0, scene.height, show_progress);
        render_thread.join();
    }
    else if (mode == 3) {
        acc.usegpu = true;
        render_gpu(acc, img, scene, show_progress);
    }
}
}
