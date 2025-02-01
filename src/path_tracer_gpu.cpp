// comp := make
// 4a2 run  := time ./raytracer scenes/test4/cornellBRDF.test && kcat output/cornellBRDF.png
// all run  := ./raytracer scenes/tiny/cornellCosine.test && ./raytracer scenes/tiny/cornellBRDF.test && ./raytracer scenes/tiny/ggx.test && ./raytracer scenes/tiny/mis.test && kcat output/cornellCosine.png output/cornellBRDF.png output/ggx.png output/mis.png
// 4a1 run  := time ./raytracer scenes/test4/cornellCosine.test && kcat output/cornellCosine.png
// ggx run  := time ./raytracer scenes/test4/ggx.test && kcat output/ggx.png
// 4b1 run  := time ./raytracer scenes/test4/mis.test && kcat output/mis.png

#include <iostream>
#include "primitive.hpp"
#include <thread>
#include <variant>

#include "accelerator.hpp"
#include "image.hpp"
#include "parse_scene.hpp"
#include "path_tracer.hpp"

namespace path_tracer {

Scene scene;
Image img;
accelerator acc;
bool show_progress;

glm::vec3 get_hitpos(Ray &ray, hitobj &hit) { return ray.origin + hit.t * ray.direction + EPS * hit.normal; }

bool visible2light(glm::vec3 origin, glm::vec3 light_pos) {
    Ray ray(origin, glm::normalize(light_pos - origin));
    float dis = glm::length(light_pos - origin);
    hitobj hit = acc.intersect(ray);
    if (hit.t > 0 && hit.t + 10 * EPS < dis) return false;
    return true;
}

std::vector<bool> visible2light_gpu(std::vector<glm::vec3> origin, std::vector<glm::vec3> light_pos) {
    std::vector<Ray> rays;
    for (uint i = 0; i < origin.size(); ++i) rays.push_back(Ray(origin[i], glm::normalize(light_pos[i] - origin[i])));
    std::vector<hitobj> hits = acc.intersect_multi(rays);
    std::vector<bool> res(origin.size(), true);
    for (uint i = 0; i < origin.size(); ++i) {
        if (hits[i].t > 0 && hits[i].t + 10 * EPS < glm::length(light_pos[i] - origin[i])) res[i] = false;
    }
    return res;
}

std::pair<glm::vec3, float> importance_sample(hitobj &hit, glm::vec3 w_o) {
    std::string brdf_type = std::visit([](auto &&arg) { return arg.brdf; }, hit.shape);

    auto hemisphere_sample = [](float theta, float phi, glm::vec3 n) {
        glm::vec3 s = glm::vec3(glm::cos(phi) * glm::sin(theta), glm::sin(phi) * glm::sin(theta), glm::cos(theta));
        glm::vec3 w = glm::normalize(n);
        glm::vec3 a = (fabs(w.x) > 0.9) ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        glm::vec3 u = glm::normalize(glm::cross(a, w));
        glm::vec3 v = glm::cross(w, u);
        return s.x * u + s.y * v + s.z * w;
    };

    float epsilon = gen_real();
    float phi = 2.0f * (float)M_PI * gen_real();

    if (scene.importance_sampling == "cosine") {
        float theta = glm::acos(glm::sqrt(epsilon));
        glm::vec3 w_i = hemisphere_sample(theta, phi, hit.normal);
        return std::make_pair(w_i, glm::abs(glm::dot(hit.normal, w_i)) / (float)M_PI);
    }
    else if (scene.importance_sampling == "brdf") {
        float roughness = std::visit([](auto &&arg) { return arg.roughness; }, hit.shape);
        float shininess = std::visit([](auto &&arg) { return arg.shininess; }, hit.shape);
        glm::vec3 specular = std::visit([](auto &&arg) { return arg.specular; }, hit.shape);
        glm::vec3 diffuse = std::visit([](auto &&arg) { return arg.diffuse; }, hit.shape);
        float ks = (specular.x + specular.y + specular.z) / 3.0f;
        float kd = (diffuse.x + diffuse.y + diffuse.z) / 3.0f;
        float t = ks / (ks + kd);
        if (brdf_type == "ggx") {
            t = glm::max(0.25f, t);
            bool lt = gen_real() < t;
            float theta = lt ? glm::atan(roughness * glm::sqrt(epsilon), glm::sqrt(1.0f - epsilon)) : glm::acos(glm::sqrt(epsilon));
            glm::vec3 w_i = hemisphere_sample(theta, phi, hit.normal);
            if (lt) w_i = glm::reflect(-w_o, w_i);
            glm::vec3 half_vec = glm::normalize(glm::normalize(w_i) + glm::normalize(w_o));
            float half_angle = glm::acos(glm::min(1.0f, glm::dot(half_vec, hit.normal)));
            float d = (1.0f - t) * glm::dot(hit.normal, w_i) / (float)M_PI;
            float g = t * D(half_angle, roughness) * glm::dot(hit.normal, half_vec) / (4.0f * glm::dot(half_vec, w_i));
            return std::make_pair(w_i, d + g);
        }
        else if (brdf_type == "phong") {
            glm::vec3 reflection = (2.0f * glm::dot(hit.normal, w_o) * hit.normal - w_o);
            bool lt = gen_real() < t;
            float theta = lt ? glm::acos(glm::pow(epsilon, 1.0 / (1.0 + shininess))) : glm::acos(glm::sqrt(epsilon));
            glm::vec3 w_i = hemisphere_sample(theta, phi, lt ? reflection : hit.normal);
            float d = (1.0f - t) * glm::max(0.0f, glm::dot(w_i, hit.normal)) / (float)M_PI;
            float s = t * (shininess + 1) / (2.0f * (float)M_PI) * glm::pow(glm::max(0.0f, glm::dot(reflection, w_i)), shininess);
            return std::make_pair(w_i, d + s);
        }
        else assert(false);
    }
    else {
        float theta = glm::acos(epsilon);
        glm::vec3 w_i = hemisphere_sample(theta, phi, hit.normal);
        return std::make_pair(w_i, 1.0f / (2.0f * (float)M_PI));
    }
    assert(false);
}

glm::vec3 direct_lighting(Ray &ray, hitobj &hit) {
    auto get_p_nee = [](glm::vec3 &hitpos, glm::vec3 w_i) -> float {
        float p = 0.0f;
        for (auto light: scene.lights) {
            if (!std::holds_alternative<Quad>(light)) continue;
            Quad quad = std::get<Quad>(light);
            float t = quad_intersect(Ray(hitpos, w_i), quad);
            if (t < 0.0f) continue;

            float area = glm::length(glm::cross(quad.ab, quad.ac));
            glm::vec3 n_l = glm::normalize(glm::cross(quad.ac, quad.ab));
            p += glm::pow(t, 2.0f) / (area * glm::abs(glm::dot(w_i, n_l)));
        }
        p /= (float)scene.num_quad_lights;
        return p;
    };

    glm::vec3 hitpos = get_hitpos(ray, hit);
    glm::vec3 color = glm::vec3(0.0f);
    glm::vec3 w_o = glm::normalize(ray.origin - hitpos);
    for (auto light: scene.lights) {
        if (!std::holds_alternative<Quad>(light)) continue;
        Quad quad = std::get<Quad>(light);
        std::vector<std::pair<float, float>> samples = gen_sample(scene.samples, scene.stratified);
        for (auto [da, db]: samples) {
            glm::vec3 xp = quad.a + da * quad.ab + db * quad.ac;
            glm::vec3 n_l = glm::normalize(glm::cross(quad.ac, quad.ab));
            float area = glm::length(glm::cross(quad.ab, quad.ac));
            glm::vec3 w_i = glm::normalize(xp - hitpos);
            glm::vec3 f = brdf(hit, w_i, w_o);
            if (!visible2light(hitpos, xp)) continue;
            float g = gterm(hitpos, hit.normal, xp, n_l);
            float p = 1.0f;
            if (scene.nee == "mis") {
                float p_nee = glm::pow(get_p_nee(hitpos, w_i), 2);
                float p_brdf = glm::pow(pdf(hit, w_i, w_o), 2);
                p = p_nee / (p_nee + p_brdf);
            }
            color += area * quad.intensity * f * g * p / (float)scene.samples;
        }
    }
    if (scene.nee == "mis") {
        auto [w_i, _] = importance_sample(hit, w_o);
        hitobj new_hit = acc.intersect(Ray(hitpos, glm::normalize(w_i)));
        if (!new_hit.hit()) return color;
        glm::vec3 hit_emission = std::visit([](auto &&arg) { return arg.emission; }, new_hit.shape);
        glm::vec3 f = brdf(hit, w_i, w_o);
        glm::vec3 T = f * glm::abs(glm::dot(w_i, hit.normal));
        float p_nee = glm::pow(get_p_nee(hitpos, w_i), 2);
        float p_brdf = glm::pow(pdf(hit, w_i, w_o), 2);
        color += T * hit_emission * p_brdf / pdf(hit, w_i, w_o) / (p_nee + p_brdf);
    }
    return color;
}

std::vector<glm::vec3> direct_lighting_gpu(std::vector<Ray> &rays, std::vector<hitobj> &hits) {
    auto get_p_nee = [](glm::vec3 &hitpos, glm::vec3 w_i) -> float {
        float p = 0.0f;
        for (auto light: scene.lights) {
            if (!std::holds_alternative<Quad>(light)) continue;
            Quad quad = std::get<Quad>(light);
            float t = quad_intersect(Ray(hitpos, w_i), quad);
            if (t < 0.0f) continue;

            float area = glm::length(glm::cross(quad.ab, quad.ac));
            glm::vec3 n_l = glm::normalize(glm::cross(quad.ac, quad.ab));
            p += glm::pow(t, 2.0f) / (area * glm::abs(glm::dot(w_i, n_l)));
        }
        p /= (float)scene.num_quad_lights;
        return p;
    };

    std::vector<glm::vec3> colors(rays.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> hitpos_vec, xp_vec;
    for (uint i = 0; i < rays.size(); ++i) {
        Ray &ray = rays[i];
        hitobj &hit = hits[i];
        glm::vec3 hitpos = get_hitpos(ray, hit);
        glm::vec3 w_o = glm::normalize(ray.origin - hitpos);
        for (auto light: scene.lights) {
            if (!std::holds_alternative<Quad>(light)) continue;
            Quad quad = std::get<Quad>(light);
            float da = gen_real(), db = gen_real();
            glm::vec3 xp = quad.a + da * quad.ab + db * quad.ac;

            hitpos_vec.push_back(hitpos);
            xp_vec.push_back(xp);
        }
    }
    std::vector<bool> visibles = visible2light_gpu(hitpos_vec, xp_vec);
    for (uint i = 0; i < rays.size(); ++i) {
        Ray &ray = rays[i];
        hitobj &hit = hits[i];
        uint j = i * scene.num_quad_lights - 1;
        glm::vec3 hitpos = get_hitpos(ray, hit);
        glm::vec3 color = glm::vec3(0.0f);
        glm::vec3 w_o = glm::normalize(ray.origin - hitpos);
        for (auto light: scene.lights) {
            if (!std::holds_alternative<Quad>(light)) continue;
            Quad quad = std::get<Quad>(light);
            ++j;
            // std::cout << j << " " << visibles.size() << std::endl;
            if (!visibles[j]) continue;
            glm::vec3 xp = xp_vec[j];
            glm::vec3 n_l = glm::normalize(glm::cross(quad.ac, quad.ab));
            float area = glm::length(glm::cross(quad.ab, quad.ac));
            glm::vec3 w_i = glm::normalize(xp - hitpos);
            glm::vec3 f = brdf(hit, w_i, w_o);
            float g = gterm(hitpos, hit.normal, xp, n_l);
            float p = 1.0f;
            if (scene.nee == "mis") {
                float p_nee = glm::pow(get_p_nee(hitpos, w_i), 2);
                float p_brdf = glm::pow(pdf(hit, w_i, w_o), 2);
                p = p_nee / (p_nee + p_brdf);
            }
            color += area * quad.intensity * f * g * p / (float)scene.samples;
        }
        colors[i] = color;
    }
    if (scene.nee == "mis") {
        std::vector<Ray> new_rays;
        for (uint i = 0; i < rays.size(); ++i) {
            Ray &ray = rays[i];
            hitobj &hit = hits[i];
            glm::vec3 hitpos = get_hitpos(ray, hit);
            glm::vec3 w_o = glm::normalize(ray.origin - hitpos);
            auto [w_i, _] = importance_sample(hit, w_o);
            new_rays.push_back(Ray(hitpos, glm::normalize(w_i)));
        }
        std::vector<hitobj> new_hits = acc.intersect_multi(new_rays);
        for (uint i = 0; i < rays.size(); ++i) {
            Ray &ray = rays[i];
            hitobj &hit = hits[i];
            glm::vec3 hitpos = get_hitpos(ray, hit);
            glm::vec3 w_o = glm::normalize(ray.origin - hitpos);
            glm::vec3 w_i = new_rays[i].direction;
            hitobj &new_hit = new_hits[i];
            if (!new_hit.hit()) continue;
            glm::vec3 hit_emission = std::visit([](auto &&arg) { return arg.emission; }, new_hit.shape);
            glm::vec3 f = brdf(hit, w_i, w_o);
            glm::vec3 T = f * glm::abs(glm::dot(w_i, hit.normal));
            float p_nee = glm::pow(get_p_nee(hitpos, w_i), 2);
            float p_brdf = glm::pow(pdf(hit, w_i, w_o), 2);
            colors[i] += T * hit_emission * p_brdf / pdf(hit, w_i, w_o) / (p_nee + p_brdf);
        }
    }
    return colors;
}

glm::vec3 trace(Ray &ray, uint depth) {
    glm::vec3 color = glm::vec3(0.0f);
    hitobj hit = acc.intersect(ray);
    glm::vec3 hitpos = get_hitpos(ray, hit);
    bool is_light_source = std::holds_alternative<Triangle>(hit.shape) && std::get<Triangle>(hit.shape).light_source;
    glm::vec3 emission = std::visit([](auto &&arg) { return arg.emission; }, hit.shape);
    if (!hit.hit()) return color;

    if (scene.nee == "on" || scene.nee == "mis") {
        if (is_light_source && !depth && glm::dot(hit.normal, ray.direction) <= 0.0f) color += emission;
        if (is_light_source && (depth || glm::dot(hit.normal, ray.direction) > 0.0f)) return glm::vec3(0.0f);
        if (!scene.rr && depth >= scene.maxdepth) return glm::vec3(0.0f);
        color += direct_lighting(ray, hit);
    }
    else if (is_light_source || (!scene.rr && depth > scene.maxdepth)) return emission;
    glm::vec3 w_o = glm::normalize(ray.origin - hitpos);
    auto [w_i, p] = importance_sample(hit, w_o);
    glm::vec3 f = brdf(hit, w_i, w_o);
    glm::vec3 T = f * glm::abs(glm::dot(w_i, hit.normal)) / p;
    float coef = 1.0f;
    if (scene.rr) {
        float q = 1 - glm::min(glm::max(glm::max(T.x, T.y), T.z), 1.0f);
        if (q > gen_real()) return color;
        coef /= 1 - q;
    }
    Ray new_ray = Ray(hitpos, w_i, hit.normal);
    color += coef * T * trace(new_ray, depth + 1);
    return color;
}

std::vector<glm::vec3> trace_gpu(std::vector<Ray> &rays, uint depth) {
    std::vector<hitobj> hits = acc.intersect_multi(rays);
    std::vector<glm::vec3> colors(rays.size(), glm::vec3(0.0f));

    std::vector<Ray> remained_rays;
    std::vector<hitobj> remained_hits;
    std::vector<uint> remained_idx;

    for (uint i = 0; i < rays.size(); ++i) {
        Ray &ray = rays[i];
        hitobj &hit = hits[i];
        if (!hit.hit()) continue;
        glm::vec3 hitpos = get_hitpos(ray, hit);
        bool is_light_source = std::holds_alternative<Triangle>(hit.shape) && std::get<Triangle>(hit.shape).light_source;
        glm::vec3 emission = std::visit([](auto &&arg) { return arg.emission; }, hit.shape);
        if (scene.nee == "on" || scene.nee == "mis") {
            if (is_light_source && !depth && glm::dot(hit.normal, ray.direction) <= 0.0f) colors[i] += emission;
            if (is_light_source && (depth || glm::dot(hit.normal, ray.direction) > 0.0f)) continue;
            if (!scene.rr && depth >= scene.maxdepth) continue;
        }
        else if (is_light_source || (!scene.rr && depth > scene.maxdepth)) continue;

        remained_rays.push_back(ray);
        remained_hits.push_back(hit);
        remained_idx.push_back(i);
    }
    if (remained_rays.size() && (scene.nee == "on" || scene.nee == "mis")) {
        std::vector<glm::vec3> nee_colors = direct_lighting_gpu(remained_rays, remained_hits);
        for (uint i = 0; i < remained_rays.size(); ++i) {
            colors[remained_idx[i]] += nee_colors[i];
        }
    }
    std::vector<Ray> new_rays;
    for (uint idx = 0; idx < remained_rays.size(); ++idx) {
        Ray &ray = remained_rays[idx];
        hitobj &hit = remained_hits[idx];
        glm::vec3 hitpos = get_hitpos(ray, hit);
        glm::vec3 w_o = glm::normalize(ray.origin - get_hitpos(ray, hit));
        auto [w_i, p] = importance_sample(hit, w_o);
        glm::vec3 f = brdf(hit, w_i, w_o);
        glm::vec3 T = f * glm::abs(glm::dot(w_i, hit.normal)) / p;
        float coef = 1.0f;
        if (scene.rr) {
            float q = 1 - glm::min(glm::max(glm::max(T.x, T.y), T.z), 1.0f);
            if (q > gen_real()) continue;
            coef /= 1 - q;
        }
        new_rays.push_back(Ray(hitpos, w_i, hit.normal));
    }
    if (new_rays.empty()) return colors;
    std::vector<glm::vec3> new_colors = trace_gpu(new_rays, depth + 1);
    for (uint i = 0; i < new_colors.size(); ++i) colors[remained_idx[i]] += new_colors[i];
    return colors;
}

uint cnt = 0;
void render_segment(uint start, uint end) {
    for (uint i = start; i < end; ++i) {
        if (show_progress) std::cout << ++cnt << "/" << scene.height << std::endl;
        for (uint j = 0; j < scene.width; ++j) {
            std::vector<std::pair<float, float>> samples = gen_sample(scene.spp, false);
            glm::vec3 color = glm::vec3(0.0f);
            for (auto [da, db]: samples) {
                Ray ray = ray_thru_pixel(scene, i + da, j + db);
                color += trace(ray, 0);
            }
            color /= (float)scene.spp;
            img.set(i, j, color);
        }
    }
}

uint cnt_gpu = 0;
void render_gpu() {
    for (uint k = 0; k < scene.spp; ++k) {
        if (show_progress) std::cout << ++cnt_gpu << "/" << scene.spp << std::endl;
        std::vector<Ray> rays;
        for (uint i = 0; i < scene.height; ++i) for (uint j = 0; j < scene.width; ++j) {
            float da = gen_real(), db = gen_real();
            rays.push_back(ray_thru_pixel(scene, i + da, j + db));
        }
        std::vector<glm::vec3> colors = trace_gpu(rays, 0);
        for (uint i = 0; i < scene.height; ++i) for (uint j = 0; j < scene.width; ++j) {
            img.set(i, j, img.get(i, j) + colors[i * scene.width + j] / (float)scene.spp);
        }
    }
}

// NOTE: mode 0: cpu single
// NOTE: mode 1: cpu multi-thread
// NOTE: mode 2: gpu single
// NOTE: mode 3: gpu multi-thread
Image run(Scene &s, int mode, bool show) {
    path_tracer::scene = s;
    path_tracer::img.set_size(scene.width, scene.height);
    path_tracer::acc.build(scene, false);
    path_tracer::show_progress = show;

    // D_BEGIN:
    mode = 3;
    // D_END:
    if (mode == 0) {
        std::thread render_thread(render_segment, 0, scene.height);
        render_thread.join();
    }
    else if (mode == 1) {
        uint num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads(num_threads);
        uint rows_per_thread = scene.height / num_threads;

        for (uint i = 0; i < num_threads; ++i) {
            int start = i * rows_per_thread;
            int end = (i == num_threads - 1) ? scene.height : (i + 1) * rows_per_thread;
            threads[i] = std::thread(render_segment, start, end);
        }
        for (auto &thread: threads) thread.join();
    }
    else if (mode == 2) {
        acc.usegpu = true;
        std::thread render_thread(render_segment, 0, scene.height);
        render_thread.join();
    }
    else if (mode == 3) {
        acc.usegpu = true;
        render_gpu();
    }
    img.gamma_correct(scene.gamma);

    return img;
}
}
