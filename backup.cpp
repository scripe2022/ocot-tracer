// comp := make
// all run  := ./raytracer scenes/tiny/cornellCosine.test && ./raytracer scenes/tiny/cornellBRDF.test && ./raytracer scenes/tiny/ggx.test && ./raytracer scenes/tiny/mis.test && kcat output/cornellCosine.png output/cornellBRDF.png output/ggx.png output/mis.png
// 4b1 run  := time ./raytracer scenes/test4/mis.test && kcat output/mis.png
// ggx run  := time ./raytracer scenes/test4/ggx.test && kcat output/ggx.png
// 4a2 run  := time ./raytracer scenes/test4/cornellBRDF.test && kcat output/cornellBRDF.png
// 4a1 run  := time ./raytracer scenes/test4/cornellCosine.test && kcat output/cornellCosine.png

#include <chrono>
#include <iostream>
#include <primitive.hpp>
#include <random>
#include <thread>
#include <variant>

#include "accelerator.hpp"
#include "image.hpp"
#include "parse_scene.hpp"
#include "path_tracer.hpp"

#define PI (3.14159265358979323846f)
#define TWO_PI (6.28318530717958647692f)
#define INV_PI (0.31830988618379067154f)

namespace path_tracer {

Scene scene;
Image img;
accelerator acc;
bool show_progress;

struct Mat {
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
    glm::vec3 emission;
    std::string brdf;
    float roughness;
    bool is_light_source;
};

Mat get_mat(Shape &shape) {
    Mat mat;
    mat.ambient = std::visit([](auto &&arg) { return arg.ambient; }, shape);
    mat.diffuse = std::visit([](auto &&arg) { return arg.diffuse; }, shape);
    mat.specular = std::visit([](auto &&arg) { return arg.specular; }, shape);
    mat.shininess = std::visit([](auto &&arg) { return arg.shininess; }, shape);
    mat.emission = std::visit([](auto &&arg) { return arg.emission; }, shape);
    mat.brdf = std::visit([](auto &&arg) { return arg.brdf; }, shape);
    mat.roughness = std::visit([](auto &&arg) { return arg.roughness; }, shape);
    mat.is_light_source = std::holds_alternative<Triangle>(shape) && std::get<Triangle>(shape).light_source;
    return mat;
}

float intersectRayParallelogram(Ray &ray, Quad quad) {
    // quad.a -= 0.01f * quad.ab + 0.01f * quad.ac;
    // quad.ab *= 1.01f;
    // quad.ac *= 1.01f;
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

bool visible2light(glm::vec3 origin, glm::vec3 lightPosition) {
    Ray ray(origin, glm::normalize(lightPosition - origin));
    float dis = glm::length(lightPosition - origin);
    hitobj hit = acc.intersect(ray);
    if (hit.t > 0 && hit.t + 10 * EPS < dis) return false;
    return true;
}

float microfacetDistribution(float halfAngle, Mat material) {
    float a_squared = material.roughness * material.roughness;
    float denominator = PI * glm::pow(glm::cos(halfAngle), 4.0f) * glm::pow(a_squared + glm::pow(glm::tan(halfAngle), 2.0f), 2.0f);
    return a_squared / denominator;
}

float microfacetSelfShadowing(glm::vec3 normal, glm::vec3 view, Mat material) {
    if (glm::dot(view, normal) <= 0) return 0;
    float thetaV = glm::acos(glm::dot(view, normal));
    return 2.0f / (1.0f + glm::sqrt(glm::max(0.0f, 1.0f + glm::pow(material.roughness, 2.0f) * glm::pow(glm::tan(thetaV), 2.0f))));
}

glm::vec3 fresnel(glm::vec3 w_in, glm::vec3 halfVector, Mat material) { return material.specular + (glm::vec3(1.0f) - material.specular) * glm::pow(1.0f - glm::dot(w_in, halfVector), 5.0f); }

glm::vec3 ggx_brdf(glm::vec3 normal, glm::vec3 w_in, glm::vec3 w_out, Mat material) {
    if (glm::dot(normal, w_in) < 0 || glm::dot(normal, w_out) < 0) { return glm::vec3(0); }
    glm::vec3 halfVector = glm::normalize(glm::normalize(w_in) + glm::normalize(w_out));
    float halfAngle = glm::acos(glm::min(1.0f, glm::dot(halfVector, normal)));
    float D = microfacetDistribution(halfAngle, material);
    float G = microfacetSelfShadowing(normal, w_in, material) * microfacetSelfShadowing(normal, w_out, material);
    glm::vec3 F = fresnel(w_in, normal, material);
    float normalization = 4 * glm::dot(w_in, normal) * glm::dot(w_out, normal);
    return material.diffuse / PI + D * G * F / normalization;
}

glm::vec3 phong_brdf(glm::vec3 normal, glm::vec3 w_in, glm::vec3 w_out, Mat material) {
    glm::vec3 reflection = 2 * glm::dot(normal, w_out) * normal - w_out;
    glm::vec3 diffuse = material.diffuse * INV_PI;
    glm::vec3 specular = material.specular * (material.shininess + 2) / (2 * PI) * glm::pow(glm::dot(reflection, w_in), material.shininess);
    return diffuse + specular;
}

glm::vec3 brdf(glm::vec3 surfaceNormal, glm::vec3 w_in, glm::vec3 w_out, Mat mat) {
    if (mat.brdf == "ggx") return ggx_brdf(surfaceNormal, w_in, w_out, mat);
    else
        return phong_brdf(surfaceNormal, w_in, w_out, mat);
}

float geometry(glm::vec3 surfacePoint, glm::vec3 surfaceNormal, glm::vec3 lightPoint, glm::vec3 lightNormal) {
    glm::vec3 lightVector = glm::normalize(lightPoint - surfacePoint);
    float cosThetai = glm::abs(glm::dot(lightVector, surfaceNormal));
    float cosThetal = glm::abs(glm::dot(lightVector, -lightNormal));
    cosThetai = (cosThetai < 0) ? 0 : cosThetai;
    cosThetal = (cosThetal < 0) ? 0 : cosThetal;
    return cosThetai * cosThetal / glm::pow(glm::length(lightPoint - surfacePoint), 2);
}

float neePDF(glm::vec3 position, glm::vec3 w_in) {
    float p = 0.0f;
    Ray ray(position, w_in);
    // hitobj hit = acc.intersect(ray);
    uint quad_cnt = 0;
    for (auto light: scene.lights) {
        if (!std::holds_alternative<Quad>(light)) continue;
        ++quad_cnt;
        Quad quad = std::get<Quad>(light);
        float t = intersectRayParallelogram(ray, quad);
        if (t < 0.0f) continue;

        float lightArea = glm::length(glm::cross(quad.ab, quad.ac));
        glm::vec3 lightNormal = glm::normalize(glm::cross(quad.ac, quad.ab));
        p += glm::pow(t, 2.0f) / (lightArea * glm::abs(glm::dot(w_in, lightNormal)));
    }
    p /= (float)quad_cnt;
    return p;
}

float averageVector(glm::vec3 vec) {
    float avg = 0;
    avg += vec.x;
    avg += vec.y;
    avg += vec.z;
    return avg / 3.0f;
}

float ggx_pdf(glm::vec3 normal, glm::vec3 w_in, glm::vec3 w_out, Mat material) {
    glm::vec3 halfVector = glm::normalize(w_in + w_out);
    float halfAngle = glm::acos(glm::min(1.0f, glm::dot(halfVector, normal)));
    float k_s = averageVector(material.specular);
    float k_d = averageVector(material.diffuse);
    float t = glm::max(0.25f, k_s / (k_s + k_d));
    float diffuseTerm = (1.0f - t) * glm::dot(normal, w_in) / PI;
    float ggxTerm = t * microfacetDistribution(halfAngle, material) * glm::dot(normal, halfVector) / (4.0f * glm::dot(halfVector, w_in));
    return diffuseTerm + ggxTerm;
}

float phong_pdf(glm::vec3 normal, glm::vec3 w_in, glm::vec3 w_out, Mat material) {
    float k_s = averageVector(material.specular);
    float k_d = averageVector(material.diffuse);
    float t = k_s / (k_s + k_d);
    glm::vec3 reflection = (2 * glm::dot(normal, w_out) * normal - w_out);
    float cosTerm = glm::max(0.0f, glm::dot(w_in, normal));
    float diffuse = (1 - t) * cosTerm / PI;
    float specular = t * (material.shininess + 1) / TWO_PI * glm::pow(glm::max(0.0f, glm::dot(reflection, w_in)), material.shininess);
    return diffuse + specular;
}

float pdf(glm::vec3 normal, glm::vec3 w_in, glm::vec3 w_out, Mat material) {
    if (material.brdf == "ggx") return ggx_pdf(normal, w_in, w_out, material);
    else
        return phong_pdf(normal, w_in, w_out, material);
}

float brdfMisWeighting(glm::vec3 position, glm::vec3 normal, glm::vec3 w_in, glm::vec3 w_out, Mat material, bool use_brdf) {
    float neePDFValue = neePDF(position, w_in);
    float brdfPDFValue = pdf(normal, w_in, w_out, material);
    float neePDF2 = glm::pow(neePDFValue, 2);
    float brdfPDF2 = glm::pow(brdfPDFValue, 2);
    float denom = neePDF2 + brdfPDF2;
    if (use_brdf) { return brdfPDF2 / denom; }
    else { return neePDF2 / denom; }
}

glm::vec3 nextEventEstimation(glm::vec3 position, glm::vec3 normal, Mat material, glm::vec3 origin, float &pdfNormalization) {
    glm::vec3 outputColor = glm::vec3(0.0f);
    pdfNormalization = 0.0f;
    uint quad_cnt = 0;
    for (auto light: scene.lights) {
        if (!std::holds_alternative<Quad>(light)) continue;
        ++quad_cnt;
        Quad quad = std::get<Quad>(light);
        std::vector<std::pair<float, float>> samples = gen_sample(scene.samples, scene.stratified);
        for (auto [da, db]: samples) {
            glm::vec3 lightPosition = quad.a + da * quad.ab + db * quad.ac;
            glm::vec3 lightNormal = glm::normalize(glm::cross(quad.ac, quad.ab));
            float lightArea = glm::length(glm::cross(quad.ab, quad.ac));
            glm::vec3 w_in = glm::normalize(lightPosition - position);
            glm::vec3 w_out = glm::normalize(origin - position);
            glm::vec3 F = brdf(normal, w_in, w_out, material);
            if (!visible2light(position + EPS * normal, lightPosition)) continue;
            float G = geometry(position, normal, lightPosition, lightNormal);
            pdfNormalization = scene.nee == "mis" ? brdfMisWeighting(position, normal, w_in, w_out, material, false) : 1.0f;
            outputColor += lightArea * quad.intensity * F * G * pdfNormalization / (float)scene.samples;
        }
    }
    pdfNormalization /= (float)quad_cnt;
    return outputColor;
}

glm::vec3 sphereCoordsToVector(float theta, float phi, glm::vec3 samplingSpaceCenter) {
    glm::vec3 s = glm::vec3(glm::cos(phi) * glm::sin(theta), glm::sin(phi) * glm::sin(theta), glm::cos(theta));
    glm::vec3 w = samplingSpaceCenter;
    glm::vec3 a = glm::vec3(0, 1, 0);
    if (glm::length(w - a) < 0.001 || glm::length(w + a) < 0.001) a = glm::vec3(0, 0, 1);
    glm::vec3 u = glm::normalize(glm::cross(a, w));
    glm::vec3 v = glm::cross(w, u);
    return s.x * u + s.y * v + s.z * w;
}

glm::vec3 phong_importanceSample(glm::vec3 normal, glm::vec3 w_out, Mat material, float &pdfNormalization) {
    float epsilon1 = gen_real();
    float epsilon2 = gen_real();
    float theta = 0;
    float phi = 0;
    float k_s = averageVector(material.specular);
    float k_d = averageVector(material.diffuse);
    float t = k_s / (k_s + k_d);
    glm::vec3 samplingSpaceCenter = normal;
    glm::vec3 reflection = (2 * glm::dot(normal, w_out) * normal - w_out);
    if (gen_real() < t) {
        theta = glm::acos(glm::pow(epsilon1, 1.0 / (1.0 + material.shininess)));
        phi = TWO_PI * epsilon2;
        samplingSpaceCenter = reflection;
    }
    else {
        theta = glm::acos(glm::sqrt(epsilon1));
        phi = TWO_PI * epsilon2;
    }
    glm::vec3 w_in = sphereCoordsToVector(theta, phi, samplingSpaceCenter);
    float cosTerm = glm::max(0.0f, glm::dot(w_in, normal));
    float diffuse = (1 - t) * cosTerm / PI;
    float specular = t * (material.shininess + 1) / TWO_PI * glm::pow(glm::max(0.0f, glm::dot(reflection, w_in)), material.shininess);
    pdfNormalization = diffuse + specular;
    return w_in;
}

glm::vec3 ggx_importanceSample(glm::vec3 normal, glm::vec3 w_out, Mat material, float &pdfNormalization) {
    float epsilon1 = gen_real();
    float epsilon2 = gen_real();
    float epsilon3 = gen_real();
    float theta = 0;
    float phi = 0;
    float k_s = averageVector(material.specular);
    float k_d = averageVector(material.diffuse);
    float t = glm::max(0.25f, k_s / (k_s + k_d));
    glm::vec3 w_in;
    if (epsilon3 < t) {
        theta = glm::atan(material.roughness * glm::sqrt(epsilon1), glm::sqrt(1 - epsilon1));
        phi = TWO_PI * epsilon2;
        glm::vec3 halfVector = sphereCoordsToVector(theta, phi, normal);
        w_in = glm::reflect(-w_out, halfVector);
    }
    else {
        theta = glm::acos(glm::sqrt(epsilon1));
        phi = TWO_PI * epsilon2;
        w_in = sphereCoordsToVector(theta, phi, normal);
    }
    glm::vec3 halfVector = glm::normalize(w_in + w_out);
    float halfAngle = glm::acos(glm::min(1.0f, glm::dot(halfVector, normal)));
    float diffuseTerm = (1.0f - t) * glm::dot(normal, w_in) / PI;
    float ggxTerm = t * microfacetDistribution(halfAngle, material) * glm::dot(normal, halfVector) / (4.0f * glm::dot(halfVector, w_in));
    pdfNormalization = diffuseTerm + ggxTerm;
    return w_in;
}

glm::vec3 importanceSample(glm::vec3 normal, glm::vec3 w_out, Mat material, float &pdfNormalization) {
    glm::vec3 w_in;
    float epsilon1 = gen_real();
    float epsilon2 = gen_real();
    float theta = 0;
    float phi = 0;

    glm::vec3 samplingSpaceCenter = normal;
    if (scene.importance_sampling == "cosine") {
        theta = glm::acos(glm::sqrt(epsilon1));
        phi = TWO_PI * epsilon2;
        w_in = sphereCoordsToVector(theta, phi, samplingSpaceCenter);
        pdfNormalization = glm::abs(glm::dot(normal, w_in)) / PI;
    }
    else if (scene.importance_sampling == "brdf") {
        if (material.brdf == "ggx") w_in = ggx_importanceSample(normal, w_out, material, pdfNormalization);
        else
            w_in = phong_importanceSample(normal, w_out, material, pdfNormalization);
    }
    else {
        theta = glm::acos(epsilon1);
        phi = TWO_PI * epsilon2;
        w_in = sphereCoordsToVector(theta, phi, samplingSpaceCenter);
        pdfNormalization = 1.0f / TWO_PI;
    }
    return w_in;
}

// FIX:
glm::vec3 ggxDirect(glm::vec3 position, glm::vec3 normal, Mat material, glm::vec3 origin, float &pdfNormalization) {
    glm::vec3 outputColor = glm::vec3(0);
    glm::vec3 w_out = glm::normalize(origin - position);
    float dummy;
    glm::vec3 w_in = importanceSample(normal, w_out, material, dummy);
    glm::vec3 hitPosition;
    glm::vec3 hitNormal;
    Mat hitMaterial;
    hitobj hit = acc.intersect(Ray(position + EPS * normal, glm::normalize(w_in)));

    hitPosition = position + glm::normalize(w_in) * hit.t;
    hitNormal = hit.normal;
    hitMaterial = get_mat(hit.shape);

    if (hit.t < EPS || hit.t > INF) return outputColor;
    glm::vec3 f = brdf(normal, w_in, w_out, material);
    glm::vec3 T = f * glm::abs(glm::dot(w_in, normal));
    pdfNormalization = brdfMisWeighting(position, normal, w_in, w_out, material, true);
    outputColor = T * hitMaterial.emission / pdf(normal, w_in, w_out, material);
    return outputColor;
}

glm::vec3 trace(glm::vec3 origin, glm::vec3 direction, int numBounces);
glm::vec3 indirectLighting(glm::vec3 position, glm::vec3 normal, Mat material, glm::vec3 origin, int numBounces) {
    glm::vec3 outputColor = glm::vec3(0);
    glm::vec3 w_out = glm::normalize(origin - position);

    for (int i = 0; i < 1; i++) {
        float pdfNormalization = 1;
        glm::vec3 w_in = importanceSample(normal, w_out, material, pdfNormalization);

        glm::vec3 f = brdf(normal, w_in, w_out, material);
        glm::vec3 T = f * glm::abs(glm::dot(w_in, normal)) / pdfNormalization;

        if (scene.rr) {
            float p = 1 - glm::min(glm::max(T.x, glm::max(T.y, T.z)), 1.0f);

            if (p > gen_real()) {
                continue;
            }
            else {
                float boost = 1.0f / (1.0f - p);
                outputColor += boost * T * trace(position + w_in * EPS, w_in, numBounces);
            }
        }
        else { outputColor += T * trace(position + w_in * EPS, w_in, numBounces); }
    }
    return outputColor / ((float)1.0f);
}

glm::vec3 trace(glm::vec3 origin, glm::vec3 direction, int numBounces) {
    glm::vec3 outputColor = glm::vec3(0.0f);
    hitobj hit = acc.intersect(Ray(origin, direction));
    glm::vec3 hitPosition = origin + direction * hit.t + EPS * hit.normal;
    glm::vec3 hitNormal = hit.normal;
    Mat hitMaterial = get_mat(hit.shape);

    if (hit.t < EPS || hit.t > INF) return outputColor;

    if (scene.nee == "on" || scene.nee == "mis") {
        if (hitMaterial.is_light_source) {
            if (numBounces == 0 && glm::dot(hitNormal, direction) <= 0.0f) { outputColor += hitMaterial.emission; }
            else { return glm::vec3(0.0f); }
        }
        if (numBounces >= scene.maxdepth) return glm::vec3(0.0f);
    }
    else {
        if (hitMaterial.is_light_source || numBounces > scene.maxdepth) { return hitMaterial.emission; }
    }
    if (scene.nee == "mis") {
        float neeWeighting;
        glm::vec3 neeColor = nextEventEstimation(hitPosition, hitNormal, hitMaterial, origin, neeWeighting);
        float brdfWeighting;
        glm::vec3 brdfColor = ggxDirect(hitPosition, hitNormal, hitMaterial, origin, brdfWeighting);
        outputColor += neeColor;
        outputColor += brdfWeighting * brdfColor;
    }
    else if (scene.nee == "on") {
        float neePDF;
        outputColor += nextEventEstimation(hitPosition, hitNormal, hitMaterial, origin, neePDF);
    }
    outputColor += indirectLighting(hitPosition, hitNormal, hitMaterial, origin, numBounces + 1);
    return outputColor;
}

int cnt = 0;
void render_segment(uint start, uint end) {
    for (uint i = start; i < end; ++i) {
        if (show_progress) std::cout << ++cnt << "/" << scene.height << std::endl;
        for (uint j = 0; j < scene.width; ++j) {
            std::vector<std::pair<float, float>> samples = gen_sample(scene.spp, false);
            glm::vec3 color = glm::vec3(0.0f);
            for (auto [da, db]: samples) {
                Ray ray = ray_thru_pixel(scene, i + da, j + db);
                color += trace(ray.origin, ray.direction, 0);
            }
            color /= (float)scene.spp;
            img.set(i, j, color);
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
    mode = 1;
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
    // else if (mode == 2) {
    //     acc.usegpu = true;
    //     std::thread render_thread(render_segment, 0, scene.height);
    //     render_thread.join();
    // }
    // else if (mode == 3) {
    //     acc.usegpu = true;
    //     render_gpu();
    // }
    else if (mode == 9) {
        uint line = 47;
        uint col = 233;
        for (uint i = line; i < line + 1; ++i) {
            for (uint j = col; j < col + 1; ++j) {
                std::vector<std::pair<float, float>> samples = gen_sample(1, false);
                glm::vec3 color = glm::vec3(0.0f);
                for (auto [da, db]: samples) {
                    Ray ray = ray_thru_pixel(scene, i + da, j + db);
                    color += trace(ray.origin, ray.direction, 0);
                    break;
                }
                color /= (float)scene.spp;
                color = glm::clamp(color, 0.0f, 1.0f);
                img.set(i, j, color);
            }
        }
    }
    img.gamma_correct(scene.gamma);

    return img;
}
}
