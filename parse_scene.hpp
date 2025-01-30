#ifndef PARSE_SCENE_HPP
#define PARSE_SCENE_HPP
#include <variant>
#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <vector>

struct Ray {
    glm::vec3 origin = glm::vec3(0.0f);
    glm::vec3 direction = glm::vec3(0.0f);
    Ray();
    Ray(glm::vec3 origin, glm::vec3 direction);
    Ray(glm::vec3 origin, glm::vec3 direction, glm::vec3 normal);
};

struct Camera {
    glm::vec3 pos = glm::vec3(0.0f), lookat = glm::vec3(0.0f), up = glm::vec3(0.0f);
    float fovy = 0;
};

struct Sphere {
    glm::vec3 center;
    float radius;
    glm::mat4 transform;
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
    glm::vec3 emission;
    std::string brdf;
    float roughness;
};

struct Triangle {
    glm::vec3 v0, v1, v2;
    bool has_normal = false;
    glm::vec3 n0, n1, n2;
    glm::vec3 surface_normal = glm::vec3(0.0f);
    glm::mat4 transform;
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
    glm::vec3 emission;
    bool light_source = false;
    std::string brdf;
    float roughness;
};

using Shape = std::variant<Sphere, Triangle>;

struct Directional {
    glm::vec3 direction;
    glm::vec3 color;
};

struct Point {
    glm::vec3 position;
    glm::vec3 color;
};

struct Quad {
    glm::vec3 a, ab, ac;
    glm::vec3 intensity;
};

using Light = std::variant<Directional, Point, Quad>;

struct Scene {
    uint width, height;
    uint maxdepth = 5;
    std::string integrator = "raytracer";
    std::string output_filename = "raytrace.png";
    Camera camera;
    std::vector<Shape> shapes;
    std::vector<Light> lights;
    // std::vector<std::pair<Triangle, Triangle>> quad_lights;
    glm::vec3 attenuation = glm::vec3(1.0f, 0.0f, 0.0f);
    uint samples = 1;
    uint spp = 1;
    bool stratified = false;
    std::string nee = "off";
    bool rr = false;
    std::string importance_sampling = "hemisphere";
    float gamma = 1.0f;
    uint num_quad_lights = 0;
};

Scene parse_scene(std::string filename);
#endif

