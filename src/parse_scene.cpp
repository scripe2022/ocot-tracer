#include "parse_scene.hpp"
#include "primitive.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stack>

Ray::Ray() {}
Ray::Ray(glm::vec3 o, glm::vec3 d) : origin(o), direction(d) {}
Ray::Ray(glm::vec3 o, glm::vec3 d, glm::vec3 n) : origin(o + EPS*n), direction(d) {}

Scene parse_scene(std::string filename) {
    Scene scene;
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Error opening file " << filename << std::endl;
        return scene;
    }

    auto string_trim = [](std::string s) -> std::string {
        const auto str_begin = s.find_first_not_of(" \t");
        if (str_begin == std::string::npos) return "";
        const auto str_end = s.find_last_not_of(" \t");
        return s.substr(str_begin, str_end - str_begin + 1);
    };

    glm::mat4 current_transform = glm::mat4(1.0f);
    std::stack <glm::mat4> transform_stack;
    std::vector<glm::vec3> vertices;
    std::vector<std::pair<glm::vec3, glm::vec3>> vertices_norm;
    glm::vec3 current_ambient = glm::vec3(0.2f, 0.2f, 0.2f);
    glm::vec3 current_diffuse = glm::vec3(0.0f), current_specular = glm::vec3(0.0f), current_emission = glm::vec3(0.0f);
    float current_shininess = 1.0f;
    std::string current_brdf = "phong";
    float current_roughness = 0.0f;

    for (std::string line; std::getline(fin, line); ) {
        auto comment_pos = line.find("#");
        if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);
        line = string_trim(line);
        if (line == "") continue;
        std::istringstream iss(line);
        std::string instruction; iss >> instruction;
        if (instruction == "size") {
            iss >> scene.width >> scene.height;
        }
        else if (instruction == "maxdepth") {
            iss >> scene.maxdepth;
        }
        else if (instruction == "output") {
            iss >> scene.output_filename;
        }
        else if (instruction == "integrator") {
            iss >> scene.integrator;
        }
        else if (instruction == "camera") {
            iss >> scene.camera.pos.x >> scene.camera.pos.y >> scene.camera.pos.z;
            iss >> scene.camera.lookat.x >> scene.camera.lookat.y >> scene.camera.lookat.z;
            iss >> scene.camera.up.x >> scene.camera.up.y >> scene.camera.up.z;
            iss >> scene.camera.fovy;
        }
        else if (instruction == "translate") {
            glm::vec3 translation; iss >> translation.x >> translation.y >> translation.z;
            current_transform = glm::translate(current_transform, translation);
        }
        else if (instruction == "rotate") {
            glm::vec3 axis; float angle; iss >> axis.x >> axis.y >> axis.z >> angle;
            current_transform = glm::rotate(current_transform, glm::radians(angle), axis);
        }
        else if (instruction == "scale") {
            glm::vec3 scale; iss >> scale.x >> scale.y >> scale.z;
            current_transform = glm::scale(current_transform, scale);
        }
        else if (instruction == "pushTransform") {
            transform_stack.push(current_transform);
        }
        else if (instruction == "popTransform") {
            current_transform = transform_stack.top();
            transform_stack.pop();
        }
        else if (instruction == "sphere") {
            Sphere sphere;
            iss >> sphere.center.x >> sphere.center.y >> sphere.center.z >> sphere.radius;
            sphere.transform = current_transform;
            sphere.ambient = current_ambient;
            sphere.diffuse = current_diffuse;
            sphere.specular = current_specular;
            sphere.shininess = current_shininess;
            sphere.emission = current_emission;
            sphere.brdf = current_brdf;
            sphere.roughness = current_roughness;
            scene.shapes.push_back(sphere);
        }
        else if (instruction == "maxverts" || instruction == "maxvertnorms") {
            // Do nothing
        }
        else if (instruction == "vertex") {
            glm::vec3 v; iss >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        }
        else if (instruction == "vertexnormal") {
            glm::vec3 v, n; iss >> v.x >> v.y >> v.z >> n.x >> n.y >> n.z;
            vertices_norm.push_back({v, n});
        }
        else if (instruction == "tri") {
            uint v0, v1, v2; iss >> v0 >> v1 >> v2;
            Triangle tri;
            tri.v0 = vertices[v0];
            tri.v1 = vertices[v1];
            tri.v2 = vertices[v2];
            tri.transform = current_transform;
            tri.ambient = current_ambient;
            tri.diffuse = current_diffuse;
            tri.specular = current_specular;
            tri.shininess = current_shininess;
            tri.emission = current_emission;
            tri.brdf = current_brdf;
            tri.roughness = current_roughness;
            scene.shapes.push_back(tri);
        }
        else if (instruction == "trinormal") {
            uint v0, v1, v2; iss >> v0 >> v1 >> v2;
            Triangle tri;
            tri.v0 = vertices_norm[v0].first;
            tri.v1 = vertices_norm[v1].first;
            tri.v2 = vertices_norm[v2].first;
            tri.has_normal = true;
            tri.n0 = vertices_norm[v0].second;
            tri.n1 = vertices_norm[v1].second;
            tri.n2 = vertices_norm[v2].second;
            tri.transform = current_transform;
            tri.ambient = current_ambient;
            tri.diffuse = current_diffuse;
            tri.specular = current_specular;
            tri.shininess = current_shininess;
            tri.emission = current_emission;
            tri.brdf = current_brdf;
            tri.roughness = current_roughness;
            scene.shapes.push_back(tri);
        }
        else if (instruction == "ambient") {
            iss >> current_ambient.x >> current_ambient.y >> current_ambient.z;
        }
        else if (instruction == "directional") {
            Directional dir;
            iss >> dir.direction.x >> dir.direction.y >> dir.direction.z;
            iss >> dir.color.x >> dir.color.y >> dir.color.z;
            scene.lights.push_back(dir);
        }
        else if (instruction == "point") {
            Point point;
            iss >> point.position.x >> point.position.y >> point.position.z;
            iss >> point.color.x >> point.color.y >> point.color.z;
            scene.lights.push_back(point);
        }
        else if (instruction == "quadLight") {
            Quad quad;
            iss >> quad.a.x >> quad.a.y >> quad.a.z;
            iss >> quad.ab.x >> quad.ab.y >> quad.ab.z;
            iss >> quad.ac.x >> quad.ac.y >> quad.ac.z;
            iss >> quad.intensity.x >> quad.intensity.y >> quad.intensity.z;
            scene.lights.push_back(quad);
            // TODO: Add quad light to scene shapes
            glm::vec3 sn = glm::normalize(glm::cross(quad.ab, quad.ac));
            Triangle tri0, tri1;
            tri0.v0 = quad.a, tri0.v1 = quad.a + quad.ab + quad.ac, tri0.v2 = quad.a + quad.ab;
            tri0.has_normal = false, tri0.transform = glm::mat4(1.0f), tri0.emission = quad.intensity, tri0.surface_normal = sn;
            tri0.ambient = glm::vec3(0.0f), tri0.diffuse = glm::vec3(1.0f), tri0.specular = glm::vec3(0.0f), tri0.shininess = 1.0f, tri0.brdf = "phong", tri0.roughness = 0.0f;
            tri0.light_source = true;
            scene.shapes.push_back(tri0);
            tri1.v0 = quad.a, tri1.v1 = quad.a + quad.ac, tri1.v2 = quad.a + quad.ab + quad.ac;
            tri1.has_normal = false, tri1.transform = glm::mat4(1.0f), tri1.emission = quad.intensity, tri1.surface_normal = sn;
            tri1.ambient = glm::vec3(0.0f), tri1.diffuse = glm::vec3(1.0f), tri1.specular = glm::vec3(0.0f), tri1.shininess = 1.0f, tri1.brdf = "phong", tri1.roughness = 0.0f;
            tri1.light_source = true;
            scene.shapes.push_back(tri1);
            // scene.quad_lights.push_back(std::make_pair(tri0, tri1));
            ++scene.num_quad_lights;
        }
        else if (instruction == "attenuation") {
            iss >> scene.attenuation.x >> scene.attenuation.y >> scene.attenuation.z;
        }
        else if (instruction == "diffuse") {
            iss >> current_diffuse.x >> current_diffuse.y >> current_diffuse.z;
        }
        else if (instruction == "specular") {
            iss >> current_specular.x >> current_specular.y >> current_specular.z;
        }
        else if (instruction == "shininess") {
            iss >> current_shininess;
        }
        else if (instruction == "emission") {
            iss >> current_emission.x >> current_emission.y >> current_emission.z;
        }
        else if (instruction == "brdf") {
            iss >> current_brdf;
        }
        else if (instruction == "roughness") {
            iss >> current_roughness;
        }
        else if (instruction == "lightsamples") {
            iss >> scene.samples;
        }
        else if (instruction == "spp") {
            iss >> scene.spp;
        }
        else if (instruction == "lightstratify") {
            std::string s; iss >> s;
            scene.stratified = s == "on";
        }
        else if (instruction == "nexteventestimation") {
            iss >> scene.nee;
        }
        else if (instruction == "russianroulette") {
            std::string s; iss >> s;
            scene.rr = s == "on";
        }
        else if (instruction == "importancesampling") {
            iss >> scene.importance_sampling;
        }
        else if (instruction == "gamma") {
            iss >> scene.gamma;
        }
        else {
            std::cerr << "Unknown instruction: " << instruction << std::endl;
        }
    }
    return scene;
}


