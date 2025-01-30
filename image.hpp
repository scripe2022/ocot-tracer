#ifndef IMAGE_HPP
#define IMAGE_HPP
#include <png.h>
#include <vector>
#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

struct Image {
    int width, height;
    std::vector<glm::vec3> data;
    Image();
    void set_size(int w, int h);
    Image(int w, int h);
    glm::vec3 get(int x, int y);
    void set(int x, int y, glm::vec3 color);
    bool dumppng(const std::string &filename);
    void dumptxt(const std::string &filename);
    void gamma_correct(float gamma);
};
#endif
