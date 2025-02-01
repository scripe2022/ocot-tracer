#include "image.hpp"
#include <iostream>

Image::Image() {}

void Image::set_size(int w, int h) {
    this->width = w;
    this->height = h;
    data.resize(w * h);
}

Image::Image(int w, int h) {
    set_size(w, h);
}

glm::vec3 Image::get(int x, int y) {
    return data[x * width + y];
}

void Image::set(int x, int y, glm::vec3 color) {
    data[x * width + y] = color;
}

bool Image::dumppng(const std::string &filename) {
    FILE *fp = fopen(filename.c_str(), "wb");
    if(!fp) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return false;
    }
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        std::cerr << "Cannot create PNG write structure" << std::endl;
        fclose(fp);
        return false;
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        std::cerr << "Cannot create PNG info structure" << std::endl;
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        fclose(fp);
        return false;
    }
    if (setjmp(png_jmpbuf(png_ptr))) {
        std::cerr << "Error during PNG creation" << std::endl;
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);
    std::vector<png_byte> row(width * 3);
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            glm::vec3 color = get(x, y);
            int ti = y * 3;
            row[ti] = static_cast<png_byte>((color.r * 255));
            row[ti + 1] = static_cast<png_byte>((color.g * 255));
            row[ti + 2] = static_cast<png_byte>((color.b * 255));
        }
        png_write_row(png_ptr, row.data());
    }
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    return true;
}

void Image::gamma_correct(float gamma) {
    for (int i = 0; i < width*height; ++i) {
        // data[i] = glm::clamp(data[i], 0.0f, 1.0f);
        data[i].r = pow(data[i].r, 1.0f / gamma);
        data[i].g = pow(data[i].g, 1.0f / gamma);
        data[i].b = pow(data[i].b, 1.0f / gamma);
        data[i] = glm::clamp(data[i], 0.0f, 1.0f);
    }
}

void Image::dumptxt(const std::string &filename) {
    FILE *fp = fopen(filename.c_str(), "w");
    for (int i = 0; i < width*height; ++i) {
        fprintf(fp, "%f %f %f\n", data[i].r, data[i].g, data[i].b);
    }
}
