// comp := make
// run  := time ./raytracer test/dragon.test && kcat dragon.png
// run  := time ./raytracer test/sphere.test && kcat sphere.png
// run  := time ./raytracer test/cornell.test && kcat cornell.png
#include <iostream>
#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

#include "analytic_direct.hpp"
#include "direct_light.hpp"
#include "image.hpp"
#include "parse_scene.hpp"
#include "path_tracer.hpp"
#include "ray_tracer.hpp"

int32_t main(int32_t argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <scene_file>" << std::endl;
        return 1;
    }
    Scene scene = parse_scene(argv[1]);
    Image img(scene.width, scene.height);
    if (scene.integrator == "raytracer") raytracer::run(scene, img, false, true);
    else if (scene.integrator == "analyticdirect")
        analytic_direct::run(scene, img);
    else if (scene.integrator == "direct")
        direct::run(scene, img, 3, true);
    else if (scene.integrator == "pathtracer") {
        #ifdef GPU_RENDERING
        img = path_tracer::run(scene, 3, true);
        #else
        img = path_tracer::run(scene, 1, true);
        #endif
    }

    img.dumppng("./output/" + scene.output_filename);
    img.dumptxt("./output/rgb/" + scene.output_filename + ".rgb");
}
