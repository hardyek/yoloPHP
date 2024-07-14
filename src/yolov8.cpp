#include "yolov8.h"
#include <stb_image.h>
#include <stb_image_resize.h>
#include <stb_image_write.h>
#include <torch/script.h>
#include <iostream>
#include <vector>

extern "C" {
    struct YOLOv8 {
        torch::jit::script::Module module;
    };

    YOLOv8* load_model(const char* model_path) {
        YOLOv8* model = new YOLOv8();
        try {
            model->module = torch::jit::load(model_path);
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            delete model;
            return nullptr;
        }
        return model;
    }

    void process_frame(YOLOv8* model, const char* frame_path, const char* output_path) {
        int width, height, channels;
        unsigned char* data = stbi_load(frame_path, &width, &height, &channels, 3);
        if (!data) {
            std::cerr << "Failed to read the image\n";
            return;
        }

        std::cout << "Image loaded: " << width << "x" << height << " Channels: " << channels << std::endl;

        int new_width = 640;
        int new_height = 640;
        unsigned char* resized_data = new unsigned char[new_width * new_height * 3];

        stbir_resize_uint8(data, width, height, 0, resized_data, new_width, new_height, 0, 3);
        stbi_image_free(data);

        std::cout << "Image resized." << std::endl;

        auto input_tensor = torch::from_blob(resized_data, {1, new_height, new_width, 3}, torch::kUInt8);
        input_tensor = input_tensor.permute({0, 3, 1, 2}).to(torch::kFloat);
        input_tensor = input_tensor.div(255);

        std::cout << "Tensor prepared." << std::endl;

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        at::Tensor output;
        try {
            output = model->module.forward(inputs).toTensor();
        } catch (const c10::Error& e) {
            std::cerr << "Error during model inference: " << e.what() << std::endl;
            delete[] resized_data;
            return;
        }

        std::cout << "Model inference done." << std::endl;

        for (int i = 0; i < output.size(1); ++i) {
            float* data = output[0][i].data_ptr<float>();
            int left = static_cast<int>(data[0] * new_width);
            int top = static_cast<int>(data[1] * new_height);
            int right = static_cast<int>((data[0] + data[2]) * new_width);
            int bottom = static_cast<int>((data[1] + data[3]) * new_height);

            for (int y = top; y < bottom; ++y) {
                for (int x = left; x < right; ++x) {
                    resized_data[y * new_width * 3 + x * 3] = 0;
                    resized_data[y * new_width * 3 + x * 3 + 1] = 255;
                    resized_data[y * new_width * 3 + x * 3 + 2] = 0;
                }
            }
        }

        std::cout << "Drawing rectangles done." << std::endl;

        stbi_write_jpg(output_path, new_width, new_height, 3, resized_data, 100);
        delete[] resized_data;

        std::cout << "Image saved to " << output_path << std::endl;
    } 

    void release_model(YOLOv8* model) {
        delete model;
    }
}