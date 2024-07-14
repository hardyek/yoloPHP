#include "yolov8.h"
#include "stb_image.h"
#include "stb_image_write.h"
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
        unsigned char* img_data = stbi_load(frame_path, &width, &height, &channels, 3);
        if (!img_data) {
            std::cerr << "Failed to read the image\n";
            return;
        }

        std::vector<float> img_float(width * height * 3);
        for (int i = 0; i < width * height * 3; ++i) {
            img_float[i] = img_data[i] / 255.0f;
        }

        stbi_image_free(img_data);

        auto input_tensor = torch::from_blob(img_float.data(), {1, height, width, 3});
        input_tensor = input_tensor.permute({0, 3, 1, 2});

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        at::Tensor output = model->module.forward(inputs).toTensor();

        img_data = new unsigned char[width * height * 3];
        for (int i = 0; i < width * height * 3; ++i) {
            img_data[i] = static_cast<unsigned char>(img_float[i] * 255);
        }

        for (int i = 0; i < output.size(1); ++i) {
            float* data = output[0][i].data_ptr<float>();
            int x = static_cast<int>(data[0]);
            int y = static_cast<int>(data[1]);
            int w = static_cast<int>(data[2]);
            int h = static_cast<int>(data[3]);

            for (int xx = x; xx < x + w; ++xx) {
                img_data[(y * width + xx) * 3] = 0; // Red
                img_data[(y * width + xx) * 3 + 1] = 255; // Green
                img_data[(y * width + xx) * 3 + 2] = 0; // Blue

                img_data[((y + h) * width + xx) * 3] = 0;
                img_data[((y + h) * width + xx) * 3 + 1] = 255;
                img_data[((y + h) * width + xx) * 3 + 2] = 0;
            }
            for (int yy = y; yy < y + h; ++yy) {
                img_data[(yy * width + x) * 3] = 0;
                img_data[(yy * width + x) * 3 + 1] = 255;
                img_data[(yy * width + x) * 3 + 2] = 0;

                img_data[(yy * width + x + w) * 3] = 0;
                img_data[(yy * width + x + w) * 3 + 1] = 255;
                img_data[(yy * width + x + w) * 3 + 2] = 0;
            }
        }

        stbi_write_jpg(output_path, width, height, 3 , img_data, 100);
        delete[] img_data;
    }

    void release_model(YOLOv8* model) {
        delete model;
    }
}