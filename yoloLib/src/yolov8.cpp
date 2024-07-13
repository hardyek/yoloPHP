#include "yolov8.h"
#include <torch/script.h>
#include <preprocessingLib.h>
#include <iostream>

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
        cv::Mat frame = preprocessingLib::imread(frame_path);
        if (frame.empty()) {
            std::cerr << "Failed to read the image\n";
            return;
        }

        cv::Mat img_float;
        frame.convertTo(img_float, CV_32F, 1.0 / 255);

        auto input_tensor = torch::from_blob(img_float.data, {1, frame.rows, frame.cols, 3});
        input_tensor = input_tensor.permute({0, 3, 1, 2});

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        at::Tensor output = model->module.forward(inputs).toTensor();

        for (int i = 0; i < output.size(1); ++i) {
            float* data = output[0][i].data_ptr<float>();
            cv::Rect box(data[0], data[1], data[2], data[3]);
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        }

        cv::imwrite(output_path, frame);
    }

    void release_model(YOLOv8* model) {
        delete model;
    }
}
