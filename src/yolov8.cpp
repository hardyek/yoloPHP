#include "yolov8.h"
#include <stb_image.h>
#include <stb_image_resize.h>
#include <stb_image_write.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <algorithm>

extern "C" {
    struct YOLOv8 {
        torch::jit::script::Module module;
    };

    // Load model from torchscript file
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

    std::tuple<float, int> find_max_score(const std::vector<float>& scores) {
        float maxScore = scores[0];
        int maxIndex = 0;
        for (int i = 1; i < scores.size(); ++i) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxIndex = i;
            }
        }
        return std::make_tuple(maxScore, maxIndex);
    }

    float iou(const std::array<float, 4>& boxA, const std::array<float, 4>& boxB) {
        float xA = std::max(boxA[0], boxB[0]);
        float yA = std::max(boxA[1], boxB[1]);
        float xB = std::min(boxA[2], boxB[2]);
        float yB = std::min(boxA[3], boxB[3]);

        float interArea = std::max(0.0f, xB - xA + 1) * std::max(0.0f, yB - yA + 1);
        float boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);
        float boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1);

        return interArea / (boxAArea + boxBArea - interArea);
    }

    std::vector<int> apply_nms(
        const std::vector<std::array<float, 4>>& boxes,
        const std::vector<float>& scores,
        const std::vector<int>& class_ids,
        float score_threshold, float nms_threshold
    ) {
        std::vector<int> indices(boxes.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [&scores](int i1, int i2) {
            return scores[i1] > scores[i2];
        });

        std::vector<int> keep;
        std::vector<bool> suppressed(boxes.size(), false);

        for (int i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            if (suppressed[idx] || scores[idx] < score_threshold) continue;
            keep.push_back(idx);
            for (int j = i + 1; j < indices.size(); ++j) {
                int next_idx = indices[j];
                if (iou(boxes[idx], boxes[next_idx]) > nms_threshold) {
                    suppressed[next_idx] = true;
                }
            }
        }

        return keep;
    }

    void draw_rectangles(unsigned char* image_data, int width, int height, const std::vector<std::array<float, 4>>& boxes) {
        for (const auto& box : boxes) {
            int left = static_cast<int>(box[0] * width);
            int top = static_cast<int>(box[1] * height);
            int right = static_cast<int>(box[2] * width);
            int bottom = static_cast<int>((box[3] * height));

            left = std::max(0, std::min(left, width - 1));
            top = std::max(0, std::min(top, height - 1));
            right = std::max(0, std::min(right, width - 1));
            bottom = std::max(0, std::min(bottom, height - 1));

            for (int y = top; y < bottom; ++y) {
                for (int x = left; x < right; ++x) {
                    image_data[y * width * 3 + x * 3] = 0;
                    image_data[y * width * 3 + x * 3 + 1] = 255;
                    image_data[y * width * 3 + x * 3 + 2] = 0;
                }
            }
        }
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

        try {
            std::vector<std::array<float, 4>> boxes;
            std::vector<float> scores;
            std::vector<int> class_ids;

            int num_outputs = output.size(1);

            for (int i = 0; i < num_outputs; ++i) {
                auto data = output[0][i].data_ptr<float>();
                std::vector<float> class_scores(data + 5, data + output.size(2));
                auto [maxScore, maxClassIndex] = find_max_score(class_scores);
                if (maxScore >= 0.25) {
                    std::array<float, 4> box = {data[0] - (0.5 * data[2]), data[1] - (0.5 * data[3]), data[0] + (0.5 * data[2]), data[1] + (0.5 * data[3])};
                    boxes.push_back(box);
                    scores.push_back(maxScore);
                    class_ids.push_back(maxClassIndex);
                }
            }

            auto keep = apply_nms(boxes, scores, class_ids, 0.25, 0.45);
            std::vector<std::array<float, 4>> nms_boxes;
            for (int idx : keep) {
                nms_boxes.push_back(boxes[idx]);
            }

            draw_rectangles(resized_data, new_width, new_height, nms_boxes);
            std::cout << "Drawing rectangles done." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error during post-processing: " << e.what() << std::endl;
            delete[] resized_data;
            return;
        }

        stbi_write_jpg(output_path, new_width, new_height, 3, resized_data, 100);
        delete[] resized_data;

        std::cout << "Image saved to " << output_path << std::endl;
    }


    void release_model(YOLOv8* model) {
        delete model;
    }
}