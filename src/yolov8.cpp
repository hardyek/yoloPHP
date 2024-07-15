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

    float iou(const std::array<float, 4>& box1, const std::array<float, 4>& box2) {
         // Convert (x, y, w, h) to (x1, y1, x2, y2)
        float x1_1 = box1[0];
        float y1_1 = box1[1];
        float x2_1 = box1[0] + box1[2];
        float y2_1 = box1[1] + box1[3];

        float x1_2 = box2[0];
        float y1_2 = box2[1];
        float x2_2 = box2[0] + box2[2];
        float y2_2 = box2[1] + box2[3];

        float inter_x1 = std::max(x1_1, x1_2);
        float inter_y1 = std::max(y1_1, y1_2);
        float inter_x2 = std::min(x2_1, x2_2);
        float inter_y2 = std::min(y2_1, y2_2);

        float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
        float box1_area = (x2_1 - x1_1) * (y2_1 - y1_1);
        float box2_area = (x2_2 - x1_2) * (y2_2 - y1_2);

    return inter_area / (box1_area + box2_area - inter_area);
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

    std::vector<unsigned char> draw_rectangles(unsigned char* image_data, int width, int height, const std::vector<std::tuple<std::array<float, 4>, float, int>>& nms_boxes) {
        // Calculate scale factors
        float scale_x = static_cast<float>(width) / 640.0;
        float scale_y = static_cast<float>(height) / 640.0;
        int outline_width = 5;

        for (size_t i = 0; i < nms_boxes.size(); ++i) {
            const auto& box_info = nms_boxes[i];
            const auto& box = std::get<0>(box_info);
            float score = std::get<1>(box_info);
            int class_id = std::get<2>(box_info);
            
            std::cout << "Box " << i << ": ["
                    << "x=" << box[0] << ", "
                    << "y=" << box[1] << ", "
                    << "w=" << box[2] << ", "
                    << "h=" << box[3] << "], "
                    << "score=" << score << ", "
                    << "class_id=" << class_id << std::endl;
        }

        for (const auto& box_info : nms_boxes) {
            auto box = std::get<0>(box_info);
            
            // Scale the bounding box coordinates
            int left = static_cast<int>(box[0] * scale_x);
            int top = static_cast<int>(box[1] * scale_y);
            int right = static_cast<int>((box[0] + box[2]) * scale_x);
            int bottom = static_cast<int>((box[1] + box[3]) * scale_y);

            std::cout << "Valid box: [" << left << ", " << top << ", " << right << ", " << bottom << "]" << std::endl;
             for (int y = top; y < top + outline_width && y < height; ++y) {
                for (int x = left; x < right && x < width; ++x) {
                    int index = y * width * 3 + x * 3;
                    image_data[index] = 0;         // Red channel
                    image_data[index + 1] = 255;   // Green channel
                    image_data[index + 2] = 0;     // Blue channel
                }
            }

            for (int y = bottom - outline_width; y < bottom && y < height; ++y) {
                for (int x = left; x < right && x < width; ++x) {
                    int index = y * width * 3 + x * 3;
                    image_data[index] = 0;         // Red channel
                    image_data[index + 1] = 255;   // Green channel
                    image_data[index + 2] = 0;     // Blue channel
                }
            }

            // Draw left and right borders
            for (int y = top; y < bottom && y < height; ++y) {
                for (int x = left; x < left + outline_width && x < width; ++x) {
                    int index = y * width * 3 + x * 3;
                    image_data[index] = 0;         // Red channel
                    image_data[index + 1] = 255;   // Green channel
                    image_data[index + 2] = 0;     // Blue channel
                }

                for (int x = right - outline_width; x < right && x < width; ++x) {
                    int index = y * width * 3 + x * 3;
                    image_data[index] = 0;         // Red channel
                    image_data[index + 1] = 255;   // Green channel
                    image_data[index + 2] = 0;     // Blue channel
                }
            }
        }
        return image_data;
    }

    void process_frame(YOLOv8* model, const char* frame_path, const char* output_path) {
        int width, height, channels;
        unsigned char* original_data = stbi_load(frame_path, &width, &height, &channels, 3);
        if (!original_data) {
            std::cerr << "Failed to read the image\n";
            return;
        }

        std::cout << "Image loaded: " << width << "x" << height << " Channels: " << channels << std::endl;

        int new_width = 640;
        int new_height = 640;
        unsigned char* resized_data = new unsigned char[new_width * new_height * 3];

        stbir_resize_uint8(original_data, width, height, 0, resized_data, new_width, new_height, 0, 3);

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

        std::vector<std::array<float, 4>> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        at::Tensor transposed_output = output[0].transpose(1, 0).contiguous();
        int rows = transposed_output.size(0);

        for (int i = 0; i < rows; ++i) {
            auto data = transposed_output[i].data_ptr<float>();
            std::vector<float> class_scores(data + 4, data + transposed_output.size(1));
            auto [maxScore, maxClassIndex] = find_max_score(class_scores);
            if (maxScore >= 0.25) {
                std::array<float, 4> box = {data[0] - (0.5 * data[2]), data[1] - (0.5 * data[3]), data[2], data[3]};
                boxes.push_back(box);
                scores.push_back(maxScore);
                class_ids.push_back(maxClassIndex);
            }
        }

        auto keep = apply_nms(boxes, scores, class_ids, 0.25, 0.45);

        std::vector<std::tuple<std::array<float, 4>, float, int>> nms_boxes;
        for (auto idx : keep) {
            nms_boxes.emplace_back(boxes[idx], scores[idx], class_ids[idx]);
        }

        std::vector<unsigned char> image_with_boxes = draw_rectangles(original_data, width, height, nms_boxes);
        std::cout << "Drawing rectangles done." << std::endl;
        std::cout << "S1" << std::endl;
        stbi_write_jpg(output_path, width, height, 3, image_with_boxes, 100);
        std::cout << "S2" << std::endl;
        delete[] original_data;
        delete[] resized_data;
        std::cout << "S3" << std::endl;
        stbi_image_free(image_with_boxes);
        std::cout << "S4" << std::endl;
    }


    void release_model(YOLOv8* model) {
        delete model;
    }
}