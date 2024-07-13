#ifndef YOLOV8_H
#define YOLOV8_H

#include <torch/script.h>
#include <string>

struct YOLOv8 {
    torch::jit::script::Module module;
};

extern "C" {
    YOLOv8* load_model(const char* model_path);
    void process_frame(YOLOv8)
}