#ifndef YOLOV8_H
#define YOLOV8_H

#ifdef __cplusplus
extern "C" {
#endif

struct YOLOv8;
YOLOv8* load_model(const char* model_path);
void process_frame(YOLOv8* model, const char* frame_path, const char* output_path);
void release_model(YOLOv8* model);

#ifdef __cplusplus
}
#endif

#endif // YOLOV8_H
