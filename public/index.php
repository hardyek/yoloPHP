<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['image'])) {
    $imagePath = $_FILES['image']['tmp_name'];
    $outputPath = "../output/output.jpg"; // Path to save the processed image

    echo "Processing image: " . $imagePath . " Output path: " . $outputPath . "<br>";

    // Load the shared library
    $ffi = FFI::cdef("
        typedef struct YOLOv8 YOLOv8;
        YOLOv8* load_model(const char* model_path);
        void process_frame(YOLOv8* model, const char* framePath, const char* outputPath);
        void release_model(YOLOv8* model);
    ", "/home/hardy/projects/yoloPHP/build/libYOLO.so");

    // Load the YOLOv8 model
    $model = $ffi->load_model("/home/hardy/projects/yoloPHP/model/yolov8n.torchscript");

    if ($model === null) {
        echo "Failed to load model.";
        return;
    }

    echo "Model loaded successfully.<br>";

    // Process the image
    $ffi->process_frame($model, $imagePath, $outputPath);

    echo "Image processed.<br>";

    // Release the model
    $ffi->release_model($model);

    echo "Image processing complete. Check output.jpg.";
} else {
    echo "Please upload an image.";
}
?>
