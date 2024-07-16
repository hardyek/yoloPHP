<h1>YOLOPHP</h1>

<h3>YOLOv8 functionality on a PHP server using the PHP FFI and LibTorch</h3>

<h2>Directory Structure<h2>

```
├── CMakeLists.txt              # Build options and configuration
├── README.md                               
├── build                       # Directory for build outputs
│   ├── CMakeCache.txt
│   ├── CMakeFiles
│   │   ├── ...
│   ├── ... 
│   └── libYOLO.so              # Compiled shared library
├── include                     # Directory containing header files
│   ├── stb                     # Libraries for image processing
│   │   ├── stb_image.h
│   │   ├── stb_image_resize.h
│   │   └── stb_image_write.h
│   └── yolov8.h
├── libtorch                    # Directory for the libtorch library
│   ├── ...
├── model                       # Directory for model-related files 
│   ├── modelExport.py          # Script for TorchScript export
│   ├── yolov8n.pt              # YOLOv8n from ultralytics website
│   └── yolov8n.torchscript     # TorchScript export of "yolov8n.pt"
├── notebooks                   # Notebooks for experimentation
│   └── outputs.ipynb           # Post-processing steps needed
├── output                      # Directory for generated outputs
│   └── output.jpg              # Example output image
├── public                      # Directory for web server files
│   └── index.php               # Example PHP file for web interface
├── src                         # Source files for the C++ library
│   ├── stb_image_impl.cpp      # File for stb library image handling
│   └── yolov8.cpp              # Main source file
└── testInput.jpg               # Example input image for testing
```
<h1>
Setup
<h2>CUDA and CUDNN</h2>
1.  Install CUDA 12.1 from the <a href=https://developer.nvidia.com/cuda-12-1-0-download-archive>NVIDIA website</a>. <br>
2.  Install CUDNN for CUDA 12 from the <a href=https://developer.nvidia.com/cudnn-downloads>NVIDIA website</a>. <br>
<h2>(Optional) Install Python and Dependencies</h2>
Only required for model export so if using YOLOv8n then ignore this step. <br>
1.  Install Python (3.7 or higher).<br>
2.  Install ultralytics (torch and other dependencies should come bundled as requirements of ultralytics): <br>
<code>pip install ultralytics</code>
<h2>Install PHP and enable FFI</h2>
1. Install PHP (7.4 or higher). <br>
<code>sudo apt install php php-cli php-ffi</code> <br>
1. Open php.ini file (location depends on your installation, e.g., `/etc/php/7.4/cli/php.ini` for Ubuntu). <br>
2. Add or uncomment the following:<br>
<code>
[ffi] <br>
extension=ffi <br>
ffi.enable=true
</code>
<h2>Install CMake</h2>
<code>
sudo apt update <br>
sudo apt install cmake
</code>
<h2>Clone Repository</h2>
<code>
git clone https://github.com/hardyek/YOLOPHP.git <br>
cd YOLOPHP
</code>
<h2>Check file paths are correct</h2>
Search codebase for comment FILEPATH and ensure they are all correct.
</h1>
<hr>

<h1>
Check Correct Setup
<h2>Empty Build Directory</h2>
Navigate to the build directory and empty its contents: <br>
<code>
cd build <br>
rm -rf *
</code>
<h2> Rebuild libYOLO.so </h2>
<code>
cmake .. <br>
make
</code>
<h2> Navigate to public directory and run PHP server </h2>
<code>
cd .. <br>
cd public <br>
php -S localhost:8000
</code>
<h2> Test server using curl </h2>
You might want to delete output.jpg in the output directory before you do this just to confirm that it is working correctly and it isn't just the old image. <br>
1. Open a new terminal window <br>
2. Navigate to the yoloPHP directory <br>
3. Use curl to test the server with testInput.jpg <br>
<code>
curl -X POST -F "image=@testInput.jpg" http://localhost:8000/index.php
</code> <br>
or test with any image you like: <br>
<code>
curl -X POST -F "image=@/path/to/your/image.jpg" http://localhost:8000/index.php
</code>
</h1>
<hr>
<h3> 
Currently whilst the classes are determined for each box (as well as confidence score) they are not put onto the output. This is mostly done as for each use case there will be different processing done onto the detections themselves. Modify yolov8.cpp file to generate the output you would like wether it be the raw detections or an image, this is why the <code>draw_rectangles</code> function is seperate in yolov8.cpp and can easily be removed/replaced with another post-processing function.
</h3>