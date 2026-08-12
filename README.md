# CUDA Image Batch Accelerator

A compact GPU Specialization Capstone project using custom CUDA kernels to process 1,000 small RGB PPM images. Each image is converted to grayscale and then processed with a Sobel edge detector.

## Requirements
- NVIDIA GPU and CUDA Toolkit (`nvcc`)
- CMake 3.18+ or `make`

## Build
```bash
mkdir -p build && cd build
cmake ..
cmake --build . --config Release
cd ..
```
or `make`.

## Generate data
```bash
python3 scripts/generate_dataset.py --output data/input --count 1000 --width 128 --height 128
```

## Run
```bash
./build/cuda_capstone --input data/input --output data/output --limit 1000 --threads 256
```
Arguments: `--input`, `--output`, `--limit`, and `--threads`.

The program reports GPU name, image count, pixels, CUDA kernel time, total runtime, and throughput.

## CUDA
Two custom kernels are used: RGB-to-grayscale and Sobel edge detection. GPU threads process pixels independently. CPU handles file I/O and orchestration.

## Evidence
```bash
mkdir -p artifacts
./build/cuda_capstone --input data/input --output data/output --limit 1000 --threads 256 | tee artifacts/execution_log.txt
cp data/input/image_0000.ppm artifacts/input_sample.ppm
cp data/output/image_0000.ppm artifacts/output_sample.ppm
```
Use the real output from the CUDA lab; do not invent benchmark values.
