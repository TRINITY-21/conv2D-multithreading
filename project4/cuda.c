#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#define BMP_HEADER_SIZE 54

__constant__ float d_kernel[9];

void startTimer(cudaEvent_t *start)
{
    cudaEventRecord(*start);
}

float stopTimer(cudaEvent_t *start, cudaEvent_t *stop)
{
    cudaEventRecord(*stop);
    cudaEventSynchronize(*stop);
    float time_ms;
    cudaEventElapsedTime(&time_ms, *start, *stop);
    return time_ms;
}

unsigned char *readBMP(const char *filename, int *width, int *height, int *stride)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        printf("Error opening file %s\n", filename);
        return NULL;
    }

    unsigned char header[BMP_HEADER_SIZE];
    fread(header, sizeof(unsigned char), BMP_HEADER_SIZE, file);

    *width = *(int *)&header[18];
    *height = *(int *)&header[22];
    *stride = (*width * 3 + 3) & (~3);
    printf("BMP: %d x %d\n", *width, *height);

    unsigned char *data = (unsigned char *)malloc(*height * *stride);
    fseek(file, *(int *)&header[10], SEEK_SET);
    fread(data, sizeof(unsigned char), *height * *stride, file);
    fclose(file);

    return data;
}

void writeBMP(const char *filename, unsigned char *data, int width, int height, int stride)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        printf("Error opening file %s for writing\n", filename);
        return;
    }

    unsigned char header[BMP_HEADER_SIZE] = {0};
    header[0] = 'B';
    header[1] = 'M';
    *(int *)&header[2] = BMP_HEADER_SIZE + height * stride;
    *(int *)&header[10] = BMP_HEADER_SIZE;
    *(int *)&header[14] = 40;
    *(int *)&header[18] = width;
    *(int *)&header[22] = height;
    header[26] = 1;
    header[28] = 24;

    fwrite(header, sizeof(unsigned char), BMP_HEADER_SIZE, file);
    fwrite(data, sizeof(unsigned char), height * stride, file);
    fclose(file);
}

// CUDA kernel for 2D convolution with ReLU
__global__ void conv2DKernel(unsigned char *input, unsigned char *output,
                             int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width)
        return;

    float sum = 0.0f;

    // Convolution
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            int r = row + i;
            int c = col + j;

            if (r >= 0 && r < height && c >= 0 && c < width)
            {
                int inIdx = r * width + c;
                sum += input[inIdx] * d_kernel[(i + 1) * 3 + (j + 1)];
            }
        }
    }

    // ReLU
    sum = fmaxf(0.0f, fminf(sum, 255.0f));

    // Store output
    output[row * width + col] = (unsigned char)sum;
}

// Process channels with CUDA streams
void runConvolutionWithStreams(FILE *csvFile, int threadsPerBlock, unsigned char *input,
                               int width, int height, int stride)
{
    size_t channelSize = width * height;
    size_t imageSize = height * stride;
    unsigned char *output = (unsigned char *)malloc(imageSize);

    unsigned char *h_r = (unsigned char *)malloc(channelSize);
    unsigned char *h_g = (unsigned char *)malloc(channelSize);
    unsigned char *h_b = (unsigned char *)malloc(channelSize);

    // Extract RGB channels on CPU
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int srcIdx = y * stride + x * 3;
            int dstIdx = y * width + x;
            h_r[dstIdx] = input[srcIdx];
            h_g[dstIdx] = input[srcIdx + 1];
            h_b[dstIdx] = input[srcIdx + 2];
        }
    }

    unsigned char *d_input_r, *d_input_g, *d_input_b;
    unsigned char *d_output_r, *d_output_g, *d_output_b;

    cudaMalloc(&d_input_r, channelSize);
    cudaMalloc(&d_input_g, channelSize);
    cudaMalloc(&d_input_b, channelSize);
    cudaMalloc(&d_output_r, channelSize);
    cudaMalloc(&d_output_g, channelSize);
    cudaMalloc(&d_output_b, channelSize);

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    dim3 threads(threadsPerBlock, threadsPerBlock);
    dim3 blocks((width + threads.x) / threads.x,
                (height + threads.y) / threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemsetAsync(d_output_r, 0, channelSize, stream1);
    cudaMemsetAsync(d_output_g, 0, channelSize, stream2);
    cudaMemsetAsync(d_output_b, 0, channelSize, stream3);

    float totalTime = 0.0f;
    int runs = 5;
    for (int i = 0; i < runs; i++)
    {
        cudaDeviceSynchronize();
        startTimer(&start);

        cudaMemcpyAsync(d_input_r, h_r, channelSize, cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_input_g, h_g, channelSize, cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d_input_b, h_b, channelSize, cudaMemcpyHostToDevice, stream3);

        conv2DKernel<<<blocks, threads, 0, stream1>>>(d_input_r, d_output_r, width, height);
        conv2DKernel<<<blocks, threads, 0, stream2>>>(d_input_g, d_output_g, width, height);
        conv2DKernel<<<blocks, threads, 0, stream3>>>(d_input_b, d_output_b, width, height);

        cudaDeviceSynchronize();
        totalTime += stopTimer(&start, &stop);
    }

    float avgTime = totalTime / runs;

    // Record results
    printf("Threads per block: %dx%d -> Avg time: %.3f ms\n",
           threadsPerBlock, threadsPerBlock, avgTime);
    fprintf(csvFile, "%d,%.3f\n", threadsPerBlock, avgTime);

    unsigned char *h_out_r = (unsigned char *)malloc(channelSize);
    unsigned char *h_out_g = (unsigned char *)malloc(channelSize);
    unsigned char *h_out_b = (unsigned char *)malloc(channelSize);

    cudaMemcpyAsync(h_out_r, d_output_r, channelSize, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_out_g, d_output_g, channelSize, cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(h_out_b, d_output_b, channelSize, cudaMemcpyDeviceToHost, stream3);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int srcIdx = y * width + x;
            int dstIdx = y * stride + x * 3;

            output[dstIdx] = h_out_r[srcIdx];
            output[dstIdx + 1] = h_out_g[srcIdx];
            output[dstIdx + 2] = h_out_b[srcIdx];
        }
    }

    char outName[32];
    sprintf(outName, "output_threads_%d.bmp", threadsPerBlock);
    writeBMP(outName, output, width, height, stride);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_input_r);
    cudaFree(d_input_g);
    cudaFree(d_input_b);
    cudaFree(d_output_r);
    cudaFree(d_output_g);
    cudaFree(d_output_b);

    free(h_r);
    free(h_g);
    free(h_b);
    free(h_out_r);
    free(h_out_g);
    free(h_out_b);
    free(output);
}

int main()
{
    const char *inputFile = "lena.bmp";
    const char *csvOutputFile = "timing_results.csv";
    int width, height, stride;
    unsigned char *input = readBMP(inputFile, &width, &height, &stride);

    float kernel[9] = {
        -2, -1, 0,
        -1, 1, 1,
        0, 1, 2};

    cudaMemcpyToSymbol(d_kernel, kernel, sizeof(kernel));

    FILE *csvFile = fopen(csvOutputFile, "w");
    fprintf(csvFile, "Threads,Time(ms)\n");

    int threadCounts[] = {1, 2, 4, 8, 12};
    int n = sizeof(threadCounts) / sizeof(threadCounts[0]);

    for (int i = 0; i < n; i++)
    {
        runConvolutionWithStreams(csvFile, threadCounts[i], input, width, height, stride);
    }

    fclose(csvFile);
    free(input);
    return 0;
}