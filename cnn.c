#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>

#define BMP_HEADER_SIZE 54  // Standard BMP header size (for BMP images)
#define MAX_THREADS 12      // Maximum number of threads (1,3,6,9,12)

// Structure to store thread-related data
typedef struct {
    int thread_id;   // Thread ID
    int start_row;   // Start row of image assigned to thread
    int end_row;     // End row of image assigned to thread
} ThreadData;

// Global variables for image processing
unsigned char *image, *output_image;  // Pointers to store original and processed image data
int width, height, stride;        // Image dimensions and row padding

// Convolution kernel (3x3 sharpening filter)
int kernel[3][3] = {
    { 0, -1, 0 },
    { -1, 50, -1 },
    { 0, -1, 0 }
};

// Function to read a BMP image file
unsigned char* read_bmp(const char *filename) {
    printf("\n[Task 1: Reading BMP Image] - Started\n");
    
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return NULL;
    }
    
    // Read BMP header
    unsigned char *header = (unsigned char *)malloc(BMP_HEADER_SIZE);
    fread(header, 1, BMP_HEADER_SIZE, file);
    
    // Extract width and height from header
    width = *(int*)&header[18];
    height = *(int*)&header[22];

    // Ensure row size is a multiple of 4 (BMP format padding)
    stride = (width * 3 + 3) & (~3); // Each pixel in a 24-bit BMP uses 3 bytes (1 byte per color: Red, Green, Blue).
                                        // Adds 3 extra bytes to ensure that rounding up to the nearest multiple of 4 is possible.
                                        // Applying a bitwise AND with this value rounds down the result to the nearest multiple of 4.
    // Allocate memory for image data
    image = (unsigned char *)malloc(height * stride);
    output_image = (unsigned char *)malloc(height * stride);

    // Read pixel data
    fread(image, sizeof(unsigned char), height * stride, file);
    fclose(file);
    
    printf("[Task 1: Reading BMP Image] - Completed\n");
    return header;
}

// Function to apply convolution and ReLU activation (Executed by Threads)
void* apply_filter(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    printf("   [Thread %d] - Processing rows %d to %d\n", data->thread_id, data->start_row, data->end_row);

    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 1; j < width - 1; j++) {
            for (int color = 0; color < 3; color++) {
                int sum = 0;

                // Apply convolution kernel
                for (int ki = -1; ki <= 1; ki++) {
                    for (int kj = -1; kj <= 1; kj++) {
                        int row = i + ki;
                        int col = (j + kj) * 3 + color;
                        sum += image[row * stride + col] * kernel[ki + 1][kj + 1];
                    }
                }

                // Apply ReLU activation (Negative values are set to 0)
                sum = sum < 0 ? 0 : (sum > 255 ? 255 : sum);
                output_image[i * stride + j * 3 + color] = (unsigned char)sum;
            }
        }
    }

    printf("   [Thread %d] - Completed processing\n", data->thread_id);
    pthread_exit(NULL);
}

// Function to save processed BMP image
void save_bmp(const char *filename, unsigned char *header) {
    printf("\n[Task 4: Saving Processed BMP Image] - Started\n");

    FILE *file = fopen(filename, "wb");
    fwrite(header, 1, BMP_HEADER_SIZE, file);
    fwrite(output_image, sizeof(unsigned char), height * stride, file);
    fclose(file);
    
    printf("[Task 4: Saving Processed BMP Image] - Completed\n");
}

// Function to run threading experiments and measure execution time
double run_experiment(const char *input_filename, const char *output_filename, int num_threads) {
    printf("\n======================================\n");
    printf("[Experiment] Running with %d threads\n", num_threads);
    printf("======================================\n");

    unsigned char *header = read_bmp(input_filename);
    if (!header) return -1;

    pthread_t threads[MAX_THREADS];
    ThreadData thread_data[MAX_THREADS];
    struct timespec start, end;

    printf("\n[Task 2 and 3: Creating Threads & Processing Image with RELU] - Started\n");

    clock_gettime(CLOCK_MONOTONIC, &start);

    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].start_row = (height / num_threads) * i;
        thread_data[i].end_row = (i == num_threads - 1) ? height : (height / num_threads) * (i + 1);

        pthread_create(&threads[i], NULL, apply_filter, &thread_data[i]);
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    
    printf("[Task 2 and 3: Creating Threads & Processing Image] - Completed\n");

    // Calculate execution time
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("\n[Task 4: Combining] - Execution Time: %f sec\n", time_taken);

    // Save the output image
    save_bmp(output_filename, header);

    // Free allocated memory
    free(header);
    free(image);
    free(output_image);

    return time_taken;
}


// Main function to run experiments
int main(int argc, char *argv[]) {
    // Ensure a BMP file is provided
    if (argc != 2) {
        printf("Usage: %s <input BMP file>\n", argv[0]);
        return 1;
    }

    const char *input_filename = argv[1];  // Read BMP file from command line
    const char *output_filename_template = "output_%d_threads.bmp";

    int threads[] = {1, 3, 6, 9, 12};
    FILE *log_file = fopen("timing_results.txt", "w");

    printf("\n[Program Start] Image Processing Begins\n");

    int num_thread_configs = sizeof(threads) / sizeof(threads[0]);

    for (int i = 0; i < num_thread_configs; i++) {
        char output_filename[50];
        sprintf(output_filename, output_filename_template, threads[i]);

        // Run the experiment with the given number of threads
        double time_taken = run_experiment(input_filename, output_filename, threads[i]);

        // Log execution time if valid
        if (time_taken > 0) {
            fprintf(log_file, "%d %f\n", threads[i], time_taken);
        }
    }
    
    fclose(log_file);
    printf("\n[Program End] All Experiments Completed\n");

    return 0;
}
