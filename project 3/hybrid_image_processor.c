#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

#define BMP_HEADER_SIZE 54  // Standard BMP header size (for BMP images)
#define MAX_THREADS 12      // Maximum number of threads per process

// Structure to store thread-related data
typedef struct {
    int thread_id;         // Thread ID
    int start_row;         // Start row of image segment assigned to thread
    int end_row;           // End row of image segment assigned to thread
    int mpi_rank;          // MPI rank for this thread's process
    int process_start_row; // Start row of the entire segment assigned to this process
} ThreadData;

// Global variables for image processing
unsigned char *image = NULL;            // Original image data
unsigned char *output_image = NULL;     // Processed image data
unsigned char *process_segment = NULL;  // Segment of image assigned to this process
unsigned char *process_output = NULL;   // Output segment for this process
int width, height, stride;              // Image dimensions and row padding
int num_threads = 1;                    // Number of threads per process

// Convolution kernel (3x3 sharpening filter)
int kernel[3][3] = {
    { 0, -1, 0 },
    { -1, 5, -1 },
    { 0, -1, 0 }
};

// Function to read a BMP image file (only rank 0 reads the file)
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
    stride = (width * 3 + 3) & (~3);
    
    // Allocate memory for image data
    image = (unsigned char *)malloc(height * stride);
    
    // Read pixel data
    fread(image, sizeof(unsigned char), height * stride, file);
    fclose(file);
    
    printf("[Task 1: Reading BMP Image] - Completed\n");
    return header;
}

// Function for each thread to apply convolution and ReLU activation
void* apply_filter_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    
    // Calculate the actual position in the process_segment
    int local_start = data->start_row - data->process_start_row;
    int local_end = data->end_row - data->process_start_row;
    
    printf("   [Process %d, Thread %d] - Processing local rows %d to %d (global rows %d to %d)\n", 
           data->mpi_rank, data->thread_id, local_start, local_end, 
           data->start_row, data->end_row);

    // Process each pixel in the assigned rows
    for (int i = local_start; i < local_end; i++) {
        // Skip first and last columns to avoid edge effects
        for (int j = 1; j < width - 1; j++) {
            for (int color = 0; color < 3; color++) {
                int sum = 0;

                // Apply convolution kernel
                for (int ki = -1; ki <= 1; ki++) {
                    for (int kj = -1; kj <= 1; kj++) {
                        // Calculate row in the process_segment
                        int row = i + ki;
                        // Calculate column and color channel
                        int col = (j + kj) * 3 + color;
                        
                        // Skip out-of-bounds pixels
                        if (row >= 0 && row < (data->end_row - data->process_start_row) &&
                            row + data->process_start_row >= 0 && row + data->process_start_row < height) {
                            sum += process_segment[row * stride + col] * kernel[ki + 1][kj + 1];
                        }
                    }
                }

                // Apply ReLU activation (Negative values are set to 0)
                sum = sum < 0 ? 0 : (sum > 255 ? 255 : sum);
                process_output[i * stride + j * 3 + color] = (unsigned char)sum;
            }
        }
    }

    pthread_exit(NULL);
}

// Function to save processed BMP image (only rank 0 saves the file)
void save_bmp(const char *filename, unsigned char *header) {
    printf("\n[Task 5: Saving Processed BMP Image] - Started\n");

    FILE *file = fopen(filename, "wb");
    fwrite(header, 1, BMP_HEADER_SIZE, file);
    fwrite(output_image, sizeof(unsigned char), height * stride, file);
    fclose(file);
    
    printf("[Task 5: Saving Processed BMP Image] - Completed\n");
}

int main(int argc, char *argv[]) {
    int rank, size;
    double start_time, end_time;
    unsigned char *header = NULL;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Ensure correct command line arguments
    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: %s <input BMP file> <threads per process>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    const char *input_filename = argv[1];  // Input BMP file
    num_threads = atoi(argv[2]);           // Number of threads per process
    
    if (num_threads < 1) num_threads = 1;
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    
    char output_filename[100];
    sprintf(output_filename, "output_hybrid_%dp_%dt.bmp", size, num_threads);
    
    // Variables for work distribution
    int rows_per_process;
    int process_start_row, process_end_row;
    
    // Root process reads the image
    if (rank == 0) {
        printf("\n[Program Start] Hybrid MPI+Pthread Image Processing Begins with %d processes x %d threads\n", 
               size, num_threads);
        header = read_bmp(input_filename);
        if (!header) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Allocate memory for the output image
        output_image = (unsigned char *)malloc(height * stride);
    }
    
    // Broadcast image dimensions to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&stride, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate work distribution (rows per process)
    rows_per_process = height / size;
    process_start_row = rank * rows_per_process;
    process_end_row = (rank == size - 1) ? height : process_start_row + rows_per_process;
    
    // Allocate memory for the process segment
    int segment_size = (process_end_row - process_start_row) * stride;
    process_segment = (unsigned char *)malloc(segment_size);
    process_output = (unsigned char *)malloc(segment_size);
    
    if (rank == 0) {
        printf("\n[Task 2: Distributing Work to Processes] - Started\n");
    }
    
    // Distribute image segments to processes
    if (rank == 0) {
        // Root process already has the full image
        // Copy its own segment
        memcpy(process_segment, image + process_start_row * stride, segment_size);
        
        // Send segments to other processes
        for (int i = 1; i < size; i++) {
            int i_start_row = i * rows_per_process;
            int i_end_row = (i == size - 1) ? height : i_start_row + rows_per_process;
            int i_segment_size = (i_end_row - i_start_row) * stride;
            
            MPI_Send(image + i_start_row * stride, i_segment_size, MPI_UNSIGNED_CHAR, 
                     i, 0, MPI_COMM_WORLD);
        }
    } else {
        // Receive segment from root
        MPI_Recv(process_segment, segment_size, MPI_UNSIGNED_CHAR, 
                 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    if (rank == 0) {
        printf("[Task 2: Distributing Work to Processes] - Completed\n");
        printf("\n[Task 3: Creating Threads & Processing Image with RELU] - Started\n");
    }
    
    // Synchronize before starting timer
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Create threads for processing
    pthread_t threads[MAX_THREADS];
    ThreadData thread_data[MAX_THREADS];
    
    int rows_per_thread = (process_end_row - process_start_row) / num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].mpi_rank = rank;
        thread_data[i].process_start_row = process_start_row;
        
        // Calculate row range for this thread
        thread_data[i].start_row = process_start_row + (i * rows_per_thread);
        thread_data[i].end_row = (i == num_threads - 1) ? 
                                 process_end_row : 
                                 process_start_row + ((i + 1) * rows_per_thread);
        
        pthread_create(&threads[i], NULL, apply_filter_thread, &thread_data[i]);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Synchronize and gather results
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("[Task 3: Processing Image with Threads] - Completed\n");
        printf("\n[Task 4: Gathering Results from Processes] - Started\n");
    }
    
    // Gather processed segments back to root
    if (rank == 0) {
        // Copy root's processed segment to the output image
        memcpy(output_image + process_start_row * stride, process_output, segment_size);
        
        // Receive processed segments from other processes
        for (int i = 1; i < size; i++) {
            int i_start_row = i * rows_per_process;
            int i_end_row = (i == size - 1) ? height : i_start_row + rows_per_process;
            int i_segment_size = (i_end_row - i_start_row) * stride;
            
            MPI_Recv(output_image + i_start_row * stride, i_segment_size, MPI_UNSIGNED_CHAR, 
                     i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        // Send processed segment to root
        MPI_Send(process_output, segment_size, MPI_UNSIGNED_CHAR, 
                 0, 1, MPI_COMM_WORLD);
    }
    
    // End timing
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("[Task 4: Gathering Results from Processes] - Completed\n");
        printf("\n[Task 5: Execution Time] - %f seconds\n", end_time - start_time);
        
        // Save the processed image
        save_bmp(output_filename, header);
        
        // Log the timing results
        FILE *log_file = fopen("hybrid_timing_results.txt", "a");
        fprintf(log_file, "%d %d %f\n", size, num_threads, end_time - start_time);
        fclose(log_file);
        
        printf("\n[Program End] Hybrid MPI+Pthread Image Processing Completed\n");
        
        // Free header and full image memory
        free(header);
        free(image);
        free(output_image);
    }
    
    // Free process segment memory
    free(process_segment);
    free(process_output);
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}