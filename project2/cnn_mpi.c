#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <mpi.h>

#define BMP_HEADER_SIZE 54  // Standard BMP header size (for BMP images)

// Declare global variables to store image data
// Pointer to store the original input image data
unsigned char *image = NULL;
// Pointer to store the processed output image data
unsigned char *output_image = NULL;
// Variables to store image dimensions and row padding
int width, height, stride;

// Define a 3x3 convolution kernel for image sharpening
// This filter emphasizes the center pixel and subtracts adjacent pixels
int kernel[3][3] = {
    { 0, -1, 0 },   // Top row: only attenuates the pixel above
    { -1, 6, -1 },  // Middle row: emphasizes center pixel (5) and attenuates horizontal neighbors
    { 0, -1, 0 }    // Bottom row: only attenuates the pixel below
};

// Function to read a BMP image file (only executed by the root process with rank 0)
unsigned char* read_bmp(const char *filename) {
    // Print a message indicating the start of the image reading task
    printf("\n[Task 1: Reading BMP Image] - Started\n");
    
    // Open the BMP file in binary read mode
    FILE *file = fopen(filename, "rb");
    // Check if file was opened successfully
    if (!file) {
        // Print error message if file opening failed
        printf("Error: Could not open file %s\n", filename);
        // Return NULL to indicate failure
        return NULL;
    }
    
    // Allocate memory for the BMP header (54 bytes)
    unsigned char *header = (unsigned char *)malloc(BMP_HEADER_SIZE);
    // Read the header from the file into the allocated memory
    fread(header, 1, BMP_HEADER_SIZE, file);
    
    // Extract the width from bytes 18-21 of the header (little-endian)
    width = *(int*)&header[18];
    // Extract the height from bytes 22-25 of the header (little-endian)
    height = *(int*)&header[22];

    // Calculate row stride (padding to ensure each row is a multiple of 4 bytes)
    // This is a BMP file format requirement
    stride = (width * 3 + 3) & (~3);
    
    // Allocate memory for the entire image data
    image = (unsigned char *)malloc(height * stride);
    
    // Read the pixel data from the file into the allocated memory
    fread(image, sizeof(unsigned char), height * stride, file);
    // Close the file after reading
    fclose(file);
    
    // Print a message indicating the completion of the image reading task
    printf("[Task 1: Reading BMP Image] - Completed\n");
    // Return the header data to the caller
    return header;
}

// Function to apply convolution filter and ReLU activation to a portion of the image
// Each MPI process executes this on its assigned rows
// Function to apply convolution filter and ReLU activation to a portion of the image
// Creates separate matrices for R, G, and B channels during convolution

// Modified convolution function with proper edge handling
// Updated apply_filter function with proper boundary handling
void apply_filter(int start_row, int end_row) {
    // Define temporary matrices to store convolution results for each channel
    int **red_matrix = (int **)malloc((end_row - start_row) * sizeof(int *));
    int **green_matrix = (int **)malloc((end_row - start_row) * sizeof(int *));
    int **blue_matrix = (int **)malloc((end_row - start_row) * sizeof(int *));
    
    // Allocate memory for each row in the matrices
    for (int i = 0; i < (end_row - start_row); i++) {
        red_matrix[i] = (int *)malloc(width * sizeof(int));
        green_matrix[i] = (int *)malloc(width * sizeof(int));
        blue_matrix[i] = (int *)malloc(width * sizeof(int));
    }
    
    // Process each color channel separately
    for (int color = 0; color < 3; color++) {
        // Select which matrix to use based on color channel
        int **current_matrix;
        if (color == 0) current_matrix = red_matrix;       // Red channel
        else if (color == 1) current_matrix = green_matrix; // Green channel
        else current_matrix = blue_matrix;                 // Blue channel
        
        // Loop through each assigned row
        for (int i = start_row; i < end_row; i++) {
            // Relative row index in our matrix
            int local_i = i - start_row;
            
            // Loop through each column (pixel) in the row
            for (int j = 0; j < width; j++) {
                // Initialize accumulator for convolution result
                int sum = 0;
                
                // Apply the 3x3 convolution kernel
                for (int ki = -1; ki <= 1; ki++) {
                    for (int kj = -1; kj <= 1; kj++) {
                        // Calculate the corresponding image row and column
                        int row = i + ki;
                        int col = j + kj;
                        
                        // Only process pixels that are within the image bounds
                        if (row >= 0 && row < height && col >= 0 && col < width) {
                            // Extract the color value at the current position
                            unsigned char pixel_value = image[row * stride + col * 3 + color];
                            
                            // Apply the kernel weight to the pixel and add to sum
                            sum += pixel_value * kernel[ki + 1][kj + 1];
                        }
                        else {
                            // For out-of-bounds pixels, use the value of the closest valid pixel
                            // This prevents artifacts at the image boundaries
                            int clamped_row = row < 0 ? 0 : (row >= height ? height - 1 : row);
                            int clamped_col = col < 0 ? 0 : (col >= width ? width - 1 : col);
                            
                            unsigned char pixel_value = image[clamped_row * stride + clamped_col * 3 + color];
                            sum += pixel_value * kernel[ki + 1][kj + 1];
                        }
                    }
                }
                
                // Store the convolution result in the appropriate matrix
                current_matrix[local_i][j] = sum;
            }
        }
    }
    
    // Apply ReLU activation and store results in output image
    for (int i = start_row; i < end_row; i++) {
        int local_i = i - start_row;
        for (int j = 0; j < width; j++) {
            // Get values from each channel matrix
            int red_val = red_matrix[local_i][j];
            int green_val = green_matrix[local_i][j];
            int blue_val = blue_matrix[local_i][j];
            
            // Apply ReLU activation (clip values to 0-255 range)
            red_val = red_val < 0 ? 0 : (red_val > 255 ? 255 : red_val);
            green_val = green_val < 0 ? 0 : (green_val > 255 ? 255 : green_val);
            blue_val = blue_val < 0 ? 0 : (blue_val > 255 ? 255 : blue_val);
            
            // Store the results in the output image
            output_image[i * stride + j * 3] = (unsigned char)red_val;       // Red
            output_image[i * stride + j * 3 + 1] = (unsigned char)green_val; // Green
            output_image[i * stride + j * 3 + 2] = (unsigned char)blue_val;  // Blue
        }
    }
    
    // Free the allocated memory for the matrices
    for (int i = 0; i < (end_row - start_row); i++) {
        free(red_matrix[i]);
        free(green_matrix[i]);
        free(blue_matrix[i]);
    }
    free(red_matrix);
    free(green_matrix);
    free(blue_matrix);
}


// Function to save the processed image as a BMP file
// Only the root process (rank 0) performs this operation
void save_bmp(const char *filename, unsigned char *header) {
    // Print a message indicating the start of the image saving task
    printf("\n[Task 4: Saving Processed BMP Image] - Started\n");

    // Open a file for binary writing
    FILE *file = fopen(filename, "wb");
    // Write the original BMP header to the file
    fwrite(header, 1, BMP_HEADER_SIZE, file);
    // Write the processed image data to the file
    fwrite(output_image, sizeof(unsigned char), height * stride, file);
    // Close the file after writing
    fclose(file);
    
    // Print a message indicating the completion of the image saving task
    printf("[Task 4: Saving Processed BMP Image] - Completed\n");
}

// Main function - entry point of the program
int main(int argc, char *argv[]) {
    // Variables to store MPI process information
    int rank;  // The ID of the current process
    int size;  // The total number of processes
    // Variables for timing the execution
    double start_time, end_time;
    // Pointer to store the BMP header
    unsigned char *header = NULL;
    
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // Get the rank (ID) of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Check if a BMP file is provided as a command line argument
    if (argc != 2) {
        // If no file is provided, show usage information (only by root process)
        if (rank == 0) {
            printf("Usage: %s <input BMP file>\n", argv[0]);
        }
        // Terminate MPI and exit the program
        MPI_Finalize();
        return 1;
    }
    
    // Get the input filename from command line arguments
    const char *input_filename = argv[1];
    // Create a string for the output filename that includes the number of processes
    char output_filename[100];
    sprintf(output_filename, "output_mpi_%d_processes.bmp", size);
    
    // Variables for work distribution
    int rows_per_process;  // Number of rows each process will handle
    int start_row, end_row;  // Start and end rows for the current process
    
    // Only the root process (rank 0) reads the input image
    if (rank == 0) {
        // Print a message indicating the start of the program
        printf("\n[Program Start] MPI Image Processing Begins with %d processes\n", size);
        // Read the BMP image and get the header
        header = read_bmp(input_filename);
        // Check if image reading was successful
        if (!header) {
            // If reading failed, abort all MPI processes
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    // Broadcast image dimensions to all processes so they know image size
    // Send the width from root to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Send the height from root to all processes
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Send the stride from root to all processes
    MPI_Bcast(&stride, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Non-root processes need to allocate memory for the image
    if (rank != 0) {
        // Allocate memory for the image data
        image = (unsigned char *)malloc(height * stride);
    }
    // All processes allocate memory for their portion of the output image
    output_image = (unsigned char *)malloc(height * stride);
    
    // Broadcast the entire image data from root to all processes
    // MPI_Bcast(image, width * height * stride, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(image, height * stride, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Calculate how many rows each process should handle
    rows_per_process = height / size;
    // Calculate the starting row for the current process
    start_row = rank * rows_per_process;
    // Calculate the ending row for the current process
    // The last process may get extra rows if height is not evenly divisible by size
    end_row = (rank == size - 1) ? height : start_row + rows_per_process;
    
    // Print a message indicating the start of image processing (only by root)
    if (rank == 0) {
        printf("\n[Task 2 and 3: Distributing Work & Processing Image with RELU] - Started\n");
    }
    
    // Synchronize all processes before starting the timer
    MPI_Barrier(MPI_COMM_WORLD);

    
    // Root process prints which rows it's processing
    if (rank == 0) {
            // Record the start time
    start_time = MPI_Wtime();
        printf("   [Process %d] - Processing rows %d to %d\n", rank, start_row, end_row);
    }
    // Apply the filter to the assigned rows
    apply_filter(start_row, end_row);
    
    // Gather the processed rows from all processes back to the root process
    MPI_Gather(output_image + start_row * stride, rows_per_process * stride, MPI_UNSIGNED_CHAR,
               output_image + start_row * stride, rows_per_process * stride, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);
               
    // Handle the case where the last process has extra rows
    // (when height is not evenly divisible by the number of processes)
    if (rank == size - 1 && end_row > start_row + rows_per_process) {
        // Last process sends its extra rows to the root
        MPI_Send(output_image + (start_row + rows_per_process) * stride, 
                 (end_row - start_row - rows_per_process) * stride, 
                 MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    
    // Root process receives the extra rows from the last process
    if (rank == 0 && height % size != 0) {
        MPI_Recv(output_image + (size * rows_per_process) * stride,
                 (height - size * rows_per_process) * stride,
                 MPI_UNSIGNED_CHAR, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Synchronize all processes before stopping the timer
    MPI_Barrier(MPI_COMM_WORLD);
    // Record the end time
    end_time = MPI_Wtime();
    
    // Root process handles final operations
    if (rank == 0) {
        // Print a message indicating the completion of image processing
        printf("[Task 2 and 3: Processing Image] - Completed\n");
        // Print the execution time
        printf("\n[Task 4: Execution Time] - %f seconds\n", end_time - start_time);
        
        // Save the processed image to a file
        save_bmp(output_filename, header);
        
        // Log the timing results to a file for performance analysis
        FILE *log_file = fopen("mpi_timing_results.txt", "a");
        fprintf(log_file, "%d %f\n", size, end_time - start_time);
        fclose(log_file);
        
        // Print a message indicating the end of the program
        printf("\n[Program End] MPI Image Processing Completed\n");
        
        // Free the memory allocated for the header
        free(header);
    }
    
    // Free the memory allocated for the image data
    free(image);
    // Free the memory allocated for the output image data
    free(output_image);
    
    // Finalize MPI and clean up MPI environment
    MPI_Finalize();
    // Return success status
    return 0;
}