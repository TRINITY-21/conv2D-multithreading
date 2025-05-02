#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>

#define BMP_HEADER_SIZE 54

int width, height, stride;
unsigned char *image = NULL;
unsigned char *output_image = NULL;
int kernel[3][3] = {
    { 0, -1, 0 },
    { -1, 6, -1 },
    { 0, -1, 0 }
};

int **allocate_matrix(int h, int w) {
    int **mat = (int **)malloc(h * sizeof(char *));
    mat[0] = (int *)malloc(h * w * sizeof(char));
    for (int i = 1; i < h; i++) {
        mat[i] = mat[0] + i * w;
    }
    return mat;
}

unsigned char* read_bmp(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }
    unsigned char *header = (unsigned char *)malloc(BMP_HEADER_SIZE);
    fread(header, 1, BMP_HEADER_SIZE, file);
    width = *(int*)&header[18];
    height = *(int*)&header[22];
    stride = (width * 3 + 3) & ~3;
    image = (unsigned char *)malloc(height * stride);
    fread(image, sizeof(unsigned char), height * stride, file);
    fclose(file);
    return header;
}

void save_bmp(const char *filename, unsigned char *header) {
    FILE *file = fopen(filename, "wb");
    fwrite(header, 1, BMP_HEADER_SIZE, file);
    fwrite(output_image, sizeof(unsigned char), height * stride, file);
    fclose(file);
    printf("Output saved: %s\n", filename);
}

void apply_convolution(int **input, int **output, int h, int w, int kernel[3][3]) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int sum = 0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int ni = i + ki;
                    int nj = j + kj;
                    if (ni < 0) ni = 0;
                    if (nj < 0) nj = 0;
                    if (ni >= h) ni = h - 1;
                    if (nj >= w) nj = w - 1;
                    sum += input[ni][nj] * kernel[ki + 1][kj + 1];
                }
            }
            output[i][j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double start_time = 0, end_time = 0;
    unsigned char *header = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s <input BMP>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *input_filename = argv[1];
    char output_filename[100];
    sprintf(output_filename, "output_rgb_split_%d.bmp", size);

    if (rank == 0) {
        header = read_bmp(input_filename);
        if (!header) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&stride, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        image = (unsigned char *)malloc(height * stride);
    }

    // MPI_Bcast(image, height * stride, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    int **channel = allocate_matrix(height, width);
    int **processed = allocate_matrix(height, width);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) start_time = MPI_Wtime();

    if (rank == 0) {
        int **R = allocate_matrix(height, width);
        int **G = allocate_matrix(height, width);
        int **B = allocate_matrix(height, width);

        int **R_out = allocate_matrix(height, width);
        int **G_out = allocate_matrix(height, width);
        int **B_out = allocate_matrix(height, width);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                R[i][j] = image[i * stride + j * 3];
                G[i][j] = image[i * stride + j * 3 + 1];
                B[i][j] = image[i * stride + j * 3 + 2];
            }
        }

        // Send or process based on available ranks
        if (size > 1) {
            MPI_Send(&(R[0][0]), height * width, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        } else {
            apply_convolution(R, R_out, height, width, kernel);
        }

        if (size > 2) {
            MPI_Send(&(G[0][0]), height * width, MPI_BYTE, 2, 0, MPI_COMM_WORLD);
        } else {
            apply_convolution(G, G_out, height, width, kernel);
        }

        if (size > 3) {
            MPI_Send(&(B[0][0]), height * width, MPI_BYTE, 3, 0, MPI_COMM_WORLD);
        } else {
            apply_convolution(B, B_out, height, width, kernel);
        }

        // Receive processed data
        if (size > 1) {
            MPI_Recv(&(R_out[0][0]), height * width, MPI_BYTE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (size > 2) {
            MPI_Recv(&(G_out[0][0]), height * width, MPI_BYTE, 2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (size > 3) {
            MPI_Recv(&(B_out[0][0]), height * width, MPI_BYTE, 3, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        output_image = (unsigned char *)malloc(height * stride);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int r = R_out[i][j];
                int g = G_out[i][j];
                int b = B_out[i][j];
                output_image[i * stride + j * 3]     = (unsigned char)(r < 0 ? 0 : (r > 255 ? 255 : r));
                output_image[i * stride + j * 3 + 1] = (unsigned char)(g < 0 ? 0 : (g > 255 ? 255 : g));
                output_image[i * stride + j * 3 + 2] = (unsigned char)(b < 0 ? 0 : (b > 255 ? 255 : b));
            }
        }

        save_bmp(output_filename, header);
        end_time = MPI_Wtime();

        FILE *log_file = fopen("mpi_timing_results.txt", "a");
        fprintf(log_file, "%d %.6f\n", size, end_time - start_time);
        fclose(log_file);

        free(header);
        free(output_image);
    }
    else if (rank == 1 || rank == 2 || rank == 3) {
        MPI_Recv(&(channel[0][0]), height * width, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        apply_convolution(channel, processed, height, width, kernel);
        MPI_Send(&(processed[0][0]), height * width, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
    }

    free(image);
    free(channel[0]); free(channel);
    free(processed[0]); free(processed);

    MPI_Finalize();
    return 0;
}
