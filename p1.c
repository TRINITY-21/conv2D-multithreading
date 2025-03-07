#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

unsigned char* readBMP(char* filename);

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: %s <BMP file>\n", argv[0]);
        return 1;
    }

    unsigned char* data_ptr = readBMP(argv[1]);
    if(data_ptr == NULL) {
        return 1;
    }

    // Process data_ptr as needed...
    
    // Free the allocated memory
    free(data_ptr);

    return 0;
}

unsigned char* readBMP(char* filename)
{
    int i;
    FILE* f = fopen(filename, "rb");
    if (f == NULL) {
        perror("Error opening file");
        return NULL;
    }

    unsigned char info[54];

    // Read the 54-byte header
    if(fread(info, sizeof(unsigned char), 54, f) != 54) {
        printf("Error reading BMP header\n");
        fclose(f);
        return NULL;
    }

    // Extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];
    printf("Image size: [%d, %d]\n", width, height);

    // Allocate 3 bytes per pixel (assuming 24-bit BMP)
    int size = 3 * width * height;
    unsigned char* data = (unsigned char*)malloc(size);
    if (data == NULL) {
        printf("Memory allocation failed\n");
        fclose(f);
        return NULL;
    }

    // Read the pixel data at once
    if(fread(data, sizeof(unsigned char), size, f) != size) {
        printf("Error reading BMP data\n");
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);

    // Flip the order of every 3 bytes (BGR to RGB conversion)
    for(i = 0; i < size; i += 3)
    {
        unsigned char tmp = data[i];
        data[i] = data[i+2];
        data[i+2] = tmp;
    }

    return data;
}
