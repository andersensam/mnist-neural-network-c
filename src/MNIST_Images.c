/*  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 * /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 * \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 *  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 *   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 *    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 *     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 *                                                                                                               
 * Project: Basic Neural Network in C
 * @author : Samuel Andersen
 * @version: 2024-10-12
 *
 * General Notes: MNIST file reading inspired by: https://github.com/AndrewCarterUK/mnist-neural-network-plain-c/blob/master/mnist_file.c 
 *
 * TODO: Continue adding functionality 
 */

#include "include/MNIST_Images.h"

// Include the type of Matrix that we want to use
#define MATRIX_TYPE_NAME PixelMatrix
#define MATRIX_TYPE float
#define MATRIX_STRING "%1.3f"
#include "Matrix.c"

MNIST_Images* MNIST_Images_init(const char* path) {

    // Open the path to where the images are stored, in read-only mode
    FILE* images_file = fopen(path, "rb");

    if (images_file == NULL) {

        fprintf(stderr, "ERR: <MNIST_Images_init> Unable to open the path to the MNIST imagess\n");
        exit(EXIT_FAILURE);
    }

    // We want to read in four values at once here to avoid using fread again and again
    uint32_t image_buffer[4] = {0, 0, 0, 0};

    // Read in 8 bytes from the file, grabbing the magic number and the number of items contained
    if (fread(&image_buffer, sizeof(uint32_t), 4, images_file) != 4){

        fprintf(stderr, "ERR: <MNIST_Images_init> Unable to read headers from MNIST images file\n");

        fclose(images_file);
        exit(EXIT_FAILURE);
    }

    // The first entry in the array is used for the magic number; the second is for the number of items
    if (map_uint32(image_buffer[0]) != MNIST_IMAGE_MAGIC) {

        fprintf(stderr, "ERR: <MNIST_Images_init> Mistmatched magic number in the MNIST images file header\n");

        if (MNIST_IMAGES_DEBUG) {

            fprintf(stderr, "DEBUG: <MNIST_Images_init> Got value %u but was expecting %u\n", map_uint32(image_buffer[0]),
                MNIST_IMAGE_MAGIC);
        }
        
        fclose(images_file);
        exit(EXIT_FAILURE);
    }

    // Validate the image size matches what we're expecting
    if (map_uint32(image_buffer[2]) != MNIST_IMAGE_HEIGHT || map_uint32(image_buffer[3]) != MNIST_IMAGE_WIDTH) {

        fprintf(stderr, "ERR: <MNIST_Images_init>: Unexpected image dimensions provided. Check include/MNIST_Images.h\n");

        if (MNIST_IMAGES_DEBUG) { fprintf(stderr, "DEBUG: <MNIST_Images_init> Detected: %u x %u\n", 
            map_uint32(image_buffer[2]), map_uint32(image_buffer[3])); }

        fclose(images_file);
        exit(EXIT_FAILURE);
    }

    // Allocate the memory to store the images
    MNIST_Images* target = malloc(sizeof(MNIST_Images));

    if (target == NULL) {

        fprintf(stderr, "ERR: <MNIST_Images_init> Unable to allocate memory to create MNIST_Images\n");
        
        fclose(images_file);
        exit(EXIT_FAILURE);
    }

    // Persist the flipped number of items before reading in
    target->num_images = map_uint32(image_buffer[1]);

    // Allocate the memory for storing the labels themselves
    target->images = calloc(target->num_images, sizeof(PixelMatrix*));

    uint8_t pixel_value = 0;

    // Iterate over the expected number of images
    for (uint32_t i = 0; i < target->num_images; ++i) {

        // For each expected image, create a PixelMatrix to store it in
        target->images[i] = PixelMatrix_init(MNIST_IMAGE_HEIGHT, MNIST_IMAGE_WIDTH);

        // For each expected pixel, grab the value and store it in the PixelMatrix
        for (size_t j = 0; j < MNIST_IMAGE_HEIGHT; ++j) {

            for (size_t k = 0; k < MNIST_IMAGE_WIDTH; ++k) {

                if (fread(&pixel_value, sizeof(uint8_t), 1, images_file) != 1) {

                    fprintf(stderr, "ERR: <MNIST_Images_init> Error reading pixel data @ index (%u, %zu, %zu)\n", i,  j, k);

                    fclose(images_file);
                    exit(EXIT_FAILURE);
                }

                // Copy the value of the pixel intensity to the newly created PixelMatrix
                target->images[i]->set(target->images[i], j, k, MNIST_Images_pixel_to_float(&pixel_value));
            }
        }
    }

    fclose(images_file);

    target->get = MNIST_Images_get;
    target->clear = MNIST_Images_clear;

    return target;
}

PixelMatrix* MNIST_Images_get(const MNIST_Images* target, uint32_t index) {
    
    if (target == NULL) {

        fprintf(stderr, "ERR: <MNIST_Images_get> Invalid MNIST_Images pointer provided\n");
        exit(EXIT_FAILURE);
    }

    if (index >= target->num_images) {

        fprintf(stderr, "ERR: <MNIST_Images_get> Invalid image index provided\n");

        if (MNIST_IMAGES_DEBUG) {

            fprintf(stderr, "DEBUG: <MNIST_Images_get> Got %u but images has a max index of %u\n", index, target->num_images);
        }
        
        exit(EXIT_FAILURE);
    }

    return target->images[index];
}

void MNIST_Images_clear(MNIST_Images* target) {

    if (target == NULL) { return; }

    if (target->images == NULL) { free(target); return; }

    for (uint32_t i = 0; i < target->num_images; ++i) {

        if (target->images[i] != NULL) {

            target->images[i]->clear(target->images[i]);
        }
    }

    free(target->images);
    free(target);
}

float MNIST_Images_pixel_to_float(const uint8_t* pixel) {

    return (float)*pixel / 255.0f;
}
