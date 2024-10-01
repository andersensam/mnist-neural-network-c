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
 * @version: 2024-09-30
 *
 * General Notes: MNIST file reading inspired by: https://github.com/AndrewCarterUK/mnist-neural-network-plain-c/blob/master/mnist_file.c 
 *
 * TODO: Continue adding functionality 
 */

#ifndef MNIST_IMAGES_H
#define MNIST_IMAGES_H

#define MNIST_IMAGE_MAGIC 0x00000803
#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_SIZE MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT
#define MNIST_IMAGES_DEBUG 1

/* Standard dependencies */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Local dependencies */
#include "utils.h"

/* Matrix Definition */
#define MATRIX_CREATE_HEADER
#define MATRIX_TYPE_NAME pixelMatrix
#define MATRIX_TYPE float
#include "Matrix.h"

/* Definitions */

typedef struct MNIST_Images {

    /* Data members */
    uint32_t num_images;
    pixelMatrix** images;

    /* Methods */

    /**
     * Get a pixelMatrix from the image dataset
     * @param target The MNIST_Images instance to fetch from
     * @param index The number of the image / pixelMatrix we want to fetch
     */
    pixelMatrix* (*get)(const struct MNIST_Images* target, uint32_t index);

    /**
     * Method to clean up a MNSIST_Images instance and free memory
     * @param target The instance to clean up
     */
    void (*clear)(struct MNIST_Images* target);
} MNIST_Images;

/**
 * Method to create an instance of MNIST_Images, reading in the images and creating the associated pixelMatrix objects
 * @param path Path to the file containing the images
 * @returns Returns a new MNIST_Images instance pointer
 */
MNIST_Images* init_MNIST_images(const char* path);

/**
 * Get a pixelMatrix from the image dataset
 * @param target The MNIST_Images instance to fetch from
 * @param index The number of the image / pixelMatrix we want to fetch
 */
pixelMatrix* MNIST_Images_get(const MNIST_Images* target, uint32_t index);

/**
 * Method to clean up a MNSIST_Images instance and free memory
 * @param target The instance to clean up
 */
void MNIST_Images_clear(MNIST_Images* target);

/**
 * Convert uint8_t to a float representing pixel intensity
 * @param pixel A pointer to the uint8_t pixel representation to convert
 * @returns Returns a float from 0 to 1 representing pixel / 255
 */
float MNIST_Images_pixel_to_float(const uint8_t* pixel);

#endif
