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
 * @version: 2024-10-08
 *
 * General Notes: MNIST file reading inspired by: https://github.com/AndrewCarterUK/mnist-neural-network-plain-c/blob/master/mnist_file.c 
 *
 * TODO: Continue adding functionality 
 */

#include "include/MNIST_Labels.h"

MNIST_Labels* MNIST_Labels_init(const char* path) {

    // Open the path to where the labels are stored, in read-only mode
    FILE* label_file = fopen(path, "rb");

    if (label_file == NULL) {

        if (MNIST_LABELS_DEBUG) { fprintf(stderr, "ERR: Unable to open the path to the MNIST labels\n"); }
        exit(EXIT_FAILURE);
    }

    // We want to read in two values at once here to avoid using fread twice
    uint32_t label_buffer[2] = {0, 0};

    // Read in 8 bytes from the file, grabbing the magic number and the number of items contained
    if (fread(&label_buffer, sizeof(uint32_t), 2, label_file) != 2){

        if (MNIST_LABELS_DEBUG) { fprintf(stderr, "ERR: Unable to read headers from MNIST label file\n"); }

        fclose(label_file);
        exit(EXIT_FAILURE);
    }

    // The first entry in the array is used for the magic number; the second is for the number of items
    if (map_uint32(label_buffer[0]) != MNIST_LABEL_MAGIC) {

        if (MNIST_LABELS_DEBUG) { fprintf(stderr, "ERR: Mistmatched magic number in the MNIST label file header\n"); }
        
        fclose(label_file);
        exit(EXIT_FAILURE);
    }

    // Allocate the memory to store the labels
    MNIST_Labels* target = calloc(1, sizeof(MNIST_Labels));

    if (target == NULL) {

        if (MNIST_LABELS_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory to create MNIST_Labels\n"); }
        
        fclose(label_file);
        exit(EXIT_FAILURE);
    }

    // Persist the flipped number of items before reading in
    target->num_labels = map_uint32(label_buffer[1]);

    // Allocate the memory for storing the labels themselves
    target->labels = calloc(target->num_labels, sizeof(uint8_t));

    if (target->labels == NULL) {

        if (MNIST_LABELS_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for the labels\n"); }

        free(target);
        fclose(label_file);
        exit(EXIT_FAILURE);
    }

    // Read the labels from the file
    if (fread(target->labels, sizeof(uint8_t), target->num_labels, label_file) != target->num_labels) {

        if (MNIST_LABELS_DEBUG) { fprintf(stderr, "ERR: Reading labels was unsuccessful. Returning\n"); }

        free(target->labels);
        free(target);
        fclose(label_file);
        exit(EXIT_FAILURE);
    }

    // Ensure we clean up the file reference
    fclose(label_file);

    target->get = MNIST_Labels_get;
    target->clear = MNIST_Labels_clear;

    return target;
}

uint8_t MNIST_Labels_get(const MNIST_Labels* target, uint32_t index) {

    if (target == NULL) {

        if (MNIST_LABELS_DEBUG) { fprintf(stderr, "ERR: Invalid MNIST_Labels pointer provided. Returning 0\n"); }
        exit(EXIT_FAILURE);
    }

    if (index >= target->num_labels) {

        if (MNIST_LABELS_DEBUG) { fprintf(stderr, "ERR: Invalid label index provided. Got %u and expecting max %u\n", index, target->num_labels); }
        exit(EXIT_FAILURE);
    }

    return target->labels[index];
}

void MNIST_Labels_clear(MNIST_Labels* target) {

    if (target == NULL) { return; }

    if (target->labels == NULL) { free(target); return; }

    free(target->labels);
    free(target);
}
