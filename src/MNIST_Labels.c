/*  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 * /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 * \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 *  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 *   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 *    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 *     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 *                                                                                                               
 * Project: Neural Network in C
 * @author : Samuel Andersen
 * @version: 2024-10-22
 *
 * General Notes: MNIST file reading inspired by: https://github.com/AndrewCarterUK/mnist-neural-network-plain-c/blob/master/mnist_file.c 
 *
 * TODO: Continue adding functionality 
 */

#include "include/MNIST_Labels.h"

MNIST_Labels* MNIST_Labels_alloc(const char* path) {

    // Open the path to where the labels are stored, in read-only mode
    FILE* label_file = fopen(path, "rb");

    if (label_file == NULL) {

        fprintf(stderr, "ERR: <MNIST_Labels_alloc> Unable to open the path to the MNIST labels\n");
        exit(EXIT_FAILURE);
    }

    // We want to read in two values at once here to avoid using fread twice
    uint32_t label_buffer[2] = {0, 0};

    // Read in 8 bytes from the file, grabbing the magic number and the number of items contained
    if (fread(&label_buffer, sizeof(uint32_t), 2, label_file) != 2){

       fprintf(stderr, "ERR: <MNIST_Labels_alloc> Unable to read headers from MNIST label file\n");

        fclose(label_file);
        exit(EXIT_FAILURE);
    }

    // The first entry in the array is used for the magic number; the second is for the number of items
    if (map_uint32(label_buffer[0]) != MNIST_LABEL_MAGIC) {

        fprintf(stderr, "ERR: <MNIST_Labels_alloc> Mistmatched magic number in the MNIST label file header\n");

        if (MNIST_LABELS_DEBUG) {

            fprintf(stderr, "DEBUG: <MNIST_Labels_alloc> Got %u but expected %u\n", map_uint32(label_buffer[0]), MNIST_LABEL_MAGIC);
        }
        
        fclose(label_file);
        exit(EXIT_FAILURE);
    }

    // Allocate the memory to store the labels
    MNIST_Labels* target = malloc(sizeof(MNIST_Labels));

    if (target == NULL) {

        fprintf(stderr, "ERR: <MNIST_Labels_alloc> Unable to allocate memory to create MNIST_Labels\n");
        
        fclose(label_file);
        exit(EXIT_FAILURE);
    }

    // Persist the flipped number of items before reading in
    target->num_labels = map_uint32(label_buffer[1]);

    // Allocate the memory for storing the labels themselves
    target->labels = calloc(target->num_labels, sizeof(uint8_t));

    if (target->labels == NULL) {

        fprintf(stderr, "ERR: <MNIST_Labels_alloc> Unable to allocate memory for the labels\n");

        free(target);
        fclose(label_file);
        exit(EXIT_FAILURE);
    }

    // Read the labels from the file
    if (fread(target->labels, sizeof(uint8_t), target->num_labels, label_file) != target->num_labels) {

        fprintf(stderr, "ERR: <MNIST_Labels_alloc> Reading labels was unsuccessful\n");

        free(target->labels);
        free(target);
        fclose(label_file);
        exit(EXIT_FAILURE);
    }

    // Ensure we clean up the file reference
    fclose(label_file);

    target->get = MNIST_Labels_get;
    target->clear = MNIST_Labels_clear;
    target->size = MNIST_Labels_size;

    return target;
}

uint8_t MNIST_Labels_get(const MNIST_Labels* target, uint32_t index) {

    if (target == NULL) {

        fprintf(stderr, "ERR: <MNIST_Labels_get> Invalid MNIST_Labels pointer provided\n");
        exit(EXIT_FAILURE);
    }

    if (index >= target->num_labels) {

        fprintf(stderr, "ERR: <MNIST_Labels_get> Invalid label index provided\n");

        if (MNIST_LABELS_DEBUG) {

            fprintf(stderr, "DEBUG: <MNIST_Labels_get> Got %u but max index is %u\n", index, target->num_labels);
        }

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

size_t MNIST_Labels_size(const MNIST_Labels* target) {

    if (target == NULL) {

        fprintf(stderr, "ERR: <MNIST_Labels_size>: Invalid Labels_Images instance passed to size\n");
        exit(EXIT_FAILURE);
    }

    return sizeof(*target) + (sizeof(uint8_t) * target->num_labels);
}
