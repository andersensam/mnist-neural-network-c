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

#ifndef MNIST_LABELS_H
#define MNIST_LABELS_H

#define MNIST_LABEL_MAGIC 0x00000801
#define MNIST_LABELS 10
#define MNIST_LABELS_DEBUG 1

/* Standard dependencies */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Local dependencies */
#include "utils.h"

/* Definitions */
typedef struct MNIST_Labels {

    /* Data members */
    uint32_t num_labels;
    uint8_t* labels;

    /* Methods */

    /**
     * Get a label from the label dataset
     * @param target The MNIST_Labels instance to fetch from
     * @param index The number of the label we want to fetch
     */
    uint8_t (*get)(const struct MNIST_Labels* target, uint32_t index);

    /**
     * Clean up a MNIST_Labels instance and free associated memory
     * @param target The MNIST_Labels instance to clean up
     */
    void (*clear)(struct MNIST_Labels* target);

} MNIST_Labels;

/**
 * Read the labels from the MNIST dataset
 * @param path Path to the file we're reading the labels from
 * @return Returns an instance of MNIST_Labels containing the labels
 */
MNIST_Labels* MNIST_Labels_init(const char* path);

/**
 * Get a label from the label dataset
 * @param target The MNIST_Labels instance to fetch from
 * @param index The number of the label we want to fetch
 */
uint8_t MNIST_Labels_get(const MNIST_Labels* target, uint32_t index);

/**
 * Clean up a MNIST_Labels instance and free associated memorry
 * @param target The MNIST_Labels instance to clean up
 */
void MNIST_Labels_clear(MNIST_Labels* target);

#endif
