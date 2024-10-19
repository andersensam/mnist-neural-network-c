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
 * @version: 2024-10-19
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef INFERENCE_H
#define INFERENCE_H

/* Configure multithreading */
#define INFERENCE_MAX_THREADS 4

/* Standard dependencies */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* Local dependencies */
#include "MNIST_Images.h"
#include "MNIST_Labels.h"
#include "Neural_Network.h"
#include "Neural_Network_Threading.h"
#include "utils.h"

/* Definitions */

/**
 * Function to perform inference on a pretrained model
 * @param labels_path Path to the labels for the data
 * @param images_path Path to the images
 * @param model_path Path to the pretained model
 * @param num_predict Number of images to predict
 */
void inference(const char* labels_path, const char* images_path, const char* model_path, size_t num_predict);

/**
 * Function to perform batched inference on a pretrained model (for multithreading)
 * @param labels_path Path to the labels for the data
 * @param images_path Path to the images
 * @param model_path Path to the pretained model
 * @param num_predict Number of images to predict
 */
void threaded_inference(const char* labels_path, const char* images_path, const char* model_path, size_t num_predict);

#endif
