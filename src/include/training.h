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

#ifndef TRAINING_H
#define TRAINING_H

/* Configure multithreading */
#define TRAINING_MAX_THREADS 4
#define THREAD_EPOCHS_BEFORE_COMBINE 1

/* Show epochs when 1, does not when 0 */
#define SHOW_EPOCH 1

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

/**
 * Function to train a new model from scratch, saving it to a file when complete
 * @param labels_path Path to the labels for the data
 * @param images_path Path to the images
 * @param num_layers Number of layers to create in the network
 * @param nn_config An array of size_t containing the number of neurons in each layer
 * @param learning_rate Hyperparameter learning rate
 * @param generate_biases Boolean of whether or not to create random biases
 * @param num_training_images Number of training images to train on
 * @param epochs Number of epochs to run
 * @param model_path Path to save the model once it has been trained
 */
void train_new_model(const char* labels_path, const char* images_path, size_t num_layers, const size_t* nn_config,
    float learning_rate, bool generate_biases, size_t num_training_images, size_t epochs, const char* model_path);

/**
 * Function to train a new model from scratch using batching, saving it to a file when complete
 * @param labels_path Path to the labels for the data
 * @param images_path Path to the images
 * @param num_layers Number of layers to create in the network
 * @param nn_config An array of size_t containing the number of neurons in each layer
 * @param learning_rate Hyperparameter learning rate
 * @param lambda Hyperparameter for regularization
 * @param num_training_images Number of training images to train on
 * @param batch_size Batch size to use, does not have to divide evenly over dataset
 * @param epochs Number of epochs to run
 * @param model_path Path to save the model once it has been trained
 */
void train_new_model_batched(const char* labels_path, const char* images_path, size_t num_layers, const size_t* nn_config,
    float learning_rate, float lambda, bool generate_biases, size_t num_training_images, size_t batch_size, size_t epochs, const char* model_path);

/**
 * Function to train a new model from scratch using batching, saving it to a file when complete. Uses threading
 * @param labels_path Path to the labels for the data
 * @param images_path Path to the images
 * @param num_layers Number of layers to create in the network
 * @param nn_config An array of size_t containing the number of neurons in each layer
 * @param learning_rate Hyperparameter learning rate
 * @param lambda Hyperparameter for regularization
 * @param num_training_images Number of training images to train on
 * @param batch_size Batch size to use, does not have to divide evenly over dataset
 * @param epochs Number of epochs to run
 * @param model_path Path to save the model once it has been trained
 */
void train_new_model_batched_threaded(const char* labels_path, const char* images_path, size_t num_layers, const size_t* nn_config,
    float learning_rate, float lambda, bool generate_biases, size_t num_training_images, size_t batch_size, size_t epochs, const char* model_path);

/**
 * Function to run additional training on an existing, saving it to a file when complete
 * @param labels_path Path to the labels for the data
 * @param images_path Path to the images
 * @param num_training_images Number of training images to train on
 * @param epochs Number of epochs to run
 * @param model_path Path to the existing model
 * @param updated_model_path Path to save the updated model
 */
void train_existing_model(const char* labels_path, const char* images_path, size_t num_training_images, size_t epochs,
    const char* model_path, const char* updated_model_path);

/**
 * Function to run additional training on an existing, saving it to a file when complete
 * @param labels_path Path to the labels for the data
 * @param images_path Path to the images
 * @param num_training_images Number of training images to train on
 * @param batch_size Batch size to use for training
 * @param epochs Number of epochs to run
 * @param model_path Path to the existing model
 * @param updated_model_path Path to save the updated model
 */
void train_existing_model_batched(const char* labels_path, const char* images_path, size_t num_training_images, 
    size_t batch_size, size_t epochs, const char* model_path, const char* updated_model_path);

#endif
