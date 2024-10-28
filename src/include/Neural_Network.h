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
 * @version: 2024-10-28
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#define NEURAL_NETWORK_DEBUG 1

/* Cost Function and Derivative Definition */
#define NEURAL_NETWORK_ACTIVATION Neural_Network_sigmoid
#define NEURAL_NETWORK_COST_DERIVATIVE Neural_Network_cross_entropy
#define NEURAL_NETWORK_OUTPUT_DELTA Neural_Network_sigmoid_delta

/* Markers to help with loading / saving Neural Networks */
#define NN_HEADER_MAGIC 0x0000AA00
#define NN_WEIGHTS_MAGIC 0x00000A00
#define NN_WEIGHT_BEGIN 0x00000A01
#define NN_WEIGHT_END 0x00000A02
#define NN_BIASES_MAGIC 0x00000F00
#define NN_BIAS_BEGIN 0x00000F01
#define NN_BIAS_END 0x00000F02

/**
 * File structure for model
 * 
 * uint32_t NN_HEADER_MAGIC
 * float learning_rate (generally 0.1)
 * float lambda
 * uint32_t includes_biases (0 or 1)
 * size_t number_of_layers
 * size_t[number_of_layers] number_of_neurons
 * 
 * uint32_t NN_WEIGHTS_MAGIC
 *   uint32_t NN_WEIGHT_BEGIN
 *   float[number_of_neurons * previous_layer_neurons] weight
 *   uint32_t NN_WEIGHT_END
 * 
 * uint32_t NN_BIASES_MAGIC
 *   uint32_t NN_BIAS_BEGIN
 *   float[number_of_neurons] bias
 *   uint32_t NN_BIAS_END
 */

 /* Standard dependencies */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Local dependencies */
#include "utils.h"
#include "MNIST_Images.h"
#include "MNIST_Labels.h"

/* Matrix Definition */
#define MATRIX_CREATE_HEADER
#define MATRIX_TYPE_NAME FloatMatrix
#define MATRIX_TYPE float
#include "Matrix.h"

/* Definitions */

typedef struct Neural_Network_Layer {

    /* Data elements */
    
    size_t num_neurons;
    FloatMatrix* weights;
    FloatMatrix* biases;
    FloatMatrix* outputs;
    FloatMatrix* errors;
    FloatMatrix* new_weights;
    FloatMatrix* z;

    /* Methods */

    /**
     * Copy a Neural Network Layer
     * @param self Neural Network Layer to copy
     * @returns Returns a copy of the specified Neural Network Layer
     */
    struct Neural_Network_Layer* (*copy)(const struct Neural_Network_Layer* self);

    /**
     * Cleans up a Neural Network Layer instance
     * @param target The Neural Network Layer instance to clean up
     */
    void (*clear)(struct Neural_Network_Layer* target);

    /**
     * Get the size of a Neural Network Layer
     * @param target the Neural Network Layer to get the size of
     * @returns Returns the total size of the allocated layer
     */
    size_t (*size)(const struct Neural_Network_Layer* target);

} Neural_Network_Layer;

/**
 * Initialize a Neural Network Layer
 * @param num_neurons Number of neurons in the layer
 * @param previous_layer_neurons Number of neurons in the previous layer. Set to 0 for the input layer
 * @param generate_biases Boolean of whether or not to generate biases
 * @param import Boolean of whether or not weights / biases are being read in
 * @returns Returns a pointer to a Neural Network Layer
 */
Neural_Network_Layer* Neural_Network_Layer_alloc(size_t num_neurons, size_t previous_layer_neurons, bool generate_biases, bool import);

/**
 * Copy a Neural Network Layer
 * @param target Neural Network Layer to copy
 * @returns Returns a copy of the specified Neural Network Layer
 */
Neural_Network_Layer* Neural_Network_Layer_copy(const Neural_Network_Layer* self);

/**
 * Cleans up a Neural Network Layer instance
 * @param target The Neural Network Layer instance to clean up
 */
void Neural_Network_Layer_clear(Neural_Network_Layer* target);

/**
 * Get the size of a Neural Network Layer
 * @param target the Neural Network Layer to get the size of
 * @returns Returns the total size of the allocated layer
 */
size_t Neural_Network_Layer_size(const Neural_Network_Layer* target);

typedef struct Neural_Network {

    /* Data elements */

    size_t num_layers;
    Neural_Network_Layer** layers;
    float learning_rate;
    float lambda;

    /* Methods */

    /**
     * Run inference for a given input
     * @param self The Neural Network to execute inference on
     * @param image The Image to process
     * @returns Returns a FloatMatrix containing probabilities for the output
     */
    FloatMatrix* (*predict)(const struct Neural_Network* self, const PixelMatrix* image);

    /**
     * Execute the training of the Neural Network
     * @param self The Neural Network to train
     * @param image The PixelMatrix representation of the input image
     * @param label The uint8_t label for the corresponding image
     */
    void (*train)(struct Neural_Network* self, const PixelMatrix* image, uint8_t label);

    /**
     * Execute a batch training operation on the Neural Network
     * @param self The Neural Network ot train
     * @param images The MNIST_Images instance to pull from
     * @param labels The MNIST_Labels instance to validate labels from
     * @param num_train The number of images to train on from the dataset
     * @param target_batch_size The number of images to train against in this batch
     */
    void (*batch_train)(struct Neural_Network* self, const MNIST_Images* images, const MNIST_Labels* labels, size_t num_train, size_t target_batch_size);

    /**
     * Save the Neural Network to a file
     * @param self The Neural Network to export
     * @param include_biases Boolean of whether or not to include the biases in the export
     * @param filename The name of the file to export. Will export as filename.model
     */
    void (*save)(const struct Neural_Network* self, bool include_biases, const char* filename);

    /**
     * Create a copy of a Neural Network
     * @param self The Neural Network to copy
     * @returns Returns a copy of the Neural Network
     */
    struct Neural_Network* (*copy)(const struct Neural_Network* self);

    /**
     * The cost function to apply. This is a placeholder, using the function defined in the header above
     * This should be used with Matrix->apply
     * @param z Float to apply the cost function to
     * @returns Returns a float of the cost function.
     */
    float (*activation)(float z);

    /**
     * The derivative of the cost function to apply
     * @param target A FloatMatrix to apply calculate the derivative of
     * @param is_final_layer Boolean handling special considerations for final layer
     */
    FloatMatrix* (*cost_derivative)(const FloatMatrix* target, bool is_final_layer);

    /**
     * Delta placeholder function
     * @param actual Actual value(s)
     * @param output Output(s) from the final layer
     * @returns Returns a FloatMatrix with the delta (however that is calculate for your activation)
     */
    FloatMatrix* (*delta)(const FloatMatrix* actual, const FloatMatrix* output);

    /**
     * Cleans up a Neural Network instance
     * @param target The Neural Network to clean up
     */
    void (*clear)(struct Neural_Network* target);

    /**
     * Get the size of a Neural Network
     * @param target The Neural Network to get the size of
     * @returns Returns the actual size of the Neural Network
     */
    size_t (*size)(const struct Neural_Network* target);

} Neural_Network;

/**
 * Initialize a Neural Network instance
 * @param num_layers Number of layers to create, including input and output
 * @param layer_info Pointer to an array of size_t containing the sizes of each layer
 * @param learning_rate Float 0 - 1 of the learning rate to use
 * @param generate_biases Generate random biases if true, set to zero if false
 * @param lambda Hyperparameter for regularization
 * @returns Returns a pointer to a Neural Network
 */
Neural_Network* Neural_Network_alloc(size_t num_layers, const size_t* layer_info, float learning_rate, bool generate_biases, float lambda);

/**
 * Cleans up a Neural Network instance
 * @param target The Neural Network to clean up
 */
void Neural_Network_clear(Neural_Network* target);

/**
 * Run inference for a given input
 * @param self The Neural Network to execute inference on
 * @param image The Image to process
 * @returns Returns a FloatMatrix containing probabilities for the output
 */
FloatMatrix* Neural_Network_predict(const Neural_Network* self, const PixelMatrix* image);

/**
 * Calculate the sigmoid of a given float
 * @param z The float value we want to calculate the sigmoid of
 * @returns Returns a float, containing the sigmoid
 */
float Neural_Network_sigmoid(float z);

/**
 * Calculate the sigmoid prime of a Matrix
 * @param target The Matrix to calculate the sigmoid prime of -- this should already have sigmoid applied to it
 * @param is_final_layer Boolean handling special considerations for final layer
 * @returns Returns a new Matrix of sigmoid prime values
 */
FloatMatrix* Neural_Network_sigmoid_prime(const FloatMatrix* target, bool is_final_layer);

/**
 * Calculate the "derivative" for cross-entropy (i.e. return a Matrix of 1)
 * @param target FloatMatrix to calculate the "derivative" for -- this will not be modified at all
 * @param is_final_layer Boolean handling special considerations for final layer
 * @return Returns a Matrix of ones
 */
FloatMatrix* Neural_Network_cross_entropy(const FloatMatrix* target, bool is_final_layer);

/**
 * Calculate the error / delta for sigmoid output layers
 * @param actual Actual value(s)
 * @param output The output(s) from the NN
 * @returns Returns a FloatMatrix containing `actual - output`
 */
FloatMatrix* Neural_Network_sigmoid_delta(const FloatMatrix* actual, const FloatMatrix* output);

/**
 * Calculate the softmax of a Matrix
 * @param target The Matrix to calculate the softmax for
 * @returns Returns a new Matrix of softmax
 */
FloatMatrix* Neural_Network_softmax(const FloatMatrix* target);

/**
 * Convert a PixelMatrix to FloatMatrix
 * @param pixels PixelMatrix instance to read from
 * @returns Returns a new FloatMatrix
 */
FloatMatrix* Neural_Network_convert_PixelMatrix(const PixelMatrix* pixels);

/**
 * Create a FloatMatrix containing the right label
 * @param label The label for the image
 * @returns Returns a FloatMatrix with a singular 1 for the right label
 */
FloatMatrix* Neural_Network_create_label(uint8_t label);

/**
 * Run inference against a single image and update the outputs without running softmax
 * @param self The Neural Network to run training on
 * @param flat_image The image (already flat and in FloatMatrix format) to run inference on
 */
void Neural_Network_training_predict(Neural_Network* self, const FloatMatrix* flat_image);

/**
 * Execute the training of the Neural Network
 * @param self The Neural Network to train
 * @param image The PixelMatrix representation of the input image
 * @param label The uint8_t label for the corresponding image
 */
void Neural_Network_train(Neural_Network* self, const PixelMatrix* image, uint8_t label);

/**
 * Execute a batch of training
 * @param self The Neural Network ot train
 * @param images The MNIST_Images instance to pull from
 * @param labels The MNIST_Labels instance to validate labels from
 * @param num_train The number of images to train on from the dataset
 * @param target_batch_size The number of images to train against in this batch
 */
void Neural_Network_batch_train(Neural_Network* self, const MNIST_Images* images, const MNIST_Labels* labels, size_t num_train, size_t target_batch_size);

/**
 * Save the Neural Network to a file
 * @param self The Neural Network to export
 * @param include_biases Boolean of whether or not to include the biases in the export
 * @param filename The name of the file to export. Will export as filename.model
 */
void Neural_Network_save(const Neural_Network* self, bool include_biases, const char* filename);

/**
 * Import a Neural Network from a file
 * @param filename The filename of the Neural Network to import
 */
Neural_Network* Neural_Network_import(const char* filename);

/**
 * Create a copy of a Neural Network
 * @param self The Neural Network to copy
 * @returns Returns a copy of the Neural Network
 */
Neural_Network* Neural_Network_copy(const Neural_Network* self);

/**
 * Expand the regular bias vector to a Matrix for use with (mini)batches
 * @param current_bias The FloatMatrix containing the current bias vector
 * @param batch_size The size of the batch, i.e. the number of copies of the vector we want in the Matrix
 */
FloatMatrix* Neural_Network_expand_bias(const FloatMatrix* current_bias, size_t batch_size);

/**
 * Get the size of a Neural Network
 * @param target The Neural Network to get the size of
 * @returns Returns a size_t of the actual size of the network
 */
size_t Neural_Network_size(const Neural_Network* target);

#endif
