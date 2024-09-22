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
 * @version: 2024-09-22
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#define NEURAL_NETWORK_DEBUG 1

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
#define MATRIX_TYPE_NAME floatMatrix
#define MATRIX_TYPE float
#include "Matrix.h"

/* Definitions */

typedef struct Neural_Network_Layer {

    /* Data elements */
    size_t num_neurons;
    floatMatrix* weights;
    floatMatrix* biases;
    floatMatrix* outputs;
    floatMatrix* errors;
    floatMatrix* new_weights;

    /* Methods */

    /**
     * Normalize a layer
     * @param target Pointer to the floatMatrix to normalize
     * @returns Returns a floatMatrix with its values normalized
     */
    floatMatrix* (*normalize)(const floatMatrix* target);

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

} Neural_Network_Layer;

/**
 * Initialize a Neural Network Layer
 * @param num_neurons Number of neurons in the layer
 * @param previous_layer_neurons Number of neurons in the previous layer. Set to 0 for the input layer
 * @param generate_biases Boolean of whether or not to generate biases
 * @param import Boolean of whether or not weights / biases are being read in
 * @returns Returns a pointer to a Neural Network Layer
 */
Neural_Network_Layer* init_Neural_Network_Layer(size_t num_neurons, size_t previous_layer_neurons, bool generate_biases, bool import);

/**
 * Normalize a Neural Network Layer
 * @param target The pointer to the floatMatrix we want to normalize
 * @returns Returns a new floatMatrix with its values normalized
 */
floatMatrix* Neural_Network_Layer_normalize_layer(const floatMatrix* target);

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

typedef struct Threaded_Inference_Result {

    /* Data elements */
    const struct Neural_Network* nn;
    const MNIST_Images* images;
    size_t image_start_index;
    size_t num_images;
    floatMatrix* results;

    /* Methods */

    /**
     * Cleans up Threaded Inference Result
     */
    void (*clear)(struct Threaded_Inference_Result* target);

} Threaded_Inference_Result;

typedef struct Neural_Network {

    /* Data elements */
    size_t num_layers;
    Neural_Network_Layer** layers;
    float learning_rate;

    /* Methods */

    /**
     * Run inference for a given input
     * @param self The Neural Network to execute inference on
     * @param image The Image to process
     * @returns Returns a floatMatrix containing probabilities for the output
     */
    floatMatrix* (*predict)(const struct Neural_Network* self, const pixelMatrix* image);

    /**
     * Run inference on a batch of images. This is set up for threading
     * @param thread_void A void pointer-cast Threaded_Inference_Results instance containing all the info for executing batch inference
     */
    void* (*threaded_predict)(void* thread_void);

    /**
     * Execute the training of the Neural Network
     * @param self The Neural Network to train
     * @param image The pixelMatrix representation of the input image
     * @param label The uint8_t label for the corresponding image
     */
    void (*train)(struct Neural_Network* self, const pixelMatrix* image, uint8_t label);

    /**
     * Execute a batch training operation on the Neural Network
     * @param self The Neural Network ot train
     * @param images The MNIST_Images instance to pull from
     * @param labels The MNIST_Labels instance to validate labels from
     * @param batch_size The number of images to train against in this batch
     */
    void (*batch_train)(struct Neural_Network* self, const MNIST_Images* images, const MNIST_Labels* labels, size_t batch_size);

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
     * Cleans up a Neural Network instance
     * @param target The Neural Network to clean up
     */
    void (*clear)(struct Neural_Network* target);

} Neural_Network;

/**
 * Initialize a Neural Network instance
 * @param num_layers Number of layers to create, including input and output
 * @param layer_info Pointer to an array of size_t containing the sizes of each layer
 * @param learning_rate Float 0 - 1 of the learning rate to use
 * @param generate_biases Generate random biases if true, set to zero if false
 * @returns Returns a pointer to a Neural Network
 */
Neural_Network* init_Neural_Network(size_t num_layers, const size_t* layer_info, float learning_rate, bool generate_biases);

/**
 * Cleans up a Neural Network instance
 * @param target The Neural Network to clean up
 */
void Neural_Network_clear(Neural_Network* target);

/**
 * Run inference for a given input
 * @param self The Neural Network to execute inference on
 * @param image The Image to process
 * @returns Returns a floatMatrix containing probabilities for the output
 */
floatMatrix* Neural_Network_predict(const Neural_Network* self, const pixelMatrix* image);

/**
 * Run inference on a batch of images. This is set up for threading
 * @param thread_void A void pointer-cast Threaded_Inference_Results instance containing all the info for executing batch inference
 */
void* Neural_Network_threaded_predict(void* thread_void);

/**
 * Calculate the sigmoid prime of a Matrix
 * @param target The Matrix to calculate the sigmoid prime of
 * @returns Returns a new Matrix of sigmoid prime values
 */
floatMatrix* Neural_Network_sigmoid_prime(const floatMatrix* target);

/**
 * Calculate the softmax of a Matrix
 * @param target The Matrix to calculate the softmax for
 * @returns Returns a new Matrix of softmax
 */
floatMatrix* Neural_Network_softmax(const floatMatrix* target);

/**
 * Convert a pixelMatrix to floatMatrix
 * @param pixels pixelMatrix instance to read from
 * @returns Returns a new floatMatrix
 */
floatMatrix* Neural_Network_convert_pixelMatrix(const pixelMatrix* pixels);

/**
 * Create a floatMatrix containing the right label
 * @param label The label for the image
 * @returns Returns a floatMatrix with a singular 1 for the right label
 */
floatMatrix* Neural_Network_create_label(uint8_t label);

/**
 * Run inference against a single image and update the outputs without running softmax
 * @param self The Neural Network to run training on
 * @param image The image to run inference on
 */
void Neural_Network_training_predict(Neural_Network* self, const pixelMatrix* image);

/**
 * Execute the training of the Neural Network
 * @param self The Neural Network to train
 * @param image The pixelMatrix representation of the input image
 * @param label The uint8_t label for the corresponding image
 */
void Neural_Network_train(Neural_Network* self, const pixelMatrix* image, uint8_t label);

/**
 * Execute a batch of training
 * @param self The Neural Network ot train
 * @param images The MNIST_Images instance to pull from
 * @param labels The MNIST_Labels instance to validate labels from
 * @param batch_size The number of images to train against in this batch
 */
void Neural_Network_batch_train(Neural_Network* self, const MNIST_Images* images, const MNIST_Labels* labels, size_t batch_size);

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
Neural_Network* import_Neural_Network(const char* filename);

/**
 * Create a copy of a Neural Network
 * @param self The Neural Network to copy
 * @returns Returns a copy of the Neural Network
 */
Neural_Network* Neural_Network_copy(const Neural_Network* self);

/**
 * Cleans up Threaded Inference Result
 */
void Threaded_Inference_Result_clear( Threaded_Inference_Result* target);

/**
 * Allocate a Threaded Inference Result and set the basic info
 * @param nn Neural Network to run inference on
 * @param images Pointer to MNIST_Images
 * @param image_start_index Start index of the images to be processed
 * @param num_images Number of images processed by this thread
 * @returns Returns a Threaded Inference Result and sets up the floatMatrix inside
 */
Threaded_Inference_Result* init_Threaded_Inference_Result(const Neural_Network* nn, const MNIST_Images* images,
    size_t image_start_index, size_t num_images);

#endif

