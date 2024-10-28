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
 * @version: 2024-10-15
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */

#ifndef NN_THREADING_H
#define NN_THREADING_H

#define NN_THREADING_DEBUG 0

/* Standard dependencies */
#include <string.h>

/* Local dependencies */
#include "Neural_Network.h"
#include "utils.h"

typedef struct Threaded_Inference_Result {

    /* Data elements */
    const Neural_Network* nn;
    const MNIST_Images* images;
    size_t image_start_index;
    size_t num_images;
    FloatMatrix* results;

    /* Methods */

    /**
     * Cleans up Threaded Inference Result
     */
    void (*clear)(struct Threaded_Inference_Result* target);

} Threaded_Inference_Result;

typedef struct Threaded_Training {

    /* Data elements */
    Neural_Network* nn;
    const MNIST_Images* images;
    const MNIST_Labels* labels;
    size_t num_images;
    size_t batch_size;
    size_t epochs;
    size_t thread_id;

} Threaded_Training;

/**
 * Allocate a Threaded Inference Result and set the basic info
 * @param nn Neural Network to run inference on
 * @param images Pointer to MNIST_Images
 * @param image_start_index Start index of the images to be processed
 * @param num_images Number of images processed by this thread
 * @returns Returns a Threaded Inference Result and sets up the FloatMatrix inside
 */
Threaded_Inference_Result* Threaded_Inference_Result_alloc(const Neural_Network* nn, const MNIST_Images* images,
    size_t image_start_index, size_t num_images);

/**
 * Cleans up Threaded Inference Result
 */
void Threaded_Inference_Result_clear( Threaded_Inference_Result* target);

/**
 * Run inference on a batch of images. This is set up for threading
 * @param thread_void A void pointer-cast Threaded_Inference_Results instance containing all the info for executing batch inference
 */
void* Neural_Network_Threading_predict(void* thread_void);

/**
 * Allocate a Threaded_Training instance and set any required info
 * @param nn Neural Network to run training on
 * @param images Pointer to MNIST_Images (to be shared with other instances)
 * @param labels Pointer to MNIST_Labels (also to be shared)
 * @param num_images Number of images to process
 * @param batch_size Target batch size
 * @param epochs Number of epochs to run for
 * @param thread_id Id of the tread running (good for debugging)
 * @returns Returns a Threaded_Training instance
 */
Threaded_Training* Threaded_Training_alloc(Neural_Network* nn, const MNIST_Images* images, const MNIST_Labels* labels,
    size_t num_images, size_t batch_size, size_t epochs, size_t thread_id);

/**
 * Run batch training in threaded mode
 * @param thread_void A void pointer-case Threaded_Training instance containing the info to properly execute batch training
 */
void* Neural_Network_Threading_train(void* thread_void);

/**
 * Combine Neural Networks together from threaded training
 * @param networks Array of Neural Networks to combine together
 * @param network_count Number of networks to combine together
 */
void Neural_Network_Threading_combine(Neural_Network** networks, size_t network_count);

#endif
