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

#include "include/Neural_Network_Threading.h"

Threaded_Inference_Result* Threaded_Inference_Result_alloc(const Neural_Network* nn, const MNIST_Images* images,
    size_t image_start_index, size_t num_images) {

    if (nn == NULL || images == NULL) {

        fprintf(stderr, "ERR: <Threaded_Inference_Result_alloc> Invalid Neural Network or MNIST_Images passed to Threaded_Inference_Result_alloc\n");
        exit(EXIT_FAILURE);
    }

    Threaded_Inference_Result* target = malloc(sizeof(Threaded_Inference_Result));

    if (target == NULL) {

        fprintf(stderr, "ERR: <Threaded_Inference_Result_alloc> Unable to allocate Threaded_Inference_Result\n");
        exit(EXIT_FAILURE);
    }

    target->nn = nn;
    target->images = images;
    target->image_start_index = image_start_index;
    target->num_images = num_images;

    target->results = FloatMatrix_alloc(MNIST_LABELS, num_images);

    target->clear = Threaded_Inference_Result_clear;

    return target;
}

void Threaded_Inference_Result_clear( Threaded_Inference_Result* target) {

    if (target == NULL) { return; }

    if (target->results == NULL) { free(target); return; }

    target->results->clear(target->results);
    free(target);

    return;
}

void* Neural_Network_Threading_predict(void* thread_void) {

    if (thread_void == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_Threading_predict> Invalid Threaded_Inference_Results provided to batch_predict\n");
        exit(EXIT_FAILURE);
    }

    Threaded_Inference_Result* thread = (Threaded_Inference_Result*)thread_void;
    
    // Create a FloatMatrix reference to use as we iterate over the images
    FloatMatrix* current_result = NULL;

    for (size_t i = 0; i < thread->num_images; ++i) {

        current_result = Neural_Network_predict(thread->nn, thread->images->get(thread->images, i + thread->image_start_index));

        for (size_t j = 0; j < current_result->num_rows; ++j) {

            // For the output of each image, copy each row of the Vector into the corresponding coordinate in the Matrix
            thread->results->set(thread->results, j, i, current_result->get(current_result, j, 0));
        }

        current_result->clear(current_result);
    }

    return NULL;
}

Threaded_Training* Threaded_Training_alloc(Neural_Network* nn, const MNIST_Images* images, const MNIST_Labels* labels,
    size_t num_images, size_t batch_size, size_t epochs, size_t thread_id) {

    if (nn == NULL || images == NULL || labels == NULL) {

        fprintf(stderr, "ERR: <Threaded_Training_alloc> Invalid Neural Network, images, or labels provided to Threaded_Training_alloc\n");
        exit(EXIT_FAILURE);
    }

    Threaded_Training* target = malloc(sizeof(Threaded_Training));

    if (target == NULL) {

        fprintf(stderr, "ERR: <Threaded_Training_alloc> Unable to allocate memory for Threaded_Training\n");
        exit(EXIT_FAILURE);
    }

    target->nn = nn;
    target->images = images;
    target->labels = labels;
    target->num_images = num_images;
    target->batch_size = batch_size;
    target->epochs = epochs;
    target->thread_id = thread_id;

    return target;
}

void* Neural_Network_Threading_train(void* thread_void) {

    if (thread_void == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_Threading_train> Invalid Threaded_Training instance provided to train\n");
        exit(EXIT_FAILURE);
    }

    Threaded_Training* thread = (Threaded_Training*)thread_void;

    // Use a buffer for snprintf
    char buffer[100];
    memset(buffer, '\0', 100);

    for (size_t i = 0; i < thread->epochs; ++i) {

        if (NN_THREADING_DEBUG) {

            snprintf(buffer, 100, "DEBUG: <<Thread %zu>> Starting batch training epoch %zu. # of images: %zu", thread->thread_id, i, thread->num_images);
            log_message(buffer);
        }

        thread->nn->batch_train(thread->nn, thread->images, thread->labels, thread->num_images, thread->batch_size);
    }

    return NULL;
}

void Neural_Network_Threading_combine(Neural_Network** networks, size_t network_count) {

    if (networks == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_Threading_combine> Invalid network array provided to combine\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < network_count; ++i) {

        if (networks[i] == NULL) {

            fprintf(stderr, "ERR: <Neural_Network_Threading_combine> At least one invalid network provided to combine\n");
            exit(EXIT_FAILURE);
        }
    }

    // Have the base network available for easy referencing
    Neural_Network* base = networks[0];

    // We waste some memory here with having an extra FloatMatrix pointer, but it makes matching the index easier
    FloatMatrix** new_weights = calloc(base->num_layers, sizeof(FloatMatrix*));
    FloatMatrix** new_biases = calloc(base->num_layers, sizeof(FloatMatrix*));

    if (new_weights == NULL || new_biases == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_Threading_combine> Unable to allocate memory for new weights / biases FloatMatrix\n");
        exit(EXIT_FAILURE);
    }

    // Iterate over the layers and create zero'd out FloatMatrix to add up
    for (size_t i = 1; i < base->num_layers; ++i) {

        new_weights[i] = base->layers[i]->weights->copy(base->layers[i]->weights);
        new_biases[i] = base->layers[i]->biases->copy(base->layers[i]->biases);

        new_weights[i]->populate(new_weights[i], 0);
        new_biases[i]->populate(new_biases[i], 0);
    }

    // Add up all the weights and biases from each network
    for (size_t i = 0; i < network_count; ++i) {

        // Go layer by layer, ignoring the input layer
        for (size_t j = 1; j < base->num_layers; ++j) {

            new_weights[j]->add_o(new_weights[j], networks[i]->layers[j]->weights);
            new_biases[j]->add_o(new_biases[j], networks[i]->layers[j]->biases);
        }
    }

    // Scale the new weights and biases properly
    for (size_t i = 1; i < base->num_layers; ++i) {

        new_weights[i]->scale_o(new_weights[i], (float)(1.0 / network_count));
        new_biases[i]->scale_o(new_biases[i], (float)(1.0 / network_count));
    }

    // Replace the old weights and biases in each network
    for (size_t i = 0; i < network_count; ++i) {

        for (size_t j = 1; j < base->num_layers; ++j) {

            // Clean up the old weights and biases first
            networks[i]->layers[j]->weights->clear(networks[i]->layers[j]->weights);
            networks[i]->layers[j]->biases->clear(networks[i]->layers[j]->biases);

            // Copy the new values to the respective Neural Networks
            networks[i]->layers[j]->weights = new_weights[j]->copy(new_weights[j]);
            networks[i]->layers[j]->biases = new_biases[j]->copy(new_biases[j]);
        }
    }

    // Clean up
    for (size_t i = 1; i < base->num_layers; ++i) {

        new_weights[i]->clear(new_weights[i]);
        new_biases[i]->clear(new_biases[i]);
    }

    free(new_weights);
    free(new_biases);
}
