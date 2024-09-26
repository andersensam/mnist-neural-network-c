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
 * @version: 2024-09-25
 *
 * General Notes:
 *
 * TODO: Continue adding functionality 
 */
 
#include "include/Neural_Network.h"

// Include the type of Matrix that we want to use
#define MATRIX_TYPE_NAME floatMatrix
#define MATRIX_TYPE float
#define MATRIX_STRING "%1.28f"
#include "Matrix.c"

void Neural_Network_Layer_clear(Neural_Network_Layer* target) {

    if (target == NULL) { return; }

    if (target->weights != NULL) { target->weights->clear(target->weights); }

    if (target->biases != NULL) { target->biases->clear(target->biases); }

    if (target->outputs != NULL) { target->outputs->clear(target->outputs); }

    if (target->errors != NULL) { target->errors->clear(target->errors); }

    if (target->new_weights != NULL) { target->new_weights->clear(target->new_weights); }

    free(target);
}

void Neural_Network_clear(Neural_Network* target) {

    if (target == NULL) { return; }

    if (target->layers == NULL) { free(target); return; }

    for (size_t i = 0; i < target->num_layers; ++i) {

        if (target->layers[i] != NULL) { target->layers[i]->clear(target->layers[i]); }
    }

    free(target->layers);
    free(target);
}

Neural_Network_Layer* init_Neural_Network_Layer(size_t num_neurons, size_t previous_layer_neurons, bool generate_biases, bool import) {

    Neural_Network_Layer* target = malloc(sizeof(Neural_Network_Layer));

    if (target == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for new Neural_Network_Layer\n"); }
        return NULL;
    }

    // Persist the number of neurons
    target->num_neurons = num_neurons;

    if (previous_layer_neurons == 0 || import) {

        target->weights = NULL;
        target->biases = NULL;
        target->outputs = NULL;
        target->errors = NULL;
        target->new_weights = NULL;

        target->clear = Neural_Network_Layer_clear;
        target->copy = Neural_Network_Layer_copy;

        return target;
    }

    // Create a weights Matrix with the proper dimensions of current_neurons x previous_neurons
    target->weights = floatMatrix_allocate(num_neurons, previous_layer_neurons);

    // The bias Matrix is always only one column wide
    target->biases = floatMatrix_allocate(num_neurons, 1);

    if (target->weights == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for weights in Neural_Network_Layer\n"); }
        return NULL;
    }

    if (target->biases == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for biases in Neural_Network_Layer\n"); }

        target->weights->clear(target->weights);
        return NULL;
    }

    for (size_t i = 0; i < num_neurons; ++i) {

        // If we want to generate biases, grab random floats from [-1, 1]
        if (generate_biases) { target->biases->set(target->biases, i, 0, random_float()); }

        // Otherwise, set all biases to 0 (i.e. no bias)
        else {target->biases->set(target->biases, i, 0, 0); }

        for (size_t j = 0; j < previous_layer_neurons; ++j) {

            // Fill the weights Matrix with random values to start
            target->weights->set(target->weights, i, j, random_float());
        }
    }

    target->outputs = NULL;
    target->errors = NULL;
    target->new_weights = NULL;

    target->clear = Neural_Network_Layer_clear;
    target->copy = Neural_Network_Layer_copy;

    return target;
}

floatMatrix* Neural_Network_Layer_normalize_layer(const floatMatrix* target) {

    if (target == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid floatMatrix provided to normalize\n"); }
        return NULL;
    }

    float mean = target->sum(target) / target->num_rows;

    // Subtract the mean from the original value
    floatMatrix* subtract_mean = target->add_scalar(target, (-1.0) * mean);

    // Square the original value - mean: (value - mean) ^ 2
    floatMatrix* squared = subtract_mean->apply_second(subtract_mean, powf, 2.0);

    // Calculate the variance
    float var_mean = squared->sum(squared) / squared->num_rows;

    // Calculate the standard deviation of the data, including a small value to prevent a 0
    float stdev = sqrtf(var_mean + powf(10, -10.0));

    // "Divide" the values in the Matrix by scaling by 1 / stdev
    floatMatrix* divided = subtract_mean->scale(subtract_mean, 1.0 / stdev);

    // Clean up
    subtract_mean->clear(subtract_mean);
    squared->clear(squared);

    return divided;
}

Neural_Network_Layer* Neural_Network_Layer_copy(const Neural_Network_Layer* self) {

    if (self == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid Neural Network Layer provided to copy\n"); }
        return NULL;
    }

    // Take advantage of the import function to just allocate memory and setup the function pointers
    Neural_Network_Layer* target = init_Neural_Network_Layer(self->num_neurons, 0, false, true);

    if (self->weights != NULL) { target->weights = self->weights->copy(self->weights); }

    if (self->biases != NULL) { target->biases = self->biases->copy(self->biases); }

    if (self->outputs != NULL) { target->outputs = self->outputs->copy(self->outputs); }

    if (self->errors != NULL) { target->errors = self->errors->copy(self->errors); }

    if (self->new_weights != NULL) { target->new_weights = self->new_weights->copy(self->new_weights); }

    return target;
}

floatMatrix* Neural_Network_predict(const Neural_Network* self, const pixelMatrix* image) {

    if (self == NULL || image == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid Neural_Network or pixelMatrix provided to predict\n"); }
        return NULL;
    }

    // Create a floatMatrix to store the pixelMatrix data
    floatMatrix* converted_image = Neural_Network_convert_pixelMatrix(image);

    // Flatten the floatMatrix containing the image data, converting to a column vector
    floatMatrix* flat_image = converted_image->flatten(converted_image, COLUMN);
    converted_image->clear(converted_image);

    // Define some variables we're going to use again and again in loops
    floatMatrix* layer_input = NULL;

    // Setup storage for the outputs of each layer
    floatMatrix** outputs = calloc(self->num_layers, sizeof(floatMatrix*));

    if (outputs == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for output storage in predict\n"); }
        return NULL;
    }

    for (size_t i = 1; i < self->num_layers; ++i) {

        // Special processing for the first hidden layer
        if (i == 1) {

            layer_input = self->layers[i]->weights->dot(self->layers[i]->weights, flat_image);
        }
        else {

            layer_input = self->layers[i]->weights->dot(self->layers[i]->weights, 
                outputs[i - 1]);
        }

        // Add the biases to the input of the layer
        layer_input->add_o(layer_input, self->layers[i]->biases);

        // Apply the sigmoid actication function and store as the layer's output
        layer_input->apply_o(layer_input, sigmoid);

        outputs[i] = layer_input;
        layer_input = NULL;
    }

    floatMatrix* final_output = Neural_Network_softmax(outputs[self->num_layers - 1]);

    for (size_t i = 1; i < self->num_layers; ++i) {

        outputs[i]->clear(outputs[i]);
    }

    flat_image->clear(flat_image);
    free(outputs);

    return final_output;
}

void* Neural_Network_threaded_predict(void* thread_void) {

    if (thread_void == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Threaded_Inference_Results provided to batch_predict\n"); }
        return NULL;
    }

    Threaded_Inference_Result* thread = (Threaded_Inference_Result*)thread_void;
    
    // Create a floatMatrix reference to use as we iterate over the images
    floatMatrix* current_result = NULL;

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

floatMatrix* Neural_Network_sigmoid_prime(const floatMatrix* target) {

    if (target == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid floatMatrix provided to sigmoid_prime\n"); }
        return NULL;
    }

    floatMatrix* one = floatMatrix_allocate(target->num_rows, target->num_cols);

    // Populate the Matrix with 1.0 in each coordinate
    one->populate(one, 1);

    // Perform 1 - target
    one->subtract_o(one, target);

    // Multiple the subtracted Matrix with the original
    one->multiply_o(one, target);

    return one;
}

floatMatrix* Neural_Network_softmax(const floatMatrix* target) {

    if (target == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid floatMatrix provided to softmax\n"); }
        return NULL;
    }

    float total = 0;

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            total += exp(target->get(target, i, j));
        }
    }

    floatMatrix* result = floatMatrix_allocate(target->num_rows, target->num_cols);

    if (result == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for new floatMatrix in softmax\n"); }
        return NULL;
    }

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            result->set(result, i, j, exp(target->get(target, i, j)) / total);
        }
    }

    return result;
}

floatMatrix* Neural_Network_convert_pixelMatrix(const pixelMatrix* pixels) {

    if (pixels == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid pixelMatrix provided for conversion\n"); }
        return NULL;
    }

    floatMatrix* target = floatMatrix_allocate(pixels->num_rows, pixels->num_cols);

    if (target == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to allocate floatMatrix for pixelMatrix conversion\n"); }
        return NULL;
    }

    for (size_t i = 0; i < pixels->num_rows; ++i) {

        for (size_t j = 0; j < pixels->num_cols; ++j) {

            target->set(target, i, j, pixels->get(pixels, i, j));
        }
    }

    return target;
}

floatMatrix* Neural_Network_create_label(uint8_t label) {

    floatMatrix* target = floatMatrix_allocate(MNIST_LABELS, 1);

    if (target == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for label floatMatrix\n"); }
        return NULL;
    }

    // Zero out the values of the floatMatrix
    target->populate(target, 0);

    // Set only the row representing the value as a one, leaving the rest as zeros
    target->set(target, label, 0, 1);

    return target;
}

void Neural_Network_training_predict(Neural_Network* self, const floatMatrix* flat_image) {

    if (self == NULL || flat_image == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid Neural_Network or floatMatrix provided to training_predict\n"); }
    }

    // Store the input layer's output (aka the flattened image)
    self->layers[0]->outputs = flat_image->copy(flat_image);

    /* Begin the feed forward part (same as inference without softmax at the end) */
    floatMatrix* hidden_inputs = NULL;

    // For each layer, do the required transformations, skipping the input layer and output layer
    for (size_t i = 1; i < self->num_layers; ++i) {

        // Special processing for the first hidden layer
        if (i == 1) {

            // Dot product of the first hidden layer's weights by the image input
            hidden_inputs = self->layers[i]->weights->dot(self->layers[i]->weights, flat_image);
            
        }
        else {

            // The next layer's inputs are the dot of this layer's weights by the previous layer's output
            hidden_inputs = self->layers[i]->weights->dot(self->layers[i]->weights, self->layers[i - 1]->outputs);
        }

        // Apply the bias before proceeding
        hidden_inputs->add_o(hidden_inputs, self->layers[i]->biases);

        // The output of this layer is the input with the sigmoid applied
        hidden_inputs->apply_o(hidden_inputs, sigmoid);

        // Copy the sigmoid outputs and store it in the layer for use later
        self->layers[i]->outputs = hidden_inputs;
    }
}

void Neural_Network_train(Neural_Network* self, const pixelMatrix* image, uint8_t label) {

    if (self == NULL || image == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid Neural_Network or pixelMatrix provided to train\n"); }
    }

    /* Do inference and store all the layer outputs in the respective layers */
    floatMatrix* converted_image = Neural_Network_convert_pixelMatrix(image);
    floatMatrix* flat_image = converted_image->flatten(converted_image, COLUMN);
    converted_image->clear(converted_image);

    Neural_Network_training_predict(self, flat_image);
    flat_image->clear(flat_image);

    /* Begin the backpropagation part of the training process */

    // Convert the label to a floatMatrix containing the right value
    floatMatrix* output = Neural_Network_create_label(label);

    floatMatrix* transpose = NULL;
    floatMatrix* sigmoid_prime = NULL;
    floatMatrix* multiplied = NULL;
    floatMatrix* dot = NULL;
    floatMatrix* scaled = NULL;
    floatMatrix* added = NULL;

    for (size_t i = self->num_layers - 1; i >= 1; --i) {

        // If starting at the final output layer
        if (i == self->num_layers - 1) {

            // Calculate the error between the proper label and the predicted output
            self->layers[i]->errors = output->subtract(output, self->layers[self->num_layers - 1]->outputs);
        }
        else {

            // Get the error as the transpose of the next layer's weights (dot) next layer's error
            transpose = self->layers[i + 1]->weights->transpose(self->layers[i + 1]->weights);
            self->layers[i]->errors = transpose->dot(transpose, self->layers[i + 1]->errors);

            transpose->clear(transpose);
        }

        // Sig prime
        sigmoid_prime = Neural_Network_sigmoid_prime(self->layers[i]->outputs);

        // Mult
        multiplied = self->layers[i]->errors->multiply(self->layers[i]->errors, sigmoid_prime);

        // Transpose
        transpose = self->layers[i - 1]->outputs->transpose(self->layers[i - 1]->outputs);

        // Dot
        dot = multiplied->dot(multiplied, transpose);
        //printf("Size of dot for layer [%zu]: %zu x %zu\n", i, dot->num_rows, dot->num_cols);

        // Scale
        scaled = dot->scale(dot, self->learning_rate);

        // Add
        added = scaled->add(scaled, self->layers[i]->weights);

        // Persist
        self->layers[i]->new_weights = added->copy(added);

        sigmoid_prime->clear(sigmoid_prime);
        multiplied->clear(multiplied);
        transpose->clear(transpose);
        dot->clear(dot);
        scaled->clear(scaled);
        added->clear(added);
    }

    // Clean up and move the weights
    for (size_t i = 1; i < self->num_layers; ++i) {

        self->layers[i]->weights->clear(self->layers[i]->weights);
        self->layers[i]->weights = self->layers[i]->new_weights;
        self->layers[i]->new_weights = NULL;

        self->layers[i]->outputs->clear(self->layers[i]->outputs);
        self->layers[i]->outputs = NULL;

        self->layers[i]->errors->clear(self->layers[i]->errors);
        self->layers[i]->errors = NULL;
    }

    output->clear(output);

    // Clean up the special input layer
    if (self->layers[0]->outputs != NULL) {

        self->layers[0]->outputs->clear(self->layers[0]->outputs);
        self->layers[0]->outputs = NULL;
    }
}

void Neural_Network_batch_train(Neural_Network* self, const MNIST_Images* images, const MNIST_Labels* labels, size_t num_train, size_t batch_size) {

    if (self == NULL || images == NULL || labels == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid Neural Network, MNIST_Images, or MNIST_Labels provided to batch_train\n"); }

        return;
    }

    // Define variables we're going to use throughout the loops
    floatMatrix* inputs = NULL;
    floatMatrix* converted_image = NULL;
    floatMatrix* flat_image = NULL;
    floatMatrix* expanded_bias = NULL;
    floatMatrix* sigmoid_prime = NULL;
    floatMatrix* transpose = NULL;
    floatMatrix* outputs = NULL;
    floatMatrix** nabla_w = calloc(self->num_layers, sizeof(floatMatrix*));

    // Initialize values across nabla_w
    for (size_t i = 1; i < self->num_layers; ++i) {

        nabla_w[i] = NULL;
    }

    // Information about the batches
    size_t num_batches = num_train / batch_size;
    size_t final_batch_size = num_train - (batch_size * num_batches);
    size_t current_index = 0;

    // Iterate over the number of batches, ensuring we catch the last (potentially smaller) batch
    for (size_t i = 0; i <= num_batches; ++i) {

        // Handle the last batch differently
        if (i == num_batches) {

            if (final_batch_size == 0) { continue; }

            // Set up the inputs and correct outputs 
            inputs = floatMatrix_allocate(MNIST_IMAGE_SIZE, final_batch_size);
            outputs = floatMatrix_allocate(MNIST_LABELS, final_batch_size);
            outputs->populate(outputs, 0);

            // Insert the image data into the larger input Matrix
            for (size_t j = 0; j < final_batch_size; ++j) {

                converted_image = Neural_Network_convert_pixelMatrix(images->get(images, current_index + j));
                flat_image = converted_image->flatten(converted_image, COLUMN);

                converted_image->clear(converted_image);
                converted_image = NULL;

                for (size_t k = 0; k < flat_image->num_rows; ++k) {

                    inputs->set(inputs, k, j, flat_image->get(flat_image, k, 0));
                }

                flat_image->clear(flat_image);
                flat_image = NULL;

                // Set the correct output as 1.0 in the output Matrix
                outputs->set(outputs, labels->get(labels, current_index + j), j, 1);
            }

            // Update the bias vectors to be Matrix, ignoring the first layer of course
            for (size_t j = 1; j < self->num_layers; ++j) {

                expanded_bias = Neural_Network_expand_bias(self->layers[j]->biases, final_batch_size);

                self->layers[j]->biases->clear(self->layers[j]->biases);
                self->layers[j]->biases = expanded_bias;
            }
        }
        else {

            // Allocate a floatMatrix that will contain batch_size number of images
            inputs = floatMatrix_allocate(MNIST_IMAGE_SIZE, batch_size);
            outputs = floatMatrix_allocate(MNIST_LABELS, batch_size);
            outputs->populate(outputs, 0);

            // Insert the image data into the larger input Matrix
            for (size_t j = 0; j < batch_size; ++j) {

                converted_image = Neural_Network_convert_pixelMatrix(images->get(images, current_index + j));
                flat_image = converted_image->flatten(converted_image, COLUMN);

                converted_image->clear(converted_image);
                converted_image = NULL;

                for (size_t k = 0; k < flat_image->num_rows; ++k) {

                    inputs->set(inputs, k, j, flat_image->get(flat_image, k, 0));
                }

                flat_image->clear(flat_image);
                flat_image = NULL;

                // Set the correct output as 1.0 in the output Matrix
                outputs->set(outputs, labels->get(labels, current_index + j), j, 1);
            }

            // Update the bias vectors to be Matrix, ignoring the first layer of course
            for (size_t j = 1; j < self->num_layers; ++j) {

                expanded_bias = Neural_Network_expand_bias(self->layers[j]->biases, batch_size);

                self->layers[j]->biases->clear(self->layers[j]->biases);
                self->layers[j]->biases = expanded_bias;
            }
        }

        /* Run inference on the Matrix of inputs and store their outputs in each layer */
        Neural_Network_training_predict(self, inputs);

        /* Begin the backpropagation component */
        for (size_t j = self->num_layers - 1; j >= 1; --j) {

            if (j == self->num_layers - 1) {

                // Calculate the error between the proper labels and the predicted outputs
                self->layers[j]->errors = outputs->subtract(outputs, self->layers[self->num_layers - 1]->outputs);

            }
            else {

                // Get the error as the transpose of the next layer's weights (dot) next layer's error
                transpose = self->layers[j + 1]->weights->transpose(self->layers[j + 1]->weights);
                self->layers[j]->errors = transpose->dot(transpose, self->layers[j + 1]->errors);

                transpose->clear(transpose);
            }

            // Sigmoid prime of the outputs
            sigmoid_prime = Neural_Network_sigmoid_prime(self->layers[j]->outputs);

            // Multiply the errors with the sigmoid prime
            sigmoid_prime->multiply_o(sigmoid_prime, self->layers[j]->errors);

            // Transpose the outputs from the last layer
            transpose = self->layers[j - 1]->outputs->transpose(self->layers[j - 1]->outputs);

            // Get the dot product of the transposed outputs and the errors * sigmoid
            nabla_w[j] = sigmoid_prime->dot(sigmoid_prime, transpose);

            sigmoid_prime->clear(sigmoid_prime);
            transpose->clear(transpose);
        }

        /* Begin the final calculations for the new weights */

        // Update the weights for each layer
        for (size_t j = 1; j < self->num_layers; ++j) {

            // Handle the last batch with special care since it may be smaller than the others
            if (i == num_batches) {

                // Divide the sum of deltas per layer by the batch size and multiply by learning rate
                nabla_w[j]->scale_o(nabla_w[j], self->learning_rate / (float)final_batch_size);

            }
            else {

                nabla_w[j]->scale_o(nabla_w[j], self->learning_rate / (float)batch_size);
            }

            // Add the processed changes to the original weights
            self->layers[j]->weights->add_o(self->layers[j]->weights, nabla_w[j]);

            // Clean up nabla_w
            nabla_w[j]->clear(nabla_w[j]);
            nabla_w[j] = NULL;

            // Clean up errors and outputs
            self->layers[j]->outputs->clear(self->layers[j]->outputs);
            self->layers[j]->outputs = NULL;

            self->layers[j]->errors->clear(self->layers[j]->errors);
            self->layers[j]->errors = NULL;

            // Convert the bias Matrix back to being one column wide
            expanded_bias = self->layers[j]->biases->get_col(self->layers[j]->biases, 0);

            self->layers[j]->biases->clear(self->layers[j]->biases);
            self->layers[j]->biases = expanded_bias;

            expanded_bias = NULL;
        }

        // Clean up the first layer's 'output', i.e. the image Matrix
        self->layers[0]->outputs->clear(self->layers[0]->outputs);
        self->layers[0]->outputs = NULL;

        // Increment the current index for the next batch process
        current_index += batch_size;

        // Clean up input and output Matrix
        inputs->clear(inputs);
        inputs = NULL;

        outputs->clear(outputs);
        outputs = NULL;
    }

    free(nabla_w);
}

void Neural_Network_save(const Neural_Network* self, bool include_biases, const char* filename) {

    if (self == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid Neural Network to save\n"); }
        return;
    }

    FILE* model = fopen(filename, "w");

    if (model == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to open file %s to save model\n", filename); }
        return;
    }

    uint32_t header_magic = NN_HEADER_MAGIC;
    uint32_t weights_magic = NN_WEIGHTS_MAGIC;
    uint32_t weights_begin = NN_WEIGHT_BEGIN;
    uint32_t weights_end = NN_WEIGHT_END;

    // Write the header magic before starting anything else
    fwrite(&header_magic, sizeof(uint32_t), 1, model);

    // Write the learning rate of the model
    fwrite(&(self->learning_rate), sizeof(float), 1, model);

    // Determine whether or not biases are to be included
    uint32_t has_biases = include_biases ? 1 : 0;
    fwrite(&has_biases, sizeof(uint32_t), 1, model);

    // Write the number of layers
    fwrite(&(self->num_layers), sizeof(size_t), 1, model);

    // For each layer, write the number of neurons
    for (size_t i = 0; i < self->num_layers; ++i) {

        fwrite(&(self->layers[i]->num_neurons), sizeof(size_t), 1, model);
    }

    // Write the magic for the start of the weights section
    fwrite(&weights_magic, sizeof(uint32_t), 1, model);

    // Set up a float to store the weights / biases as they are read
    float current_value = 0;

    // Iterate over the layers, ignoring the input layer since it has no weights or biases
    for (size_t i = 1; i < self->num_layers; ++i) {

        // Signal the beginning of a weights Matrix
        fwrite(&weights_begin, sizeof(uint32_t), 1, model);

        for (size_t j = 0; j < self->layers[i]->weights->num_rows; ++j) {

            for (size_t k = 0; k < self->layers[i]->weights->num_cols; ++k) {

                current_value = self->layers[i]->weights->get(self->layers[i]->weights, j, k);
                fwrite(&current_value, sizeof(float), 1, model);
            }
        }

        // Signal the end of a weights Matrix
        fwrite(&weights_end, sizeof(uint32_t), 1, model);
    }

    if (!include_biases) {

        fclose(model);
        return;
    }

    uint32_t biases_magic = NN_BIASES_MAGIC;
    uint32_t bias_begin = NN_BIAS_BEGIN;
    uint32_t bias_end = NN_BIAS_END;

    // Write the magic for the start of the biases section
    fwrite(&biases_magic, sizeof(uint32_t), 1, model);

    // Iterate over the layers, ignoring the input layer since it has no weights or biases
    for (size_t i = 1; i < self->num_layers; ++i) {

        // Signal the beginning of a weights Matrix
        fwrite(&bias_begin, sizeof(uint32_t), 1, model);

        for (size_t j = 0; j < self->layers[i]->biases->num_rows; ++j) {

            current_value = self->layers[i]->biases->get(self->layers[i]->biases, j, 0);
            fwrite(&current_value, sizeof(float), 1, model);
        }

        // Signal the end of a weights Matrix
        fwrite(&bias_end, sizeof(uint32_t), 1, model);
    }

    fclose(model);
}

Neural_Network* import_Neural_Network(const char* filename) {

    if (filename == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid char* for filename provided to import\n"); }
        return NULL;
    }

    // Open the path to the model file
    FILE* model = fopen(filename, "ro");

    if (model == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to read model file\n"); }
        return NULL;
    }

    uint32_t header_magic = 0;

    // Read in the header and grab the magic number
    if (fread(&header_magic, sizeof(uint32_t), 1, model) != 1) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to read header info from model\n"); }

        fclose(model);
        return NULL;
    }

    if (header_magic != NN_HEADER_MAGIC) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid magic number in model header\n"); }

        fclose(model);
        return NULL;
    }

    // Get ready to get the learning rate for the model
    float learning_rate = 0;

    if (fread(&learning_rate, sizeof(float), 1, model) != 1) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to read learning rate from model\n"); }

        fclose(model);
        return NULL;
    }

    // Check to see if biases were included in the model, either 0 for no, or 1 for yes
    uint32_t includes_biases = 0;

    if (fread(&includes_biases, sizeof(uint32_t), 1, model) != 1) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to read bias info from model\n"); }

        fclose(model);
        return NULL;
    }

    // Get the number of layers in the model
    size_t num_layers = 0;

    if (fread(&num_layers, sizeof(size_t), 1, model) != 1) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to read number of layers from model\n"); }

        fclose(model);
        return NULL;
    }

    // Allocate an array of size_t to get the size of the layers
    size_t* layer_info = calloc(num_layers, sizeof(size_t));

    if (fread(layer_info, sizeof(size_t), num_layers, model) != num_layers) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to read layer info\n"); }

        fclose(model);
        return NULL;
    }

    // Setup a buffer to grab values from the file without making new variables
    uint32_t buffer = 0;

    if (fread(&buffer, sizeof(uint32_t), 1, model) != 1) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to read weight magic number\n"); }

        fclose(model);
        return NULL;
    }

    if (buffer != NN_WEIGHTS_MAGIC) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid weight magic number\n"); }

        fclose(model);
        return NULL;
    }

    // Once we have all the information, create a new Neural Network
    Neural_Network* target = malloc(sizeof(Neural_Network));

    if (target == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for the Neural Network\n"); }

        fclose(model);
        return NULL;
    }

    // Much of this should look the same as allocate_Neural_Network
    target->num_layers = num_layers;
    target->learning_rate = learning_rate;

    target->predict = Neural_Network_predict;
    target->threaded_predict = Neural_Network_threaded_predict;
    target->train = Neural_Network_train;
    target->batch_train = Neural_Network_batch_train;
    target->save = Neural_Network_save;
    target->copy = Neural_Network_copy;
    target->clear = Neural_Network_clear;

    // Setup the array of pointers for each layer
    target->layers = calloc(num_layers, sizeof(Neural_Network_Layer*));

    if (target->layers == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for layers\n"); }

        fclose(model);
        return NULL;
    }

    // Special processing for the input layer, since it has no weights or biases
    target->layers[0] = init_Neural_Network_Layer(layer_info[0], 0, false, true);

    for (size_t i = 1; i < num_layers; ++i) {

        // Allocate each normal layer
        target->layers[i] = init_Neural_Network_Layer(layer_info[i], layer_info[i - 1], (bool)includes_biases, true);
    }

    floatMatrix* current_weights = NULL;
    float float_buffer = 0;

    // Iterate over the number of layers, processing each of their weight matrices
    for (size_t i = 1; i < num_layers; ++i) {

        if (fread(&buffer, sizeof(uint32_t), 1, model) != 1) {

            if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Error reading model layer %zu\n", i); }

            fclose(model);
            return NULL;
        }

        if (buffer != NN_WEIGHT_BEGIN) {

            if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid start weights value for layer %zu\n", i); }

            fclose(model);
            return NULL;
        }

        // Allocate a new floatMatrix for the weights of this layer
        current_weights = floatMatrix_allocate(layer_info[i], layer_info[i - 1]);

        for (size_t j = 0; j < current_weights->num_rows; ++j) {

            for (size_t k = 0; k < current_weights->num_cols; ++k) {

                if (fread(&float_buffer, sizeof(float), 1, model) != 1) {

                    if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Error reading (%zu, %zu) for model layer %zu\n", j, k, i); }

                    fclose(model);
                    return NULL;
                }

                // Copy the float value of the weight to the new floatMatrix
                current_weights->set(current_weights, j, k, float_buffer);
            }
        }

        if (fread(&buffer, sizeof(uint32_t), 1, model) != 1) {

            if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Error reading end of model layer %zu\n", i); }

            fclose(model);
            return NULL;
        }

        if (buffer != NN_WEIGHT_END) {

            if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid end weights value for layer %zu\n", i); }

            fclose(model);
            return NULL;
        }

        // Persist the weights to the layer directly
        target->layers[i]->weights = current_weights;
        current_weights = NULL;

    }

    // If we don't include any biases, clean up and return
    if (includes_biases == 0) {

        fclose(model);
        free(layer_info);
        return target;
    }

    // Begin processing the biases and check the magic number
    if (fread(&buffer, sizeof(uint32_t), 1, model) != 1) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to read bias magic number\n"); }

        fclose(model);
        return NULL;
    }

    if (buffer != NN_BIASES_MAGIC) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid bias magic number\n"); }

        fclose(model);
        return NULL;
    }

    // Store the layer's biases somewhere
    floatMatrix* current_biases = NULL;

    for (size_t i = 1; i < num_layers; ++i) {

        if (fread(&buffer, sizeof(uint32_t), 1, model) != 1) {

            if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Error reading bias for layer %zu\n", i); }

            fclose(model);
            return NULL;
        }

        if (buffer != NN_BIAS_BEGIN) {

            if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid start bias value for layer %zu\n", i); }

            fclose(model);
            return NULL;
        }

        current_biases = floatMatrix_allocate(layer_info[i], 1);

        // Only one nested loop since the biases are stored in a vector (1-column matrix)
        for (size_t j = 0; j < current_biases->num_rows; ++j) {

            if (fread(&float_buffer, sizeof(float), 1, model) != 1) {

                if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Error reading (%zu, 0) for bias layer %zu\n", j, i); }

                fclose(model);
                return NULL;
            }

            current_biases->set(current_biases, j, 0, float_buffer);
        }

        if (fread(&buffer, sizeof(uint32_t), 1, model) != 1) {

            if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Error reading end of bias layer %zu\n", i); }

            fclose(model);
            return NULL;
        }

        if (buffer != NN_BIAS_END) {

            if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid end biases value for layer %zu\n", i); }

            fclose(model);
            return NULL;
        }

        target->layers[i]->biases = current_biases;
        current_biases = NULL;
    }

    fclose(model);
    free(layer_info);
    return target;
}

Neural_Network* init_Neural_Network(size_t num_layers, const size_t* layer_info, float learning_rate, bool generate_biases) {

    Neural_Network* target = malloc(sizeof(Neural_Network));

    if (target == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for new Neural_Network\n"); }
        return NULL;
    }

    target->num_layers = num_layers;
    target->learning_rate = learning_rate;

    target->layers = calloc(num_layers, sizeof(Neural_Network_Layer*));

    target->layers[0] = init_Neural_Network_Layer(layer_info[0], 0, generate_biases, false);

    for (size_t i = 1; i < num_layers; ++i) {

        target->layers[i] = init_Neural_Network_Layer(layer_info[i], layer_info[i - 1], generate_biases, false);
    }

    target->predict = Neural_Network_predict;
    target->threaded_predict = Neural_Network_threaded_predict;
    target->train = Neural_Network_train;
    target->batch_train = Neural_Network_batch_train;
    target->save = Neural_Network_save;
    target->copy = Neural_Network_copy;
    target->clear = Neural_Network_clear;

    return target;
}

Neural_Network* Neural_Network_copy(const Neural_Network* self) {
    
    if (self == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid Neural Network provided to copy\n"); }
        return NULL;
    }

    Neural_Network* target = malloc(sizeof(Neural_Network));

    if (target == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for Neural Network copy\n"); }
        return NULL;
    }

    target->num_layers = self->num_layers;
    target->learning_rate = self->learning_rate;
    
    target->layers = calloc(target->num_layers, sizeof(Neural_Network_Layer*));

    if (target->layers == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for Neural Network Layer array in copy\n"); }

        free(target);
        return NULL;
    }

    for (size_t i = 0; i < target->num_layers; ++i) {

        target->layers[i] = self->layers[i]->copy(self->layers[i]);
    }

    target->predict = Neural_Network_predict;
    target->threaded_predict = Neural_Network_threaded_predict;
    target->train = Neural_Network_train;
    target->batch_train = Neural_Network_batch_train;
    target->save = Neural_Network_save;
    target->copy = Neural_Network_copy;
    target->clear = Neural_Network_clear;

    return target;
}

void Threaded_Inference_Result_clear( Threaded_Inference_Result* target) {

    if (target == NULL) { return; }

    if (target->results == NULL) { free(target); return; }

    target->results->clear(target->results);
    free(target);

    return;
}

floatMatrix* Neural_Network_expand_bias(const floatMatrix* current_bias, size_t batch_size) {

    if (current_bias == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid bias Matrix provided to expand_bias\n"); }
        return NULL;
    }

    floatMatrix* expanded = floatMatrix_allocate(current_bias->num_rows, batch_size);
    float bias_value = 0;

    for (size_t i = 0; i < expanded->num_rows; ++i) {

        bias_value = current_bias->get(current_bias, i, 0);

        for (size_t j = 0; j < expanded->num_cols; ++j) {

            expanded->set(expanded, i, j, bias_value);
        }
    }

    return expanded;
}

Threaded_Inference_Result* init_Threaded_Inference_Result(const Neural_Network* nn, const MNIST_Images* images,
    size_t image_start_index, size_t num_images) {

    if (nn == NULL || images == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid Neural Network or MNIST_Images passed to init_Threaded_Inference_Result\n"); }
        return NULL;
    }

    Threaded_Inference_Result* target = malloc(sizeof(Threaded_Inference_Result));

    if (target == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to allocate Threaded_Inference_Result\n"); }
        return NULL;
    }

    target->nn = nn;
    target->images = images;
    target->image_start_index = image_start_index;
    target->num_images = num_images;

    target->results = floatMatrix_allocate(MNIST_LABELS, num_images);

    if (target->results == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Unable to allocate floatMatrix for Threaded_Inference_Result\n"); }

        free(target);
        return NULL;
    }

    target->clear = Threaded_Inference_Result_clear;

    return target;
}
