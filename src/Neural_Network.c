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
 
#include "include/Neural_Network.h"

// Include the type of Matrix that we want to use
#define MATRIX_TYPE_NAME FloatMatrix
#define MATRIX_TYPE float
#define MATRIX_STRING "%1.10f"
#include "Matrix.c"

void Neural_Network_Layer_clear(Neural_Network_Layer* target) {

    if (target == NULL) { return; }

    if (target->weights != NULL) { target->weights->clear(target->weights); }

    if (target->biases != NULL) { target->biases->clear(target->biases); }

    if (target->outputs != NULL) { target->outputs->clear(target->outputs); }

    if (target->errors != NULL) { target->errors->clear(target->errors); }

    if (target->new_weights != NULL) { target->new_weights->clear(target->new_weights); }

    if (target->z != NULL) { target->z->clear(target->z); }

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

Neural_Network_Layer* Neural_Network_Layer_init(size_t num_neurons, size_t previous_layer_neurons, bool generate_biases, bool import) {

    Neural_Network_Layer* target = malloc(sizeof(Neural_Network_Layer));

    if (target == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_Layer_init> Unable to allocate memory for new Neural_Network_Layer\n");
        exit(EXIT_FAILURE);
    }

    // Persist the number of neurons
    target->num_neurons = num_neurons;

    if (previous_layer_neurons == 0 || import) {

        target->weights = NULL;
        target->biases = NULL;
        target->outputs = NULL;
        target->errors = NULL;
        target->new_weights = NULL;
        target->z = NULL;

        target->clear = Neural_Network_Layer_clear;
        target->copy = Neural_Network_Layer_copy;

        return target;
    }

    // Create a weights Matrix with the proper dimensions of current_neurons x previous_neurons
    target->weights = FloatMatrix_init(num_neurons, previous_layer_neurons);

    // The bias Matrix is always only one column wide
    target->biases = FloatMatrix_init(num_neurons, 1);

    for (size_t i = 0; i < num_neurons; ++i) {

        // If we want to generate biases, grab random floats from [-1, 1]
        if (generate_biases) { target->biases->set(target->biases, i, 0, random_float()); }

        // Otherwise, set all biases to 0 (i.e. no bias)
        else {target->biases->set(target->biases, i, 0, 0); }

        for (size_t j = 0; j < previous_layer_neurons; ++j) {

            // Fill the weights Matrix with random values to start, using smaller weights (compared to `random_float()`)
            target->weights->set(target->weights, i, j, random_weight_float(previous_layer_neurons));
        }
    }

    target->outputs = NULL;
    target->errors = NULL;
    target->new_weights = NULL;
    target->z = NULL;

    target->clear = Neural_Network_Layer_clear;
    target->copy = Neural_Network_Layer_copy;

    return target;
}

FloatMatrix* Neural_Network_Layer_normalize_layer(const FloatMatrix* target) {

    if (target == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_Layer_normalize_layer> Invalid FloatMatrix provided to normalize\n");
        exit(EXIT_FAILURE);
    }

    float mean = target->sum(target) / target->num_rows;

    // Subtract the mean from the original value
    FloatMatrix* subtract_mean = target->add_scalar(target, (-1.0) * mean);

    // Square the original value - mean: (value - mean) ^ 2
    FloatMatrix* squared = subtract_mean->apply_second(subtract_mean, powf, 2.0);

    // Calculate the variance
    float var_mean = squared->sum(squared) / squared->num_rows;

    // Calculate the standard deviation of the data, including a small value to prevent a 0
    float stdev = sqrtf(var_mean + powf(10, -10.0));

    // "Divide" the values in the Matrix by scaling by 1 / stdev
    FloatMatrix* divided = subtract_mean->scale(subtract_mean, 1.0 / stdev);

    // Clean up
    subtract_mean->clear(subtract_mean);
    squared->clear(squared);

    return divided;
}

Neural_Network_Layer* Neural_Network_Layer_copy(const Neural_Network_Layer* self) {

    if (self == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_Layer_copy> Invalid Neural Network Layer provided to copy\n");
        exit(EXIT_FAILURE);
    }

    // Take advantage of the import function to just allocate memory and setup the function pointers
    Neural_Network_Layer* target = Neural_Network_Layer_init(self->num_neurons, 0, false, true);

    if (self->weights != NULL) { target->weights = self->weights->copy(self->weights); }

    if (self->biases != NULL) { target->biases = self->biases->copy(self->biases); }

    if (self->outputs != NULL) { target->outputs = self->outputs->copy(self->outputs); }

    if (self->errors != NULL) { target->errors = self->errors->copy(self->errors); }

    if (self->new_weights != NULL) { target->new_weights = self->new_weights->copy(self->new_weights); }

    if (self->z != NULL) { target->z = self->z->copy(self->z); }

    return target;
}

FloatMatrix* Neural_Network_predict(const Neural_Network* self, const PixelMatrix* image) {

    if (self == NULL || image == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_predict> Invalid Neural_Network or PixelMatrix provided to predict\n");
        exit(EXIT_FAILURE);
    }

    // Create a FloatMatrix to store the PixelMatrix data
    FloatMatrix* flat_image = Neural_Network_convert_PixelMatrix(image);

    // Flatten the FloatMatrix containing the image data, converting to a column vector
    flat_image->flatten_o(flat_image, COLUMN);

    // Define some variables we're going to use again and again in loops
    FloatMatrix* layer_input = NULL;

    // Setup storage for the outputs of each layer
    FloatMatrix** outputs = calloc(self->num_layers, sizeof(FloatMatrix*));

    if (outputs == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_predict> Unable to allocate memory for output storage in predict\n");
        exit(EXIT_FAILURE);
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
        layer_input->apply_o(layer_input, self->activation);

        outputs[i] = layer_input;
        layer_input = NULL;
    }

    FloatMatrix* final_output = Neural_Network_softmax(outputs[self->num_layers - 1]);

    for (size_t i = 1; i < self->num_layers; ++i) {

        outputs[i]->clear(outputs[i]);
    }

    flat_image->clear(flat_image);
    free(outputs);

    return final_output;
}

float Neural_Network_sigmoid(float z) {

    return 1.0f / (1 + exp(-1 * z));
}

FloatMatrix* Neural_Network_sigmoid_prime(const FloatMatrix* target, bool is_final_layer) {

    if (target == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_sigmoid_prime> Invalid FloatMatrix provided to sigmoid_prime\n");
        exit(EXIT_FAILURE);
    }

    // Suppress warning for not using is_final_layer
    if (is_final_layer) { is_final_layer = true; }

    FloatMatrix* one = FloatMatrix_init(target->num_rows, target->num_cols);

    // Populate the Matrix with 1.0 in each coordinate
    one->populate(one, 1);

    // Perform 1 - target
    one->subtract_o(one, target);

    // Multiple the subtracted Matrix with the original
    one->multiply_o(one, target);

    return one;
}

FloatMatrix* Neural_Network_cross_entropy(const FloatMatrix* target, bool is_final_layer) {

    if (target == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_cross_entropy> Invalid FloatMatrix provided to cross_entropy\n");
        exit(EXIT_FAILURE);
    }

    // Since we still use sigmoid prime for all other layers, simply refer to that method and return
    if (!is_final_layer) { return Neural_Network_sigmoid_prime(target, true); }

    FloatMatrix* one = FloatMatrix_init(target->num_rows, target->num_cols);

    // Populate the Matrix with 1.0 in each coordinate
    one->populate(one, 1);

    return one;
}

FloatMatrix* Neural_Network_sigmoid_delta(const FloatMatrix* actual, const FloatMatrix* output) {

    if (actual == NULL || output == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_sigmoid_delta> Invalid FloatMatrix(s) provided to sigmoid_delta\n");
        exit(EXIT_FAILURE);
    }

    return output->subtract(output, actual);
}

FloatMatrix* Neural_Network_softmax(const FloatMatrix* target) {

    if (target == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_softmax> Invalid FloatMatrix provided to softmax\n");
        exit(EXIT_FAILURE);
    }

    float total = 0;

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            total += exp(target->get(target, i, j));
        }
    }

    total = 1 / total;

    FloatMatrix* result = FloatMatrix_init(target->num_rows, target->num_cols);

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            result->set(result, i, j, exp(target->get(target, i, j)) * total);
        }
    }

    return result;
}

FloatMatrix* Neural_Network_convert_PixelMatrix(const PixelMatrix* pixels) {

    if (pixels == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_convert_PixelMatrix> Invalid PixelMatrix provided for conversion\n");
        exit(EXIT_FAILURE);
    }

    // Get a copy of the underlying data from the PixelMatrix
    float* data = pixels->expose(pixels);

    // Initialize a new FloatMatrix
    FloatMatrix* target = FloatMatrix_init(pixels->num_rows, pixels->num_cols);

    // Free the original FloatMatrix data
    free(target->data);

    // Use the copied data from the PixelMatrix
    target->data = data;

    return target;
}

FloatMatrix* Neural_Network_create_label(uint8_t label) {

    FloatMatrix* target = FloatMatrix_init(MNIST_LABELS, 1);

    // Zero out the values of the FloatMatrix
    target->populate(target, 0);

    // Set only the row representing the value as a one, leaving the rest as zeros
    target->set(target, label, 0, 1);

    return target;
}

void Neural_Network_training_predict(Neural_Network* self, const FloatMatrix* flat_image) {

    if (self == NULL || flat_image == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_training_predict> Invalid Neural_Network or FloatMatrix provided to training_predict\n");
        exit(EXIT_FAILURE);
    }

    // Store the input layer's output (aka the flattened image)
    self->layers[0]->outputs = flat_image->copy(flat_image);

    /* Begin the feed forward part (same as inference without softmax at the end) */
    FloatMatrix* hidden_inputs = NULL;

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

        // Copy the pre-activation values to the Z vector
        self->layers[i]->z = hidden_inputs->copy(hidden_inputs);

        // The output of this layer is the input with the sigmoid applied
        hidden_inputs->apply_o(hidden_inputs, self->activation);

        // Copy the sigmoid outputs and store it in the layer for use later
        self->layers[i]->outputs = hidden_inputs;
    }
}

void Neural_Network_train(Neural_Network* self, const PixelMatrix* image, uint8_t label) {

    if (self == NULL || image == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_train> Invalid Neural_Network or PixelMatrix provided to train\n");
        exit(EXIT_FAILURE);
    }

    /* Do inference and store all the layer outputs in the respective layers */
    FloatMatrix* flat_image = Neural_Network_convert_PixelMatrix(image);
    flat_image->flatten_o(flat_image, COLUMN);

    Neural_Network_training_predict(self, flat_image);
    flat_image->clear(flat_image);

    /* Begin the backpropagation part of the training process */

    // Convert the label to a FloatMatrix containing the right value
    FloatMatrix* output = Neural_Network_create_label(label);

    FloatMatrix* transpose = NULL;
    FloatMatrix* derivative = NULL;

    for (size_t i = self->num_layers - 1; i >= 1; --i) {

        // If starting at the final output layer
        if (i == self->num_layers - 1) {

            // Calculate the error between the proper label and the predicted output
            self->layers[i]->errors = self->delta(output, self->layers[i]->outputs);
            
            derivative = self->cost_derivative(self->layers[i]->outputs, true);
        }
        else {

            // Get the error as the transpose of the next layer's weights (dot) next layer's error
            transpose = self->layers[i + 1]->weights->transpose(self->layers[i + 1]->weights);
            self->layers[i]->errors = transpose->dot(transpose, self->layers[i + 1]->errors);

            transpose->clear(transpose);

            derivative = self->cost_derivative(self->layers[i]->outputs, false);
        }        

        // Multiply the errors by the sigmoid prime
        derivative->multiply_o(derivative, self->layers[i]->errors);

        // Transpose the previous layer's output
        transpose = self->layers[i - 1]->outputs->transpose(self->layers[i - 1]->outputs);

        // Dot product 
        self->layers[i]->new_weights = derivative->dot(derivative, transpose);

        // Scale
        self->layers[i]->new_weights->scale_o(self->layers[i]->new_weights, self->learning_rate);

        // Subtract
        self->layers[i]->new_weights->subtract_o(self->layers[i]->new_weights, self->layers[i]->weights);

        // Clean up
        derivative->clear(derivative);
        transpose->clear(transpose);
    }

    // Clean up and move the weights
    for (size_t i = 1; i < self->num_layers; ++i) {

        self->layers[i]->weights->clear(self->layers[i]->weights);
        self->layers[i]->weights = self->layers[i]->new_weights;
        self->layers[i]->new_weights = NULL;

        // Update the biases too, now that we're done with the errors we can scale by learning rate before cleaning up
        self->layers[i]->errors->scale_o(self->layers[i]->errors, self->learning_rate);
        self->layers[i]->biases->subtract_o(self->layers[i]->biases, self->layers[i]->errors);

        self->layers[i]->outputs->clear(self->layers[i]->outputs);
        self->layers[i]->outputs = NULL;

        self->layers[i]->errors->clear(self->layers[i]->errors);
        self->layers[i]->errors = NULL;

        // Clean up the Z vector
        self->layers[i]->z->clear(self->layers[i]->z);
        self->layers[i]->z = NULL;
    }

    output->clear(output);

    // Clean up the special input layer
    if (self->layers[0]->outputs != NULL) {

        self->layers[0]->outputs->clear(self->layers[0]->outputs);
        self->layers[0]->outputs = NULL;
    }
}

void Neural_Network_batch_train(Neural_Network* self, const MNIST_Images* images, const MNIST_Labels* labels, size_t num_train, size_t target_batch_size) {

    if (self == NULL || images == NULL || labels == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_batch_train> Invalid Neural Network, MNIST_Images, or MNIST_Labels provided to batch_train\n");
        exit(EXIT_FAILURE);
    }
    
    // Define variables we're going to use throughout the loops
    FloatMatrix* inputs = NULL;
    FloatMatrix* flat_image = NULL;
    FloatMatrix* expanded_bias = NULL;
    FloatMatrix* derivative = NULL;
    FloatMatrix* transpose = NULL;
    FloatMatrix* outputs = NULL;
    FloatMatrix* ones = NULL;
    FloatMatrix** nabla_w = calloc(self->num_layers, sizeof(FloatMatrix*));
    FloatMatrix** nabla_b = calloc(self->num_layers, sizeof(FloatMatrix*));

    // Initialize values across nabla_w
    for (size_t i = 1; i < self->num_layers; ++i) {

        nabla_w[i] = NULL;
        nabla_b[i] = NULL;
    }

    // Information about the batches
    size_t num_batches = num_train / target_batch_size;
    size_t final_batch_size = num_train - (target_batch_size * num_batches);
    size_t current_index = 0;

    // Create the shuffled index array
    size_t* shuffled = create_index_array(num_train);

    // Iterate over the number of batches, ensuring we catch the last (potentially smaller) batch
    for (size_t i = 0; i <= num_batches; ++i) {

        size_t batch_size = (i == num_batches) ? final_batch_size : target_batch_size;

        // If the final_batch_size is zero, we know the batches divided the number of training images evenly
        if (batch_size == 0) { continue; }

        // Allocate a FloatMatrix that will contain batch_size number of images
        inputs = FloatMatrix_init(MNIST_IMAGE_SIZE, batch_size);
        outputs = FloatMatrix_init(MNIST_LABELS, batch_size);
        outputs->populate(outputs, 0);

        // Set up a Matrix of ones for getting the bias error
        ones = FloatMatrix_init(batch_size, 1);
        ones->populate(ones, 1);

        // Insert the image data into the larger input Matrix
        for (size_t j = 0; j < batch_size; ++j) {

            flat_image = Neural_Network_convert_PixelMatrix(images->get(images, shuffled[current_index + j]));
            flat_image->flatten_o(flat_image, COLUMN);

            for (size_t k = 0; k < flat_image->num_rows; ++k) {

                inputs->set(inputs, k, j, flat_image->get(flat_image, k, 0));
            }

            flat_image->clear(flat_image);
            flat_image = NULL;

            // Set the correct output as 1.0 in the output Matrix
            outputs->set(outputs, labels->get(labels, shuffled[current_index + j]), j, 1);
        }

        // Update the bias vectors to be Matrix, ignoring the first layer of course
        for (size_t j = 1; j < self->num_layers; ++j) {

            expanded_bias = Neural_Network_expand_bias(self->layers[j]->biases, batch_size);

            self->layers[j]->biases->clear(self->layers[j]->biases);
            self->layers[j]->biases = expanded_bias;
        }

        /* Run inference on the Matrix of inputs and store their outputs in each layer */
        Neural_Network_training_predict(self, inputs);

        /* Begin the backpropagation component */
        for (size_t j = self->num_layers - 1; j >= 1; --j) {

            if (j == self->num_layers - 1) {

                // Calculate the error between the proper labels and the predicted outputs
                self->layers[j]->errors = self->delta(outputs, self->layers[j]->outputs);

                // Get the cost derivative, special handling for last layer
                derivative = self->cost_derivative(self->layers[j]->outputs, true);

            }
            else {

                // Get the error as the transpose of the next layer's weights (dot) next layer's error
                transpose = self->layers[j + 1]->weights->transpose(self->layers[j + 1]->weights);
                self->layers[j]->errors = transpose->dot(transpose, self->layers[j + 1]->errors);

                transpose->clear(transpose);

                // Get the cost derivative
                derivative = self->cost_derivative(self->layers[j]->outputs, false);
            }

            // Multiply the errors with the sigmoid prime
            derivative->multiply_o(derivative, self->layers[j]->errors);

            // Transpose the outputs from the last layer
            transpose = self->layers[j - 1]->outputs->transpose(self->layers[j - 1]->outputs);

            // Get the dot product of the transposed outputs and the errors * sigmoid
            nabla_w[j] = derivative->dot(derivative, transpose);

            // Get the sum of the errors for processing later
            nabla_b[j] = self->layers[j]->errors->dot(self->layers[j]->errors, ones);

            derivative->clear(derivative);
            transpose->clear(transpose);
        }

        /* Begin the final calculations for the new weights */

        // Update the weights for each layer
        for (size_t j = 1; j < self->num_layers; ++j) {

            nabla_w[j]->scale_o(nabla_w[j], self->learning_rate / (float)batch_size);
            nabla_b[j]->scale_o(nabla_b[j], self->learning_rate / (float)batch_size);

            // Add the processed changes to the original weights
            //self->layers[j]->weights->subtract_o(self->layers[j]->weights, nabla_w[j]);
            self->layers[j]->weights->scale_o(self->layers[j]->weights, 1.0f - ((self->learning_rate * self->lambda) / (float)num_train));
            self->layers[j]->weights->subtract_o(self->layers[j]->weights, nabla_w[j]);

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
            self->layers[j]->biases->subtract_o(self->layers[j]->biases, nabla_b[j]);

            expanded_bias = NULL;

            nabla_b[j]->clear(nabla_b[j]);
            nabla_b[j] = NULL;

            // Clean up the Z vector
            self->layers[j]->z->clear(self->layers[j]->z);
            self->layers[j]->z = NULL;
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

        ones->clear(ones);
        ones = NULL;
    }

    free(nabla_w);
    free(nabla_b);
    free(shuffled);
}

void Neural_Network_save(const Neural_Network* self, bool include_biases, const char* filename) {

    if (self == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_save> Invalid Neural Network to save\n");
        exit(EXIT_FAILURE);
    }

    FILE* model = fopen(filename, "wb");

    if (model == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_save> Unable to open file %s to save model\n", filename);
        exit(EXIT_FAILURE);
    }

    uint32_t header_magic = NN_HEADER_MAGIC;
    uint32_t weights_magic = NN_WEIGHTS_MAGIC;
    uint32_t weights_begin = NN_WEIGHT_BEGIN;
    uint32_t weights_end = NN_WEIGHT_END;

    // Write the header magic before starting anything else
    fwrite(&header_magic, sizeof(uint32_t), 1, model);

    // Write the learning rate of the model
    fwrite(&(self->learning_rate), sizeof(float), 1, model);

    // Write the lambda used for the model
    fwrite(&(self->lambda), sizeof(float), 1, model);

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

Neural_Network* Neural_Network_import(const char* filename) {

    if (filename == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_import> Invalid char* for filename provided to import\n");
        exit(EXIT_FAILURE);
    }

    // Open the path to the model file
    FILE* model = fopen(filename, "rb");

    if (model == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_import> Unable to read model file\n");
        exit(EXIT_FAILURE);
    }

    uint32_t header_magic = 0;

    // Read in the header and grab the magic number
    if (fread(&header_magic, sizeof(uint32_t), 1, model) != 1) {

        fprintf(stderr, "ERR: <Neural_Network_import> Unable to read header info from model\n");

        fclose(model);
        exit(EXIT_FAILURE);
    }

    if (header_magic != NN_HEADER_MAGIC) {

        fprintf(stderr, "ERR: <Neural_Network_import> Invalid magic number in model header\n");

        if (NEURAL_NETWORK_DEBUG) {

            fprintf(stderr, "DEBUG: <Neural_Network_import> Got magic number %u but expected %u\n", header_magic, NN_HEADER_MAGIC);
        }

        fclose(model);
        exit(EXIT_FAILURE);
    }

    // Get ready to get the learning rate for the model
    float learning_rate = 0;

    if (fread(&learning_rate, sizeof(float), 1, model) != 1) {

        fprintf(stderr, "ERR: <Neural_Network_import> Unable to read learning rate from model\n");

        fclose(model);
        exit(EXIT_FAILURE);
    }

    float lambda = 0;

    if (fread(&lambda, sizeof(float), 1, model) != 1) {

        fprintf(stderr, "ERR: <Neural_Network_import> Unable to read lambda from model\n");

        fclose(model);
        exit(EXIT_FAILURE);
    }

    // Check to see if biases were included in the model, either 0 for no, or 1 for yes
    uint32_t includes_biases = 0;

    if (fread(&includes_biases, sizeof(uint32_t), 1, model) != 1) {

        fprintf(stderr, "ERR: <Neural_Network_import> Unable to read bias info from model\n");

        fclose(model);
        exit(EXIT_FAILURE);
    }

    // Get the number of layers in the model
    size_t num_layers = 0;

    if (fread(&num_layers, sizeof(size_t), 1, model) != 1) {

        fprintf(stderr, "ERR: <Neural_Network_import> Unable to read number of layers from model\n");

        fclose(model);
        exit(EXIT_FAILURE);
    }

    // Allocate an array of size_t to get the size of the layers
    size_t* layer_info = calloc(num_layers, sizeof(size_t));

    if (layer_info == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_import> Unable to allocate memory to read layer info\n");

        fclose(model);
        exit(EXIT_FAILURE);
    }

    if (fread(layer_info, sizeof(size_t), num_layers, model) != num_layers) {

        fprintf(stderr, "ERR: <Neural_Network_import> Unable to read layer info\n");

        fclose(model);
        exit(EXIT_FAILURE);
    }

    // Setup a buffer to grab values from the file without making new variables
    uint32_t buffer = 0;

    if (fread(&buffer, sizeof(uint32_t), 1, model) != 1) {

        fprintf(stderr, "ERR: <Neural_Network_import> Unable to read weight magic number\n");

        fclose(model);
        exit(EXIT_FAILURE);
    }

    if (buffer != NN_WEIGHTS_MAGIC) {

        fprintf(stderr, "ERR: <Neural_Network_import> Invalid weight magic number\n");

        if (NEURAL_NETWORK_DEBUG) {

            fprintf(stderr, "DEBUG: <Neural_Network_import> Got %u but was expecting %u\n", buffer, NN_WEIGHTS_MAGIC);
        }

        fclose(model);
        exit(EXIT_FAILURE);
    }

    // Once we have all the information, create a new Neural Network
    Neural_Network* target = malloc(sizeof(Neural_Network));

    if (target == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_import> Unable to allocate memory for the Neural Network\n");

        fclose(model);
        exit(EXIT_FAILURE);
    }

    // Much of this should look the same as allocate_Neural_Network
    target->num_layers = num_layers;
    target->learning_rate = learning_rate;
    target->lambda = lambda;

    target->predict = Neural_Network_predict;
    target->train = Neural_Network_train;
    target->batch_train = Neural_Network_batch_train;
    target->save = Neural_Network_save;
    target->copy = Neural_Network_copy;
    target->clear = Neural_Network_clear;

    // Set the cost and derivative functions
    target->activation = NEURAL_NETWORK_ACTIVATION;
    target->cost_derivative = NEURAL_NETWORK_COST_DERIVATIVE;
    target->delta = NEURAL_NETWORK_OUTPUT_DELTA;

    // Setup the array of pointers for each layer
    target->layers = calloc(num_layers, sizeof(Neural_Network_Layer*));

    if (target->layers == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_import> Unable to allocate memory for layers\n");

        fclose(model);
        exit(EXIT_FAILURE);
    }

    // Special processing for the input layer, since it has no weights or biases
    target->layers[0] = Neural_Network_Layer_init(layer_info[0], 0, false, true);

    for (size_t i = 1; i < num_layers; ++i) {

        // Allocate each normal layer
        target->layers[i] = Neural_Network_Layer_init(layer_info[i], layer_info[i - 1], (bool)includes_biases, true);
    }

    FloatMatrix* current_weights = NULL;
    float float_buffer = 0;

    // Iterate over the number of layers, processing each of their weight matrices
    for (size_t i = 1; i < num_layers; ++i) {

        if (fread(&buffer, sizeof(uint32_t), 1, model) != 1) {

            fprintf(stderr, "ERR: <Neural_Network_import> Error reading model layer %zu\n", i);

            fclose(model);
            exit(EXIT_FAILURE);
        }

        if (buffer != NN_WEIGHT_BEGIN) {

            fprintf(stderr, "ERR: <Neural_Network_import> Invalid start weights value for layer %zu\n", i);

            if (NEURAL_NETWORK_DEBUG) {

                fprintf(stderr, "DEBUG: Got %u but was expecting %u\n", buffer, NN_WEIGHT_BEGIN);
            }

            fclose(model);
            exit(EXIT_FAILURE);
        }

        // Allocate a new FloatMatrix for the weights of this layer
        current_weights = FloatMatrix_init(layer_info[i], layer_info[i - 1]);

        for (size_t j = 0; j < current_weights->num_rows; ++j) {

            for (size_t k = 0; k < current_weights->num_cols; ++k) {

                if (fread(&float_buffer, sizeof(float), 1, model) != 1) {

                    fprintf(stderr, "ERR: <Neural_Network_import> Error reading (%zu, %zu) for model layer %zu\n", j, k, i);

                    fclose(model);
                    exit(EXIT_FAILURE);
                }

                // Copy the float value of the weight to the new FloatMatrix
                current_weights->set(current_weights, j, k, float_buffer);
            }
        }

        if (fread(&buffer, sizeof(uint32_t), 1, model) != 1) {

            fprintf(stderr, "ERR: <Neural_Network_import> Error reading end of model layer %zu\n", i);

            fclose(model);
            exit(EXIT_FAILURE);
        }

        if (buffer != NN_WEIGHT_END) {

            fprintf(stderr, "ERR: <Neural_Network_import> Invalid end weights value for layer %zu\n", i);

            if (NEURAL_NETWORK_DEBUG) {

                fprintf(stderr, "DEBUG: <Neural_Network_import> Got %u but was expecting %u\n", buffer, NN_WEIGHT_END);
            }

            fclose(model);
            exit(EXIT_FAILURE);
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

        fprintf(stderr, "ERR: <Neural_Network_import> Unable to read bias magic number\n");

        fclose(model);
        exit(EXIT_FAILURE);
    }

    if (buffer != NN_BIASES_MAGIC) {

        fprintf(stderr, "ERR: <Neural_Network_import> Invalid bias magic number\n");

        if (NEURAL_NETWORK_DEBUG) {

            fprintf(stderr, "DEBUG: <Neural_Network_import> Got %u but was expecting %u\n", buffer, NN_BIASES_MAGIC);
        }

        fclose(model);
        exit(EXIT_FAILURE);
    }

    // Store the layer's biases somewhere
    FloatMatrix* current_biases = NULL;

    for (size_t i = 1; i < num_layers; ++i) {

        if (fread(&buffer, sizeof(uint32_t), 1, model) != 1) {

            fprintf(stderr, "ERR: <Neural_Network_import> Error reading bias for layer %zu\n", i);

            fclose(model);
            exit(EXIT_FAILURE);
        }

        if (buffer != NN_BIAS_BEGIN) {

            fprintf(stderr, "ERR: <Neural_Network_import> Invalid start bias value for layer %zu\n", i);

            if (NEURAL_NETWORK_DEBUG) {

                fprintf(stderr, "DEBUG: <Neural_Network_import> Got %u but was expecting %u\n", buffer, NN_BIAS_BEGIN);
            }

            fclose(model);
            exit(EXIT_FAILURE);
        }

        current_biases = FloatMatrix_init(layer_info[i], 1);

        // Only one nested loop since the biases are stored in a vector (1-column matrix)
        for (size_t j = 0; j < current_biases->num_rows; ++j) {

            if (fread(&float_buffer, sizeof(float), 1, model) != 1) {

                fprintf(stderr, "ERR: <Neural_Network_import> Error reading (%zu, 0) for bias layer %zu\n", j, i);

                fclose(model);
                exit(EXIT_FAILURE);
            }

            current_biases->set(current_biases, j, 0, float_buffer);
        }

        if (fread(&buffer, sizeof(uint32_t), 1, model) != 1) {

            fprintf(stderr, "ERR: <Neural_Network_import> Error reading end of bias layer %zu\n", i);

            fclose(model);
            exit(EXIT_FAILURE);
        }

        if (buffer != NN_BIAS_END) {

            fprintf(stderr, "ERR: <Neural_Network_import> Invalid end biases value for layer %zu\n", i);

            if (NEURAL_NETWORK_DEBUG) {

                fprintf(stderr, "DEBUG: <Neural_Network_import> Got %u but was expecting %u\n", buffer, NN_BIAS_END);
            }

            fclose(model);
            exit(EXIT_FAILURE);
        }

        target->layers[i]->biases = current_biases;
        current_biases = NULL;
    }

    fclose(model);
    free(layer_info);
    return target;
}

Neural_Network* Neural_Network_init(size_t num_layers, const size_t* layer_info, float learning_rate, bool generate_biases, float lambda) {

    Neural_Network* target = malloc(sizeof(Neural_Network));

    if (target == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_init> Unable to allocate memory for new Neural_Network\n");
        exit(EXIT_FAILURE);
    }

    target->num_layers = num_layers;
    target->learning_rate = learning_rate;
    target->lambda = lambda;

    target->layers = calloc(num_layers, sizeof(Neural_Network_Layer*));

    target->layers[0] = Neural_Network_Layer_init(layer_info[0], 0, generate_biases, false);

    for (size_t i = 1; i < num_layers; ++i) {

        target->layers[i] = Neural_Network_Layer_init(layer_info[i], layer_info[i - 1], generate_biases, false);
    }

    target->predict = Neural_Network_predict;
    target->train = Neural_Network_train;
    target->batch_train = Neural_Network_batch_train;
    target->save = Neural_Network_save;
    target->copy = Neural_Network_copy;
    target->clear = Neural_Network_clear;

    // Set the cost and derivative functions
    target->activation = NEURAL_NETWORK_ACTIVATION;
    target->cost_derivative = NEURAL_NETWORK_COST_DERIVATIVE;
    target->delta = NEURAL_NETWORK_OUTPUT_DELTA;

    return target;
}

Neural_Network* Neural_Network_copy(const Neural_Network* self) {
    
    if (self == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_copy> Invalid Neural Network provided to copy\n");
        exit(EXIT_FAILURE);
    }

    Neural_Network* target = malloc(sizeof(Neural_Network));

    if (target == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_copy> Unable to allocate memory for Neural Network copy\n");
        exit(EXIT_FAILURE);
    }

    target->num_layers = self->num_layers;
    target->learning_rate = self->learning_rate;
    target->lambda = self->lambda;
    
    target->layers = calloc(target->num_layers, sizeof(Neural_Network_Layer*));

    if (target->layers == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_copy> Unable to allocate memory for Neural Network Layer array in copy\n");

        free(target);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < target->num_layers; ++i) {

        target->layers[i] = self->layers[i]->copy(self->layers[i]);
    }

    target->predict = Neural_Network_predict;
    target->train = Neural_Network_train;
    target->batch_train = Neural_Network_batch_train;
    target->save = Neural_Network_save;
    target->copy = Neural_Network_copy;
    target->clear = Neural_Network_clear;

    // Set the cost and derivative functions
    target->activation = NEURAL_NETWORK_ACTIVATION;
    target->cost_derivative = NEURAL_NETWORK_COST_DERIVATIVE;
    target->delta = NEURAL_NETWORK_OUTPUT_DELTA;

    return target;
}

FloatMatrix* Neural_Network_expand_bias(const FloatMatrix* current_bias, size_t batch_size) {

    if (current_bias == NULL) {

        fprintf(stderr, "ERR: <Neural_Network_expand_bias> Invalid bias Matrix provided to expand_bias\n");
        exit(EXIT_FAILURE);
    }

    FloatMatrix* expanded = FloatMatrix_init(current_bias->num_rows, batch_size);
    float bias_value = 0;

    for (size_t i = 0; i < expanded->num_rows; ++i) {

        bias_value = current_bias->get(current_bias, i, 0);

        for (size_t j = 0; j < expanded->num_cols; ++j) {

            expanded->set(expanded, i, j, bias_value);
        }
    }

    return expanded;
}
