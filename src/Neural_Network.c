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
 * @version: 2024-09-18
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
    floatMatrix* add_bias = NULL;
    floatMatrix* layer_output = NULL;
    floatMatrix* normalized = NULL;

    for (size_t i = 1; i < self->num_layers; ++i) {

        // Special processing for the first hidden layer
        if (i == 1) {

            layer_input = self->layers[i]->weights->dot(self->layers[i]->weights, flat_image);
        }
        else {

            layer_input = self->layers[i]->weights->dot(self->layers[i]->weights, 
                self->layers[i - 1]->outputs);
        }

        add_bias = layer_input->add(layer_input, self->layers[i]->biases);

        // We currently have layer normalization disabled, but can be turned on by uncommenting the line below
        //normalized = Neural_Network_Layer_normalize_layer(add_bias);

        // Since normalization is disabled, just copy the Matrix
        normalized = add_bias->copy(add_bias);

        // Apply the sigmoid actication function
        layer_output = normalized->apply(normalized, sigmoid);

        // Copy the result of the sigmoid to the output of  the layer
        self->layers[i]->outputs = layer_output->copy(layer_output);

        add_bias->clear(add_bias);
        layer_input->clear(layer_input);
        layer_output->clear(layer_output);
        normalized->clear(normalized);
    }

    floatMatrix* final_output = Neural_Network_softmax(self->layers[self->num_layers - 1]->outputs);

    for (size_t i = 1; i < self->num_layers; ++i) {

        self->layers[i]->outputs->clear(self->layers[i]->outputs);
        self->layers[i]->outputs = NULL;
    }

    flat_image->clear(flat_image);

    return final_output;
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
    floatMatrix* subtracted = one->subtract(one, target);
    one->clear(one);

    // Multiple the subtracted Matrix with the original
    floatMatrix* multiplied = target->multiply(target, subtracted);
    subtracted->clear(subtracted);

    return multiplied;
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

void Neural_Network_train(Neural_Network* self, const pixelMatrix* image, uint8_t label) {

    if (self == NULL || image == NULL) {

        if (NEURAL_NETWORK_DEBUG) { fprintf(stderr, "ERR: Invalid Neural_Network or pixelMatrix provided to train\n"); }
    }

    // Convert the image to a floatMatrix
    floatMatrix* converted_image = Neural_Network_convert_pixelMatrix(image);

    // Flatten the floatMatrix containing the image data, converting to a column vector
    floatMatrix* flat_image = converted_image->flatten(converted_image, COLUMN);
    converted_image->clear(converted_image);

    // Store the input layer's output (aka the flattened image)
    self->layers[0]->outputs = flat_image->copy(flat_image);

    /* Begin the feed forward part (same as inference without softmax at the end) */
    floatMatrix* hidden_inputs = NULL;
    floatMatrix* hidden_outputs = NULL;
    floatMatrix* add_bias = NULL;
    floatMatrix* normalized = NULL;

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
        add_bias = hidden_inputs->add(hidden_inputs, self->layers[i]->biases);

        //normalized = Neural_Network_Layer_normalize_layer(add_bias);
        normalized = add_bias->copy(add_bias);

        // The output of this layer is the input with the sigmoid applied
        hidden_outputs = normalized->apply(normalized, sigmoid);

        // Copy the sigmoid outputs and store it in the layer for use later
        self->layers[i]->outputs = hidden_outputs->copy(hidden_outputs);

        hidden_outputs->clear(hidden_outputs);
        hidden_inputs->clear(hidden_inputs);
        add_bias->clear(add_bias);
        normalized->clear(normalized);
    }

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
    flat_image->clear(flat_image);

    // Clean up the special input layer
    if (self->layers[0]->outputs != NULL) {

        self->layers[0]->outputs->clear(self->layers[0]->outputs);
        self->layers[0]->outputs = NULL;
    }
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
    target->train = Neural_Network_train;
    target->save = Neural_Network_save;
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
    target->train = Neural_Network_train;
    target->save = Neural_Network_save;
    target->clear = Neural_Network_clear;

    return target;
}