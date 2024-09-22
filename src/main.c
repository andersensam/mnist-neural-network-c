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

#include "include/main.h"

void train_new_model(const char* labels_path, const char* images_path, size_t num_layers, const size_t* nn_config,
    float learning_rate, bool generate_biases, const char* model_path, size_t num_training_images) {

    // Load image dataset and associated labels
    log_message("Starting to load MNIST labels");
    MNIST_Labels* labels = init_MNIST_labels(labels_path);
    log_message("Finished loading MNIST labels");

    log_message("Starting to load MNIST images");
    MNIST_Images* images = init_MNIST_images(images_path);
    log_message("Finished loading MNIST images");

    if (num_training_images > images->num_images) {

        fprintf(stderr, "ERR: num_training_images cannot be greater than number of images in dataset\n");
        images->clear(images);
        labels->clear(labels);

        return;
    }

    if (labels->num_labels != images->num_images) {

        fprintf(stderr, "ERR: Number of labels and images must match. Num labels: %u. Num images: %u\n", labels->num_labels, images->num_images);
        images->clear(images);
        labels->clear(labels);

        return;
    }

    // Initalize the Neural Network
    Neural_Network* nn = init_Neural_Network(num_layers, nn_config, learning_rate, generate_biases);

    // Setup some variables that we'll use again and again in the loop
    pixelMatrix* current_image = NULL;
    uint8_t current_label = 0;

    // Define where to start in the training dataset and how many samples to train on
    size_t training_start_idx = 0;

    log_message("Starting model training");

    // Set i to the range of images + labels we want to train on
    for (size_t i = training_start_idx; i < training_start_idx + num_training_images; ++i) {

        current_image = images->get(images, i);
        current_label = labels->get(labels, i);

        // Execute the training on the current image + label
        nn->train(nn, current_image, current_label);
    }

    log_message("Finished model training");

    log_message("Saving model");
    nn->save(nn, true, model_path);
    log_message("Finished saving model");

    labels->clear(labels);
    images->clear(images);
    nn->clear(nn);

    return;
}

void inference(const char* labels_path, const char* images_path, const char* model_path, size_t num_predict) {

    // Load image dataset and associated labels
    log_message("Starting to load MNIST labels");
    MNIST_Labels* labels = init_MNIST_labels(labels_path);
    log_message("Finished loading MNIST labels");

    log_message("Starting to load MNIST images");
    MNIST_Images* images = init_MNIST_images(images_path);
    log_message("Finished loading MNIST images");

    if (num_predict > images->num_images) {

        fprintf(stderr, "ERR: num_predict cannot be greater than number of images in dataset\n");
        images->clear(images);
        labels->clear(labels);

        return;
    }

    if (labels->num_labels != images->num_images) {

        fprintf(stderr, "ERR: Number of labels and images must match. Num labels: %u. Num images: %u\n", labels->num_labels, images->num_images);
        images->clear(images);
        labels->clear(labels);

        return;
    }

    log_message("Starting to load model from file");
    Neural_Network* nn = import_Neural_Network(model_path);
    log_message("Finished loading model from file");

    // Setup some variables that we'll use again and again in the loop
    pixelMatrix* current_image = NULL;
    uint8_t current_label = 0;
    floatMatrix* current_prediction = NULL;
    size_t current_guess = 0;
    size_t number_correct = 0;
    size_t start_index = 0;

    log_message("Starting inference");

    for (size_t i = start_index; i < start_index + num_predict; ++i) {

        current_image = images->get(images, i);
        current_label = labels->get(labels, i);

        // Run inference on the trained model
        current_prediction = nn->predict(nn, current_image);

        // Extract the prediction
        current_guess = current_prediction->max_idx(current_prediction, COLUMN, 0);

        if (current_guess == current_label) { 
            
            ++number_correct;
        }

        current_prediction->clear(current_prediction);
    }

    log_message("Finished inference");

    printf("\nStatistics:\nModel path: %s\nImages predicted: %zu\nImages predicted correctly: %zu\nPercentage correct: %3.5f%c\n",
        model_path, num_predict, number_correct, ((float)number_correct / (float)num_predict) * 100.0f, '%');

    labels->clear(labels);
    images->clear(images);
    nn->clear(nn);

    return;
}

void threaded_inference(const char* labels_path, const char* images_path, const char* model_path, size_t num_predict) {

    // Load image dataset and associated labels
    log_message("Starting to load MNIST labels");
    MNIST_Labels* labels = init_MNIST_labels(labels_path);
    log_message("Finished loading MNIST labels");

    log_message("Starting to load MNIST images");
    MNIST_Images* images = init_MNIST_images(images_path);
    log_message("Finished loading MNIST images");

    if (num_predict > images->num_images) {

        fprintf(stderr, "ERR: num_predict cannot be greater than number of images in dataset\n");
        images->clear(images);
        labels->clear(labels);

        return;
    }

    if (labels->num_labels != images->num_images) {

        fprintf(stderr, "ERR: Number of labels and images must match. Num labels: %u. Num images: %u\n", labels->num_labels, images->num_images);
        images->clear(images);
        labels->clear(labels);

        return;
    }

    log_message("Starting to load model from file");
    Neural_Network* nn = import_Neural_Network(model_path);
    log_message("Finished loading model from file");

    // Define the threads we'll use (configure in main.h)
    pthread_t* thread_ids = calloc(INFERENCE_MAX_THREADS, sizeof(pthread_t));

    // Setup the threading info
    Threaded_Inference_Result** thread_results = calloc(INFERENCE_MAX_THREADS, sizeof(Threaded_Inference_Result*));
    Neural_Network** thread_nn = calloc(INFERENCE_MAX_THREADS, sizeof(Neural_Network*));

    for (size_t i = 0; i < INFERENCE_MAX_THREADS; ++i) {

        if (i == 0) {

            thread_nn[i] = nn;
        }
        else {

            // Make copies of the Neural Network to avoid memory access issues
            thread_nn[i] = nn->copy(nn);
        }
    }

    // Setup the 'batch' size for the threads
    size_t images_per_thread = num_predict / INFERENCE_MAX_THREADS;
    size_t final_thread_images = num_predict - (images_per_thread * (INFERENCE_MAX_THREADS - 1));

    log_message("Starting threaded inference");

    for (size_t i = 0; i < INFERENCE_MAX_THREADS; ++i) {

        if (i == 0) {

            thread_results[i] = init_Threaded_Inference_Result(thread_nn[i], images, 0, images_per_thread);
        }
        else if (i == INFERENCE_MAX_THREADS - 1) {

            thread_results[i] = init_Threaded_Inference_Result(thread_nn[i], images, thread_results[i - 1]->image_start_index + images_per_thread, final_thread_images);
        }
        else {

            thread_results[i] = init_Threaded_Inference_Result(thread_nn[i], images, thread_results[i - 1]->image_start_index + images_per_thread, images_per_thread);
        }

        pthread_create(&(thread_ids[i]), NULL, thread_nn[i]->threaded_predict, (void*)(thread_results[i]));
    }

    for (size_t i = 0; i < INFERENCE_MAX_THREADS; ++i) {

        pthread_join(thread_ids[i], NULL);
    }

    size_t number_correct = 0;
    size_t current_guess = 0;
    uint32_t current_label = 0;

    for (size_t i = 0; i < INFERENCE_MAX_THREADS; ++i) {

        for (size_t j = 0; j < thread_results[i]->num_images; ++j) {

            current_label = labels->get(labels, thread_results[i]->image_start_index + j);

            current_guess = thread_results[i]->results->max_idx(thread_results[i]->results, COLUMN, j);

            if (current_guess == current_label) {

                ++number_correct;
            }
        }
    }

    log_message("Finished threaded inference");

    printf("\nStatistics:\nModel path: %s\nImages predicted: %zu\nImages predicted correctly: %zu\nPercentage correct: %3.5f%c\n",
        model_path, num_predict, number_correct, ((float)number_correct / (float)num_predict) * 100.0f, '%');

    // Clean up
    for (size_t i = 0; i < INFERENCE_MAX_THREADS; ++i) {

        thread_results[i]->clear(thread_results[i]);
        thread_nn[i]->clear(thread_nn[i]);
    }

    free(thread_results);
    free(thread_nn);
    free(thread_ids);

    labels->clear(labels);
    images->clear(images);
}

int main(int argc, char* argv[]) {

    if (argc <= 1) {

        fprintf(stderr, "Too few arguments provided. Use main help to get a help menu. Exiting\n");
        exit(-1);
    }

    // Ensure this is only executed once
    srand(time(0));

    if (strncmp(argv[1], "train", 5) == 0) {

        /*
         * For training, expect: labels_path (2), images_path (3), learning_rate (4), include_biases (5), 
         * num_layers (6), [layer_info] (7..n-2), model_name (n-1), num_images (n)
         */

        if (argc <= 9) {

            fprintf(stderr, "ERR: Too few arguments provided to train\n");
            return -1;
        }

        float learning_rate = 0;
        sscanf(argv[4], "%f", &learning_rate);

        bool generate_biases = false;

        if (strncmp(argv[5], "true", 4) == 0) { generate_biases = true; }

        size_t num_layers = (size_t)(atoi(argv[6]));
        size_t* nn_config = calloc(num_layers, sizeof(size_t));

        // Iterate over the number of layers and get the neuron counts for each
        for (size_t i = 0; i < num_layers; ++i) {

            nn_config[i] = (size_t)atoi(argv[i + 7]);
        }

        // Train a new model with the given parameters
        train_new_model(argv[2], argv[3], num_layers, nn_config, learning_rate, generate_biases, argv[argc - 1], (size_t)atoi(argv[argc - 2]));

        free(nn_config);
        return 0;
    }
    else if (strncmp(argv[1], "predict", 7) == 0) {

        if (argc < 6) {

            fprintf(stderr, "ERR: Too few arguments provided to predict\n");
            return -1;
        }

        // Execute inference and exit
        inference(argv[2], argv[3], argv[5], (size_t)atoi(argv[4]));
        return 0;
    }
    else if (strncmp(argv[1], "threaded-predict", 16) == 0) {

        if (argc < 6) {

            fprintf(stderr, "ERR: Too few arguments provided to use threaded-predict\n");
            return -1;
        }

        // Execute inference and exit
        threaded_inference(argv[2], argv[3], argv[5], (size_t)atoi(argv[4]));
        return 0;
    }
    else if (strncmp(argv[1], "help", 4) == 0) {

        printf("Expected usage: main train labels_path images_path learning_rate include_biases num_layers [layer_info] num_training_images model_name\n");
        printf("Example for train: main train data/labels data/images 0.1 true 3 786 100 10 1000 model.model\n\n");
        printf("This example has a learning rate of 0.1, uses biases, has 3 layers, and uses 1000 images to train on\n");

        printf("Expected usage: main predict labels_path images_path num_predict model_path\n");
        printf("Example for predict: main predict data/labels data/images 100 model.model\n");
        printf("This example predicts 100 images");

        return 0;
    }

    fprintf(stderr, "ERR: Invalid argument passed to main. Exiting\n");
    return -1;
}
