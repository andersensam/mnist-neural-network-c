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

#include "include/inference.h"

void inference(const char* labels_path, const char* images_path, const char* model_path, size_t num_predict) {

    // Load image dataset and associated labels
    log_message("Starting to load MNIST labels");
    MNIST_Labels* labels = MNIST_Labels_init(labels_path);
    log_message("Finished loading MNIST labels");

    log_message("Starting to load MNIST images");
    MNIST_Images* images = MNIST_Images_init(images_path);
    log_message("Finished loading MNIST images");

    if (num_predict > images->num_images) {

        fprintf(stderr, "ERR: <inference> num_predict cannot be greater than number of images in dataset\n");
        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    if (labels->num_labels != images->num_images) {

        fprintf(stderr, "ERR: <inference> Number of labels and images must match. Num labels: %u. Num images: %u\n", labels->num_labels, images->num_images);
        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    log_message("Starting to load model from file");
    Neural_Network* nn = Neural_Network_import(model_path);
    log_message("Finished loading model from file");

    // Setup some variables that we'll use again and again in the loop
    PixelMatrix* current_image = NULL;
    uint8_t current_label = 0;
    FloatMatrix* current_prediction = NULL;
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
    MNIST_Labels* labels = MNIST_Labels_init(labels_path);
    log_message("Finished loading MNIST labels");

    log_message("Starting to load MNIST images");
    MNIST_Images* images = MNIST_Images_init(images_path);
    log_message("Finished loading MNIST images");

    if (num_predict > images->num_images) {

        fprintf(stderr, "ERR: <threaded_inference> num_predict cannot be greater than number of images in dataset\n");
        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    if (labels->num_labels != images->num_images) {

        fprintf(stderr, "ERR: <threaded_inference> Number of labels and images must match. Num labels: %u. Num images: %u\n", labels->num_labels, images->num_images);
        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    log_message("Starting to load model from file");
    Neural_Network* nn = Neural_Network_import(model_path);
    log_message("Finished loading model from file");

    // Define the threads we'll use (configure in main.h)
    pthread_t* thread_ids = calloc(INFERENCE_MAX_THREADS, sizeof(pthread_t));

    // Setup the threading info
    Threaded_Inference_Result** thread_results = calloc(INFERENCE_MAX_THREADS, sizeof(Threaded_Inference_Result*));

    // Setup the 'batch' size for the threads
    size_t images_per_thread = num_predict / INFERENCE_MAX_THREADS;
    size_t final_thread_images = num_predict - (images_per_thread * (INFERENCE_MAX_THREADS - 1));

    log_message("Starting threaded inference");

    for (size_t i = 0; i < INFERENCE_MAX_THREADS; ++i) {

        if (i == 0) {

            thread_results[i] = Threaded_Inference_Result_init(nn, images, 0, images_per_thread);
        }
        else if (i == INFERENCE_MAX_THREADS - 1) {

            thread_results[i] = Threaded_Inference_Result_init(nn, images, thread_results[i - 1]->image_start_index + images_per_thread, final_thread_images);
        }
        else {

            thread_results[i] = Threaded_Inference_Result_init(nn, images, thread_results[i - 1]->image_start_index + images_per_thread, images_per_thread);
        }

        pthread_create(&(thread_ids[i]), NULL, Neural_Network_Threading_predict, (void*)(thread_results[i]));
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
    }

    free(thread_results);
    free(thread_ids);

    labels->clear(labels);
    images->clear(images);
    nn->clear(nn);
}
