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

#include "include/training.h"

void train_new_model(const char* labels_path, const char* images_path, size_t num_layers, const size_t* nn_config,
    float learning_rate, bool generate_biases, size_t num_training_images, size_t epochs, const char* model_path) {

    // Load image dataset and associated labels
    log_message("Starting to load MNIST labels");
    MNIST_Labels* labels = MNIST_Labels_init(labels_path);
    log_message("Finished loading MNIST labels");

    log_message("Starting to load MNIST images");
    MNIST_Images* images = MNIST_Images_init(images_path);
    log_message("Finished loading MNIST images");

    if (num_training_images > images->num_images) {

        fprintf(stderr, "ERR: <train_new_model> num_training_images cannot be greater than number of images in dataset\n");
        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    if (labels->num_labels != images->num_images) {

        fprintf(stderr, "ERR: <train_new_model> Number of labels and images must match. Num labels: %u. Num images: %u\n", labels->num_labels, images->num_images);
        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    if (nn_config[num_layers - 1] != MNIST_LABELS) {

        fprintf(stderr, "ERR: <train_new_model> Output layer does not match number of labels. Got %zu and expected %d\n",
            nn_config[num_layers - 1], MNIST_LABELS);

        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    // Initalize the Neural Network -- Lambda set to 0 for SGD
    Neural_Network* nn = Neural_Network_init(num_layers, nn_config, learning_rate, generate_biases, 0);

    // Setup some variables that we'll use again and again in the loop
    PixelMatrix* current_image = NULL;
    uint8_t current_label = 0;

    // Create the shuffled index array
    size_t* shuffled = create_index_array(num_training_images);

    // Use a buffer for snprintf
    char buffer[100];
    memset(buffer, '\0', 100);

    log_message("Starting model online training");

    // Execute the training over the number of epochs
    for (size_t i = 0; i < epochs; ++i) {

        if (SHOW_EPOCH) {

            snprintf(buffer, 100, "INFO: Starting online training epoch %zu", i + 1);
            log_message(buffer);
        }

        // Set i to the range of images + labels we want to train on
        for (size_t j = 0; j < num_training_images; ++j) {

            current_image = images->get(images, shuffled[j]);
            current_label = labels->get(labels, shuffled[j]);

            // Execute the training on the current image + label
            nn->train(nn, current_image, current_label);
        }

        // Shuffle the index array again at the end of each epoch
        shuffle(shuffled, num_training_images);
    }

    log_message("Finished model online training");

    log_message("Saving model");
    nn->save(nn, true, model_path);
    log_message("Finished saving model");

    labels->clear(labels);
    images->clear(images);
    nn->clear(nn);

    free(shuffled);

    return;
}

void train_new_model_batched(const char* labels_path, const char* images_path, size_t num_layers, const size_t* nn_config,
    float learning_rate, float lambda, bool generate_biases, size_t num_training_images, size_t batch_size, size_t epochs, const char* model_path) {

    // Load image dataset and associated labels
    log_message("Starting to load MNIST labels");
    MNIST_Labels* labels = MNIST_Labels_init(labels_path);
    log_message("Finished loading MNIST labels");

    log_message("Starting to load MNIST images");
    MNIST_Images* images = MNIST_Images_init(images_path);
    log_message("Finished loading MNIST images");

    if (num_training_images > images->num_images) {

        fprintf(stderr, "ERR: <train_new_model_batched> num_training_images cannot be greater than number of images in dataset\n");
        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    if (labels->num_labels != images->num_images) {

        fprintf(stderr, "ERR: <train_new_model_batched> Number of labels and images must match. Num labels: %u. Num images: %u\n", labels->num_labels, images->num_images);
        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    if (nn_config[num_layers - 1] != MNIST_LABELS) {

        fprintf(stderr, "ERR: <train_new_model> Output layer does not match number of labels. Got %zu and expected %d\n",
            nn_config[num_layers - 1], MNIST_LABELS);

        images->clear(images);
        labels->clear(labels);
        
        exit(EXIT_FAILURE);
    }

    if (batch_size > num_training_images) {

        fprintf(stderr, "ERR: <train_new_model_batched> Batch size cannot exceed the number of training images\n");

        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    // Initalize the Neural Network
    Neural_Network* nn = Neural_Network_init(num_layers, nn_config, learning_rate, generate_biases, lambda);

    // Use a buffer for snprintf
    char buffer[100];
    memset(buffer, '\0', 100);

    log_message("Starting batch model training");

    for (size_t i = 0; i < epochs; ++i) {

        if (SHOW_EPOCH) {

            snprintf(buffer, 100, "INFO: Starting batch training epoch %zu", i + 1);
            log_message(buffer);
        }

        nn->batch_train(nn, images, labels, num_training_images, batch_size);
    }

    log_message("Finished batch model training");

    log_message("Saving model");
    nn->save(nn, true, model_path);
    log_message("Finished saving model");

    nn->clear(nn);
    labels->clear(labels);
    images->clear(images);

    return;
}

void train_new_model_batched_threaded(const char* labels_path, const char* images_path, size_t num_layers, const size_t* nn_config,
    float learning_rate, float lambda, bool generate_biases, size_t num_training_images, size_t batch_size, size_t epochs, const char* model_path) {

    // Load image dataset and associated labels
    log_message("Starting to load MNIST labels");
    MNIST_Labels* labels = MNIST_Labels_init(labels_path);
    log_message("Finished loading MNIST labels");

    log_message("Starting to load MNIST images");
    MNIST_Images* images = MNIST_Images_init(images_path);
    log_message("Finished loading MNIST images");

    if (num_training_images > images->num_images) {

        fprintf(stderr, "ERR: <train_new_model_batch_threaded> num_training_images cannot be greater than number of images in dataset\n");
        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    if (labels->num_labels != images->num_images) {

        fprintf(stderr, "ERR: <train_new_model_batch_threaded> Number of labels and images must match. Num labels: %u. Num images: %u\n", labels->num_labels, images->num_images);
        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    if (nn_config[num_layers - 1] != MNIST_LABELS) {

        fprintf(stderr, "ERR: <train_new_model_batch_threaded> Output layer does not match number of labels. Got %zu and expected %d\n",
            nn_config[num_layers - 1], MNIST_LABELS);

        images->clear(images);
        labels->clear(labels);
        
        exit(EXIT_FAILURE);
    }

    if (batch_size > num_training_images) {

        fprintf(stderr, "ERR: <train_new_model_batch_threaded> Batch size cannot exceed the number of training images\n");

        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    if (epochs < TRAINING_MAX_THREADS) {

        fprintf(stderr, "ERR: <train_new_model_batch_threaded> Number of epochs must be >= TRAINING_MAX_THREADS\n");

        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    if (epochs < TRAINING_MAX_THREADS * THREAD_EPOCHS_BEFORE_COMBINE) {

        fprintf(stderr, "ERR: <train_new_model_batch_threaded> Number of epochs must be >= TRAINING_MAX_EPOCHS * THREAD_EPOCHS_BEFORE_COMBINE\n");

        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    // Initalize the Neural Network
    Neural_Network* nn = Neural_Network_init(num_layers, nn_config, learning_rate, generate_biases, lambda);

    // Create an array of Neural Networks for threading
    Neural_Network** networks = calloc(TRAINING_MAX_THREADS, sizeof(Neural_Network*));

    if (networks == NULL) {

        fprintf(stderr, "ERR: <train_new_model_batch_threaded> Unable to allocate memory for Networks\n");
        exit(EXIT_FAILURE);
    }

    // Set the first index as the original Neural Network
    networks[0] = nn;

    // Copy the Neural Network for training
    for (size_t i = 1; i < TRAINING_MAX_THREADS; ++i) {

        networks[i] = nn->copy(nn);
    }

     // Define the threads we'll use (configure in main.h)
    pthread_t* thread_ids = calloc(TRAINING_MAX_THREADS, sizeof(pthread_t));

    if (thread_ids == NULL) {

        fprintf(stderr, "ERR: <train_new_model_batch_threaded> Unable to allocate memory for threads\n");
        exit(EXIT_FAILURE);
    }

    // Setup the threading info
    Threaded_Training** training = calloc(TRAINING_MAX_THREADS, sizeof(Threaded_Training*));

    if (training == NULL) {

        fprintf(stderr, "ERR: <train_new_model_bath_threaded> Unable to allocate memory for threading data\n");
        exit(EXIT_FAILURE);
    }

    // Use a buffer for snprintf
    char buffer[100];
    memset(buffer, '\0', 100);

    log_message("Starting threaded batch model training");

    for (size_t i = 0; i < epochs; i += (TRAINING_MAX_THREADS * THREAD_EPOCHS_BEFORE_COMBINE)) {

        // Handle the case where we don't have enough remaining epochs to spread evenly across threads
        if (i + (TRAINING_MAX_THREADS * THREAD_EPOCHS_BEFORE_COMBINE) > epochs) {

            // Essentially revert back to the regular batch training method to finish up
            for (size_t j = i; j < epochs; ++j) {

                if (SHOW_EPOCH) {

                    snprintf(buffer, 100, "INFO: Starting threaded batch training epoch %zu", j + 1);
                    log_message(buffer);
                }
                
                nn->batch_train(nn, images, labels, num_training_images, batch_size);
            }
        }
        else {

            if (SHOW_EPOCH) {

                snprintf(buffer, 100, "INFO: Starting threaded batch training epochs %zu - %zu", i + 1, i + (TRAINING_MAX_THREADS * THREAD_EPOCHS_BEFORE_COMBINE));
                log_message(buffer);

                memset(buffer, '\0', 100);
            }

            for (size_t j = 0; j < TRAINING_MAX_THREADS; ++j) {

                // Create the information for training
                training[j] = Threaded_Training_init(networks[j], images, labels, num_training_images, batch_size,
                    THREAD_EPOCHS_BEFORE_COMBINE, j);

                // Spawn the thread and begin training
                pthread_create(&(thread_ids[j]), NULL, Neural_Network_Threading_train, (void*)(training[j]));
            }

            // Join the threads back after training for the desired number of epochs
            for (size_t j = 0; j < TRAINING_MAX_THREADS; ++j) {

                pthread_join(thread_ids[j], NULL);
                free(training[j]);
            }

            // Combine the Neural Networks back together and copy their weights / biases
            Neural_Network_Threading_combine(networks, TRAINING_MAX_THREADS);
        }
    }

    log_message("Finished threaded batch model training");

    for (size_t i = 1; i < TRAINING_MAX_THREADS; ++i) {

        networks[i]->clear(networks[i]);
    }

    free(networks);
    free(thread_ids);
    free(training);

    log_message("Saving model");
    nn->save(nn, true, model_path);
    log_message("Finished saving model");

    nn->clear(nn);
    labels->clear(labels);
    images->clear(images);

    return;
}

void train_existing_model(const char* labels_path, const char* images_path, size_t num_training_images, size_t epochs, const char* model_path,
    const char* updated_model_path) {

    // Load image dataset and associated labels
    log_message("Starting to load MNIST labels");
    MNIST_Labels* labels = MNIST_Labels_init(labels_path);
    log_message("Finished loading MNIST labels");

    log_message("Starting to load MNIST images");
    MNIST_Images* images = MNIST_Images_init(images_path);
    log_message("Finished loading MNIST images");

    if (num_training_images > images->num_images) {

        fprintf(stderr, "ERR: <train_existing_model> num_training_images cannot be greater than number of images in dataset\n");
        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    if (labels->num_labels != images->num_images) {

        fprintf(stderr, "ERR: <train_existing_model> Number of labels and images must match. Num labels: %u. Num images: %u\n", labels->num_labels, images->num_images);
        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    // Initalize the Neural Network
    Neural_Network* nn = Neural_Network_import(model_path);

    // Setup some variables that we'll use again and again in the loop
    PixelMatrix* current_image = NULL;
    uint8_t current_label = 0;

    // Create the shuffled index array
    size_t* shuffled = create_index_array(num_training_images);

    // Use a buffer for snprintf
    char buffer[100];
    memset(buffer, '\0', 100);

    log_message("Starting model online training for existing model");

    // Execute the training over the number of epochs
    for (size_t i = 0; i < epochs; ++i) {

        if (SHOW_EPOCH) {

            snprintf(buffer, 100, "INFO: Starting online training epoch %zu", i + 1);
            log_message(buffer);
        }

        // Set i to the range of images + labels we want to train on
        for (size_t j = 0; j < num_training_images; ++j) {

            current_image = images->get(images, shuffled[j]);
            current_label = labels->get(labels, shuffled[j]);

            // Execute the training on the current image + label
            nn->train(nn, current_image, current_label);
        }

        shuffle(shuffled, num_training_images);
    }

    log_message("Finished model online training for existing model");

    log_message("Saving updated model");
    nn->save(nn, true, updated_model_path);
    log_message("Finished saving updated model");

    labels->clear(labels);
    images->clear(images);
    nn->clear(nn);

    free(shuffled);

    return;
}

void train_existing_model_batched(const char* labels_path, const char* images_path, size_t num_training_images, 
    size_t batch_size, size_t epochs, const char* model_path, const char* updated_model_path) {

    // Load image dataset and associated labels
    log_message("Starting to load MNIST labels");
    MNIST_Labels* labels = MNIST_Labels_init(labels_path);
    log_message("Finished loading MNIST labels");

    log_message("Starting to load MNIST images");
    MNIST_Images* images = MNIST_Images_init(images_path);
    log_message("Finished loading MNIST images");

    if (num_training_images > images->num_images) {

        fprintf(stderr, "ERR: <train_existing_model_batched> num_training_images cannot be greater than number of images in dataset\n");
        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    if (labels->num_labels != images->num_images) {

        fprintf(stderr, "ERR: <train_existing_model_batched> Number of labels and images must match. Num labels: %u. Num images: %u\n", labels->num_labels, images->num_images);
        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    if (batch_size > num_training_images) {

        fprintf(stderr, "ERR: <train_existing_model_batched> Batch size cannot exceed the number of training images\n");

        images->clear(images);
        labels->clear(labels);

        exit(EXIT_FAILURE);
    }

    // Initalize the Neural Network
    Neural_Network* nn = Neural_Network_import(model_path);

    // Use a buffer for snprintf
    char buffer[100];
    memset(buffer, '\0', 100);

    log_message("Starting batch model training for existing model");

    for (size_t i = 0; i < epochs; ++i) {

        if (SHOW_EPOCH) {

            snprintf(buffer, 100, "INFO: Starting batch training epoch %zu", i + 1);
            log_message(buffer);
        }

        nn->batch_train(nn, images, labels, num_training_images, batch_size);
    }

    log_message("Finished batch model training for existing model");

    log_message("Saving updated model");
    nn->save(nn, true, updated_model_path);
    log_message("Finished saving updated model");

    nn->clear(nn);
    labels->clear(labels);
    images->clear(images);

    return;
}
