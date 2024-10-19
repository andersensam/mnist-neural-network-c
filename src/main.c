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

#include "include/main.h"

int main(int argc, char* argv[]) {

    if (argc <= 1) {

        fprintf(stderr, "ERR: <main> Too few arguments provided. Use main help to get a help menu. Exiting\n");
        return 1;
    }

    // Ensure this is only executed once
    srand(time(0));

    if (strncmp(argv[1], "train", 5) == 0) {

        /*
         * For training, expect: labels_path (2), images_path (3), learning_rate (4), include_biases (5), 
         * num_layers (6), [layer_info] (7..n-4), num_images (n-3), epochs (n-2), model_name (n-1)
         */

        if (argc != (10 + atoi(argv[6]))) {

            fprintf(stderr, "ERR: <main> Invalid arguments provided to train\n");
            return 1;
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
        train_new_model(argv[2], argv[3], num_layers, nn_config, learning_rate, generate_biases, (size_t)atoi(argv[argc - 3]),
            (size_t)atoi(argv[argc - 2]), argv[argc - 1]);

        free(nn_config);
        return 0;
    }
    else if (strncmp(argv[1], "update-online", 13) == 0) {

        /*
         * For update-online, expect: labels_path (2), images_path (3), num_images (4), epochs (5), existing_model_path (6)
         * updated_model_path (7)
         */

        if (argc != 8) {

            fprintf(stderr, "ERR: <main> Invalid arguments provided to update-online\n");
            return 1;
        }

        train_existing_model(argv[2], argv[3], (size_t)atoi(argv[4]), (size_t)atoi(argv[5]), argv[6], argv[7]);

        return 0;
    }
    else if (strncmp(argv[1], "update-batch", 12) == 0) {

        /*
         * For update-batch, expect: labels_path (2), images_path (3), num_images (4), batch_size (5) epochs (6), 
         * existing_model_path (7), updated_model_path (8)
         */

        if (argc != 9) {

            fprintf(stderr, "ERR: <main> Invalid arguments provided to update-batch\n");
            return 1;
        }

        train_existing_model_batched(argv[2], argv[3], (size_t)atoi(argv[4]), (size_t)atoi(argv[5]), 
            (size_t)atoi(argv[6]), argv[7], argv[8]);

        return 0;
    }
    else if (strncmp(argv[1], "batch-train", 11) == 0) {

        /*
         * For training, expect: labels_path (2), images_path (3), learning_rate (4), lambda (5), include_biases (6), 
         * num_layers (7), [layer_info] (8..n-5), num_images (n-4), batch_size (n-3), epochs (n-2), model_name (n - 1)
         */

        if (argc != (12 + atoi(argv[7]))) {

            fprintf(stderr, "ERR: <main> Invalid arguments provided to batch-train\n");
            return 1;
        }

        float learning_rate = 0;
        sscanf(argv[4], "%f", &learning_rate);

        float lambda = 0;
        sscanf(argv[5], "%f", &learning_rate);

        bool generate_biases = false;

        if (strncmp(argv[6], "true", 4) == 0) { generate_biases = true; }

        size_t num_layers = (size_t)(atoi(argv[7]));
        size_t* nn_config = calloc(num_layers, sizeof(size_t));

        // Iterate over the number of layers and get the neuron counts for each
        for (size_t i = 0; i < num_layers; ++i) {

            nn_config[i] = (size_t)atoi(argv[i + 8]);
        }

        // Train a new model with the given parameters
        train_new_model_batched(argv[2], argv[3], num_layers, nn_config, learning_rate, lambda, generate_biases,
            (size_t)atoi(argv[argc - 4]), (size_t)atoi(argv[argc - 3]), (size_t)atoi(argv[argc-2]), argv[argc - 1]);

        free(nn_config);
        return 0;
    }
    else if (strncmp(argv[1], "threaded-batch", 14) == 0) {

        /*
         * For training, expect: labels_path (2), images_path (3), learning_rate (4), include_biases (6), 
         * num_layers (7), [layer_info] (8..n-5), num_images (n-4), batch_size (n-3), epochs (n-2), model_name (n - 1)
         */

        if (argc != (12 + atoi(argv[7]))) {

            fprintf(stderr, "ERR: <main> Invalid arguments provided to threaded-batch\n");
            return 1;
        }

        float learning_rate = 0;
        sscanf(argv[4], "%f", &learning_rate);

        float lambda = 0;
        sscanf(argv[5], "%f", &learning_rate);

        bool generate_biases = false;

        if (strncmp(argv[6], "true", 4) == 0) { generate_biases = true; }

        size_t num_layers = (size_t)(atoi(argv[7]));
        size_t* nn_config = calloc(num_layers, sizeof(size_t));

        // Iterate over the number of layers and get the neuron counts for each
        for (size_t i = 0; i < num_layers; ++i) {

            nn_config[i] = (size_t)atoi(argv[i + 8]);
        }

        // Train a new model with the given parameters
        train_new_model_batched_threaded(argv[2], argv[3], num_layers, nn_config, learning_rate, lambda, generate_biases,
            (size_t)atoi(argv[argc - 4]), (size_t)atoi(argv[argc - 3]), (size_t)atoi(argv[argc-2]), argv[argc - 1]);

        free(nn_config);
        return 0;
    }
    else if (strncmp(argv[1], "predict", 7) == 0) {

        if (argc != 6) {

            fprintf(stderr, "ERR: <main> Invalid arguments provided to predict\n");
            return 1;
        }

        // Execute inference and exit
        inference(argv[2], argv[3], argv[5], (size_t)atoi(argv[4]));
        return 0;
    }
    else if (strncmp(argv[1], "threaded-predict", 16) == 0) {

        if (argc != 6) {

            fprintf(stderr, "ERR: <main> Invalid arguments provided to use threaded-predict\n");
            return 1;
        }

        // Execute inference and exit
        threaded_inference(argv[2], argv[3], argv[5], (size_t)atoi(argv[4]));
        return 0;
    }
    else if (strncmp(argv[1], "help", 4) == 0) {

        printf("Expected usage: main train labels_path images_path learning_rate include_biases num_layers [layer_info] num_training_images epochs model_name\n");
        printf("Example for train: main train data/labels data/images 0.1 true 3 786 100 10 1000 3 model.model\n\n");
        printf("This example has a learning rate of 0.1, uses biases, has 3 layers, and uses 1000 images to train on, over 3 epochs\n");

        printf("\n\nExpected usage: main batch-train labels_path images_path learning_rate lambda include_biases num_layers [layer_info] num_training_images batch_size epochs model_name\n");
        printf("Example for train: main train data/labels data/images 0.1 0.0 true 3 786 100 10 1000 10 10 model.model\n\n");
        printf("This example has a learning rate of 0.1, lambda of 0, uses biases, has 3 layers,uses 1000 images to train on, with a batch size of 10, and 10 epochs\n");

        printf("\n\nExpected usage: main threaded-batch labels_path images_path learning_rate lambda include_biases num_layers [layer_info] num_training_images batch_size epochs model_name\n");
        printf("Example for threaded-batch: main threaded-batch data/labels data/images 0.1 0.0 true 3 786 100 10 1000 10 10 model.model\n");

        printf("\n\nExpected usage: main update-online labels_path images_path num_training_images epochs model_path updated_model_path\n");
        printf("Example for update: main update-online /labels data/images 1000 3 models/existing_model models/updated_model\n");

        printf("\n\nExpected usage: main update-batch labels_path images_path num_training_images batch_size epochs model_path updated_model_path\n");
        printf("Example for update: main update-batch data/labels data/images 1000 8 3 models/existing_model models/updated_model\n");

        printf("\n\nExpected usage: main predict labels_path images_path num_predict model_path\n");
        printf("Example for predict: main predict data/labels data/images 100 model.model\n\n");
        printf("This example predicts 100 images");

        printf("\n\nExpected usage: main threaded-predict labels_path images_path num_predict model_path\n");
        printf("Example for predict: main threaded-predict data/labels data/images 100 model.model\n\n");
        printf("This example predicts 100 images");

        return 0;
    }

    fprintf(stderr, "ERR: <main> Invalid argument passed to main. Exiting\n");
    return 1;
}
