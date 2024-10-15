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

#include "include/utils.h"

uint32_t map_uint32(uint32_t in) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return (
        ((in & 0xFF000000) >> 24) |
        ((in & 0x00FF0000) >>  8) |
        ((in & 0x0000FF00) <<  8) |
        ((in & 0x000000FF) << 24)
    );
#else
    return in;
#endif
}

float random_float(void) {

    return (float) (2 * (rand() / (float)(RAND_MAX))) - 1.0;
}

void log_message(const char* message) {

    // Setup a buffer and get the current time
    char buffer[100];
    time_t t = time(NULL);

    // Format a time string, storing in the buffer
    strftime(buffer, sizeof(buffer), "[%Y-%m-%d %H:%M:%S]: ", localtime(&t));

    // Print out the message with the time
    printf("%s%s\n", buffer, message);
}

void shuffle(size_t* index, size_t elements) {

    if (index == NULL) {

        fprintf(stderr, "ERR: <shuffle> Invalid index array provided to shuffle\n");
        exit(EXIT_FAILURE);
    }

    size_t random_index = 0;
    size_t current_value = 0;

    for (size_t i = elements - 1; i > 0; --i) {

        random_index = rand() % (i + 1);

        // Grab the current value and swap with the value @ random_index
        current_value = index[i];
        index[i] = index[random_index];
        index[random_index] = current_value;
    }
}

size_t* create_index_array(size_t elements) {

    size_t* target = calloc(elements, sizeof(size_t));

    if (target == NULL) {

        fprintf(stderr, "ERR: <create_index_array> Unable to allocate memory for shuffled index array\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < elements; ++i) {

        target[i] = i;
    }

    // Shuffle the index array
    shuffle(target, elements);

    return target;
}
