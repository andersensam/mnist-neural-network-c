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

#include "include/utils.h"

float sigmoid(float z) {

    //printf("sigmoid input %f\n", z);

    return 1.0f / (1 + exp(-1 * z));
}

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
