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

#ifndef UTILS_H
#define UTILS_H

/* Standard dependencies */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* Local dependencies */

/* Definitions */

/**
 * Convert from the big endian format in the dataset if we're on a little endian machine
 * @param in Number that we're taking in and examining the byte order
 * @returns uint32_t in the proper endianness
 */
uint32_t map_uint32(uint32_t in);

/**
 * Generate a random float for populating Matrix values
 * @returns A random float between 0 and 1
 */
float random_float(void);

/**
 * Log an event to the console with proper date and time
 * @param message Message to display on the console
 */
void log_message(const char* message);

/**
 * Shuffle the indicies used for pulling images and labels
 * See Fisher-Yates Shuffle: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
 * @param index Array of size_t to shuffle
 * @param elements Number of elements in the index array
 */
void shuffle(size_t* index, size_t elements);

/**
 * Generate an array of size_t that are randomly shuffled
 * @param elements
 * @returns Returns an array of size_t
 */
size_t* create_index_array(size_t elements);

#endif
