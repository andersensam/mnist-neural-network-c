/*  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 * /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 * \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 *  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 *   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 *    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 *     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 *                                                                                                               
 * Project: Matrix Library in C
 * @author : Samuel Andersen
 * @version: 2024-09-22
 *
 * Note: see upstream for Matrix @ https://github.com/andersensam/Matrix
 * 
 */

/* Include the standard things we want for the Matrix header before doing the part that repeats */

#ifndef MATRIX_H
#define MATRIX_H

#define MATRIX_DEBUG 1

/*  Standard dependencies */
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* Definitions */
typedef enum {
    ROW,
    COLUMN
} Vector_Orientation;

#endif

/* Insert the macro expansions at this point */

#if defined(MATRIX_TYPE_NAME) && defined(MATRIX_TYPE) && defined(MATRIX_CREATE_HEADER)

/* Much of the implementation for the generics were inspired by: https://itnext.io/tutorial-generics-in-c-b3362b3376a3 */
#define MATRIX_CONCAT(tag, method) tag ## _ ## method
#define MATRIX_METHOD2(tag, method) MATRIX_CONCAT(tag,method)
#define MATRIX_METHOD(method) MATRIX_METHOD2(MATRIX_TYPE_NAME, method)

typedef struct MATRIX_TYPE_NAME {

    /* Information about the Matrix */
    size_t num_rows;
    size_t num_cols;

    /* Actual data storage */
    MATRIX_TYPE **data;

    /* Methods relating to the Matrix */
    
    /** 
     * Function to get the contents of a Matrix at a specific coordinate
     * @param target The Matrix to reference
     * @param target_row The row we want to reference
     * @param target_col The column we are getting from the row
     * @returns Returns the contents, matching the type defined for the Matrix
     */
    MATRIX_TYPE (*get)(const struct MATRIX_TYPE_NAME *target, size_t target_row, size_t target_col);

    /**
     * Function to set a value at a position in the Matrix
     * @param target The Matrix to reference
     * @param target_row The row that we will set the value for
     * @param target_col The column that we are going to set within the row
     * @param data The data to set within the Matrix
     */
    void (*set)(struct MATRIX_TYPE_NAME *target, size_t target_row, size_t target_col, MATRIX_TYPE data);

    /**
     * Function to clean up a Matrix if it's no longer needed
     * @param target The Matrix we want to clean up
     */
    void (*clear)(struct MATRIX_TYPE_NAME *target);

    /**
     * Calculate the dot product of two Matrix instances
     * @param self The Matrix to call from
     * @param target The Matrix we want to calculate the dot product with
     * @return Returns a Matrix containing the dot product
     */
    struct MATRIX_TYPE_NAME *(*dot)(const struct MATRIX_TYPE_NAME *self, const struct MATRIX_TYPE_NAME *target);

    /**
     * Get a row of a Matrix
     * @param target The Matrix we want to get the row from
     * @param target_row The row we want to extract
     * @return Returns a pointer of the Matrix row
     */
    struct MATRIX_TYPE_NAME *(*get_row)(const struct MATRIX_TYPE_NAME *target, size_t target_row);

    /**
     * Get a column of a Matrix
     * @param target The Matrix we want to get the row from
     * @param target_col The col we want to extract
     * @return A new Matrix containing the contents of the target column. Must be cleaned up after
     */
    struct MATRIX_TYPE_NAME *(*get_col)(const struct MATRIX_TYPE_NAME *target, size_t target_col);

    /**
     * Print the entire Matrix
     * @param target The Matrix that we want to print out
     */
    void (*print)(const struct MATRIX_TYPE_NAME *target);

    /**
     * Get the maximum value stored in a Matrix
     * @param target The Matrix we want to calculate the max value on
     */
    MATRIX_TYPE (*max)(const struct MATRIX_TYPE_NAME *target);

    /**
     * Get the minimum value stored in a Matrix
     * @param target The Matrix we want to calculate the mmin value on
     */
    MATRIX_TYPE (*min)(const struct MATRIX_TYPE_NAME *target);

    /**
     * Flatten a Matrix to an array of pointers to each data element
     * @param target The Matrix we want to flatten
     * @param orientation Either ROW or COLUMN
     * @returns Returns an array of pointers of type MATRIX_TYPE
     */
    struct MATRIX_TYPE_NAME *(*flatten)(const struct MATRIX_TYPE_NAME *target, Vector_Orientation orientation);

    /**
     * Transpose a Matrix
     * @param target The Matrix we want to transpose
     * @returns Returns another Matrix that has been transposed
     */
    struct MATRIX_TYPE_NAME *(*transpose)(const struct MATRIX_TYPE_NAME *target);

    /**
     * Add two Matrix instances' contents together
     * @param self The first Matrix to add values from
     * @param target The second Matrix that we add
     * @return Returns another Matrix instance with the sums of their values
     */
    struct MATRIX_TYPE_NAME *(*add)(const struct MATRIX_TYPE_NAME *self, const struct MATRIX_TYPE_NAME *target);

    /**
     * Add two Matrix instances' contents together, adding to the underlying self Matrix
     * @param self The first Matrix to add values from
     * @param target The second Matrix that we add
     */
    void (*add_o)(const struct MATRIX_TYPE_NAME *self, const struct MATRIX_TYPE_NAME *target);

    /**
     * Subtract two Matrix instances' contents together
     * @param self The first Matrix to subtract values from
     * @param target The second Matrix that we subtract
     * @return Returns another Matrix instance with the result of their subtraction
     */
    struct MATRIX_TYPE_NAME *(*subtract)(const struct MATRIX_TYPE_NAME *self, const struct MATRIX_TYPE_NAME *target);

    /**
     * Subtract two Matrix instances' contents, modifying the self Matrix
     * @param self The first Matrix to subtract values from
     * @param target The second Matrix that we subtract
     */
    void (*subtract_o)(const struct MATRIX_TYPE_NAME *self, const struct MATRIX_TYPE_NAME *target);

    /**
     * Scale a Matrix by a value
     * @param target The Matrix to pull values from
     * @param scalar The value to scale by
     * @return Returns another Matrix instance with the scalar product
     */
    struct MATRIX_TYPE_NAME *(*scale)(const struct MATRIX_TYPE_NAME *target, MATRIX_TYPE scalar);

    /**
     * Scale a Matrix by a value, scaling directly on the original Matrix
     * @param target The Matrix to pull values from
     * @param scalar The value to scale by
     */
    void (*scale_o)(const struct MATRIX_TYPE_NAME *target, MATRIX_TYPE scalar);

    /**
     * Add a scalar to a Matrix
     * @param target The Matrix to pull values from
     * @param scalar The value to add
     * @return Returns another Matrix instance with the scalar addition
     */
    struct MATRIX_TYPE_NAME *(*add_scalar)(const struct MATRIX_TYPE_NAME *target, MATRIX_TYPE scalar);

    /**
     * Add a scalar to a Matrix, adding directly to the underlying Matrix
     * @param target The Matrix to pull values from
     * @param scalar The value to add
     */
    void (*add_scalar_o)(const struct MATRIX_TYPE_NAME *target, MATRIX_TYPE scalar);

    /**
     * Apply a function to a Matrix
     * @param target The Matrix we want to apply a function to
     * @param func A function pointer that we want to use. The pointer must return MATRIX_TYPE
     * @returns Returns a new Matrix with the function applied to it
     */
    struct MATRIX_TYPE_NAME *(*apply)(const struct MATRIX_TYPE_NAME *target, MATRIX_TYPE (*func)(MATRIX_TYPE));

    /**
     * Apply a function to a Matrix, modifying the target Matrix itself
     * @param target The Matrix we want to apply a function to
     * @param func A function pointer that we want to use. The pointer must return MATRIX_TYPE
     */
    void (*apply_o)(const struct MATRIX_TYPE_NAME *target, MATRIX_TYPE (*func)(MATRIX_TYPE));

    /**
     * Apply a function to a Matrix, containing two arguments
     * @param target The Matrix we want to apply a function to
     * @param func A function pointer that we want to use. The pointer must return MATRIX_TYPE
     * @param param Second parameter to apply to the Matrix
     * @returns Returns a new Matrix with the function applied to it
     */
    struct MATRIX_TYPE_NAME *(*apply_second)(const struct MATRIX_TYPE_NAME *target, MATRIX_TYPE (*func)(MATRIX_TYPE, MATRIX_TYPE), MATRIX_TYPE param);

    /**
     * Multiply two Matrix instances' contents together
     * @param self The first Matrix to multiply
     * @param target The second Matrix that multiply by
     * @return Returns another Matrix instance with the products of their values
     */
    struct MATRIX_TYPE_NAME *(*multiply)(const struct MATRIX_TYPE_NAME *self, const struct MATRIX_TYPE_NAME *target);

    /**
     * Multiply two Matrix instances' contents together, storing in Matrix self
     * @param self The first Matrix to multiply
     * @param target The second Matrix that multiply by
     */
    void (*multiply_o)(const struct MATRIX_TYPE_NAME *self, const struct MATRIX_TYPE_NAME *target);

    /**
     * Populate a Matrix with a specific value
     * @param target The Matrix to populate
     * @param value The value to propagate into the Matrix
     */
    void (*populate)(struct MATRIX_TYPE_NAME *target, MATRIX_TYPE value);

    /**
     * Create a copy of a Matrix
     * @param target The Matrix to copy
     * @returns Returns a copy of a Matrix
     */
    struct MATRIX_TYPE_NAME *(*copy)(const struct MATRIX_TYPE_NAME *target);

    /**
     * Get the sum of all values in a Matrix
     * @param target The Matrix to get the sum of
     * @returns Returns the sum
     */
    MATRIX_TYPE (*sum)(const struct MATRIX_TYPE_NAME *target);

    /**
     * Get the index of the maximum in a vector (1D Matrix)
     * @param target The Matrix we want to get the max from
     * @param orientation Orientation we want to process this in
     * @param index The row or column to search in
     * @return Returns a size_t of the index containing the max
     */
    size_t (*max_idx)(const struct MATRIX_TYPE_NAME *target, Vector_Orientation orientation, size_t index);

} __attribute__((__packed__)) MATRIX_TYPE_NAME;

/**
 * Verify that a record is within bounds for a Matrix and that the proper pointers exist
 * @param target Matrix to reference
 * @param target_row Row to access / set
 * @param target_col Column to access / set
 * @returns True if record exists, False if it does not
 */
bool MATRIX_METHOD(exists)(const struct MATRIX_TYPE_NAME *target, size_t target_row, size_t target_col);

/**
 * Function to clean up a Matrix if it's no longer needed
 * @param target The Matrix we want to clean up
 */
void MATRIX_METHOD(clear)(struct MATRIX_TYPE_NAME *target);

/** 
 * Function to get the contents of a Matrix at a specific coordinate
 * @param target The Matrix to reference
 * @param target_row The row we want to reference
 * @param target_col The column we are getting from the row
 * @returns Returns the contents, matching the type defined for the Matrix
 */
MATRIX_TYPE MATRIX_METHOD(get)(const MATRIX_TYPE_NAME *target, size_t target_row, size_t target_col);

/**
 * Function to set a value at a position in the Matrix
 * @param target The Matrix to reference
 * @param target_row The row that we will set the value for
 * @param target_col The column that we are going to set within the row
 * @param data The data to set within the Matrix
 */
void MATRIX_METHOD(set)(MATRIX_TYPE_NAME *target, size_t target_row, size_t target_col, MATRIX_TYPE data);

/**
 * Calculate the dot product of two Matrix instances
* @param self The Matrix to call from
* @param target The Matrix we want to calculate the dot product with
* @return Returns a Matrix containing the dot product
*/
MATRIX_TYPE_NAME *MATRIX_METHOD(dot)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target);

/**
 * Get a row of a Matrix
 * @param target The Matrix we want to get the row from
 * @param target_row The row we want to extract
 * @return Returns a "vector" / array of the Matrix type
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(get_row)(const struct MATRIX_TYPE_NAME *target, size_t target_row);

/**
 * Get a col of a Matrix
 * @param target The Matrix we want to get the col from
 * @param target_col The col we want to extract
 * @return A new Matrix containing the contents of the target column. Must be cleaned up after
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(get_col)(const struct MATRIX_TYPE_NAME *target, size_t target_col);

/**
 * Print the entire Matrix
 * @param target The Matrix that we want to print out
 */
void MATRIX_METHOD(print)(const MATRIX_TYPE_NAME *target);

/**
 * Get the maximum value stored in a Matrix
 * @param target The Matrix we want to calculate the max value on
 */
MATRIX_TYPE MATRIX_METHOD(max)(const MATRIX_TYPE_NAME *target);

/**
 * Get the minimum value stored in a Matrix
 * @param target The Matrix we want to calculate the mmin value on
 */
MATRIX_TYPE MATRIX_METHOD(min)(const MATRIX_TYPE_NAME *target);

/**
 * Flatten a Matrix to either one row or column, depending on desired orientation
 * @param target The Matrix we want to flatten
 * @param orientation Either ROW or COLUMN
 * @returns Returns a vector with either one row / column containing the data
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(flatten)(const MATRIX_TYPE_NAME *target, Vector_Orientation orientation);

/**
 * Transpose a Matrix
 * @param target The Matrix we want to transpose
 * @returns Returns another Matrix that has been transposed
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(transpose)(const MATRIX_TYPE_NAME *target);

/**
 * Add two Matrix instances' contents together
 * @param self The first Matrix to add values from
 * @param target The second Matrix that we add
 * @return Returns another Matrix instance with the sums of their values
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(add)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target);

/**
 * Add two Matrix instances' contents together, adding to the underlying self Matrix
 * @param self The first Matrix to add values from
 * @param target The second Matrix that we add
 */
void MATRIX_METHOD(add_o)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target);

/**
 * Subtract two Matrix instances' contents together
 * @param self The first Matrix to subtract values from
 * @param target The second Matrix that we subtract
 * @return Returns another Matrix instance with the result of their subtraction
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(subtract)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target);

/**
 * Subtract two Matrix instances' contents, modifying the self Matrix
 * @param self The first Matrix to subtract values from
 * @param target The second Matrix that we subtract
 */
void MATRIX_METHOD(subtract_o)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target);

/**
 * Scale a Matrix by a value
 * @param target The Matrix to pull values from
 * @param scalar The value to scale by
 * @return Returns another Matrix instance with the scalar product
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(scale)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE scalar);

/**
 * Scale a Matrix by a value, scaling directly on the original Matrix
 * @param target The Matrix to pull values from
 * @param scalar The value to scale by
 */
void MATRIX_METHOD(scale_o)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE scalar);

/**
 * Add a scalar to a Matrix
 * @param target The Matrix to pull values from
 * @param scalar The value to add
 * @return Returns another Matrix instance with the scalar addition
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(add_scalar)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE scalar);

/**
 * Add a scalar to a Matrix, adding directly to the underlying Matrix
 * @param target The Matrix to pull values from
 * @param scalar The value to add
 */
void MATRIX_METHOD(add_scalar_o)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE scalar);

/**
 * Apply a function to a Matrix
 * @param target The Matrix we want to apply a function to
 * @param func A function pointer that we want to use. The pointer must return MATRIX_TYPE
 * @returns Returns a new Matrix with the function applied to it
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(apply)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE (*func)(MATRIX_TYPE));

/**
 * Apply a function to a Matrix, modifying the target Matrix itself
 * @param target The Matrix we want to apply a function to
 * @param func A function pointer that we want to use. The pointer must return MATRIX_TYPE
 */
void MATRIX_METHOD(apply_o)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE (*func)(MATRIX_TYPE));

/**
 * Apply a function to a Matrix, containing two arguments
 * @param target The Matrix we want to apply a function to
 * @param func A function pointer that we want to use. The pointer must return MATRIX_TYPE
 * @param param Second parameter to apply to the Matrix
 * @returns Returns a new Matrix with the function applied to it
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(apply_second)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE (*func)(MATRIX_TYPE, MATRIX_TYPE), MATRIX_TYPE param);

/**
 * Multiply two Matrix instances' contents together
 * @param self The first Matrix to multiply
 * @param target The second Matrix that multiply by
 * @return Returns another Matrix instance with the products of their values
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(multiply)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target);

/**
 * Multiply two Matrix instances' contents together, keeping contents in Matrix self
 * @param self The first Matrix to multiply
 * @param target The second Matrix that multiply by
 */
void MATRIX_METHOD(multiply_o)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target);

/**
 * Function to allocate a Matrix of any given type. We define this last to get all the right
 * pointers to the functions we reference later on.
 * @param desired_rows Number of rows to allocate
 * @param desired_cols Number of colums to allocate per row
 * @returns Returns a Matrix with all data elements allocated
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(allocate)(size_t desired_rows, size_t desired_cols);

/**
 * Populate a Matrix with a specific value
 * @param target The Matrix to populate
 * @param value The value to propagate into the Matrix
 */
void MATRIX_METHOD(populate)(MATRIX_TYPE_NAME *target, MATRIX_TYPE value);

/**
 * Create a copy of a Matrix
 * @param target The Matrix to copy
 * @returns Returns a copy of a Matrix
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(copy)(const MATRIX_TYPE_NAME *target);

/**
 * Get the sum of all values in a Matrix
 * @param target The Matrix to get the sum of
 * @returns Returns the sum
 */
MATRIX_TYPE MATRIX_METHOD(sum)(const MATRIX_TYPE_NAME *target);

/**
 * Get the index of the maximum in a vector (1D Matrix)
 * @param target The Matrix we want to get the max from
 * @param orientation Orientation we want to process this in
 * @param index The row or column to search in
 * @return Returns a size_t of the index containing the max
 */
size_t MATRIX_METHOD(max_idx)(const MATRIX_TYPE_NAME *target, Vector_Orientation orientation, size_t index);

#undef MATRIX_TYPE_NAME
#undef MATRIX_TYPE
#undef MATRIX_CONCAT
#undef MATRIX_METHOD2
#undef MATRIX_METHOD
#undef MATRIX_STRING
#undef MATRIX_CREATE_HEADER

#endif
