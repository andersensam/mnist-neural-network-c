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
 * @version: 2024-10-15
 *
 * Note: see upstream for Matrix @ https://github.com/andersensam/Matrix
 * 
 */

#include "include/Matrix.h"

#if !defined(MATRIX_TYPE_NAME) || !defined(MATRIX_TYPE) || !defined(MATRIX_STRING)
#error Missing name of Matrix type, datatype of Matrix, or string format of Matrix
#endif

/* Much of the implementation for the generics were inspired by: https://itnext.io/tutorial-generics-in-c-b3362b3376a3 */
#define MATRIX_CONCAT(tag, method) tag ## _ ## method
#define MATRIX_METHOD2(tag, method) MATRIX_CONCAT(tag,method)
#define MATRIX_METHOD(method) MATRIX_METHOD2(MATRIX_TYPE_NAME, method)

/**
 * Verify that a record is within bounds for a Matrix and that the proper pointers exist
 * @param target Matrix to reference
 * @param target_row Row to access / set
 * @param target_col Column to access / set
 * @returns True if record exists, False if it does not
 */
bool MATRIX_METHOD(exists)(const MATRIX_TYPE_NAME *target, size_t target_row, size_t target_col) {
    
    // Verify that the target is not NULL
    if (target == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "DEBUG: <exists> Target Matrix is NULL\n"); }
        return false;
    }

    // Ensure the data element has been allocated and accessible
    if (target->data == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "DEBUG: <exists> Target->data is NULL\n"); }
        return false;
    }

    // Validate that the target_row and target_col referenced are less than the maximum defined when allocating
    if (target_row >= target->num_rows) {

        if (MATRIX_DEBUG) { fprintf(stderr, "DEBUG: <exists> Unable to access/set record at row %zu. Max row value is: %zu\n", target_row, target->num_rows - 1); }
        return false;
    }

    if (target_col >= target->num_cols) {

        if (MATRIX_DEBUG) { fprintf(stderr, "DEBUG: <exists> Unable to access/set record at column %zu. Max column value is: %zu\n", target_col, target->num_cols - 1); }
        return false;       
    }

    return true;

}

/**
 * Function to clean up a Matrix if it's no longer needed
 * @param target The Matrix we want to clean up
 */
void MATRIX_METHOD(clear)(struct MATRIX_TYPE_NAME *target) {

    // Check to see if we were passed a NULL pointer before doing anything
    if (target == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "DEBUG: Passed a NULL pointer to clean up. Returning before we break anything\n"); }
        
        return; 
    }

    if (target->data == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "DEBUG: Matrix->data is NULL. Returning after freeing Matrix\n"); }

        free(target);
        return;

    }

    free(target->data);
    free(target);
}

/** 
 * Function to get the contents of a Matrix at a specific coordinate
 * @param target The Matrix to reference
 * @param target_row The row we want to reference
 * @param target_col The column we are getting from the row
 * @returns Returns the contents, matching the type defined for the Matrix
 */
MATRIX_TYPE MATRIX_METHOD(get)(const MATRIX_TYPE_NAME *target, size_t target_row, size_t target_col) {

    if (!MATRIX_METHOD(exists)(target, target_row, target_col)) { 
        
        fprintf(stderr, "ERR: <get> Invalid Matrix or index provided\n");
        exit(EXIT_FAILURE); 
    }

    return target->data[(target_row * target->num_cols) + target_col];
}

/**
 * Function to set a value at a position in the Matrix
 * @param target The Matrix to reference
 * @param target_row The row that we will set the value for
 * @param target_col The column that we are going to set within the row
 * @param data The data to set within the Matrix
 */
void MATRIX_METHOD(set)(MATRIX_TYPE_NAME *target, size_t target_row, size_t target_col, MATRIX_TYPE data) {

    if (!MATRIX_METHOD(exists)(target, target_row, target_col)) { 

        fprintf(stderr, "ERR: <set> Invalid Matrix or index provided\n");
        exit(EXIT_FAILURE); 
    }

    target->data[(target_row * target->num_cols) + target_col] = data;
}

/**
 * Calculate the dot product of two Matrix instances
* @param self The Matrix to call from
* @param target The Matrix we want to calculate the dot product with
* @return Returns a Matrix containing the dot product
*/
MATRIX_TYPE_NAME *MATRIX_METHOD(dot)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(self, 0, 0) || !MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <dot> Invalid Matrix provided\n");
        exit(EXIT_FAILURE);
    }

    if (self->num_cols != target->num_rows) {

        fprintf(stderr, "ERR: Matrix dimension mismatch. Cannot calculate dot product\n");

        if (MATRIX_DEBUG) {
            fprintf(stderr, "DEBUG: <dot> First Matrix is [%zu x %zu], second is [%zu x %zu]\n",
                self->num_rows, self->num_cols, target->num_rows, target->num_cols);
        }

        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(init)(self->num_rows, target->num_cols);
    MATRIX_TYPE sum = 0;

    // Iterate over the expected rows of the dot Matrix
    for (size_t i = 0; i < result->num_rows; ++i) {

        for (size_t j = 0; j < result->num_cols; ++j) {

            // Sum the multiplications of the individual elements from self and target
            for (size_t k = 0; k < self->num_cols; ++k) {

                sum += self->data[(i * self->num_cols) + k] * target->data[(k * target->num_cols) + j];
            }

            result->data[(i * result->num_cols) + j] = sum;
            sum = 0;
        }
    }

    return result;
}

/**
 * Get a row of a Matrix
 * @param target The Matrix we want to get the row from
 * @param target_row The row we want to extract
 * @return Returns a "vector" / array of the Matrix type
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(get_row)(const MATRIX_TYPE_NAME *target, size_t target_row) {

    // Ensure that the target row exists inside of a valid Matrix
    if (!MATRIX_METHOD(exists)(target, target_row, 0)) {

        fprintf(stderr, "ERR: <get_row> Invalid Matrix / row provided to get_row\n");
        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(init)(1, target->num_cols);

    for (size_t i = 0; i < target->num_cols; ++i) {

        MATRIX_METHOD(set)(result, 0, i, MATRIX_METHOD(get)(target, target_row, i));
    }

    return result;
}

/**
 * Get a col of a Matrix
 * @param target The Matrix we want to get the col from
 * @param target_col The col we want to extract
 * @return A new Matrix containing the contents of the target column. Must be cleaned up after
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(get_col)(const MATRIX_TYPE_NAME *target, size_t target_col) {

    // Ensure that the target col exists inside of a valid Matrix
    if (!MATRIX_METHOD(exists)(target, 0, target_col)) {

        fprintf(stderr, "ERR: <get_col> Invalid Matrix / col provided to get_col\n");
        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(init)(target->num_rows, 1);

    for (size_t i = 0; i < target->num_rows; ++i) {

        MATRIX_METHOD(set)(result, i, 0, MATRIX_METHOD(get)(target, i, target_col));
    }

    return result;
}

/**
 * Print the entire Matrix
 * @param target The Matrix that we want to print out
 */
void MATRIX_METHOD(print)(const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <print> Invalid Matrix provided to print\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < target->num_rows; ++i) {

        // Print out the row number in brackets, for readability
        printf("[%zu]:\t", i);

        for (size_t j = 0; j < target->num_cols; ++j) {

            printf(MATRIX_STRING, MATRIX_METHOD(get)(target, i, j));
            printf(" ");
        }

        printf("\n");
    }

}

/**
 * Get the maximum value stored in a Matrix
 * @param target The Matrix we want to calculate the max value on
 */
MATRIX_TYPE MATRIX_METHOD(max)(const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <max> Invalid Matrix provided to max\n");
        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE current_max = target->data[0];

    for (size_t i = 1; i < target->num_rows * target->num_cols; ++i) {

        MATRIX_TYPE current_value = target->data[i];

        current_max = (current_max > current_value) ? current_max : current_value;
    }

    return current_max;
}

/**
 * Get the minimum value stored in a Matrix
 * @param target The Matrix we want to calculate the mmin value on
 */
MATRIX_TYPE MATRIX_METHOD(min)(const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <min> Invalid Matrix provided to max\n");
        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE current_min = target->data[0];

    for (size_t i = 1; i < target->num_rows * target->num_cols; ++i) {

        MATRIX_TYPE current_value = target->data[i];

        current_min = (current_min < current_value) ? current_min : current_value;
    }

    return current_min;
}

/**
 * Flatten a Matrix to an array of pointers to each data element
 * @param target The Matrix we want to flatten
 * @param orientation Either ROW or COLUMN
 * @returns Returns an array of pointers of type MATRIX_TYPE
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(flatten)(const MATRIX_TYPE_NAME *target, Vector_Orientation orientation) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <flatten> Invalid Matrix provided to flatten\n");
        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(copy)(target);

    MATRIX_METHOD(flatten_o)(result, orientation);

    return result;
}

/**
 * Flatten a Matrix to either one row or column, depending on desired orientation
 * @param target The Matrix we want to flatten
 * @param orientation Either ROW or COLUMN
 */
void MATRIX_METHOD(flatten_o)(MATRIX_TYPE_NAME *target, Vector_Orientation orientation) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <flatten_o> Invalid Matrix provided to flatten_o\n");
        exit(EXIT_FAILURE);
    }

    if (orientation == ROW) {

        target->num_cols = target->num_rows * target->num_cols;
        target->num_rows = 1;
    }
    else {

        target->num_rows = target->num_rows * target->num_cols;
        target->num_cols = 1;
    }
}

/**
 * Transpose a Matrix
 * @param target The Matrix we want to transpose
 * @returns Returns another Matrix that has been transposed
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(transpose)(const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <transpose> Invalid Matrix provided to transpose\n");
        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(init)(target->num_cols, target->num_rows);

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            result->data[(j * result->num_cols) + i] = target->data[(i * target->num_cols) + j];
        }
    }

    return result;

}

/**
 * Add two Matrix instances' contents together
 * @param self The first Matrix to add values from
 * @param target The second Matrix that we add
 * @return Returns another Matrix instance with the sums of their values
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(add)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(self, 0, 0) || !MATRIX_METHOD(exists)(target, 0, 0)) { 

        fprintf(stderr, "ERR: <add> Self or target Matrix is NULL. Cannot add\n");
        exit(EXIT_FAILURE);
    }

    if (self->num_cols != target->num_cols || self->num_rows != target->num_rows) {

        fprintf(stderr, "ERR: <add> Matrix dimension mismatch. Cannot add\n");

        if (MATRIX_DEBUG) {

            fprintf(stderr, "DEBUG: <add> self is [%zu x %zu] and target is [%zu x %zu]\n", self->num_rows, self->num_cols,
                target->num_rows, target->num_cols);
        }

        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(copy)(self);

    MATRIX_METHOD(add_o)(result, target);

    return result;
}

/**
 * Add two Matrix instances' contents together, adding to the underlying self Matrix
 * @param self The first Matrix to add values from
 * @param target The second Matrix that we add
 */
void MATRIX_METHOD(add_o)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(self, 0, 0) || !MATRIX_METHOD(exists)(target, 0, 0)) { 

        fprintf(stderr, "ERR: <add_o> Self or target Matrix is NULL. Cannot add_o\n");
        exit(EXIT_FAILURE);
    }

    if (self->num_cols != target->num_cols || self->num_rows != target->num_rows) {

        fprintf(stderr, "ERR: <add_o> Matrix dimension mismatch. Cannot add_o\n");

        if (MATRIX_DEBUG) {

            fprintf(stderr, "DEBUG: <add_o> self is [%zu x %zu] and target is [%zu x %zu]\n", self->num_rows, self->num_cols,
                target->num_rows, target->num_cols);
        }

        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < self->num_rows * self->num_cols; ++i) {

        self->data[i] += target->data[i];
    }
}

/**
 * Subtract two Matrix instances' contents together
 * @param self The first Matrix to subtract values from
 * @param target The second Matrix that we subtract
 * @return Returns another Matrix instance with the result of their subtraction
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(subtract)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(self, 0, 0) || !MATRIX_METHOD(exists)(target, 0, 0)) { 

        fprintf(stderr, "ERR: <subtract> Self or target Matrix is NULL. Cannot subtract\n");
        exit(EXIT_FAILURE);
    }

    if (self->num_cols != target->num_cols || self->num_rows != target->num_rows) {

        fprintf(stderr, "ERR: <subtract> Matrix dimension mismatch. Cannot subtract\n");

        if (MATRIX_DEBUG) {

            fprintf(stderr, "DEBUG: <subtract> self is [%zu x %zu] and target is [%zu x %zu]\n", self->num_rows, self->num_cols,
                target->num_rows, target->num_cols);
        }

        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(copy)(self);

    MATRIX_METHOD(subtract_o)(result, target);

    return result;
}

/**
 * Subtract two Matrix instances' contents, modifying the self Matrix
 * @param self The first Matrix to subtract values from
 * @param target The second Matrix that we subtract
 */
void MATRIX_METHOD(subtract_o)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(self, 0, 0) || !MATRIX_METHOD(exists)(target, 0, 0)) { 

        fprintf(stderr, "ERR: <subtract_o> Self or target Matrix is NULL. Cannot subtract_o\n");
        exit(EXIT_FAILURE);
    }

    if (self->num_cols != target->num_cols || self->num_rows != target->num_rows) {

        fprintf(stderr, "ERR: <subtract_o> Matrix dimension mismatch. Cannot subtract\n");

        if (MATRIX_DEBUG) {

            fprintf(stderr, "DEBUG: <subtract_o> self is [%zu x %zu] and target is [%zu x %zu]\n", self->num_rows, self->num_cols,
                target->num_rows, target->num_cols);
        }

        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < self->num_rows * self->num_cols; ++i) {

        self->data[i] -= target->data[i];
    }
}

/**
 * Scale a Matrix by a value
 * @param target The Matrix to pull values from
 * @param scalar The value to scale by
 * @return Returns another Matrix instance with the scalar product
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(scale)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE scalar) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <scale> Invalid Matrix provided to scale\n");
        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(copy)(target);

    MATRIX_METHOD(scale_o)(result, scalar);

    return result;
}

/**
 * Scale a Matrix by a value, scaling directly on the original Matrix
 * @param target The Matrix to pull values from
 * @param scalar The value to scale by
 */
void MATRIX_METHOD(scale_o)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE scalar) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <scale_o> Invalid Matrix provided to scale_o\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < target->num_rows * target->num_cols; ++i) {

        target->data[i] *= scalar;
    }

    return;
}

/**
 * Add a scalar to a Matrix
 * @param target The Matrix to pull values from
 * @param scalar The value to add
 * @return Returns another Matrix instance with the scalar addition
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(add_scalar)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE scalar) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <add_scalar> Invalid Matrix provided to add_scalar\n");
        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(copy)(target);

    MATRIX_METHOD(add_scalar_o)(result, scalar);

    return result;
}

/**
 * Add a scalar to a Matrix, adding directly to the underlying Matrix
 * @param target The Matrix to pull values from
 * @param scalar The value to add
 */
void MATRIX_METHOD(add_scalar_o)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE scalar) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <add_scalar_o> Invalid Matrix provided to add_scalar_o\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < target->num_rows * target->num_cols; ++i) {

        target->data[i] += scalar;
    }

    return;
}

/**
 * Apply a function to a Matrix
 * @param target The Matrix we want to apply a function to
 * @param func A function pointer that we want to use. The pointer must return MATRIX_TYPE
 * @returns Returns a new Matrix with the function applied to it
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(apply)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE (*func)(MATRIX_TYPE)) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <apply> Invalid Matrix provided to apply\n");
        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(copy)(target);

    MATRIX_METHOD(apply_o)(result, func);

    return result;
}

/**
 * Apply a function to a Matrix, modifying the target Matrix itself
 * @param target The Matrix we want to apply a function to
 * @param func A function pointer that we want to use. The pointer must return MATRIX_TYPE
 */
void MATRIX_METHOD(apply_o)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE (*func)(MATRIX_TYPE)) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <apply_o> Invalid Matrix provided to apply_o\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < target->num_rows * target->num_cols; ++i) {

        target->data[i] = (*func)(target->data[i]);
    }

    return;
}

/**
 * Apply a function to a Matrix, containing two arguments
 * @param target The Matrix we want to apply a function to
 * @param func A function pointer that we want to use. The pointer must return MATRIX_TYPE
 * @param param Second parameter to apply to the Matrix
 * @returns Returns a new Matrix with the function applied to it
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(apply_second)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE (*func)(MATRIX_TYPE, MATRIX_TYPE), MATRIX_TYPE param) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <apply_second> Invalid Matrix provided to apply_second\n");
        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(copy)(target);

    for (size_t i = 0; i < target->num_rows * target->num_cols; ++i) {

        result->data[i] = (*func)(target->data[i], param);
    }

    return result;
}

/**
 * Multiply two Matrix instances' contents together elementwise
 * @param self The first Matrix to multiply
 * @param target The second Matrix that multiply by
 * @return Returns another Matrix instance with the products of their values
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(multiply)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(self, 0, 0) || !MATRIX_METHOD(exists)(target, 0, 0)) { 

        fprintf(stderr, "ERR: <multiply> Self or target Matrix is NULL. Cannot multiply\n");
        exit(EXIT_FAILURE);
    }

    if (self->num_cols != target->num_cols || self->num_rows != target->num_rows) {

        if (MATRIX_DEBUG) { fprintf(stderr, "DEBUG: <multiply> Matrix dimension mismatch. Cannot multiply. [%zu x %zu] != [%zu x %zu]\n",
            self->num_rows, self->num_cols, target->num_rows, target->num_cols); }
        
        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(copy)(self);

    MATRIX_METHOD(multiply_o)(result, target);

    return result;
}

/**
 * Multiply two Matrix instances' contents together, keeping contents in Matrix self
 * @param self The first Matrix to multiply
 * @param target The second Matrix that multiply by
 */
void MATRIX_METHOD(multiply_o)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(self, 0, 0) || !MATRIX_METHOD(exists)(target, 0, 0)) { 

        fprintf(stderr, "ERR: <multiply_o> Self or target Matrix is NULL. Cannot multiply_o\n");
        exit(EXIT_FAILURE);
    }

    if (self->num_cols != target->num_cols || self->num_rows != target->num_rows) {

        if (MATRIX_DEBUG) { fprintf(stderr, "DEBUG: <multiply_o> Matrix dimension mismatch. Cannot multiply_o. [%zu x %zu] != [%zu x %zu]\n",
            self->num_rows, self->num_cols, target->num_rows, target->num_cols); }
        
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < self->num_rows * self->num_cols; ++i) {

        self->data[i] *= target->data[i];
    }

    return;
}

/**
 * Populate a Matrix with a specific value
 * @param target The Matrix to populate
 * @param value The value to propagate into the Matrix
 */
void MATRIX_METHOD(populate)(MATRIX_TYPE_NAME *target, MATRIX_TYPE value) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <populate> Invalid Matrix provided to populate\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < target->num_rows * target->num_cols; ++i) {

        target->data[i] = value;
    }
}

/**
 * Create a copy of a Matrix
 * @param target The Matrix to copy
 * @returns Returns a copy of a Matrix
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(copy)(const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <copy> Invalid Matrix provided to copy\n");
        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(init)(target->num_rows, target->num_cols);

    memcpy(&(result->data[0]), &(target->data[0]), target->num_rows * target->num_cols * sizeof(MATRIX_TYPE));

    return result;
}

/**
 * Get the sum of all values in a Matrix
 * @param target The Matrix to get the sum of
 * @returns Returns the sum
 */
MATRIX_TYPE MATRIX_METHOD(sum)(const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <sum> Invalid Matrix provided to sum\n");
        exit(EXIT_FAILURE);
    }

    MATRIX_TYPE running_sum = 0;

    for (size_t i = 0; i < target->num_rows * target->num_cols; ++i) {

        running_sum += target->data[i];
    }

    return running_sum;
}

/**
 * Get the index of the maximum in a vector (1D Matrix)
 * @param target The Matrix we want to get the max from
 * @param orientation Orientation we want to process this in
 * @param index The row or column to search in
 * @return Returns a size_t of the index containing the max
 */
size_t MATRIX_METHOD(max_idx)(const MATRIX_TYPE_NAME *target, Vector_Orientation orientation, size_t index) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <max_idx> Invalid Matrix provided to max_idx\n");
        exit(EXIT_FAILURE);
    }

    size_t max_index = 0;
    MATRIX_TYPE max_value = 0;

    // Handle the easy case where we already have a Matrix / vector of 1 row or column
    if (target->num_rows == 1 || target->num_cols == 1) {

        // Get the max value from the Matrix
        max_value = MATRIX_METHOD(max)(target);

        if (orientation == ROW) {

            for (size_t i = 0; i < target->num_cols; ++i) {

                if (target->data[i] == max_value) { return i; }
            }
        }

        for (size_t i = 0; i < target->num_rows; ++i) {

            if (target->data[i] == max_value) { return i; }
        }
    }

    // If we have a multidimensional Matrix, convert it to a row / column vector and then recursively search
    MATRIX_TYPE_NAME *search = NULL;

    if (orientation == ROW) {

        search = MATRIX_METHOD(get_row)(target, index);
    }
    else {

        search = MATRIX_METHOD(get_col)(target, index);
    }

    max_index = MATRIX_METHOD(max_idx)(search, orientation, 0);

    MATRIX_METHOD(clear)(search);

    return max_index;
}

/**
 * Create a copy of the underlying data and return the pointer to it
* @param target The Matrix we want to expose
* @returns Returns a pointer to the data
*/
MATRIX_TYPE* MATRIX_METHOD(expose)(const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        fprintf(stderr, "ERR: <expose> Invalid Matrix provided to expose\n");
        exit(EXIT_FAILURE);
    }

    // Allocate a new array for the data
    MATRIX_TYPE *data = malloc(sizeof(MATRIX_TYPE) * target->num_rows * target->num_cols);

    if (data == NULL) {

        fprintf(stderr, "ERR: <expose> Unable to allocate memory for expose\n");
        exit(EXIT_FAILURE);
    }

    memcpy(&(data[0]), &(target->data[0]), sizeof(MATRIX_TYPE) * target->num_rows * target->num_cols);

    return data;
}

/**
 * Function to allocate a Matrix of any given type. We define this last to get all the right
 * pointers to the functions we reference later on.
 * @param desired_rows Number of rows to allocate
 * @param desired_cols Number of colums to allocate per row
 * @returns Returns a Matrix with all data elements allocated
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(init)(size_t desired_rows, size_t desired_cols) {

    // Allocate the memory needed for the Matrix struct
    MATRIX_TYPE_NAME *target = malloc(sizeof(MATRIX_TYPE_NAME));

    if (target == NULL) {

        fprintf(stderr, "ERR: <init> Unable to allocate Matrix of size %zu x %zu\n", desired_rows, desired_cols);
        exit(EXIT_FAILURE);
    }

    // Allocate the proper number of rows -- i.e. an array of arrays
    target->data = calloc(desired_rows * desired_cols, sizeof(MATRIX_TYPE));

    if (target->data == NULL) {

        fprintf(stderr, "ERR: <init> Unable to data array in Matrix of size %zu x %zu\n", desired_rows, desired_cols);
        
        exit(EXIT_FAILURE);
    }

    // Store information about the Matrix before returning
    target->num_rows = desired_rows;
    target->num_cols = desired_cols;

    // Connect the proper methods to the Matrix instance
    target->get = MATRIX_METHOD(get);
    target->set = MATRIX_METHOD(set);
    target->clear = MATRIX_METHOD(clear);
    target->dot = MATRIX_METHOD(dot);
    target->get_row = MATRIX_METHOD(get_row);
    target->get_col = MATRIX_METHOD(get_col);
    target->print = MATRIX_METHOD(print);
    target->max = MATRIX_METHOD(max);
    target->min = MATRIX_METHOD(min);
    target->flatten = MATRIX_METHOD(flatten);
    target->flatten_o = MATRIX_METHOD(flatten_o);
    target->transpose = MATRIX_METHOD(transpose);
    target->add = MATRIX_METHOD(add);
    target->add_o = MATRIX_METHOD(add_o);
    target->subtract = MATRIX_METHOD(subtract);
    target->subtract_o = MATRIX_METHOD(subtract_o);
    target->scale = MATRIX_METHOD(scale);
    target->scale_o = MATRIX_METHOD(scale_o);
    target->add_scalar = MATRIX_METHOD(add_scalar);
    target->add_scalar_o = MATRIX_METHOD(add_scalar_o);
    target->apply = MATRIX_METHOD(apply);
    target->apply_o = MATRIX_METHOD(apply_o);
    target->apply_second = MATRIX_METHOD(apply_second);
    target->multiply = MATRIX_METHOD(multiply);
    target->multiply_o = MATRIX_METHOD(multiply_o);
    target->populate = MATRIX_METHOD(populate);
    target->copy = MATRIX_METHOD(copy);
    target->sum = MATRIX_METHOD(sum);
    target->max_idx = MATRIX_METHOD(max_idx);
    target->expose = MATRIX_METHOD(expose);

    return target;
}

#undef MATRIX_TYPE_NAME
#undef MATRIX_TYPE
#undef MATRIX_CONCAT
#undef MATRIX_METHOD2
#undef MATRIX_METHOD
#undef MATRIX_STRING
