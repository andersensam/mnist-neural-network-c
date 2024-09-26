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
 * @version: 2024-09-25
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
bool MATRIX_METHOD(exists)(const struct MATRIX_TYPE_NAME *target, size_t target_row, size_t target_col) {
    
    // Verify that the target is not NULL
    if (target == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Target Matrix is NULL\n"); }
        return false;
    }

    // Ensure the data element has been allocated and accessible
    if (target->data == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Target->data is NULL\n"); }
        return false;
    }

    // Validate that the target_row and target_col referenced are less than the maximum defined when allocating
    if (target_row >= target->num_rows) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to access/set record at row %zu. Max row value is: %zu\n", target_row, target->num_rows - 1); }
        return false;
    }

    if (target_col >= target->num_cols) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to access/set record at column %zu. Max column value is: %zu\n", target_col, target->num_cols - 1); }
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

        if (MATRIX_DEBUG) { fprintf(stderr, "WARN: Passed a NULL pointer to clean up. Returning before we break anything\n"); }
        
        return; 
    }

    if (target->data == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "WARN: Matrix->data is NULL. Returning after freeing Matrix\n"); }

        free(target);
        return;

    }

    for (size_t i = 0; i < target->num_rows; ++i) {

        // Check to see if the row itself is NULL. Free if it's not, show the warning if it is
        if (target->data[i] != NULL) { free(target->data[i]); }
        else { if (MATRIX_DEBUG) { fprintf(stderr, "WARN: Row %zu is NULL. Ignoring during cleanup\n", i); } }
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

    if (!MATRIX_METHOD(exists)(target, target_row, target_col)) { return (MATRIX_TYPE)0; }

    return target->data[target_row][target_col];
}

/**
 * Function to set a value at a position in the Matrix
 * @param target The Matrix to reference
 * @param target_row The row that we will set the value for
 * @param target_col The column that we are going to set within the row
 * @param data The data to set within the Matrix
 */
void MATRIX_METHOD(set)(MATRIX_TYPE_NAME *target, size_t target_row, size_t target_col, MATRIX_TYPE data) {

    if (!MATRIX_METHOD(exists)(target, target_row, target_col)) { return; }

    target->data[target_row][target_col] = data;
}

/**
 * Calculate the dot product of two Matrix instances
* @param self The Matrix to call from
* @param target The Matrix we want to calculate the dot product with
* @return Returns a Matrix containing the dot product
*/
MATRIX_TYPE_NAME *MATRIX_METHOD(dot)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target) {

    if (self == NULL || target == NULL) { 

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Self or target Matrix is NULL. Cannot calculate dot product\n"); }
        return NULL;
    }

    if (self->num_cols != target->num_rows) {

        if (MATRIX_DEBUG) { 

            fprintf(stderr, "ERR: Matrix dimension mismatch. Cannot calculate dot product. ");
            fprintf(stderr, "First Matrix is [%zu x %zu], second is [%zu x %zu]\n",
                self->num_rows, self->num_cols, target->num_rows, target->num_cols);
        }
        return NULL;
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(allocate)(self->num_rows, target->num_cols);
    MATRIX_TYPE sum = 0;

    // Iterate over the expected rows of the dot Matrix
    for (size_t i = 0; i < result->num_rows; ++i) {

        for (size_t j = 0; j < result->num_cols; ++j) {

            // Sum the multiplications of the individual elements from self and target
            for (size_t k = 0; k < self->num_cols; ++k) {

                sum += self->data[i][k] * target->data[k][j];
            }

            result->data[i][j] = sum;
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
MATRIX_TYPE_NAME *MATRIX_METHOD(get_row)(const struct MATRIX_TYPE_NAME *target, size_t target_row) {

    // Ensure that the target row exists inside of a valid Matrix
    if (!MATRIX_METHOD(exists)(target, target_row, 0)) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix / row provided to get_row\n"); }
        return NULL;
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(allocate)(1, target->num_cols);

    for (size_t i = 0; i < target->num_cols; ++i) {

        result->set(result, 0, i, target->get(target, target_row, i));
    }

    return result;
}

/**
 * Get a col of a Matrix
 * @param target The Matrix we want to get the col from
 * @param target_col The col we want to extract
 * @return A new Matrix containing the contents of the target column. Must be cleaned up after
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(get_col)(const struct MATRIX_TYPE_NAME *target, size_t target_col) {

    // Ensure that the target col exists inside of a valid Matrix
    if (!MATRIX_METHOD(exists)(target, 0, target_col)) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix / col provided to get_col\n"); }
        return NULL;
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(allocate)(target->num_rows, 1);

    for (size_t i = 0; i < target->num_rows; ++i) {

        result->set(result, i, 0, target->get(target, i, target_col));
    }

    return result;
}

/**
 * Print the entire Matrix
 * @param target The Matrix that we want to print out
 */
void MATRIX_METHOD(print)(const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to print\n"); }
        return;
    }

    for (size_t i = 0; i < target->num_rows; ++i) {

        // Print out the row number in brackets, for readability
        printf("[%zu]:\t", i);

        for (size_t j = 0; j < target->num_cols; ++j) {

            printf(MATRIX_STRING, target->get(target, i, j));
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

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to max\n"); }
        return (MATRIX_TYPE)0;
    }

    MATRIX_TYPE current_max = target->get(target, 0, 0);

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            MATRIX_TYPE current_value = target->get(target, i, j);

            current_max = (current_max > current_value) ? current_max : current_value;
        }
    }

    return current_max;
}

/**
 * Get the minimum value stored in a Matrix
 * @param target The Matrix we want to calculate the mmin value on
 */
MATRIX_TYPE MATRIX_METHOD(min)(const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to max\n"); }
        return (MATRIX_TYPE)0;
    }

    MATRIX_TYPE current_min = target->get(target, 0, 0);

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            MATRIX_TYPE current_value = target->get(target, i, j);

            current_min = (current_min < current_value) ? current_min : current_value;
        }
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

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to flatten\n"); }
        return NULL;
    }

    MATRIX_TYPE_NAME *result = NULL;

    if (orientation == ROW) {

        result = MATRIX_METHOD(allocate)(1, target->num_cols * target->num_rows);
    }
    else {

        result = MATRIX_METHOD(allocate)(target->num_cols * target->num_rows, 1);
    }

    if (result == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for flattened Matrix\n"); }
        return NULL;
    }

    size_t index = 0;

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            // Get the addess in memory of the data and put its pointer in the result array
            if (orientation == ROW) {

                result->data[0][index] = target->data[i][j];
            }
            else {

                result->data[index][0] = target->data[i][j];
            }

            // Increment the index
            ++index;
        }
    }

    return result;
}

/**
 * Transpose a Matrix
 * @param target The Matrix we want to transpose
 * @returns Returns another Matrix that has been transposed
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(transpose)(const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to transpose\n"); }
        return NULL;
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(allocate)(target->num_cols, target->num_rows);

    if (result == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for the transpose operation\n"); }
        return NULL;
    }

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            result->data[j][i] = target->data[i][j];
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

    if (self == NULL || target == NULL) { 

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Self or target Matrix is NULL. Cannot add\n"); }
        return NULL;
    }

    if (self->num_cols != target->num_cols || self->num_rows != target->num_rows) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Matrix dimension mismatch. Cannot add\n"); }
        return NULL;
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(allocate)(self->num_rows, self->num_cols);

    if (result == NULL) { 

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for addition Matrix result\n"); }
        return NULL;
    }

    for (size_t i = 0; i < self->num_rows; ++i) {

        for (size_t j = 0; j < self->num_cols; ++j) {

            result->data[i][j] = self->data[i][j] + target->data[i][j];
        }
    }

    return result;
}

/**
 * Add two Matrix instances' contents together, adding to the underlying self Matrix
 * @param self The first Matrix to add values from
 * @param target The second Matrix that we add
 */
void MATRIX_METHOD(add_o)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target) {

    if (self == NULL || target == NULL) { 

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Self or target Matrix is NULL. Cannot add_o\n"); }
        return;
    }

    if (self->num_cols != target->num_cols || self->num_rows != target->num_rows) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Matrix dimension mismatch. Cannot add_o\n"); }
        return;
    }

    for (size_t i = 0; i < self->num_rows; ++i) {

        for (size_t j = 0; j < self->num_cols; ++j) {

            self->data[i][j] += target->data[i][j];
        }
    }
}

/**
 * Subtract two Matrix instances' contents together
 * @param self The first Matrix to subtract values from
 * @param target The second Matrix that we subtract
 * @return Returns another Matrix instance with the result of their subtraction
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(subtract)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target) {

    if (self == NULL || target == NULL) { 

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Self or target Matrix is NULL. Cannot subtract\n"); }
        return NULL;
    }

    if (self->num_cols != target->num_cols || self->num_rows != target->num_rows) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Matrix dimension mismatch. Cannot subtract\n"); }
        return NULL;
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(allocate)(self->num_rows, self->num_cols);

    if (result == NULL) { 

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for subtraxtion Matrix result\n"); }
        return NULL;
    }

    for (size_t i = 0; i < self->num_rows; ++i) {

        for (size_t j = 0; j < self->num_cols; ++j) {

            result->data[i][j] = self->data[i][j] - target->data[i][j];
        }
    }

    return result;
}

/**
 * Subtract two Matrix instances' contents, modifying the self Matrix
 * @param self The first Matrix to subtract values from
 * @param target The second Matrix that we subtract
 */
void MATRIX_METHOD(subtract_o)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target) {

    if (self == NULL || target == NULL) { 

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Self or target Matrix is NULL. Cannot subtract_o\n"); }
        return;
    }

    if (self->num_cols != target->num_cols || self->num_rows != target->num_rows) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Matrix dimension mismatch. Cannot subtract_o\n"); }
        return;
    }

    for (size_t i = 0; i < self->num_rows; ++i) {

        for (size_t j = 0; j < self->num_cols; ++j) {

            self->data[i][j] -= target->data[i][j];
        }
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

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to scale\n"); }
        return NULL;
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(allocate)(target->num_rows, target->num_cols);

    if (result == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for the scale operation\n"); }
        return NULL;
    }

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            result->data[i][j] = target->data[i][j] * scalar;
        }
    }

    return result;
}

/**
 * Scale a Matrix by a value, scaling directly on the original Matrix
 * @param target The Matrix to pull values from
 * @param scalar The value to scale by
 */
void MATRIX_METHOD(scale_o)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE scalar) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to scale_o\n"); }
        return;
    }

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            target->data[i][j] *= scalar;
        }
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

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to add_scalar\n"); }
        return NULL;
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(allocate)(target->num_rows, target->num_cols);

    if (result == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for the add scalar operation\n"); }
        return NULL;
    }

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            result->data[i][j] = target->data[i][j] + scalar;
        }
    }

    return result;
}

/**
 * Add a scalar to a Matrix, adding directly to the underlying Matrix
 * @param target The Matrix to pull values from
 * @param scalar The value to add
 */
void MATRIX_METHOD(add_scalar_o)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE scalar) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to add_scalar_o\n"); }
        return;
    }

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            target->data[i][j] += scalar;
        }
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

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to apply\n"); }
        return NULL;
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(allocate)(target->num_rows, target->num_cols);

    if (result == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for the apply operation\n"); }
        return NULL;
    }

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            result->data[i][j] = (*func)(target->data[i][j]);
        }
    }

    return result;
}

/**
 * Apply a function to a Matrix, modifying the target Matrix itself
 * @param target The Matrix we want to apply a function to
 * @param func A function pointer that we want to use. The pointer must return MATRIX_TYPE
 */
void MATRIX_METHOD(apply_o)(const MATRIX_TYPE_NAME *target, MATRIX_TYPE (*func)(MATRIX_TYPE)) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to apply_o\n"); }
        return;
    }

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            target->data[i][j] = (*func)(target->data[i][j]);
        }
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

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to apply\n"); }
        return NULL;
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(allocate)(target->num_rows, target->num_cols);

    if (result == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for the apply operation\n"); }
        return NULL;
    }

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            result->data[i][j] = (*func)(target->data[i][j], param);
        }
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

    if (self == NULL || target == NULL) { 

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Self or target Matrix is NULL. Cannot multiply\n"); }
        return NULL;
    }

    if (self->num_cols != target->num_cols || self->num_rows != target->num_rows) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Matrix dimension mismatch. Cannot multiply. [%zu x %zu] != [%zu x %zu]\n",
            self->num_rows, self->num_cols, target->num_rows, target->num_cols); }
        
        return NULL;
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(allocate)(self->num_rows, self->num_cols);

    if (result == NULL) { 

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for multiplication Matrix result\n"); }
        return NULL;
    }

    for (size_t i = 0; i < self->num_rows; ++i) {

        for (size_t j = 0; j < self->num_cols; ++j) {

            result->data[i][j] = self->data[i][j] * target->data[i][j];
        }
    }

    return result;
}

/**
 * Multiply two Matrix instances' contents together, keeping contents in Matrix self
 * @param self The first Matrix to multiply
 * @param target The second Matrix that multiply by
 */
void MATRIX_METHOD(multiply_o)(const MATRIX_TYPE_NAME *self, const MATRIX_TYPE_NAME *target) {

    if (self == NULL || target == NULL) { 

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Self or target Matrix is NULL. Cannot multiply_o\n"); }
        return;
    }

    if (self->num_cols != target->num_cols || self->num_rows != target->num_rows) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Matrix dimension mismatch. Cannot multiply_o. [%zu x %zu] != [%zu x %zu]\n",
            self->num_rows, self->num_cols, target->num_rows, target->num_cols); }
        
        return;
    }

    for (size_t i = 0; i < self->num_rows; ++i) {

        for (size_t j = 0; j < self->num_cols; ++j) {

            self->data[i][j] *= target->data[i][j];
        }
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

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to populate\n"); }
        return;
    }

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            target->data[i][j] = value;
        }
    }
}

/**
 * Create a copy of a Matrix
 * @param target The Matrix to copy
 * @returns Returns a copy of a Matrix
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(copy)(const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to copy\n"); }
        return NULL;
    }

    MATRIX_TYPE_NAME *result = MATRIX_METHOD(allocate)(target->num_rows, target->num_cols);

    if (result == NULL) { 

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to allocate memory for copying Matrix\n"); }
        return NULL;
    }

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            result->data[i][j] = target->data[i][j];
        }
    }

    return result;
}

/**
 * Get the sum of all values in a Matrix
 * @param target The Matrix to get the sum of
 * @returns Returns the sum
 */
MATRIX_TYPE MATRIX_METHOD(sum)(const MATRIX_TYPE_NAME *target) {

    if (!MATRIX_METHOD(exists)(target, 0, 0)) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to sum\n"); }
        return (MATRIX_TYPE)0;
    }

    MATRIX_TYPE running_sum = 0;

    for (size_t i = 0; i < target->num_rows; ++i) {

        for (size_t j = 0; j < target->num_cols; ++j) {

            running_sum += target->data[i][j];
        }
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

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Invalid Matrix provided to max_idx\n"); }
        return (MATRIX_TYPE)0;
    }

    size_t max_index = 0;
    MATRIX_TYPE max_value = 0;

    if (target->num_rows == 1 || target->num_cols == 1) {

        max_value = target->max(target);

        if (orientation == ROW) {

            for (size_t i = 0; i < target->num_cols; ++i) {

                if (target->data[0][i] == max_value) { return i; }
            }
        }

        for (size_t i = 0; i < target->num_rows; ++i) {

            if (target->data[i][0] == max_value) { return i; }
        }
    }

    MATRIX_TYPE_NAME *search = NULL;

    if (orientation == ROW) {

        search = target->get_row(target, index);
    }
    else {

        search = target->get_col(target, index);
    }

    max_index = MATRIX_METHOD(max_idx)(search, orientation, 0);

    search->clear(search);

    return max_index;
}

/**
 * Function to allocate a Matrix of any given type. We define this last to get all the right
 * pointers to the functions we reference later on.
 * @param desired_rows Number of rows to allocate
 * @param desired_cols Number of colums to allocate per row
 * @returns Returns a Matrix with all data elements allocated
 */
MATRIX_TYPE_NAME *MATRIX_METHOD(allocate)(size_t desired_rows, size_t desired_cols) {

    // Allocate the memory needed for the Matrix struct
    MATRIX_TYPE_NAME *target = malloc(sizeof(MATRIX_TYPE_NAME));

    if (target == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to allocate Matrix of size %zu x %zu\n", desired_rows, desired_cols); }
        return NULL;
    }

    // Allocate the proper number of rows -- i.e. an array of arrays
    target->data = calloc(desired_rows, sizeof(MATRIX_TYPE *));

    if (target->data == NULL) {

        if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to data array in Matrix of size %zu x %zu\n", desired_rows, desired_cols); }
        
        return NULL;
    }

    // Iterate over the rows and start allocating the proper columns
    for (size_t i = 0; i < desired_rows; ++i) {

        target->data[i] = calloc(desired_cols, sizeof(MATRIX_TYPE));

        if (target->data[i] == NULL) {

            if (MATRIX_DEBUG) { fprintf(stderr, "ERR: Unable to allocate row in Matrix of size %zu x %zu\n", desired_rows, desired_cols); }
            return NULL;
        }
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

    return target;
}

#undef MATRIX_TYPE_NAME
#undef MATRIX_TYPE
#undef MATRIX_CONCAT
#undef MATRIX_METHOD2
#undef MATRIX_METHOD
#undef MATRIX_STRING
