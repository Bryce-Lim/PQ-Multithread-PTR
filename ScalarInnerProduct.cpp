#include "ScalarInnerProduct.h"

size_t compute(const float *points, const size_t point_count, const float *centroids, const size_t centroid_count, const size_t dimension, float *distances) {
    // Input validation
    if (!points || !centroids || !distances || point_count == 0 || centroid_count == 0 || dimension == 0) {
        return 0;
    }
    
    size_t distance_idx = 0;
    
    // For each point
    for (size_t point_idx = 0; point_idx < point_count; ++point_idx) {
        const float *current_point = points + (point_idx * dimension);
        
        // For each centroid
        for (size_t centroid_idx = 0; centroid_idx < centroid_count; ++centroid_idx) {
            const float *current_centroid = centroids + (centroid_idx * dimension);
            
            // Compute scalar inner product (dot product)
            float inner_product = 0.0f;
            for (size_t dim = 0; dim < dimension; ++dim) {
                inner_product += current_point[dim] * current_centroid[dim];
            }
            
            // Store the result
            distances[distance_idx++] = inner_product;
        }
    }
    
    return distance_idx;
}
