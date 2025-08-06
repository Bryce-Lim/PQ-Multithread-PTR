#ifndef SCALAR_INNER_PRODUCT_H
#define SCALAR_INNER_PRODUCT_H

#include <cstddef>

/**
 * Computes scalar inner products (dot products) between points and centroids.
 * 
 * @param points Pointer to array of points, stored as [point0_dim0, point0_dim1, ..., point0_dimN-1, point1_dim0, ...]
 * @param point_count Number of points
 * @param centroids Pointer to array of centroids, stored as [centroid0_dim0, centroid0_dim1, ..., centroid0_dimN-1, centroid1_dim0, ...]
 * @param centroid_count Number of centroids
 * @param dimension Dimensionality of each point and centroid
 * @param distances Output array to store computed distances, must be pre-allocated with size >= point_count * centroid_count
 * 
 * @return Number of distances computed and stored (should equal point_count * centroid_count)
 */
size_t compute(const float *points, const size_t point_count, const float *centroids, const size_t centroid_count, const size_t dimension, float *distances);

#endif // SCALAR_INNER_PRODUCT_H
