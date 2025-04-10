// from Aizip
//
//  KalmanFilter.swift
//  YOLO
//
//  Kalman filter implementation for object tracking.
//  Implements 8-dimensional Kalman filter following ByteTrack.

import Foundation
import Accelerate

/**
 * KalmanFilter
 *
 * Maps to Python Implementation:
 * - Primary Correspondence: `KalmanFilterXYAH` class in `ultralytics/trackers/kalman_filter.py`
 * - Core functionality:
 *   - Provides motion prediction for object tracking
 *   - Implements 8-dimensional state vector tracking position and velocity
 *   - State: [x, y, aspect_ratio, height, vx, vy, va, vh]
 *
 * Implementation Details:
 * - Uses the same mathematical formulation as Python implementation
 * - Implements standard Kalman filter prediction and update steps
 * - Uses Accelerate framework for matrix operations (replacing numpy)
 * - Matches the Python implementation's handling of process and measurement noise
 */
/// Kalman filter for motion prediction in object tracking
public class KalmanFilter: @unchecked Sendable {
    // MARK: - Constants
    
    /// State transition matrix dimension
    private let ndim: Int = 8
    
    /// Measurement dimension (x, y, aspect ratio, height)
    private let measDim: Int = 4
    
    /// Standard weight position (corresponds to _std_weight_position in Python)
    private let stdWeightPosition: Float = 1.0 / 20.0
    
    /// Standard weight velocity (corresponds to _std_weight_velocity in Python)
    private let stdWeightVelocity: Float = 1.0 / 160.0
    
    // MARK: - Matrices for Kalman Filter
    
    /// Motion model matrix (8x8)
    private var motionMat: [Float]
    
    /// Update matrix, from measurement to state (8x4)
    private var updateMat: [Float]
    
    /// Measurement matrix, from state to measurement (4x8)
    private var measurementMat: [Float]
    
    // MARK: - Initialization
    
    public init() {
        // Initialize motion model matrix (F)
        // [ 1 0 0 0 1 0 0 0 ]
        // [ 0 1 0 0 0 1 0 0 ]
        // [ 0 0 1 0 0 0 1 0 ]
        // [ 0 0 0 1 0 0 0 1 ]
        // [ 0 0 0 0 1 0 0 0 ]
        // [ 0 0 0 0 0 1 0 0 ]
        // [ 0 0 0 0 0 0 1 0 ]
        // [ 0 0 0 0 0 0 0 1 ]
        motionMat = Array(repeating: 0, count: ndim * ndim)
        for i in 0..<ndim {
            motionMat[i * ndim + i] = 1.0  // Diagonal
        }
        for i in 0..<4 {
            motionMat[i * ndim + i + 4] = 1.0  // Velocity components
        }
        
        // Initialize update matrix (U)
        updateMat = Array(repeating: 0, count: ndim * measDim)
        for i in 0..<measDim {
            updateMat[i * measDim + i] = 1.0
        }
        
        // Initialize measurement matrix (H)
        measurementMat = Array(repeating: 0, count: measDim * ndim)
        for i in 0..<measDim {
            measurementMat[i * ndim + i] = 1.0
        }
    }
    
    // MARK: - Kalman Filter Operations
    
    /**
     * initiate
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `initiate()` method in `KalmanFilterXYAH` class
     * - Creates initial mean and covariance from first detection
     * - Sets initial velocity components to zero
     */
    /// Initialize Kalman filter with first measurement
    /// - Parameter measurement: Initial measurement [x, y, aspect_ratio, height]
    /// - Returns: Tuple of mean state vector and covariance matrix
    public func initiate(measurement: [Float]) -> ([Float], [Float]) {
        // Initial state: [x, y, a, h, vx, vy, va, vh]
        var mean = Array(repeating: 0.0 as Float, count: ndim)
        
        // First 4 elements of mean are the measurement values
        for i in 0..<min(measurement.count, 4) {
            mean[i] = measurement[i]
        }
        
        // Velocities are set to 0
        // mean[4...7] are already initialized to 0
        
        // Following Python implementation for initial covariance
        let std = [
            2 * stdWeightPosition * measurement[3],      // x position uncertainty
            2 * stdWeightPosition * measurement[3],      // y position uncertainty
            1e-2 as Float,                               // aspect ratio uncertainty
            2 * stdWeightPosition * measurement[3],      // height uncertainty
            10 * stdWeightVelocity * measurement[3],     // x velocity uncertainty
            10 * stdWeightVelocity * measurement[3],     // y velocity uncertainty
            1e-5 as Float,                               // aspect ratio change uncertainty
            10 * stdWeightVelocity * measurement[3]      // height change uncertainty
        ]
        
        // Initial covariance matrix: diagonal with calculated variances
        var covariance = Array(repeating: 0.0 as Float, count: ndim * ndim)
        
        // Set diagonal elements based on calculated standard deviations
        for i in 0..<ndim {
            covariance[i * ndim + i] = std[i] * std[i]
        }
        
        return (mean, covariance)
    }
    
    /**
     * predict
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `predict()` method in `KalmanFilterXYAH` class
     * - Predicts new state based on motion model
     * - Applies process noise based on current state
     */
    /// Predict new state based on motion model
    /// - Parameters:
    ///   - mean: Current mean state
    ///   - covariance: Current covariance matrix
    /// - Returns: Predicted mean and covariance
    public func predict(mean: [Float], covariance: [Float]) -> ([Float], [Float]) {
        // Calculate process noise exactly as in Python implementation
        let std = calculateProcessNoise(mean: mean)
        
        // Calculate motion noise covariance matrix Q
        var motionCov = Array(repeating: 0.0 as Float, count: ndim * ndim)
        for i in 0..<ndim {
            motionCov[i * ndim + i] = std[i] * std[i]
        }
        
        // Predict new mean: x' = F * x
        var newMean = Array(repeating: 0.0 as Float, count: ndim)
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(ndim), Int32(ndim),
                   1.0, motionMat, Int32(ndim), mean, 1, 0.0, &newMean, 1)
        
        // Predict new covariance: P' = F * P * F^T + Q
        var newCovariance = Array(repeating: 0.0 as Float, count: ndim * ndim)
        
        // temp = F * P
        var temp = Array(repeating: 0.0 as Float, count: ndim * ndim)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   Int32(ndim), Int32(ndim), Int32(ndim),
                   1.0, motionMat, Int32(ndim), covariance, Int32(ndim),
                   0.0, &temp, Int32(ndim))
        
        // P' = temp * F^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                   Int32(ndim), Int32(ndim), Int32(ndim),
                   1.0, temp, Int32(ndim), motionMat, Int32(ndim),
                   0.0, &newCovariance, Int32(ndim))
        
        // Add motion noise: P' = P' + Q
        for i in 0..<ndim*ndim {
            newCovariance[i] += motionCov[i]
        }
        
        return (newMean, newCovariance)
    }
    
    /**
     * update
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `update()` method in `KalmanFilterXYAH` class
     * - Updates state estimate with new measurement using standard Kalman update
     * - Calculates Kalman gain and applies it to update mean and covariance
     */
    /// Update state with new measurement
    /// - Parameters:
    ///   - mean: Current mean state
    ///   - covariance: Current covariance matrix
    ///   - measurement: New measurement [x, y, aspect_ratio, height]
    /// - Returns: Updated mean and covariance
    public func update(mean: [Float], covariance: [Float], measurement: [Float]) -> ([Float], [Float]) {
        // First project the state to measurement space
        let (projectedMean, projectedCov) = project(mean: mean, covariance: covariance)
        
        // Calculate innovation covariance inverse (S^-1)
        var innovationCovInv = Array(repeating: 0.0 as Float, count: measDim * measDim)
        
        // Copy projectedCov to innovationCovInv to prepare for inversion
        for i in 0..<measDim*measDim {
            innovationCovInv[i] = projectedCov[i]
        }
        
        // Invert innovation covariance
        inverseMatrix(matrix: &innovationCovInv, n: measDim)
        
        // Calculate Kalman gain: K = P * H^T * S^-1
        var kalmanGain = Array(repeating: 0.0 as Float, count: ndim * measDim)
        
        // temp = P * H^T
        var temp = Array(repeating: 0.0 as Float, count: ndim * measDim)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                   Int32(ndim), Int32(measDim), Int32(ndim),
                   1.0, covariance, Int32(ndim), measurementMat, Int32(ndim),
                   0.0, &temp, Int32(measDim))
        
        // K = temp * S^-1
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   Int32(ndim), Int32(measDim), Int32(measDim),
                   1.0, temp, Int32(measDim), innovationCovInv, Int32(measDim),
                   0.0, &kalmanGain, Int32(measDim))
        
        // Calculate innovation: y = z - H*x
        var innovation = Array(repeating: 0.0 as Float, count: measDim)
        for i in 0..<min(measurement.count, measDim) {
            innovation[i] = measurement[i] - projectedMean[i]
        }
        
        // Update state mean: x' = x + K * y
        var newMean = Array(repeating: 0.0 as Float, count: ndim)
        for i in 0..<ndim {
            newMean[i] = mean[i]
        }
        
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(ndim), Int32(measDim),
                   1.0, kalmanGain, Int32(measDim), innovation, 1, 1.0, &newMean, 1)
        
        // Update state covariance: P' = (I - K * H) * P
        // Identity matrix
        var identity = Array(repeating: 0.0 as Float, count: ndim * ndim)
        for i in 0..<ndim {
            identity[i * ndim + i] = 1.0
        }
        
        // temp = K * H
        var temp2 = Array(repeating: 0.0 as Float, count: ndim * ndim)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   Int32(ndim), Int32(ndim), Int32(measDim),
                   1.0, kalmanGain, Int32(measDim), measurementMat, Int32(ndim),
                   0.0, &temp2, Int32(ndim))
        
        // I - K * H
        var term = Array(repeating: 0.0 as Float, count: ndim * ndim)
        for i in 0..<ndim*ndim {
            term[i] = identity[i] - temp2[i]
        }
        
        // P' = (I - K * H) * P
        var newCovariance = Array(repeating: 0.0 as Float, count: ndim * ndim)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   Int32(ndim), Int32(ndim), Int32(ndim),
                   1.0, term, Int32(ndim), covariance, Int32(ndim),
                   0.0, &newCovariance, Int32(ndim))
        
        return (newMean, newCovariance)
    }
    
    /**
     * project
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `project()` method in `KalmanFilterXYAH` class
     * - Projects state distribution to measurement space
     * - Applies measurement noise
     */
    /// Project state distribution to measurement space
    /// - Parameters:
    ///   - mean: Current mean state
    ///   - covariance: Current covariance matrix
    /// - Returns: Projected mean and covariance in measurement space
    public func project(mean: [Float], covariance: [Float]) -> ([Float], [Float]) {
        // Calculate measurement noise exactly as in Python implementation
        let std = calculateMeasurementNoise(mean: mean)
        
        // Calculate innovation covariance
        var innovationCov = Array(repeating: 0.0 as Float, count: measDim * measDim)
        for i in 0..<measDim {
            innovationCov[i * measDim + i] = std[i] * std[i]
        }
        
        // Project state to measurement space: z = H * x
        var projectedMean = Array(repeating: 0.0 as Float, count: measDim)
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(measDim), Int32(ndim),
                   1.0, measurementMat, Int32(ndim), mean, 1, 0.0, &projectedMean, 1)
        
        // Project covariance to measurement space: S = H * P * H^T
        var projectedCov = Array(repeating: 0.0 as Float, count: measDim * measDim)
        
        // temp = H * P
        var temp = Array(repeating: 0.0 as Float, count: measDim * ndim)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   Int32(measDim), Int32(ndim), Int32(ndim),
                   1.0, measurementMat, Int32(ndim), covariance, Int32(ndim),
                   0.0, &temp, Int32(ndim))
        
        // S = temp * H^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                   Int32(measDim), Int32(measDim), Int32(ndim),
                   1.0, temp, Int32(ndim), measurementMat, Int32(ndim),
                   0.0, &projectedCov, Int32(measDim))
        
        // Add innovation covariance: S = S + R
        for i in 0..<measDim*measDim {
            projectedCov[i] += innovationCov[i]
        }
        
        return (projectedMean, projectedCov)
    }
    
    // MARK: - Helper methods
    
    /**
     * calculateProcessNoise
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: Logic in `predict()` method in `KalmanFilterXYAH` class
     * - Calculates process noise based on current state
     * - Uses same scaling factors as Python implementation
     */
    /// Calculate process noise based on current state
    /// - Parameter mean: Current mean state
    /// - Returns: Standard deviations for process noise
    private func calculateProcessNoise(mean: [Float]) -> [Float] {
        // Extract height for scaling
        let height = mean[3]
        
        // Process noise parameters as in Python implementation
        return [
            stdWeightPosition * height,        // position x noise
            stdWeightPosition * height,        // position y noise
            1e-2 as Float,                     // aspect ratio noise
            stdWeightPosition * height,        // height noise
            stdWeightVelocity * height,        // velocity x noise
            stdWeightVelocity * height,        // velocity y noise
            1e-5 as Float,                     // velocity aspect ratio noise
            stdWeightVelocity * height         // velocity height noise
        ]
    }
    
    /**
     * calculateMeasurementNoise
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: Logic in `project()` method in `KalmanFilterXYAH` class
     * - Calculates measurement noise based on current state
     * - Uses same scaling factors as Python implementation
     */
    /// Calculate measurement noise based on current state
    /// - Parameter mean: Current mean state
    /// - Returns: Standard deviations for measurement noise
    private func calculateMeasurementNoise(mean: [Float]) -> [Float] {
        // Use the object's height to scale measurement uncertainty as in Python
        return [
            stdWeightPosition * mean[3],    // x position noise
            stdWeightPosition * mean[3],    // y position noise
            1e-1 as Float,                  // aspect ratio noise
            stdWeightPosition * mean[3]     // height noise
        ]
    }
    
    /// Simple matrix inversion function for small matrices
    /// Not optimized, but sufficient for a 4x4 matrix
    /// - Parameters:
    ///   - matrix: Matrix to invert (in place)
    ///   - n: Matrix dimension
    private func inverseMatrix(matrix: inout [Float], n: Int) {
        // For a small 4x4 matrix, we'll use a basic Gauss-Jordan elimination
        // Create augmented matrix [A|I]
        var augmented = Array(repeating: 0.0 as Float, count: n * n * 2)
        
        // Fill the augmented matrix
        for i in 0..<n {
            for j in 0..<n {
                augmented[i * (2 * n) + j] = matrix[i * n + j]
            }
            augmented[i * (2 * n) + n + i] = 1.0
        }
        
        // Gauss-Jordan elimination
        for i in 0..<n {
            // Find pivot
            var maxVal = abs(augmented[i * (2 * n) + i])
            var maxRow = i
            for k in i+1..<n {
                let val = abs(augmented[k * (2 * n) + i])
                if val > maxVal {
                    maxVal = val
                    maxRow = k
                }
            }
            
            // Swap rows if needed
            if maxRow != i {
                for j in 0..<(2 * n) {
                    let temp = augmented[i * (2 * n) + j]
                    augmented[i * (2 * n) + j] = augmented[maxRow * (2 * n) + j]
                    augmented[maxRow * (2 * n) + j] = temp
                }
            }
            
            // Divide the pivot row by the pivot element
            let pivot = augmented[i * (2 * n) + i]
            for j in 0..<(2 * n) {
                augmented[i * (2 * n) + j] /= pivot
            }
            
            // Subtract from the other rows
            for k in 0..<n {
                if k != i {
                    let factor = augmented[k * (2 * n) + i]
                    for j in 0..<(2 * n) {
                        augmented[k * (2 * n) + j] -= factor * augmented[i * (2 * n) + j]
                    }
                }
            }
        }
        
        // Extract the inverted matrix
        for i in 0..<n {
            for j in 0..<n {
                matrix[i * n + j] = augmented[i * (2 * n) + n + j]
            }
        }
    }
} 