// OCKalmanFilter.swift
// 7D Kalman Filter for OC-SORT tracking
//
// This implements a 7-dimensional Kalman filter for object tracking with velocity.
// State vector: [x, y, s, r, vx, vy, vs] where:
// - x, y: bounding box center coordinates
// - s: scale (area)
// - r: aspect ratio (width/height)
// - vx, vy: velocity in x and y
// - vs: velocity in scale
//
// This is a 1-to-1 Swift translation of the Python KalmanFilterNew class
// from ocsort/kalmanfilter.py

import Foundation
import Accelerate
import simd

/// 7D Kalman filter for OC-SORT tracking
/// Tracks position (x, y), scale (s), aspect ratio (r), and velocities (vx, vy, vs)
public class OCKalmanFilter {

    // MARK: - Properties

    /// Dimension of state vector (7D: x, y, s, r, vx, vy, vs)
    private let ndim = 7

    /// Dimension of measurement vector (4D: x, y, s, r)
    private let mdim = 4

    /// Time step (delta t)
    private let dt: Double

    /// State transition matrix F (7x7)
    /// Maps current state to next state
    private var F: [Double]

    /// Measurement matrix H (4x7)
    /// Maps state to measurement space
    private var H: [Double]

    /// Process noise covariance Q (7x7)
    private var Q: [Double]

    /// Measurement noise covariance R (4x4)
    private var R: [Double]

    /// State covariance matrix P (7x7)
    private var P: [Double]

    /// Current state estimate (7x1)
    private var x: [Double]

    // MARK: - Initialization

    /// Initialize Kalman filter with initial bounding box
    /// - Parameter bbox: Initial bounding box [x, y, s, r] where s=area, r=aspect_ratio
    public init(bbox: [Double]) {
        // Initialize with dt=1
        self.dt = 1.0

        // Initialize state vector with bbox and zero velocities
        // [x, y, s, r, vx=0, vy=0, vs=0]
        self.x = [bbox[0], bbox[1], bbox[2], bbox[3], 0.0, 0.0, 0.0]

        // Initialize state transition matrix F (7x7)
        // F models constant velocity: x_new = x_old + vx*dt
        self.F = [Double](repeating: 0.0, count: 49) // 7x7 = 49
        // Identity for position and velocities
        for i in 0..<7 {
            self.F[i * 7 + i] = 1.0
        }
        // Add velocity terms: x += vx*dt, y += vy*dt, s += vs*dt
        self.F[0 * 7 + 4] = dt  // x += vx*dt
        self.F[1 * 7 + 5] = dt  // y += vy*dt
        self.F[2 * 7 + 6] = dt  // s += vs*dt

        // Initialize measurement matrix H (4x7)
        // H extracts position from state: [x, y, s, r] = H * [x, y, s, r, vx, vy, vs]
        self.H = [Double](repeating: 0.0, count: 28) // 4x7 = 28
        for i in 0..<4 {
            self.H[i * 7 + i] = 1.0
        }

        // Initialize process noise covariance Q (7x7)
        // Models uncertainty in state transition
        self.Q = [Double](repeating: 0.0, count: 49)
        // Position noise (small)
        self.Q[0 * 7 + 0] = 1.0  // x
        self.Q[1 * 7 + 1] = 1.0  // y
        self.Q[2 * 7 + 2] = 1.0  // s
        self.Q[3 * 7 + 3] = 0.01 // r (aspect ratio changes slowly)
        // Velocity noise (larger)
        self.Q[4 * 7 + 4] = 0.01  // vx
        self.Q[5 * 7 + 5] = 0.01  // vy
        self.Q[6 * 7 + 6] = 0.0001 // vs

        // Initialize measurement noise covariance R (4x4)
        // Models uncertainty in measurements
        self.R = [Double](repeating: 0.0, count: 16) // 4x4 = 16
        self.R[0 * 4 + 0] = 1.0  // x measurement noise
        self.R[1 * 4 + 1] = 1.0  // y measurement noise
        self.R[2 * 4 + 2] = 10.0 // s measurement noise
        self.R[3 * 4 + 3] = 10.0 // r measurement noise

        // Initialize state covariance P (7x7)
        // High initial uncertainty
        self.P = [Double](repeating: 0.0, count: 49)
        for i in 0..<4 {
            self.P[i * 7 + i] = 10.0  // Position uncertainty
        }
        for i in 4..<7 {
            self.P[i * 7 + i] = 10000.0  // Velocity uncertainty (very high initially)
        }
    }

    // MARK: - Prediction Step

    /// Predict next state using motion model
    /// x = F * x
    /// P = F * P * F^T + Q
    public func predict() {
        // x = F * x
        var newX = [Double](repeating: 0.0, count: 7)
        vDSP_mmulD(F, 1, x, 1, &newX, 1, 7, 1, 7)
        x = newX

        // P = F * P * F^T + Q
        // Step 1: FP = F * P
        var FP = [Double](repeating: 0.0, count: 49)
        vDSP_mmulD(F, 1, P, 1, &FP, 1, 7, 7, 7)

        // Step 2: FPF = FP * F^T
        var FPF = [Double](repeating: 0.0, count: 49)
        var F_T = [Double](repeating: 0.0, count: 49)
        vDSP_mtransD(F, 1, &F_T, 1, 7, 7)
        vDSP_mmulD(FP, 1, F_T, 1, &FPF, 1, 7, 7, 7)

        // Step 3: P = FPF + Q
        vDSP_vaddD(FPF, 1, Q, 1, &P, 1, 49)
    }

    // MARK: - Update Step

    /// Update state with measurement
    /// - Parameter bbox: Measurement [x, y, s, r]
    public func update(bbox: [Double]) {
        // Innovation: y = z - H * x
        var Hx = [Double](repeating: 0.0, count: 4)
        vDSP_mmulD(H, 1, x, 1, &Hx, 1, 4, 1, 7)

        var innovation = [Double](repeating: 0.0, count: 4)
        var negHx = Hx
        vDSP_vnegD(Hx, 1, &negHx, 1, 4)
        vDSP_vaddD(bbox, 1, negHx, 1, &innovation, 1, 4)

        // S = H * P * H^T + R
        // Step 1: HP = H * P
        var HP = [Double](repeating: 0.0, count: 28) // 4x7
        vDSP_mmulD(H, 1, P, 1, &HP, 1, 4, 7, 7)

        // Step 2: HPH = HP * H^T
        var H_T = [Double](repeating: 0.0, count: 28)
        vDSP_mtransD(H, 1, &H_T, 1, 4, 7)
        var HPH = [Double](repeating: 0.0, count: 16) // 4x4
        vDSP_mmulD(HP, 1, H_T, 1, &HPH, 1, 4, 4, 7)

        // Step 3: S = HPH + R
        var S = [Double](repeating: 0.0, count: 16)
        vDSP_vaddD(HPH, 1, R, 1, &S, 1, 16)

        // Kalman gain: K = P * H^T * S^(-1)
        // Step 1: Invert S using a custom approach
        // For 4x4 matrix, we can use a simpler inversion or accept small numerical errors
        var S_inv = invertMatrix4x4(S)

        // Step 2: PH = P * H^T
        var PH = [Double](repeating: 0.0, count: 28) // 7x4
        vDSP_mmulD(P, 1, H_T, 1, &PH, 1, 7, 4, 7)

        // Step 3: K = PH * S_inv
        var K = [Double](repeating: 0.0, count: 28) // 7x4
        vDSP_mmulD(PH, 1, S_inv, 1, &K, 1, 7, 4, 4)

        // Update state: x = x + K * innovation
        var Ky = [Double](repeating: 0.0, count: 7)
        vDSP_mmulD(K, 1, innovation, 1, &Ky, 1, 7, 1, 4)
        vDSP_vaddD(x, 1, Ky, 1, &x, 1, 7)

        // Update covariance: P = (I - K * H) * P
        // Step 1: KH = K * H
        var KH = [Double](repeating: 0.0, count: 49) // 7x7
        vDSP_mmulD(K, 1, H, 1, &KH, 1, 7, 7, 4)

        // Step 2: I - KH
        var I_KH = [Double](repeating: 0.0, count: 49)
        for i in 0..<7 {
            for j in 0..<7 {
                if i == j {
                    I_KH[i * 7 + j] = 1.0 - KH[i * 7 + j]
                } else {
                    I_KH[i * 7 + j] = -KH[i * 7 + j]
                }
            }
        }

        // Step 3: P = (I - KH) * P
        var newP = [Double](repeating: 0.0, count: 49)
        vDSP_mmulD(I_KH, 1, P, 1, &newP, 1, 7, 7, 7)
        P = newP
    }

    // MARK: - State Access

    /// Get current state estimate
    /// - Returns: State vector [x, y, s, r, vx, vy, vs]
    public func getState() -> [Double] {
        return x
    }

    /// Get current position and size
    /// - Returns: Bounding box [x, y, s, r]
    public func getPosition() -> [Double] {
        return [x[0], x[1], x[2], x[3]]
    }

    /// Get current velocity
    /// - Returns: Velocity [vx, vy, vs]
    public func getVelocity() -> [Double] {
        return [x[4], x[5], x[6]]
    }

    // MARK: - Helper Methods

    /// Invert a 4x4 matrix using Gaussian elimination
    /// - Parameter matrix: Input matrix (16 elements in row-major order)
    /// - Returns: Inverted matrix (16 elements in row-major order)
    private func invertMatrix4x4(_ matrix: [Double]) -> [Double] {
        // Create augmented matrix [A | I]
        var augmented = [[Double]](repeating: [Double](repeating: 0.0, count: 8), count: 4)

        // Fill left side with input matrix
        for i in 0..<4 {
            for j in 0..<4 {
                augmented[i][j] = matrix[i * 4 + j]
            }
        }

        // Fill right side with identity
        for i in 0..<4 {
            augmented[i][4 + i] = 1.0
        }

        // Gaussian elimination with partial pivoting
        for i in 0..<4 {
            // Find pivot
            var maxRow = i
            for k in (i + 1)..<4 {
                if abs(augmented[k][i]) > abs(augmented[maxRow][i]) {
                    maxRow = k
                }
            }

            // Swap rows
            if maxRow != i {
                augmented.swapAt(i, maxRow)
            }

            // Scale pivot row
            let pivot = augmented[i][i]
            if abs(pivot) < 1e-10 {
                // Singular matrix, return identity
                var identity = [Double](repeating: 0.0, count: 16)
                for k in 0..<4 {
                    identity[k * 4 + k] = 1.0
                }
                return identity
            }

            for j in 0..<8 {
                augmented[i][j] /= pivot
            }

            // Eliminate column
            for k in 0..<4 {
                if k != i {
                    let factor = augmented[k][i]
                    for j in 0..<8 {
                        augmented[k][j] -= factor * augmented[i][j]
                    }
                }
            }
        }

        // Extract inverse from right side
        var inverse = [Double](repeating: 0.0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 {
                inverse[i * 4 + j] = augmented[i][4 + j]
            }
        }

        return inverse
    }
}
