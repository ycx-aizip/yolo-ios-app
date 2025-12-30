//
//  FishCountSession.swift
//  AizipFishCount
//
//  Main protocol for fish counting session
//  This is the primary interface that iOS frontend developers will use
//

import Foundation
import CoreGraphics
import CoreVideo

// MARK: - Session Protocol

/// Main interface for fish counting operations
@MainActor
public protocol FishCountSession: AnyObject {

    // MARK: - Frame Processing

    /// Process a single video frame
    ///
    /// This is the core method for processing frames from any source (Camera, Video, UVC, etc.)
    /// Results are returned asynchronously via the delegate
    ///
    /// - Parameter pixelBuffer: The frame to process (CVPixelBuffer)
    func processFrame(_ pixelBuffer: CVPixelBuffer)

    // MARK: - Counting Configuration

    /// Set counting threshold positions
    ///
    /// - Parameter values: Array of threshold values in display coordinates (0.0-1.0)
    ///   - For vertical counting: 0.0 = top, 1.0 = bottom
    ///   - For horizontal counting: 0.0 = left, 1.0 = right
    ///   - Typically 2 values, e.g. [0.3, 0.7]
    func setThresholds(_ values: [CGFloat])

    /// Set the direction fish are expected to move
    ///
    /// - Parameter direction: The counting direction (topToBottom, bottomToTop, leftToRight, rightToLeft)
    func setCountingDirection(_ direction: CountingDirection)

    /// Get current count of fish that have crossed thresholds
    ///
    /// - Returns: Total count
    func getCount() -> Int

    /// Reset the count to zero
    func resetCount()

    /// Get current threshold values in display coordinates
    ///
    /// - Returns: Array of threshold values (0.0-1.0)
    func getThresholds() -> [CGFloat]

    /// Get current counting direction
    ///
    /// - Returns: The current counting direction
    func getCountingDirection() -> CountingDirection

    // MARK: - Detection Configuration

    /// Get current detection configuration
    ///
    /// - Returns: Current detection parameters
    func getDetectionConfig() -> DetectionConfig

    /// Update detection configuration
    ///
    /// - Parameter config: New detection parameters
    func updateDetectionConfig(_ config: DetectionConfig)

    // MARK: - Auto-Calibration

    /// Enable or disable auto-calibration
    ///
    /// Auto-calibration can automatically detect:
    /// - Threshold positions (Phase 1, using OpenCV edge detection)
    /// - Fish movement direction (Phase 2, using YOLO tracking)
    ///
    /// - Parameters:
    ///   - enabled: Whether to enable calibration
    ///   - config: Optional calibration configuration (uses defaults if nil)
    func setAutoCalibration(enabled: Bool, config: CalibrationConfig?)

    /// Check if auto-calibration is currently active
    ///
    /// - Returns: True if calibration is in progress
    func isCalibrating() -> Bool

    /// Get calibration progress
    ///
    /// - Returns: Tuple of (current frame, total frames), or nil if not calibrating
    func getCalibrationProgress() -> (current: Int, total: Int)?

    // MARK: - Session Management

    /// Get the model name being used
    ///
    /// - Returns: Name of the CoreML model
    func getModelName() -> String

    /// Set the delegate for receiving callbacks
    ///
    /// - Parameter delegate: The delegate to receive session events
    func setDelegate(_ delegate: FishCountSessionDelegate?)
}

// MARK: - Delegate Protocol

/// Delegate protocol for receiving fish counting session events
@MainActor
public protocol FishCountSessionDelegate: AnyObject {

    /// Called when new detections are available from a processed frame
    ///
    /// - Parameters:
    ///   - session: The session that produced the result
    ///   - result: Detection result containing boxes, FPS, and timing info
    func session(_ session: FishCountSession, didDetect result: DetectionResult)

    /// Called when the fish count changes
    ///
    /// - Parameters:
    ///   - session: The session tracking the count
    ///   - count: The new total count
    func session(_ session: FishCountSession, countDidChange count: Int)

    /// Called during calibration to report progress
    ///
    /// - Parameters:
    ///   - session: The session being calibrated
    ///   - current: Current frame number
    ///   - total: Total frames needed for calibration
    func session(_ session: FishCountSession, calibrationProgress current: Int, total: Int)

    /// Called when calibration completes
    ///
    /// - Parameters:
    ///   - session: The session that was calibrated
    ///   - summary: Calibration results including thresholds and detected direction
    func session(_ session: FishCountSession, calibrationDidComplete summary: CalibrationResult)

    /// Called when an error occurs
    ///
    /// - Parameters:
    ///   - session: The session that encountered the error
    ///   - error: The error that occurred
    func session(_ session: FishCountSession, didFailWithError error: FishCountError)
}

// MARK: - Optional Delegate Methods

/// Extension to make all delegate methods optional
@MainActor
public extension FishCountSessionDelegate {
    func session(_ session: FishCountSession, didDetect result: DetectionResult) {}
    func session(_ session: FishCountSession, countDidChange count: Int) {}
    func session(_ session: FishCountSession, calibrationProgress current: Int, total: Int) {}
    func session(_ session: FishCountSession, calibrationDidComplete summary: CalibrationResult) {}
    func session(_ session: FishCountSession, didFailWithError error: FishCountError) {}
}
