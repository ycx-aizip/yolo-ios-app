//
//  PublicTypes.swift
//  AizipFishCount
//
//  Public types for the AizipFishCount API
//  These types provide a clean interface for iOS frontend developers
//

import Foundation
import CoreGraphics
import CoreVideo
import QuartzCore

// MARK: - Detection Results

/// Result of a single frame detection
public struct DetectionResult {
    /// Array of detected objects in this frame
    public let boxes: [DetectionBox]

    /// Frames per second
    public let fps: Double

    /// Inference time in milliseconds
    public let inferenceTime: Double

    /// Timestamp when this frame was processed
    public let timestamp: TimeInterval

    public init(boxes: [DetectionBox], fps: Double, inferenceTime: Double, timestamp: TimeInterval = CACurrentMediaTime()) {
        self.boxes = boxes
        self.fps = fps
        self.inferenceTime = inferenceTime
        self.timestamp = timestamp
    }
}

/// A single detected object with tracking information
public struct DetectionBox {
    /// Unique tracking ID (nil if not tracked)
    public let trackId: Int?

    /// Bounding box in normalized coordinates (0.0-1.0)
    /// Format: CGRect(x: minX, y: minY, width: width, height: height)
    public let boundingBox: CGRect

    /// Detection confidence (0.0-1.0)
    public let confidence: Float

    /// Object class label
    public let label: String

    /// Whether this object has been counted
    public let isCounted: Bool

    public init(trackId: Int?, boundingBox: CGRect, confidence: Float, label: String, isCounted: Bool) {
        self.trackId = trackId
        self.boundingBox = boundingBox
        self.confidence = confidence
        self.label = label
        self.isCounted = isCounted
    }
}

// MARK: - Configuration

/// Detection configuration parameters
public struct DetectionConfig {
    /// Minimum confidence threshold for detections (0.0-1.0)
    public var confidenceThreshold: Float

    /// IoU threshold for non-maximum suppression (0.0-1.0)
    public var iouThreshold: Float

    /// Maximum number of detections to return
    public var maxDetections: Int

    public init(confidenceThreshold: Float = 0.25, iouThreshold: Float = 0.80, maxDetections: Int = 50) {
        self.confidenceThreshold = confidenceThreshold
        self.iouThreshold = iouThreshold
        self.maxDetections = maxDetections
    }
}

// MARK: - Calibration

/// Configuration for auto-calibration
public struct CalibrationConfig {
    /// Enable Phase 1: OpenCV edge detection for threshold calibration
    public var enableThresholdCalibration: Bool

    /// Enable Phase 2: Movement analysis for direction detection
    public var enableDirectionCalibration: Bool

    /// Number of frames for threshold calibration (typically 300 = 10s at 30fps)
    public var thresholdFrames: Int

    /// Number of frames for movement analysis (typically 300 = 10s at 30fps)
    public var movementFrames: Int

    public init(
        enableThresholdCalibration: Bool = false,
        enableDirectionCalibration: Bool = true,
        thresholdFrames: Int = 300,
        movementFrames: Int = 300
    ) {
        self.enableThresholdCalibration = enableThresholdCalibration
        self.enableDirectionCalibration = enableDirectionCalibration
        self.thresholdFrames = thresholdFrames
        self.movementFrames = movementFrames
    }
}

/// Result of calibration process
public struct CalibrationResult {
    /// Detected threshold values in display coordinates (0.0-1.0)
    public let thresholds: [CGFloat]

    /// Auto-detected counting direction (nil if not detected)
    public let detectedDirection: CountingDirection?

    /// Original direction before calibration
    public let originalDirection: CountingDirection

    /// Confidence of direction detection (0.0-1.0)
    public let confidence: Float

    /// Warning messages from calibration
    public let warnings: [String]

    /// Whether threshold calibration was enabled
    public let thresholdCalibrationEnabled: Bool

    /// Whether direction calibration was enabled
    public let directionCalibrationEnabled: Bool

    public init(
        thresholds: [CGFloat],
        detectedDirection: CountingDirection?,
        originalDirection: CountingDirection,
        confidence: Float,
        warnings: [String],
        thresholdCalibrationEnabled: Bool,
        directionCalibrationEnabled: Bool
    ) {
        self.thresholds = thresholds
        self.detectedDirection = detectedDirection
        self.originalDirection = originalDirection
        self.confidence = confidence
        self.warnings = warnings
        self.thresholdCalibrationEnabled = thresholdCalibrationEnabled
        self.directionCalibrationEnabled = directionCalibrationEnabled
    }
}

// MARK: - Errors

/// Errors that can occur during fish counting operations
public enum FishCountError: Error, LocalizedError {
    case modelNotFound(String)
    case invalidConfiguration(String)
    case calibrationFailed(String)
    case processingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Model not found: \(name). Please ensure the CoreML model is in your app's FishCountModels folder."
        case .invalidConfiguration(let message):
            return "Invalid configuration: \(message)"
        case .calibrationFailed(let message):
            return "Calibration failed: \(message)"
        case .processingFailed(let message):
            return "Frame processing failed: \(message)"
        }
    }
}
