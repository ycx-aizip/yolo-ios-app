// Swift file for CountingDirection enum and calibration utilities
// This file is shared between tracking and counting functionality

import Foundation
import AVFoundation
import CoreVideo
import UIKit
import CoreGraphics
// Import OpenCVWrapper directly since we're in the same module
// No need for @_exported import since we're in the same module

/// Direction for counting objects (fish)
public enum CountingDirection {
    case topToBottom
    case bottomToTop
    case leftToRight
    case rightToLeft
}

// MARK: - Auto-Calibration Configuration

/// Configuration for auto-calibration phases
public final class AutoCalibrationConfig: @unchecked Sendable {
    /// Shared singleton instance for easy developer configuration
    public static let shared = AutoCalibrationConfig()
    
    /// Enable/disable threshold detection phase (Phase 1: OpenCV edge detection)
    public var isThresholdCalibrationEnabled: Bool = true
    
    /// Enable/disable direction detection phase (Phase 2: Movement analysis)
    public var isDirectionCalibrationEnabled: Bool = true
    
    /// Number of frames for threshold detection phase
    public var thresholdCalibrationFrames: Int = 300
    
    /// Number of frames for movement analysis phase
    public var movementAnalysisFrames: Int = 300
    
    /// Minimum track length for movement analysis (frames)
    public var minTrackLengthForAnalysis: Int = 10
    
    /// Minimum confidence for movement analysis
    public var minConfidenceForAnalysis: Float = 0.6
    
    /// Percentage threshold for direction decision (60% = 0.6)
    public var directionDecisionThreshold: Float = 0.6
    
    private init() {}
    
    /// Quick setup for common configurations
    public func setConfiguration(
        thresholdCalibration: Bool = true,
        directionCalibration: Bool = true,
        thresholdFrames: Int = 300,
        movementFrames: Int = 300
    ) {
        isThresholdCalibrationEnabled = thresholdCalibration
        isDirectionCalibrationEnabled = directionCalibration
        thresholdCalibrationFrames = thresholdFrames
        movementAnalysisFrames = movementFrames
        
        print("AutoCalibrationConfig: Updated - threshold: \(thresholdCalibration), direction: \(directionCalibration)")
    }
    
    /// Get total calibration frames needed
    public var totalCalibrationFrames: Int {
        var total = 0
        if isThresholdCalibrationEnabled {
            total += thresholdCalibrationFrames
        }
        if isDirectionCalibrationEnabled {
            total += movementAnalysisFrames
        }
        return total
    }
}

// MARK: - Movement Analysis Data Structures

/// Data structure for tracking fish movement patterns
public struct FishMovementData {
    let trackId: Int
    var positions: [(x: CGFloat, y: CGFloat, timestamp: Double)]
    var movementVectors: [(dx: CGFloat, dy: CGFloat)]
    var confidence: Float
    var trackLength: Int
    var consistencyScore: Float
    
    /// Initialize with first position
    init(trackId: Int, initialPosition: (x: CGFloat, y: CGFloat), confidence: Float) {
        self.trackId = trackId
        self.positions = [(initialPosition.x, initialPosition.y, CACurrentMediaTime())]
        self.movementVectors = []
        self.confidence = confidence
        self.trackLength = 1
        self.consistencyScore = 0.0
    }
    
    /// Add a new position and calculate movement vector
    mutating func addPosition(_ position: (x: CGFloat, y: CGFloat), confidence: Float) {
        let timestamp = CACurrentMediaTime()
        
        // Calculate movement vector from last position
        if let lastPos = positions.last {
            let dx = position.x - lastPos.x
            let dy = position.y - lastPos.y
            movementVectors.append((dx: dx, dy: dy))
        }
        
        positions.append((position.x, position.y, timestamp))
        self.confidence = max(self.confidence, confidence) // Keep highest confidence
        trackLength += 1
        
        // Recalculate consistency score
        consistencyScore = MovementAnalyzer.calculateMovementConsistency(movementVectors)
        
        // Limit stored data to prevent memory growth
        if positions.count > 50 {
            positions.removeFirst()
        }
        if movementVectors.count > 49 {
            movementVectors.removeFirst()
        }
    }
}

/// Analysis result for directional movement
public struct DirectionalAnalysis {
    let predominantDirection: CountingDirection?
    let confidence: Float
    let directionWeights: [CountingDirection: Float]
    let qualifiedTracksCount: Int
    let totalMovementVectors: Int
}

/// Summary of complete calibration process
public struct CalibrationSummary {
    let thresholds: [CGFloat]
    let detectedDirection: CountingDirection?
    let originalDirection: CountingDirection
    let movementAnalysisSuccess: Bool
    let qualifiedTracksCount: Int
    let warnings: [String]
    let thresholdCalibrationEnabled: Bool
    let directionCalibrationEnabled: Bool
}

// MARK: - Movement Analysis Engine

/// Core movement analysis functionality
public class MovementAnalyzer {
    
    /// Analyze track movement patterns and determine consistency
    /// - Parameter data: Fish movement data to analyze
    /// - Returns: DirectionalAnalysis with predominant direction and confidence
    public static func analyzeTrackMovement(_ data: FishMovementData) -> DirectionalAnalysis {
        let vectors = data.movementVectors
        guard !vectors.isEmpty else {
            return DirectionalAnalysis(
                predominantDirection: nil,
                confidence: 0.0,
                directionWeights: [:],
                qualifiedTracksCount: 0,
                totalMovementVectors: 0
            )
        }
        
        // Count movement in each direction
        var directionCounts: [CountingDirection: Float] = [
            .topToBottom: 0,
            .bottomToTop: 0,
            .leftToRight: 0,
            .rightToLeft: 0
        ]
        
        for vector in vectors {
            let weight = Float(1.0) // Equal weight for each movement
            
            // Determine primary direction based on larger component
            if abs(vector.dx) > abs(vector.dy) {
                // Horizontal movement dominates
                if vector.dx > 0 {
                    directionCounts[.leftToRight]! += weight
                } else {
                    directionCounts[.rightToLeft]! += weight
                }
            } else {
                // Vertical movement dominates
                if vector.dy > 0 {
                    directionCounts[.topToBottom]! += weight
                } else {
                    directionCounts[.bottomToTop]! += weight
                }
            }
        }
        
        // Find predominant direction
        let maxEntry = directionCounts.max { $0.value < $1.value }
        let total = directionCounts.values.reduce(0, +)
        let confidence = total > 0 ? (maxEntry?.value ?? 0) / total : 0
        
        return DirectionalAnalysis(
            predominantDirection: maxEntry?.key,
            confidence: confidence,
            directionWeights: directionCounts,
            qualifiedTracksCount: 1,
            totalMovementVectors: vectors.count
        )
    }
    
    /// Calculate movement consistency score
    /// - Parameter vectors: Array of movement vectors (dx, dy)
    /// - Returns: Consistency score between 0.0 and 1.0
    public static func calculateMovementConsistency(_ vectors: [(dx: CGFloat, dy: CGFloat)]) -> Float {
        guard vectors.count >= 2 else { return 0.0 }
        
        // Calculate average direction
        let avgDx = vectors.map { $0.dx }.reduce(0, +) / CGFloat(vectors.count)
        let avgDy = vectors.map { $0.dy }.reduce(0, +) / CGFloat(vectors.count)
        let avgMagnitude = sqrt(avgDx * avgDx + avgDy * avgDy)
        
        guard avgMagnitude > 0.001 else { return 0.0 } // Avoid division by zero
        
        // Calculate how consistent each vector is with the average direction
        var consistencySum: Float = 0.0
        for vector in vectors {
            let magnitude = sqrt(vector.dx * vector.dx + vector.dy * vector.dy)
            if magnitude > 0.001 {
                // Dot product normalized by magnitudes = cosine of angle
                let dotProduct = vector.dx * avgDx + vector.dy * avgDy
                let consistency = Float(dotProduct / (magnitude * avgMagnitude))
                consistencySum += max(0, consistency) // Only positive consistency counts
            }
        }
        
        return consistencySum / Float(vectors.count)
    }
    
    /// Determine predominant direction from multiple fish tracks
    /// - Parameter movements: Array of fish movement data
    /// - Returns: Detected direction or nil if unclear
    public static func determineDirection(from movements: [FishMovementData]) -> DirectionalAnalysis {
        let config = AutoCalibrationConfig.shared
        
        // Filter qualified tracks
        let qualifiedTracks = movements.filter { track in
            return track.trackLength >= config.minTrackLengthForAnalysis &&
                   track.confidence >= config.minConfidenceForAnalysis &&
                   !track.movementVectors.isEmpty
        }
        
        guard !qualifiedTracks.isEmpty else {
            return DirectionalAnalysis(
                predominantDirection: nil,
                confidence: 0.0,
                directionWeights: [:],
                qualifiedTracksCount: 0,
                totalMovementVectors: 0
            )
        }
        
        // Aggregate weighted direction counts
        var totalDirectionWeights: [CountingDirection: Float] = [
            .topToBottom: 0,
            .bottomToTop: 0,
            .leftToRight: 0,
            .rightToLeft: 0
        ]
        
        var totalVectors = 0
        
        for track in qualifiedTracks {
            let trackAnalysis = analyzeTrackMovement(track)
            let trackWeight = Float(track.trackLength) * track.confidence * track.consistencyScore
            
            for (direction, weight) in trackAnalysis.directionWeights {
                totalDirectionWeights[direction]! += weight * trackWeight
            }
            
            totalVectors += track.movementVectors.count
        }
        
        // Find predominant direction
        let maxEntry = totalDirectionWeights.max { $0.value < $1.value }
        let totalWeight = totalDirectionWeights.values.reduce(0, +)
        let confidence = totalWeight > 0 ? (maxEntry?.value ?? 0) / totalWeight : 0
        
        // Check if confidence meets threshold
        let predominantDirection = confidence >= config.directionDecisionThreshold ? maxEntry?.key : nil
        
        return DirectionalAnalysis(
            predominantDirection: predominantDirection,
            confidence: confidence,
            directionWeights: totalDirectionWeights,
            qualifiedTracksCount: qualifiedTracks.count,
            totalMovementVectors: totalVectors
        )
    }
    
    /// Generate warnings based on analysis results
    /// - Parameter analysis: The directional analysis results
    /// - Returns: Array of warning messages
    public static func generateWarnings(from analysis: DirectionalAnalysis) -> [String] {
        var warnings: [String] = []
        
        if analysis.qualifiedTracksCount < 5 {
            warnings.append("Low track count (\(analysis.qualifiedTracksCount)) - direction may be unreliable")
        }
        
        if analysis.confidence < 0.6 {
            warnings.append("Low confidence (\(String(format: "%.1f", analysis.confidence * 100))%) - mixed movement detected")
        }
        
        if analysis.totalMovementVectors < 50 {
            warnings.append("Limited movement data (\(analysis.totalMovementVectors) vectors)")
        }
        
        return warnings
    }
}

/// Calibration utilities for auto threshold detection
public class CalibrationUtils {
    /// Default frame count for calibration (10 seconds at 30fps to match Python)
    public static let defaultCalibrationFrameCount = 300  // Matches Python implementation: 10 seconds at 30fps
    
    /// Test function to verify OpenCV access from the YOLO package
    /// This comprehensive test checks:
    /// 1. If OpenCV is properly integrated and working
    /// 2. If the version can be retrieved
    /// 3. If frame conversion works correctly
    /// 
    /// - Parameter frame: A sample frame to test with
    /// - Returns: true if all OpenCV tests passed
    public static func testOpenCVAccess(frame: CVPixelBuffer?) -> Bool {
        guard let frame = frame else {
            print("❌ No frame provided for OpenCV test")
            return false
        }
        
        print("Starting OpenCV integration tests...")
        print("Frame dimensions: \(CVPixelBufferGetWidth(frame)) x \(CVPixelBufferGetHeight(frame))")
        
        // Test 1: Basic OpenCV functionality
        let isWorking = OpenCVWrapper.isOpenCVWorking()
        print("Test 1 - Basic OpenCV functionality: \(isWorking ? "✅ PASSED" : "❌ FAILED")")
        
        // If basic test fails, don't continue
        if !isWorking {
            print("❌ Basic OpenCV test failed, aborting further tests")
            return false
        }
        
        // Test 2: Get OpenCV version
        let version = OpenCVWrapper.getOpenCVVersion()
        let versionTest = version != "OpenCV Not Accessible" && version != "Unknown"
        print("Test 2 - OpenCV version: \(versionTest ? "✅ PASSED" : "❌ FAILED") - Version: \(version)")
        
        // Test 3: Process a frame
        let frameProcessed = OpenCVWrapper.processTestFrame(frame)
        print("Test 3 - Frame processing: \(frameProcessed ? "✅ PASSED" : "❌ FAILED")")
        
        let allTestsPassed = isWorking && versionTest && frameProcessed
        print("OpenCV integration test summary: \(allTestsPassed ? "✅ ALL TESTS PASSED" : "❌ SOME TESTS FAILED")")
        
        return allTestsPassed
    }
    
    /// Calculate auto-calibration thresholds based on the provided frames
    /// This method is kept for backward compatibility but is no longer the primary calibration method.
    /// The main calibration now happens in the streaming mode in TrackingDetector.
    ///
    /// - Parameters:
    ///   - frames: Collection of frames (pixel buffers) for processing
    ///   - direction: Current counting direction
    /// - Returns: Tuple with two threshold values (0.0-1.0)
    public static func calculateThresholds(
        from frames: [CVPixelBuffer],
        direction: CountingDirection
    ) -> (threshold1: CGFloat, threshold2: CGFloat) {
        // print("Warning: Using legacy buffered calibration with \(frames.count) frames. This method is now deprecated in favor of streaming calibration.")
        
        // Return default values if no frames are provided
        if frames.isEmpty {
            print("⚠️ No frames provided for calibration, using default values")
            return (0.3, 0.7)
        }
        
        // Determine if we're using vertical or horizontal direction
        let isVerticalDirection = direction == .topToBottom || direction == .bottomToTop
        // print("Calibration direction: \(isVerticalDirection ? "vertical" : "horizontal")")
        
        // Accumulate results from each frame
        var threshold1Values: [CGFloat] = []
        var threshold2Values: [CGFloat] = []
        
        // Process all frames without skipping to match Python implementation
        // print("Processing all \(frames.count) frames for calibration without skipping")
        
        // Process each frame individually
        for (index, frame) in frames.enumerated() {
            // print("Processing calibration frame \(index+1)/\(frames.count)")
            
            // Process the frame through OpenCV
            if let thresholdArray = OpenCVWrapper.processCalibrationFrame(frame, isVerticalDirection: isVerticalDirection),
               thresholdArray.count >= 2,
               let value1 = thresholdArray[0] as? NSNumber,
               let value2 = thresholdArray[1] as? NSNumber {
                
                threshold1Values.append(CGFloat(value1.floatValue))
                threshold2Values.append(CGFloat(value2.floatValue))
            }
            
            // Memory management: Force a cleanup after each frame processing
            autoreleasepool {
                // This ensures temporary objects are released
            }
        }
        
        // Return default values if we couldn't get any valid results
        if threshold1Values.isEmpty || threshold2Values.isEmpty {
            print("⚠️ No valid threshold results from calibration, using default values")
            return (0.3, 0.7)
        }
        
        // Calculate the average thresholds from all processed frames
        let avgThreshold1 = threshold1Values.reduce(0, +) / CGFloat(threshold1Values.count)
        let avgThreshold2 = threshold2Values.reduce(0, +) / CGFloat(threshold2Values.count)
        
        print("Calibration complete - Thresholds: (\(avgThreshold1), \(avgThreshold2))")
        
        // Ensure thresholds are properly ordered (smaller one first)
        return (min(avgThreshold1, avgThreshold2), max(avgThreshold1, avgThreshold2))
    }
    
    /// Process a single frame for streaming calibration
    /// This method implements the Python equivalent of processing a frame in the calibration loop
    ///
    /// - Parameters:
    ///   - frame: The current frame to process
    ///   - direction: The counting direction
    /// - Returns: Tuple with two threshold values (0.0-1.0) from this frame, or nil if processing failed
    public static func processCalibrationFrame(
        _ frame: CVPixelBuffer,
        direction: CountingDirection
    ) -> (threshold1: CGFloat, threshold2: CGFloat)? {
        // Determine if we're using vertical or horizontal direction
        let isVerticalDirection = direction == .topToBottom || direction == .bottomToTop
        
        // Process the frame through OpenCV
        if let thresholdArray = OpenCVWrapper.processCalibrationFrame(frame, isVerticalDirection: isVerticalDirection),
           thresholdArray.count >= 2,
           let value1 = thresholdArray[0] as? NSNumber,
           let value2 = thresholdArray[1] as? NSNumber {
            
            let threshold1 = CGFloat(value1.floatValue)
            let threshold2 = CGFloat(value2.floatValue)
            
            // Ensure thresholds are properly ordered (smaller one first)
            return (min(threshold1, threshold2), max(threshold1, threshold2))
        }
        
        return nil
    }
    
    /// Calculate the progress percentage based on collected frames
    ///
    /// - Parameters:
    ///   - currentFrameCount: Number of frames collected so far
    ///   - totalFrameCount: Total number of frames needed
    /// - Returns: Progress percentage (0-100)
    public static func calculateProgress(
        currentFrameCount: Int,
        totalFrameCount: Int
    ) -> Int {
        return Int(min(100, (Double(currentFrameCount) / Double(totalFrameCount)) * 100))
    }
} 