// Swift file for CountingDirection enum and calibration utilities
// This file provides centralized configuration and utilities for fish counting

import Foundation
import AVFoundation
import CoreVideo
import UIKit
import CoreGraphics
// OpenCV integration removed - moved to app layer to enable xcframework distribution
// Phase 1 calibration (threshold detection) is disabled

/// Direction for counting objects (fish)
public enum CountingDirection {
    case topToBottom
    case bottomToTop
    case leftToRight
    case rightToLeft
}

// MARK: - Developer Configuration Interface

/// Centralized configuration for auto-calibration system
/// Developers should configure all auto-calibration settings here
public final class AutoCalibrationConfig: @unchecked Sendable {
    /// Shared singleton instance - modify this for all calibration settings
    public static let shared = AutoCalibrationConfig()
    
    // MARK: - Phase Control
    /// Enable Phase 1: OpenCV edge detection for threshold calibration
    public var isThresholdCalibrationEnabled: Bool = false
    
    /// Enable Phase 2: YOLO movement analysis for direction detection
    public var isDirectionCalibrationEnabled: Bool = true
    
    // MARK: - Frame Count Settings
    /// Number of frames for threshold detection (typically 300 = 10 seconds at 30fps)
    public var thresholdCalibrationFrames: Int = 300
    
    /// Number of frames for movement analysis (typically 300 = 10 seconds at 30fps)
    public var movementAnalysisFrames: Int = 300
    
    // MARK: - Movement Analysis Parameters
    /// Minimum track length required for direction analysis (frames)
    public var minTrackLengthForAnalysis: Int = 10
    
    /// Minimum confidence required for tracks to be considered (0.0-1.0)
    public var minConfidenceForAnalysis: Float = 0.6
    
    /// Confidence threshold for direction decision (0.6 = 60%)
    public var directionDecisionThreshold: Float = 0.6
    
    private init() {}
    
    /// Quick setup method for common configurations
    /// - Parameters:
    ///   - thresholdCalibration: Enable/disable Phase 1
    ///   - directionCalibration: Enable/disable Phase 2
    ///   - thresholdFrames: Frame count for Phase 1
    ///   - movementFrames: Frame count for Phase 2
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
        
        print("AutoCalibration: Configuration updated - Threshold: \(thresholdCalibration), Direction: \(directionCalibration)")
    }
    
    /// Total frames needed for complete calibration
    public var totalCalibrationFrames: Int {
        var total = 0
        if isThresholdCalibrationEnabled { total += thresholdCalibrationFrames }
        if isDirectionCalibrationEnabled { total += movementAnalysisFrames }
        return total
    }
}

// MARK: - Movement Analysis Data Structures

/// Data structure for tracking fish movement patterns during Phase 2
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
        self.confidence = max(self.confidence, confidence)
        trackLength += 1
        
        // Recalculate consistency score
        consistencyScore = MovementAnalyzer.calculateMovementConsistency(movementVectors)
        
        // Limit stored data to prevent memory growth (keep last 50 positions)
        if positions.count > 50 {
            positions.removeFirst()
        }
        if movementVectors.count > 49 {
            movementVectors.removeFirst()
        }
    }
}

/// Analysis result for directional movement in Phase 2
public struct DirectionalAnalysis {
    let predominantDirection: CountingDirection?
    let confidence: Float
    let directionWeights: [CountingDirection: Float]
    let qualifiedTracksCount: Int
    let totalMovementVectors: Int
}

/// Summary of complete calibration process
public struct CalibrationSummary {
    public let thresholds: [CGFloat]
    public let detectedDirection: CountingDirection?
    public let originalDirection: CountingDirection
    public let movementAnalysisSuccess: Bool
    public let qualifiedTracksCount: Int
    public let warnings: [String]
    public let thresholdCalibrationEnabled: Bool
    public let directionCalibrationEnabled: Bool
}

// MARK: - Movement Analysis Engine

/// Core movement analysis functionality for Phase 2
public class MovementAnalyzer {
    
    /// Analyze individual track movement patterns
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
            let weight = Float(1.0)
            
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
    
    /// Calculate movement consistency score for a track
    public static func calculateMovementConsistency(_ vectors: [(dx: CGFloat, dy: CGFloat)]) -> Float {
        guard vectors.count >= 2 else { return 0.0 }
        
        // Calculate average direction
        let avgDx = vectors.map { $0.dx }.reduce(0, +) / CGFloat(vectors.count)
        let avgDy = vectors.map { $0.dy }.reduce(0, +) / CGFloat(vectors.count)
        let avgMagnitude = sqrt(avgDx * avgDx + avgDy * avgDy)
        
        guard avgMagnitude > 0.001 else { return 0.0 }
        
        // Calculate how consistent each vector is with the average direction
        var consistencySum: Float = 0.0
        for vector in vectors {
            let magnitude = sqrt(vector.dx * vector.dx + vector.dy * vector.dy)
            if magnitude > 0.001 {
                let dotProduct = vector.dx * avgDx + vector.dy * avgDy
                let consistency = Float(dotProduct / (magnitude * avgMagnitude))
                consistencySum += max(0, consistency)
            }
        }
        
        return consistencySum / Float(vectors.count)
    }
    
    /// Determine predominant direction from multiple fish tracks
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

// MARK: - Calibration Utilities

/// Utilities for threshold calibration (Phase 1)
/// OpenCV integration removed to enable xcframework distribution
public class CalibrationUtils {
    /// Test OpenCV integration - DISABLED (OpenCV moved to app layer)
    public static func testOpenCVAccess(frame: CVPixelBuffer?) -> Bool {
        // OpenCV integration removed from AizipFishCount framework
        // Phase 1 calibration is disabled
        print("⚠️ OpenCV test: Phase 1 calibration disabled (OpenCV not available in framework)")
        return false

        /* Original OpenCV code - commented out for xcframework build
        guard let frame = frame else {
            print("❌ OpenCV test: No frame provided")
            return false
        }

        let isWorking = OpenCVWrapper.isOpenCVWorking()
        let version = OpenCVWrapper.getOpenCVVersion()
        let frameProcessed = OpenCVWrapper.processTestFrame(frame)

        let allTestsPassed = isWorking && frameProcessed
        if !allTestsPassed {
            print("❌ OpenCV integration failed - Version: \(version)")
        }

        return allTestsPassed
        */
    }

    /// Process single frame for streaming calibration - DISABLED (OpenCV moved to app layer)
    public static func processCalibrationFrame(
        _ frame: CVPixelBuffer,
        direction: CountingDirection
    ) -> (threshold1: CGFloat, threshold2: CGFloat)? {
        // OpenCV integration removed from AizipFishCount framework
        // Phase 1 calibration is disabled
        print("⚠️ Phase 1 calibration: Disabled (OpenCV not available in framework)")
        return nil

        /* Original OpenCV code - commented out for xcframework build
        let isVerticalDirection = direction == .topToBottom || direction == .bottomToTop

        if let thresholdArray = OpenCVWrapper.processCalibrationFrame(frame, isVerticalDirection: isVerticalDirection),
           thresholdArray.count >= 2,
           let value1 = thresholdArray[0] as? NSNumber,
           let value2 = thresholdArray[1] as? NSNumber {

            let threshold1 = CGFloat(value1.floatValue)
            let threshold2 = CGFloat(value2.floatValue)

            return (min(threshold1, threshold2), max(threshold1, threshold2))
        }

        return nil
        */
    }
} 