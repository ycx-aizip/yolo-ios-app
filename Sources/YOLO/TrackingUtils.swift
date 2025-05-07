// TrackingUtils.swift
// Centralized management of tracking parameters for fish counting application

import Foundation
import UIKit

// Import CountingDirection from CountingUtils directly
// Cannot use extension without first importing the type

/**
 * Fish Tracking System
 *
 * The tracking system in this application combines ByteTrack object tracking with specialized logic
 * for reliable fish counting. The system is designed to handle fish movement in various directions
 * and provides robust tracking through occlusions and temporary disappearances.
 *
 * Architecture Overview:
 * ---------------------
 * 1. Object Detection: YOLO-based detection provides bounding boxes for each fish
 * 2. Detection Association: ByteTrack algorithm associates detections across frames
 * 3. Track Management: Tracks lifecycle (new → tracked → lost → removed)
 * 4. Threshold Crossing: Detection of when fish cross counting lines
 * 5. Direction-aware Processing: Handles different camera viewing directions
 *
 * Key Components:
 * --------------
 * - ByteTracker: Core tracking algorithm that maintains track IDs across frames
 * - STrack: Single track representation with Kalman filtering for motion prediction
 * - TrackingDetector: Integrates tracking with threshold crossing detection
 * - MatchingUtils: Provides utility methods for detection-track association
 * - KalmanFilter: State estimation for predicting fish positions between detections
 *
 * Tracking Algorithm:
 * ------------------
 * 1. First-Stage Matching: Match high-confidence detections with existing tracks using IoU
 * 2. Second-Stage Matching: Match remaining detections with recently lost tracks
 * 3. Track Management: Create new tracks, update matched tracks, expire lost tracks
 * 4. Motion Prediction: Use Kalman filter to predict track positions for the next frame
 * 5. Threshold Crossing: Detect when tracks cross counting thresholds
 *
 * Track Lifecycle:
 * ---------------
 * 1. Potential: Detections that persist for multiple frames become candidate tracks
 * 2. New: Newly created tracks that haven't been confirmed yet
 * 3. Tracked: Active tracks currently associated with detections
 * 4. Lost: Tracks not matched in recent frames, may be recovered
 * 5. Removed: Tracks that have been lost for too long and are no longer considered
 *
 * TTL (Time-To-Live) System:
 * -------------------------
 * - Each track has a TTL counter that is decremented when not matched
 * - TTL values are adaptive based on track consistency and movement patterns
 * - Higher TTL values (15-20) for fish with consistent expected movement patterns
 * - Lower TTL values (8-12) for erratic movement or tracks being reactivated
 * - When TTL reaches zero, tracks transition from tracked to lost state
 * - Lost tracks are removed entirely after maxTimeLost frames
 *
 * Key Parameter Groups:
 * --------------------
 * 1. Association Thresholds: Control how detections are matched to existing tracks
 * 2. Track Lifecycle: Parameters for track creation, retention, and removal
 * 3. TTL Parameters: Control how long tracks persist without detection matches
 * 4. Movement Parameters: Define expected movement patterns and consistency metrics
 */

/// Centralized management of tracking parameters
@MainActor
public struct TrackingParameters {
    // General tracking thresholds
    public struct Thresholds: Sendable {
        // Association thresholds
        /// IoU threshold for first-stage matching (higher quality matches)
        /// Higher values (>0.3) require more overlap for matching, reducing ID switches but may cause more track fragmentation
        /// Lower values (<0.3) will match with less overlap, reducing track loss but may cause ID switches
        let highMatchThreshold: Float
        
        /// IoU threshold for second-stage matching (more permissive)
        /// Used for matching lost tracks with new detections
        /// Lower than highMatchThreshold to recover tracks that have drifted
        let lowMatchThreshold: Float
        
        // Detection matching
        /// Minimum distance to consider a detection-track match (in normalized coordinates)
        /// Used in spatial matching when IoU is insufficient
        /// Smaller values require closer matches (0.3 = 30% of frame width/height)
        let minMatchDistance: CGFloat
        
        /// IoU threshold for immediate track association
        /// When IoU is above this value, tracks are immediately associated without further checks
        /// Used as a fast-path for very confident matches
        let iouMatchThreshold: Float
        
        // Track lifecycle
        /// Maximum frames to keep lost tracks before removing them completely
        /// Higher values allow recovering tracks after longer occlusions but increase memory usage
        /// Typically 15-30 frames depending on frame rate and expected occlusion duration
        let maxTimeLost: Int
        
        /// Maximum frames to keep unmatched potential tracks
        /// Controls how long detections are tracked before being considered a real track
        /// Lower values create tracks faster but may create spurious tracks
        let maxUnmatchedFrames: Int
        
        /// Frames required before a potential track becomes an actual track
        /// Higher values (>1) reduce spurious tracks but delay track creation
        /// Lower values (1) create tracks immediately but may track noise
        let requiredFramesForTrack: Int
        
        /// Maximum distance for matching potential tracks to detections
        /// Controls spatial threshold for initial track creation
        /// Higher values allow more distant matches for potential tracks
        let maxMatchingDistance: CGFloat
        
        // TTL (Time-To-Live) parameters
        /// Default TTL for new tracks when first created
        /// Controls initial persistence time for new tracks
        /// Higher values give tracks more time to establish consistent movement patterns
        let defaultTTL: Int
        
        /// TTL for tracks with highly consistent movement
        /// Used for tracks showing strong alignment with expected movement direction
        /// Higher values (15-20) maintain tracks through brief occlusions
        let highConsistencyTTL: Int
        
        /// TTL for tracks with moderately consistent movement
        /// Used for tracks showing reasonable alignment with expected direction
        /// Medium values (12-15) provide balanced persistence
        let mediumConsistencyTTL: Int
        
        /// TTL for tracks with erratic movement
        /// Used for tracks that move inconsistently or against expected direction
        /// Lower values (8-10) prevent incorrect tracking of erratic objects
        let lowConsistencyTTL: Int
        
        /// TTL for reactivated tracks with high consistency
        /// Used when a lost track is recovered and shows good movement alignment
        /// More conservative than regular high consistency value to prevent incorrect reactivations
        let reactivationHighTTL: Int
        
        /// TTL for reactivated tracks with medium consistency
        /// Used when a lost track is recovered with reasonable movement alignment
        /// Balanced value to allow track continuation without excessive persistence
        let reactivationMediumTTL: Int
        
        /// TTL for reactivated tracks with low consistency
        /// Used when a lost track is recovered but shows poor movement alignment
        /// Low value to quickly remove incorrectly reactivated tracks
        let reactivationLowTTL: Int
        
        // Expected movement constraints
        /// Maximum allowed horizontal deviation for expected vertical movement
        /// Controls how much horizontal movement is allowed when moving vertically
        /// Lower values enforce straighter paths, higher values allow more zigzagging
        let maxHorizontalDeviation: CGFloat
        
        /// Maximum allowed vertical deviation for expected horizontal movement
        /// Controls how much vertical movement is allowed when moving horizontally
        /// Lower values enforce straighter paths, higher values allow more zigzagging
        let maxVerticalDeviation: CGFloat
        
        // Consistency increase/decrease rates
        /// Rate to increase movement consistency when movement matches expectations
        /// Controls how quickly tracks build up consistency score with good movement
        /// Higher values build consistency faster, making TTL increase more quickly
        let consistencyIncreaseRate: CGFloat
        
        /// Rate to decrease movement consistency when movement is unexpected
        /// Controls how quickly consistency degrades with poor movement
        /// Higher values punish inconsistent movement more severely
        let consistencyDecreaseRate: CGFloat
        
        /// Rate to increase consistency during track reactivation
        /// Controls how quickly reactivated tracks can regain consistency
        /// Lower than regular increase rate to be more conservative with reactivated tracks
        let reactivationConsistencyIncreaseRate: CGFloat
        
        /// Rate to decrease consistency during track reactivation
        /// Controls how quickly reactivated tracks lose consistency with poor movement
        /// Higher than regular decrease rate to more aggressively remove bad reactivations
        let reactivationConsistencyDecreaseRate: CGFloat
    }
    
    // Default thresholds (top-to-bottom movement)
    @MainActor
    public static let defaultThresholds = Thresholds(
        // Association thresholds
        highMatchThreshold: 0.3,  // First-stage IoU threshold for matching
        lowMatchThreshold: 0.25,  // Second-stage IoU threshold for matching
        
        // Detection matching
        minMatchDistance: 0.3,    // Minimum distance for track association
        iouMatchThreshold: 0.4,   // IoU threshold for immediate association
        
        // Track lifecycle
        maxTimeLost: 15,          // Maximum frames to keep lost tracks
        maxUnmatchedFrames: 15,   // Maximum frames to keep potential tracks
        requiredFramesForTrack: 1, // Frames required to establish a new track
        maxMatchingDistance: 0.6,  // Maximum distance for matching potential tracks
        
        // TTL parameters (Time-To-Live)
        defaultTTL: 15,           // Default TTL value for new tracks
        highConsistencyTTL: 20,   // Higher TTL for tracks with consistent movement
        mediumConsistencyTTL: 15, // Medium TTL for tracks with moderate consistency
        lowConsistencyTTL: 10,    // Lower TTL for tracks with erratic movement
        reactivationHighTTL: 15,  // TTL for reactivated tracks with high consistency
        reactivationMediumTTL: 12, // TTL for reactivated tracks with medium consistency
        reactivationLowTTL: 8,    // TTL for reactivated tracks with low consistency
        
        // Expected movement constraints
        maxHorizontalDeviation: 0.2, // Maximum allowed horizontal deviation for vertical movement
        maxVerticalDeviation: 0.2,   // Maximum allowed vertical deviation for horizontal movement
        
        // Consistency rates
        consistencyIncreaseRate: 0.2,      // Rate to increase consistency for expected movement
        consistencyDecreaseRate: 0.1,      // Rate to decrease consistency for unexpected movement
        reactivationConsistencyIncreaseRate: 0.15, // Rate to increase for reactivation
        reactivationConsistencyDecreaseRate: 0.2   // Rate to decrease for reactivation with unexpected movement
    )
    
    // Direction-specific thresholds
    @MainActor
    private static var directionThresholds: [String: Thresholds] = [
        "topToBottom": defaultThresholds,
        
        "bottomToTop": Thresholds(
            highMatchThreshold: 0.3,
            lowMatchThreshold: 0.25,
            minMatchDistance: 0.3,
            iouMatchThreshold: 0.4,
            maxTimeLost: 15,
            maxUnmatchedFrames: 15,
            requiredFramesForTrack: 1,
            maxMatchingDistance: 0.6,
            defaultTTL: 15,
            highConsistencyTTL: 20,
            mediumConsistencyTTL: 15,
            lowConsistencyTTL: 10,
            reactivationHighTTL: 15,
            reactivationMediumTTL: 12,
            reactivationLowTTL: 8,
            maxHorizontalDeviation: 0.2,
            maxVerticalDeviation: 0.2,
            consistencyIncreaseRate: 0.2,
            consistencyDecreaseRate: 0.1,
            reactivationConsistencyIncreaseRate: 0.15,
            reactivationConsistencyDecreaseRate: 0.2
        ),
        
        "leftToRight": Thresholds(
            highMatchThreshold: 0.3,
            lowMatchThreshold: 0.25,
            minMatchDistance: 0.35,  // Slightly more lenient for horizontal
            iouMatchThreshold: 0.4,
            maxTimeLost: 15,
            maxUnmatchedFrames: 15,
            requiredFramesForTrack: 1,
            maxMatchingDistance: 0.65,  // Increased for horizontal movement
            defaultTTL: 15,
            highConsistencyTTL: 20,
            mediumConsistencyTTL: 15,
            lowConsistencyTTL: 10,
            reactivationHighTTL: 15,
            reactivationMediumTTL: 12,
            reactivationLowTTL: 8,
            maxHorizontalDeviation: 0.2,
            maxVerticalDeviation: 0.2,
            consistencyIncreaseRate: 0.2,
            consistencyDecreaseRate: 0.1,
            reactivationConsistencyIncreaseRate: 0.15,
            reactivationConsistencyDecreaseRate: 0.2
        ),
        
        "rightToLeft": Thresholds(
            highMatchThreshold: 0.3,
            lowMatchThreshold: 0.25,
            minMatchDistance: 0.35,  // Slightly more lenient for horizontal
            iouMatchThreshold: 0.4,
            maxTimeLost: 15,
            maxUnmatchedFrames: 15,
            requiredFramesForTrack: 1,
            maxMatchingDistance: 0.65,  // Increased for horizontal movement
            defaultTTL: 15,
            highConsistencyTTL: 20,
            mediumConsistencyTTL: 15,
            lowConsistencyTTL: 10,
            reactivationHighTTL: 15,
            reactivationMediumTTL: 12,
            reactivationLowTTL: 8,
            maxHorizontalDeviation: 0.2,
            maxVerticalDeviation: 0.2,
            consistencyIncreaseRate: 0.2,
            consistencyDecreaseRate: 0.1,
            reactivationConsistencyIncreaseRate: 0.15,
            reactivationConsistencyDecreaseRate: 0.2
        )
    ]
    
    // Current tracking parameters (initially set to default)
    @MainActor
    private static var currentThresholds: Thresholds = defaultThresholds
    
    // Public access methods - use string to avoid direct enum dependency
    @MainActor
    public static func thresholds(forDirection direction: String) -> Thresholds {
        return directionThresholds[direction] ?? defaultThresholds
    }
    
    // Update current parameters based on direction string
    @MainActor
    public static func updateParameters(forDirection direction: String) {
        currentThresholds = thresholds(forDirection: direction)
    }
    
    // Helper function to convert CountingDirection to string (to be called from files that have CountingDirection)
    @MainActor
    public static func updateParametersForCountingDirection(_ direction: Any) {
        let directionString: String
        // Use string pattern matching instead of direct enum comparison
        if let directionEnum = direction as? Any {
            switch String(describing: directionEnum) {
            case "topToBottom":
                directionString = "topToBottom"
            case "bottomToTop":
                directionString = "bottomToTop"
            case "leftToRight":
                directionString = "leftToRight"
            case "rightToLeft":
                directionString = "rightToLeft"
            default:
                directionString = "topToBottom" // Default
            }
            updateParameters(forDirection: directionString)
        }
    }
    
    // Getter methods for general parameters
    @MainActor public static var highMatchThreshold: Float { return currentThresholds.highMatchThreshold }
    @MainActor public static var lowMatchThreshold: Float { return currentThresholds.lowMatchThreshold }
    @MainActor public static var minMatchDistance: CGFloat { return currentThresholds.minMatchDistance }
    @MainActor public static var iouMatchThreshold: Float { return currentThresholds.iouMatchThreshold }
    @MainActor public static var maxTimeLost: Int { return currentThresholds.maxTimeLost }
    @MainActor public static var maxUnmatchedFrames: Int { return currentThresholds.maxUnmatchedFrames }
    @MainActor public static var requiredFramesForTrack: Int { return currentThresholds.requiredFramesForTrack }
    @MainActor public static var maxMatchingDistance: CGFloat { return currentThresholds.maxMatchingDistance }
    
    // Getter methods for TTL parameters
    @MainActor public static var defaultTTL: Int { return currentThresholds.defaultTTL }
    @MainActor public static var highConsistencyTTL: Int { return currentThresholds.highConsistencyTTL }
    @MainActor public static var mediumConsistencyTTL: Int { return currentThresholds.mediumConsistencyTTL }
    @MainActor public static var lowConsistencyTTL: Int { return currentThresholds.lowConsistencyTTL }
    @MainActor public static var reactivationHighTTL: Int { return currentThresholds.reactivationHighTTL }
    @MainActor public static var reactivationMediumTTL: Int { return currentThresholds.reactivationMediumTTL }
    @MainActor public static var reactivationLowTTL: Int { return currentThresholds.reactivationLowTTL }
    
    // Getter methods for movement constraints
    @MainActor public static var maxHorizontalDeviation: CGFloat { return currentThresholds.maxHorizontalDeviation }
    @MainActor public static var maxVerticalDeviation: CGFloat { return currentThresholds.maxVerticalDeviation }
    
    // Getter methods for consistency rates
    @MainActor public static var consistencyIncreaseRate: CGFloat { return currentThresholds.consistencyIncreaseRate }
    @MainActor public static var consistencyDecreaseRate: CGFloat { return currentThresholds.consistencyDecreaseRate }
    @MainActor public static var reactivationConsistencyIncreaseRate: CGFloat { return currentThresholds.reactivationConsistencyIncreaseRate }
    @MainActor public static var reactivationConsistencyDecreaseRate: CGFloat { return currentThresholds.reactivationConsistencyDecreaseRate }
}
