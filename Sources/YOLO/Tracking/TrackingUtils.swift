// TrackingUtils.swift
// Centralized management of tracking parameters for fish counting application
//
// REFACTORED: Parameters now separated into:
// - SharedConfig: Common parameters for all trackers
// - ByteTrackConfig: ByteTrack-specific parameters
// - OCSortConfig: OC-SORT-specific parameters (placeholder)

import Foundation
import UIKit

/**
 * Fish Tracking System
 *
 * The tracking system in this application combines multiple object tracking algorithms
 * (ByteTrack, OC-SORT) with specialized logic for reliable fish counting. The system is
 * designed to handle fish movement in various directions and provides robust tracking
 * through occlusions and temporary disappearances.
 *
 * Architecture Overview:
 * ---------------------
 * 1. Object Detection: YOLO-based detection provides bounding boxes for each fish
 * 2. Detection Association: Tracking algorithm associates detections across frames
 * 3. Track Management: Tracks lifecycle (new → tracked → lost → removed)
 * 4. Threshold Crossing: Detection of when fish cross counting lines
 * 5. Direction-aware Processing: Handles different camera viewing directions
 *
 * Key Components:
 * --------------
 * - ByteTracker/OCSort: Core tracking algorithms that maintain track IDs across frames
 * - STrack: Single track representation with Kalman filtering for motion prediction
 * - TrackingDetector: Integrates tracking with threshold crossing detection
 * - MatchingUtils: Provides utility methods for detection-track association
 * - KalmanFilter: State estimation for predicting fish positions between detections
 *
 * Parameter Organization:
 * ----------------------
 * - SharedConfig: Parameters used by all tracking algorithms
 * - ByteTrackConfig: ByteTrack-specific parameters (two-stage matching, adaptive TTL)
 * - OCSortConfig: OC-SORT-specific parameters (observation-centric recovery)
 */

/// Centralized management of tracking parameters
public struct TrackingParameters {

    // MARK: - Shared Configuration (All Trackers)

    /// Parameters shared across all tracking algorithms
    public struct SharedConfig: Sendable {
        // Fish-specific movement constraints
        /// Maximum allowed horizontal deviation for expected vertical movement
        /// Controls how much horizontal movement is allowed when moving vertically
        /// Lower values enforce straighter paths, higher values allow more zigzagging
        let maxHorizontalDeviation: CGFloat

        /// Maximum allowed vertical deviation for expected horizontal movement
        /// Controls how much vertical movement is allowed when moving horizontally
        /// Lower values enforce straighter paths, higher values allow more zigzagging
        let maxVerticalDeviation: CGFloat

        // Spatial matching
        /// Minimum distance to consider a detection-track match (in normalized coordinates)
        /// Used in spatial matching when IoU is insufficient
        /// Smaller values require closer matches (0.3 = 30% of frame width/height)
        let minMatchDistance: CGFloat

        /// Maximum distance for matching potential tracks to detections
        /// Controls spatial threshold for initial track creation
        /// Higher values allow more distant matches for potential tracks
        let maxMatchingDistance: CGFloat

        // Memory limits (prevent unbounded growth)
        /// Maximum number of active tracks to maintain
        /// Prevents memory growth in crowded scenarios
        let maxActiveTracks: Int

        /// Maximum number of lost tracks to maintain
        /// Prevents memory growth from accumulating lost tracks
        let maxLostTracks: Int
    }

    // MARK: - ByteTrack Configuration

    /// ByteTrack-specific parameters
    public struct ByteTrackConfig: Sendable {
        // Two-stage matching thresholds
        /// IoU threshold for first-stage matching (higher quality matches)
        /// Higher values (>0.3) require more overlap for matching, reducing ID switches but may cause more track fragmentation
        /// Lower values (<0.3) will match with less overlap, reducing track loss but may cause ID switches
        let highMatchThreshold: Float

        /// IoU threshold for second-stage matching (more permissive)
        /// Used for matching lost tracks with new detections
        /// Lower than highMatchThreshold to recover tracks that have drifted
        let lowMatchThreshold: Float

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

        // Adaptive TTL (Time-To-Live) system
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

        // Movement consistency rates
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

    // MARK: - OC-SORT Configuration

    /// OC-SORT-specific parameters
    public struct OCSortConfig: Sendable {
        /// Detection confidence threshold
        /// Detections below this threshold are filtered out
        let detThresh: Float

        /// Maximum age (frames) to keep tracks without matching
        /// Similar to ByteTrack's maxTimeLost
        let maxAge: Int

        /// Minimum consecutive detections before track activation
        /// Prevents spurious tracks from brief detections
        let minHits: Int

        /// IoU threshold for association
        /// Higher values require more overlap for matching
        let iouThreshold: Float

        /// Delta-t for velocity estimation (frames apart)
        /// Used to compute velocity from observations delta_t frames ago
        let deltaT: Int

        /// Association function: "iou", "giou", "ciou", "diou", "ct_dist"
        /// Determines which distance metric to use for matching
        let assoFunc: String

        /// Inertia weight for velocity direction consistency
        /// Higher values give more weight to velocity alignment
        let inertia: Float

        /// Whether to use BYTE-style two-stage matching
        /// Enables recovery of tracks using low-confidence detections
        let useByte: Bool

        /// Velocity Direction Consistency weight (0-1)
        /// Controls influence of velocity direction in association scoring
        let vdcWeight: Float
    }

    // MARK: - Default Configurations

    /// Default shared configuration
    @MainActor
    public static let defaultSharedConfig = SharedConfig(
        // Fish-specific movement constraints
        maxHorizontalDeviation: 0.2,  // Maximum horizontal deviation for vertical movement
        maxVerticalDeviation: 0.2,    // Maximum vertical deviation for horizontal movement

        // Spatial matching
        minMatchDistance: 0.3,        // Minimum distance for track association
        maxMatchingDistance: 0.6,     // Maximum distance for matching potential tracks

        // Memory limits
        maxActiveTracks: 40,          // Maximum active tracks (optimized for mobile performance)
        maxLostTracks: 20             // Maximum lost tracks (reduced proportionally)
    )

    /// Default ByteTrack configuration
    @MainActor
    public static let defaultByteTrackConfig = ByteTrackConfig(
        // Two-stage matching thresholds
        highMatchThreshold: 0.3,      // First-stage IoU threshold
        lowMatchThreshold: 0.25,      // Second-stage IoU threshold
        iouMatchThreshold: 0.4,       // Immediate association threshold

        // Track lifecycle
        maxTimeLost: 15,              // Maximum frames to keep lost tracks
        maxUnmatchedFrames: 15,       // Maximum frames to keep potential tracks
        requiredFramesForTrack: 1,    // Frames required to establish a new track

        // Adaptive TTL system
        defaultTTL: 15,               // Default TTL for new tracks
        highConsistencyTTL: 20,       // TTL for consistent movement
        mediumConsistencyTTL: 15,     // TTL for moderate consistency
        lowConsistencyTTL: 10,        // TTL for erratic movement
        reactivationHighTTL: 15,      // TTL for reactivated high consistency
        reactivationMediumTTL: 12,    // TTL for reactivated medium consistency
        reactivationLowTTL: 8,        // TTL for reactivated low consistency

        // Movement consistency rates
        consistencyIncreaseRate: 0.2,                    // Rate to increase consistency
        consistencyDecreaseRate: 0.1,                    // Rate to decrease consistency
        reactivationConsistencyIncreaseRate: 0.15,       // Rate to increase for reactivation
        reactivationConsistencyDecreaseRate: 0.2         // Rate to decrease for reactivation
    )

    /// Default OC-SORT configuration (from Python package defaults)
    public static let defaultOCSortConfig = OCSortConfig(
        detThresh: 0.05,              // Detection confidence threshold (PYTHON DEFAULT - low for tracking)
        maxAge: 30,                   // Max age without matching
        minHits: 3,                   // Min consecutive detections before activation
        iouThreshold: 0.3,            // IoU threshold for association
        deltaT: 3,                    // Frames apart for velocity estimation
        assoFunc: "iou",              // Association function (iou, giou, ciou, diou)
        inertia: 0.2,                 // Velocity direction consistency weight
        useByte: false,               // Use BYTE two-stage matching
        vdcWeight: 0.2                // VDC weight (Python default)
    )

    /// CoreML-adapted OC-SORT configuration (iOS deployment)
    ///
    /// DIFFERENCE from Python:
    /// - Python YOLO: Returns varying confidence dets → OC-SORT splits into high/low
    /// - CoreML YOLO: Built-in NMS filters at 0.25 → only returns high-conf dets (≥0.25)
    ///
    /// Solution: Set detThresh=0.2 (below CoreML's 0.25) to ensure consistent classification
    /// All CoreML dets → high confidence bucket → Stage 1 & 3 matching active
    ///
    /// FISH-SPECIFIC TUNING (Empirically Validated):
    /// These parameters work synergistically to handle non-linear fish swimming patterns.
    /// Tested on 3008-fish ground truth video: 3105 counted (103% accuracy) vs Python defaults: 2480 (82% accuracy)
    ///
    /// - deltaT=1: Use immediate previous frame for velocity estimation
    ///   WHY: Fish make rapid, unpredictable direction changes. Using last frame captures instantaneous
    ///   swim direction without smoothing out turns. With deltaT=3, velocity averages over 3 frames and
    ///   points in wrong direction during turns, causing VDC to penalize correct matches.
    ///   SYNERGY: Accurate instantaneous velocity makes high vdcWeight effective.
    ///
    /// - vdcWeight=0.9: High weight for Velocity Direction Consistency in association cost
    ///   WHY: In dense fish schools, multiple fish have similar IoU. VDC disambiguates using swim direction.
    ///   With deltaT=1 providing accurate velocity, high VDC weight prevents ID switches when fish cross.
    ///   RESULT: Maintains track continuity (3780 tracks with proper IDs vs 3302 with fragmentation).
    ///   TRADE-OFF: Assumes fish maintain consistent direction within 1-2 frames (valid for normal swimming).
    ///
    /// - F=I (constant position): See OCSort.swift:807 for detailed rationale
    ///   WHY: Fish don't swim in straight lines. Constant velocity prediction overshoots during turns.
    ///   SYNERGY: Conservative position prediction + accurate VDC from deltaT=1 + high vdcWeight=0.9
    ///   creates robust tracking in dense, non-linear scenarios.
    public static let coreMLOCSortConfig = OCSortConfig(
        detThresh: 0.2,               // ✅ Below CoreML threshold (0.25) for consistent splitting
        maxAge: 30,                   // ✅ Python default
        minHits: 3,                   // ✅ Python default
        iouThreshold: 0.2,            // ✅ Python runtime default (smart default for OC-SORT)
        deltaT: 1,                    // ✅ Python default (was 1, changed to match Python)
        assoFunc: "iou",              // ✅ Python default
        inertia: 0.2,                 // ✅ Python default
        useByte: false,               // ✅ Python default (no low-conf dets from CoreML)
        vdcWeight: 0.9                // ⚙️ TUNED: High VDC weight for dense fish tracking
    )

    // MARK: - Direction-Specific Configurations

    /// Direction-specific shared configurations
    @MainActor
    private static var directionSharedConfigs: [String: SharedConfig] = [
        "topToBottom": defaultSharedConfig,
        "bottomToTop": defaultSharedConfig,

        // Horizontal movement may need slightly more lenient spatial thresholds
        "leftToRight": SharedConfig(
            maxHorizontalDeviation: 0.2,
            maxVerticalDeviation: 0.2,
            minMatchDistance: 0.35,    // Slightly more lenient
            maxMatchingDistance: 0.65, // Increased for horizontal
            maxActiveTracks: 40,
            maxLostTracks: 20
        ),

        "rightToLeft": SharedConfig(
            maxHorizontalDeviation: 0.2,
            maxVerticalDeviation: 0.2,
            minMatchDistance: 0.35,    // Slightly more lenient
            maxMatchingDistance: 0.65, // Increased for horizontal
            maxActiveTracks: 40,
            maxLostTracks: 20
        )
    ]

    /// Direction-specific ByteTrack configurations
    @MainActor
    private static var directionByteTrackConfigs: [String: ByteTrackConfig] = [
        "topToBottom": defaultByteTrackConfig,
        "bottomToTop": defaultByteTrackConfig,
        "leftToRight": defaultByteTrackConfig,
        "rightToLeft": defaultByteTrackConfig
    ]

    /// Direction-specific OC-SORT configurations
    @MainActor
    private static var directionOCSortConfigs: [String: OCSortConfig] = [
        "topToBottom": defaultOCSortConfig,
        "bottomToTop": defaultOCSortConfig,
        "leftToRight": defaultOCSortConfig,
        "rightToLeft": defaultOCSortConfig
    ]

    // MARK: - Current Active Configurations

    /// Current active shared configuration
    @MainActor
    private static var currentSharedConfig: SharedConfig = defaultSharedConfig

    /// Current active ByteTrack configuration
    @MainActor
    private static var currentByteTrackConfig: ByteTrackConfig = defaultByteTrackConfig

    /// Current active OC-SORT configuration
    @MainActor
    private static var currentOCSortConfig: OCSortConfig = defaultOCSortConfig

    // MARK: - Public Access Methods

    /// Get current shared configuration
    @MainActor
    public static func shared() -> SharedConfig {
        return currentSharedConfig
    }

    /// Get current ByteTrack configuration
    @MainActor
    public static func bytetrack() -> ByteTrackConfig {
        return currentByteTrackConfig
    }

    /// Get current OC-SORT configuration
    @MainActor
    public static func ocsort() -> OCSortConfig {
        return currentOCSortConfig
    }

    /// Update all configurations based on counting direction
    @MainActor
    public static func updateParametersForCountingDirection(_ direction: Any) {
        let directionString: String

        // Convert direction to string
        switch String(describing: direction) {
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

        // Update all configurations
        currentSharedConfig = directionSharedConfigs[directionString] ?? defaultSharedConfig
        currentByteTrackConfig = directionByteTrackConfigs[directionString] ?? defaultByteTrackConfig
        currentOCSortConfig = directionOCSortConfigs[directionString] ?? defaultOCSortConfig
    }

    // MARK: - Backward Compatibility (DEPRECATED)

    // These getters maintain backward compatibility with existing ByteTracker code
    // Will be removed after ByteTracker refactoring is complete

    @MainActor public static var highMatchThreshold: Float {
        return currentByteTrackConfig.highMatchThreshold
    }

    @MainActor public static var lowMatchThreshold: Float {
        return currentByteTrackConfig.lowMatchThreshold
    }

    @MainActor public static var minMatchDistance: CGFloat {
        return currentSharedConfig.minMatchDistance
    }

    @MainActor public static var iouMatchThreshold: Float {
        return currentByteTrackConfig.iouMatchThreshold
    }

    @MainActor public static var maxTimeLost: Int {
        return currentByteTrackConfig.maxTimeLost
    }

    @MainActor public static var maxUnmatchedFrames: Int {
        return currentByteTrackConfig.maxUnmatchedFrames
    }

    @MainActor public static var requiredFramesForTrack: Int {
        return currentByteTrackConfig.requiredFramesForTrack
    }

    @MainActor public static var maxMatchingDistance: CGFloat {
        return currentSharedConfig.maxMatchingDistance
    }

    // TTL parameters
    @MainActor public static var defaultTTL: Int {
        return currentByteTrackConfig.defaultTTL
    }

    @MainActor public static var highConsistencyTTL: Int {
        return currentByteTrackConfig.highConsistencyTTL
    }

    @MainActor public static var mediumConsistencyTTL: Int {
        return currentByteTrackConfig.mediumConsistencyTTL
    }

    @MainActor public static var lowConsistencyTTL: Int {
        return currentByteTrackConfig.lowConsistencyTTL
    }

    @MainActor public static var reactivationHighTTL: Int {
        return currentByteTrackConfig.reactivationHighTTL
    }

    @MainActor public static var reactivationMediumTTL: Int {
        return currentByteTrackConfig.reactivationMediumTTL
    }

    @MainActor public static var reactivationLowTTL: Int {
        return currentByteTrackConfig.reactivationLowTTL
    }

    // Movement constraints
    @MainActor public static var maxHorizontalDeviation: CGFloat {
        return currentSharedConfig.maxHorizontalDeviation
    }

    @MainActor public static var maxVerticalDeviation: CGFloat {
        return currentSharedConfig.maxVerticalDeviation
    }

    // Consistency rates
    @MainActor public static var consistencyIncreaseRate: CGFloat {
        return currentByteTrackConfig.consistencyIncreaseRate
    }

    @MainActor public static var consistencyDecreaseRate: CGFloat {
        return currentByteTrackConfig.consistencyDecreaseRate
    }

    @MainActor public static var reactivationConsistencyIncreaseRate: CGFloat {
        return currentByteTrackConfig.reactivationConsistencyIncreaseRate
    }

    @MainActor public static var reactivationConsistencyDecreaseRate: CGFloat {
        return currentByteTrackConfig.reactivationConsistencyDecreaseRate
    }
}
