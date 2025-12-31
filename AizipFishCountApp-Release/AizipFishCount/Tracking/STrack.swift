// from Aizip
//
//  STrack.swift
//  YOLO
//
//  Single object tracking representation that uses Kalman filtering for state estimation.
//  Adapts Python's STrack class from Ultralytics' ByteTrack implementation.

import Foundation
import UIKit

// CountingDirection needs to be defined in the same module (YOLO) to be accessible

/**
 * TrackState
 *
 * Maps to Python Implementation:
 * - Corresponds to track state logic in `basetrack.py` and `bytetrack.py`
 * - Replaces the Python implementation's use of class variables like `STrack.New`
 */
/// Track state enumeration for object tracking
public enum TrackState {
    case new      // Newly created track
    case tracked  // Confirmed track that is actively being tracked
    case lost     // Track that was not matched in recent frames
    case removed  // Track that is removed from tracking
}

/**
 * STrack
 *
 * Maps to Python Implementation:
 * - Primary Correspondence: `STrack` class in `ultralytics/trackers/bytetrack.py`
 * - Core functionality:
 *   - Represents a single tracked object
 *   - Maintains state and position information
 *   - Updates track position using Kalman filter
 *   - Manages track lifecycle (new → tracked → lost → removed)
 *
 * Implementation Details:
 * - Uses 8D Kalman filter (x, y, aspect ratio, height, and velocities)
 * - Provides methods for track activation, updating, and prediction
 * - Tracks object state through the detection process
 */
/// Single object tracking representation with Kalman filtering
public class STrack {
    // MARK: - Class properties
    
    /// Shared Kalman filter for efficiency
    @MainActor
    private static let sharedKalmanFilter = KalmanFilter()
    
    /// Counter for generating unique track IDs (always increasing, never reused)
    @MainActor
    private static var count: Int = 0
    
    /// Static property to determine the expected movement direction (defaults to top-to-bottom)
    @MainActor
    public static var expectedMovementDirection: CountingDirection = .topToBottom
    
    // MARK: - Instance properties
    
    /// Unique identifier for this track
    public let trackId: Int
    
    /// Current position of the tracked object (normalized coordinates)
    public var position: (x: CGFloat, y: CGFloat)
    
    /// Current state of the track
    public var state: TrackState = .new
    
    /// Flag indicating whether this object has been counted
    public var counted: Bool = false
    
    /// Time-to-live counter for the track (decremented when object not detected)
    public var ttl: Int
    
    /// The most recent detection box associated with this track
    public var lastDetection: Box?
    
    /// Current confidence score
    public var score: Float
    
    /// Class label
    public var cls: String
    
    /// Frame where the track was first detected
    public var startFrame: Int = 0
    
    /// Most recent frame where the track was detected
    public var endFrame: Int = 0
    
    /// Whether this track has been activated
    public var isActivated: Bool = false
    
    /// Length of the track history (number of frames)
    public var trackletLen: Int = 0
    
    /// Mean state vector for Kalman filter
    public var mean: [Float]?
    
    /// Covariance matrix for Kalman filter
    public var covariance: [Float]?
    
    /// Kalman filter for this track
    private var kalmanFilter: KalmanFilter?

    /// Movement direction consistency score (ByteTrack-specific, managed externally)
    /// This property is public for external trackers to read/write, but STrack doesn't modify it
    public var movementConsistency: Float = 0.0

    /// Frame count with expected movement (ByteTrack-specific, managed externally)
    /// This property is public for external trackers to read/write, but STrack doesn't modify it
    public var framesWithExpectedMovement: Int = 0

    // MARK: - OC-SORT Observation History (for observation-centric recovery)

    /// Historical observations indexed by age (frame number)
    /// Format: [age: (x, y, w, h)] in normalized coordinates
    /// Used by OC-SORT for observation-centric recovery when track is lost
    private var observations: [Int: (x: Float, y: Float, w: Float, h: Float)] = [:]

    /// Age of track (incremented each frame, used for observation indexing)
    /// Corresponds to frame count since track creation
    public private(set) var age: Int = 0

    /// Last observation bbox (for OC-SORT's observation-centric recovery)
    /// Stores the most recent detection box for fallback predictions
    public private(set) var lastObservation: Box?

    // MARK: - Initialization
    
    /// Create a new track
    /// - Parameters:
    ///   - trackId: Unique identifier for this track
    ///   - position: Initial position of the tracked object
    ///   - detection: The detection box associated with this track
    ///   - score: Detection confidence score
    ///   - cls: Class label
    @MainActor
    public init(trackId: Int, position: (x: CGFloat, y: CGFloat), detection: Box?, score: Float, cls: String) {
        self.trackId = trackId
        self.position = position
        self.lastDetection = detection
        self.score = score
        self.cls = cls
        self.ttl = TrackingParameters.defaultTTL
    }
    
    /// Backward compatibility initializer
    /// - Parameters:
    ///   - trackId: Unique identifier for this track
    ///   - position: Initial position of the tracked object
    ///   - detection: The detection box associated with this track
    @MainActor
    public convenience init(trackId: Int, position: (x: CGFloat, y: CGFloat), detection: Box?) {
        self.init(trackId: trackId, position: position, detection: detection, score: Float(detection?.conf ?? 1.0), cls: detection?.cls ?? "")
    }
    
    // MARK: - State Management
    
    /**
     * activate
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `activate()` method in `STrack` class
     * - Initializes Kalman filter state with first detection
     * - Transitions track from 'new' to 'tracked' state
     */
    @MainActor
    public func activate(kalmanFilter: KalmanFilter, frameId: Int) {
        self.kalmanFilter = kalmanFilter
        
        // Initialize Kalman filter with detection
        let xyah = convertToXYAH()
        (self.mean, self.covariance) = kalmanFilter.initiate(measurement: xyah)
        
        // Update track state
        self.state = .tracked
        self.trackletLen = 0
        
        if frameId == 1 {
            self.isActivated = true
        }
        
        self.startFrame = frameId
        self.endFrame = frameId
    }
    
    /**
     * reactivate
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `re_activate()` method in `STrack` class
     * - Updates Kalman filter with new detection for a previously lost track
     * - Transitions track from 'lost' to 'tracked' state
     */
    @MainActor
    public func reactivate(newTrack: STrack, frameId: Int, newId: Bool = false) {
        guard let kalmanFilter = self.kalmanFilter, let mean = self.mean, let covariance = self.covariance else {
            return
        }

        // Update Kalman filter with new detection
        let newMeasurement = newTrack.convertToXYAH()
        (self.mean, self.covariance) = kalmanFilter.update(mean: mean, covariance: covariance, measurement: newMeasurement)

        // Update track state
        self.state = .tracked
        self.isActivated = true
        self.trackletLen = 0

        // Update metadata
        self.endFrame = frameId
        self.score = newTrack.score
        self.cls = newTrack.cls
        self.lastDetection = newTrack.lastDetection
        self.position = newTrack.position

        // TTL will be set by the tracker (e.g., ByteTrack) based on its specific logic
        // Default to medium TTL for reactivation if not set externally
        if self.ttl == 0 {
            self.ttl = TrackingParameters.defaultTTL
        }
    }
    
    /**
     * update
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `update()` method in `STrack` class
     * - Updates track state with new detection
     * - Updates position and Kalman filter state
     */
    @MainActor
    public func update(newPosition: (x: CGFloat, y: CGFloat), detection: Box?, newScore: Float, frameId: Int) {
        guard let kalmanFilter = self.kalmanFilter, let mean = self.mean, let covariance = self.covariance else {
            return
        }

        // Update position
        self.position = newPosition
        self.lastDetection = detection
        self.score = newScore
        self.cls = detection?.cls ?? self.cls

        // Update Kalman filter with new detection
        let measurement = convertToXYAH()
        (self.mean, self.covariance) = kalmanFilter.update(mean: mean, covariance: covariance, measurement: measurement)

        // Update track state
        self.state = .tracked
        self.isActivated = true
        self.trackletLen += 1
        self.endFrame = frameId

        // TTL will be set by the tracker (e.g., ByteTrack) based on its specific logic
        // Default to medium TTL if not set externally
        if self.ttl == 0 {
            self.ttl = TrackingParameters.defaultTTL
        }
    }
    
    /// Decreases the time-to-live counter for this track.
    /// - Returns: true if the track is still alive, false if it should be considered lost.
    @MainActor
    public func decreaseTTL() -> Bool {
        ttl -= 1
        return ttl > 0
    }
    
    /// Marks this track as lost.
    public func markLost() {
        state = .lost
    }
    
    /// Marks this track as counted for fish counting purposes.
    public func markCounted() {
        counted = true
    }
    
    /// Marks this track as removed.
    @MainActor
    public func markRemoved() {
        state = .removed
    }
    
    // MARK: - Prediction
    
    /**
     * predict
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `predict()` method in `STrack` class
     * - Predicts new state using Kalman filter
     * - Updates position based on prediction
     */
    public func predict() {
        guard let kalmanFilter = self.kalmanFilter, let mean = self.mean, let covariance = self.covariance else {
            return
        }
        
        // Predict new state using Kalman filter
        (self.mean, self.covariance) = kalmanFilter.predict(mean: mean, covariance: covariance)
        
        // Convert predicted Kalman state to position
        if let mean = self.mean {
            self.position = (x: CGFloat(mean[0]), y: CGFloat(mean[1]))
        }
    }
    
    // MARK: - Helper Methods
    
    /// Converts track position to XYAH format used by Kalman filter.
    /// - Returns: [x, y, aspect_ratio, height] in normalized coordinates
    private func convertToXYAH() -> [Float] {
        let x = Float(position.x)
        let y = Float(position.y)
        
        // Default values if no detection is available
        let aspectRatio: Float = 1.0
        let height: Float = 0.1
        
        return [x, y, aspectRatio, height]
    }
    
    /// Releases any resources held by the track when it's no longer needed
    @MainActor
    public func cleanup() {
        // Release all references that might contribute to memory retention
        lastDetection = nil
        mean = nil
        covariance = nil
        kalmanFilter = nil
        
        // Reset primitive values to defaults where appropriate
        score = 0
        trackletLen = 0
        isActivated = false
        ttl = 0
        movementConsistency = 0
        framesWithExpectedMovement = 0
        
        // Note: We don't reset position, trackId, or state as they may be needed for reference
    }
    
    /// Get movement vector between current and previous position
    /// - Parameter previousPosition: The previous position to compare against
    /// - Returns: Movement vector as (dx, dy)
    public func getMovementVector(from previousPosition: (x: CGFloat, y: CGFloat)) -> (dx: CGFloat, dy: CGFloat) {
        return (dx: position.x - previousPosition.x, dy: position.y - previousPosition.y)
    }

    // MARK: - OC-SORT Specific Methods

    /**
     * Record observation at current age (for OC-SORT)
     *
     * Maps to Python Implementation:
     * - Corresponds to observation recording in OC-SORT's KalmanBoxTracker
     * - Stores historical observations for observation-centric recovery
     *
     * - Parameter bbox: Detection bounding box to record
     */
    public func recordObservation(bbox: Box) {
        // Use normalized coordinates
        let normalizedBox = bbox.xywhn
        let w = normalizedBox.maxX - normalizedBox.minX
        let h = normalizedBox.maxY - normalizedBox.minY
        let x = (normalizedBox.minX + normalizedBox.maxX) / 2
        let y = (normalizedBox.minY + normalizedBox.maxY) / 2

        observations[age] = (Float(x), Float(y), Float(w), Float(h))
        lastObservation = bbox

        // Limit observation history to prevent unbounded growth
        // Keep only the most recent observations (e.g., last 30 frames)
        let maxObservationHistory = 30
        if observations.count > maxObservationHistory {
            // Remove oldest observations
            let sortedAges = observations.keys.sorted()
            let agesToRemove = sortedAges.prefix(observations.count - maxObservationHistory)
            for ageToRemove in agesToRemove {
                observations.removeValue(forKey: ageToRemove)
            }
        }
    }

    /**
     * Get observation from k frames ago (OC-SORT's k_previous_obs)
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `k_previous_obs()` function in `ocsort.py`
     * - Retrieves observation from delta_t frames ago for velocity estimation
     *
     * - Parameter deltaT: Number of frames to look back (typically 3)
     * - Returns: Observation tuple (x, y, w, h) or nil if not available
     */
    public func getPreviousObservation(deltaT: Int) -> (x: Float, y: Float, w: Float, h: Float)? {
        // Try to find observation deltaT frames ago
        // If exact match not found, try nearby frames (OC-SORT behavior)
        for i in 0..<deltaT {
            let targetAge = age - (deltaT - i)
            if let obs = observations[targetAge] {
                return obs
            }
        }

        // Fallback to most recent observation if delta_t observation not found
        if let maxAge = observations.keys.max() {
            return observations[maxAge]
        }

        return nil
    }

    /**
     * Increment age (called each predict)
     *
     * Maps to Python Implementation:
     * - Corresponds to age increment in OC-SORT's update loop
     * - Used for indexing observations by frame
     */
    public func incrementAge() {
        age += 1
    }

    /**
     * Clear observation history (for memory management)
     *
     * Called when track is removed to free memory
     */
    public func clearObservations() {
        observations.removeAll()
        lastObservation = nil
    }

    // MARK: - Class Methods
    
    /// Gets the next available track ID (always a new, higher ID - never reuses IDs)
    @MainActor
    public static func nextId() -> Int {
        // Simply increment the counter and return the new value
        // We never reuse IDs to ensure each fish has a truly unique identifier
        count += 1
        return count
    }
    
    /// Resets the ID counter - only use when resetting the entire tracking system
    @MainActor
    public static func resetId() {
        count = 0
    }
    
    /// Predicts new states for multiple tracks
    @MainActor
    public static func multiPredict(tracks: [STrack]) {
        for track in tracks {
            track.predict()
        }
    }
    
    /// Updates multiple tracks with detections
    /// - Parameters:
    ///   - tracks: Array of tracks to update
    ///   - dets: Array of detections to update with
    ///   - frameId: Current frame ID
    @MainActor
    public static func updateMulti(tracks: [STrack], dets: [STrack], frameId: Int) {
        for i in 0..<min(tracks.count, dets.count) {
            tracks[i].update(
                newPosition: dets[i].position,
                detection: dets[i].lastDetection,
                newScore: dets[i].score,
                frameId: frameId
            )
        }
    }
} 