// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  STrack.swift
//  YOLO
//
//  Single object tracking representation that uses Kalman filtering for state estimation.
//  Adapts Python's STrack class from Ultralytics' ByteTrack implementation.

import Foundation
import UIKit

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
    
    /// Counter for generating unique track IDs
    @MainActor
    private static var count: Int = 0
    
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
    public var ttl: Int = 5
    
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
    
    // MARK: - Initialization
    
    /// Create a new track
    /// - Parameters:
    ///   - trackId: Unique identifier for this track
    ///   - position: Initial position of the tracked object
    ///   - detection: The detection box associated with this track
    ///   - score: Detection confidence score
    ///   - cls: Class label
    public init(trackId: Int, position: (x: CGFloat, y: CGFloat), detection: Box?, score: Float, cls: String) {
        self.trackId = trackId
        self.position = position
        self.lastDetection = detection
        self.score = score
        self.cls = cls
        print("STrack: Created new track with ID \(trackId) at position (\(position.x), \(position.y))")
    }
    
    /// Backward compatibility initializer
    /// - Parameters:
    ///   - trackId: Unique identifier for this track
    ///   - position: Initial position of the tracked object
    ///   - detection: The detection box associated with this track
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
    }
    
    /**
     * update
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `update()` method in `STrack` class
     * - Updates track state with new detection
     * - Updates Kalman filter with new measurement
     */
    public func update(newPosition: (x: CGFloat, y: CGFloat), detection: Box?, newScore: Float? = nil, frameId: Int) {
        // Add position smoothing for more stable tracking during camera motion
        // Use exponential moving average to smooth position updates
        // This makes tracking more robust to jitter
        let alpha: CGFloat = 0.7 // Weight for new position (higher = less smoothing)
        self.position = (
            x: newPosition.x * alpha + self.position.x * (1 - alpha),
            y: newPosition.y * alpha + self.position.y * (1 - alpha)
        )
        
        self.lastDetection = detection
        
        if let newScore = newScore {
            self.score = newScore
        }
        
        self.endFrame = frameId
        self.trackletLen += 1
        self.ttl = 5  // Reset TTL
        
        // Update Kalman filter if available
        if let kalmanFilter = self.kalmanFilter, let mean = self.mean, let covariance = self.covariance {
            let measurement = convertToXYAH()
            (self.mean, self.covariance) = kalmanFilter.update(mean: mean, covariance: covariance, measurement: measurement)
        }
        
        self.state = .tracked
        self.isActivated = true
        
        print("STrack: Updated track \(trackId) to position (\(position.x), \(position.y))")
    }
    
    /// Backward compatibility update method
    /// - Parameters:
    ///   - newPosition: The new position of the tracked object
    ///   - detection: The new detection box
    public func update(newPosition: (x: CGFloat, y: CGFloat), detection: Box?) {
        update(newPosition: newPosition, detection: detection, frameId: endFrame + 1)
    }
    
    /**
     * predict
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `predict()` method in `STrack` class
     * - Uses Kalman filter to predict new position based on motion model
     * - Called during tracking to predict position before matching
     */
    public func predict() {
        if let kalmanFilter = self.kalmanFilter, let mean = self.mean, let covariance = self.covariance {
            var stateMean = mean
            
            // If track is not actively tracked, set velocity to 0
            if self.state != .tracked {
                stateMean[7] = 0
            }
            
            (self.mean, self.covariance) = kalmanFilter.predict(mean: stateMean, covariance: covariance)
            
            // Update position from Kalman prediction
            if let mean = self.mean {
                // Extract position from mean (center_x, center_y)
                self.position = (x: CGFloat(mean[0]), y: CGFloat(mean[1]))
            }
        }
    }
    
    /**
     * multiPredict
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `multi_predict()` static method in `STrack` class
     * - Batch prediction for multiple tracks
     * - Applies Kalman prediction to all tracks in the list
     */
    @MainActor
    public static func multiPredict(tracks: [STrack]) {
        for track in tracks {
            track.predict()
        }
    }
    
    /// Mark track as lost
    public func markLost() {
        self.state = .lost
        print("STrack: Track \(trackId) marked as lost")
    }
    
    /// Mark track as removed
    public func markRemoved() {
        self.state = .removed
        print("STrack: Track \(trackId) marked as removed")
    }
    
    /// Mark this track as counted
    public func markCounted() {
        counted = true
        print("STrack: Track \(trackId) marked as counted")
    }
    
    /// Mark this track as not counted
    public func markUncounted() {
        counted = false
        print("STrack: Track \(trackId) marked as uncounted")
    }
    
    /// Decrease the TTL of the track
    /// - Returns: Whether the track is still alive
    public func decreaseTTL() -> Bool {
        ttl -= 1
        return ttl > 0
    }
    
    // MARK: - Helper methods
    
    /**
     * nextId
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: Similar to `_count` class variable in Python `STrack`
     * - Generates unique IDs for new tracks
     */
    @MainActor
    public static func nextId() -> Int {
        STrack.count += 1
        return STrack.count
    }
    
    /// Reset the track ID counter
    @MainActor
    public static func resetId() {
        STrack.count = 0
    }
    
    /**
     * convertToXYAH
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `tlwh_to_xyah()` method in `STrack` class
     * - Converts bounding box to format used by Kalman filter
     * - [x_center, y_center, aspect_ratio, height]
     */
    private func convertToXYAH() -> [Float] {
        guard let detection = lastDetection else {
            // If no detection, use position
            return [
                Float(position.x),
                Float(position.y),
                1.0,  // Default aspect ratio
                0.1   // Default height
            ]
        }
        
        let bbox = detection.xywhn
        let x = (bbox.minX + bbox.maxX) / 2
        let y = (bbox.minY + bbox.maxY) / 2
        let aspect = bbox.width / bbox.height
        let height = bbox.height
        
        return [
            Float(x),
            Float(y),
            Float(aspect),
            Float(height)
        ]
    }
    
    /// Convert XYAH format to normalized bounding box
    /// - Parameter xyah: Array containing [x, y, aspect, height]
    /// - Returns: Normalized bounding box
    private func xyahToXYWHN(xyah: [Float]) -> CGRect {
        let x = CGFloat(xyah[0])
        let y = CGFloat(xyah[1])
        let aspect = CGFloat(xyah[2])
        let height = CGFloat(xyah[3])
        let width = height * aspect
        
        return CGRect(
            x: x - width/2,
            y: y - height/2,
            width: width,
            height: height
        )
    }
    
    /// Update the position directly (used for camera motion compensation)
    /// - Parameter newPosition: The new position (x, y) for this track
    public func updatePosition(newPosition: (x: CGFloat, y: CGFloat)) {
        self.position = newPosition
    }
} 