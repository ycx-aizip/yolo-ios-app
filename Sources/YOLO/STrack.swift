// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  This file is part of the Ultralytics YOLO Package, implementing single-track object tracking.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//
//  STrack (Single Track) represents a single object track in the ByteTrack algorithm.
//  It maintains the state of a tracked object, including its position, velocity,
//  and tracking status. It works with KalmanFilter for motion prediction.

import Foundation
import CoreGraphics

/// Track state enumeration
enum TrackState {
    case new        // New track that was just created
    case tracked    // Track that has been successfully tracked in the current frame
    case lost       // Track that was not matched in the current frame
    case removed    // Track that has been lost for too many frames and should be removed
}

/// STrack class represents a single object track
public class STrack {
    /// Unique identifier for this track
    var trackId: Int
    
    /// Current bounding box of the tracked object
    var bbox: CGRect
    
    /// Confidence score
    var score: Float
    
    /// Class label
    var classId: Int
    
    /// Current tracking state
    var state: TrackState = .new
    
    /// Flag indicating if this track has been counted
    var isCounted: Bool = false
    
    /// Last known position (for threshold crossing detection)
    var lastPosition: CGPoint
    
    /// Frame count since last successful update
    var framesSinceUpdate: Int = 0
    
    /// Maximum number of frames to keep a lost track
    let maxFramesToKeep: Int = 30
    
    /// Initializes a new track
    /// - Parameters:
    ///   - trackId: Unique identifier for this track
    ///   - bbox: Initial bounding box
    ///   - score: Confidence score
    ///   - classId: Class identifier
    init(trackId: Int, bbox: CGRect, score: Float, classId: Int) {
        self.trackId = trackId
        self.bbox = bbox
        self.score = score
        self.classId = classId
        self.lastPosition = CGPoint(x: bbox.midX, y: bbox.midY)
        
        print("DEBUG: Created new STrack with ID \(trackId), box: \(bbox)")
    }
    
    /// Updates the track with a new detection
    /// - Parameter detection: New detection information
    func update(bbox: CGRect, score: Float) {
        self.bbox = bbox
        self.score = score
        self.state = .tracked
        self.framesSinceUpdate = 0
        
        // Update last position to current position
        let currentPosition = CGPoint(x: bbox.midX, y: bbox.midY)
        print("DEBUG: Updated STrack \(trackId), moved from \(lastPosition) to \(currentPosition)")
        self.lastPosition = currentPosition
    }
    
    /// Marks the track as lost if it wasn't matched in the current frame
    func markAsLost() {
        self.state = .lost
        self.framesSinceUpdate += 1
        print("DEBUG: STrack \(trackId) marked as lost, frames since update: \(framesSinceUpdate)")
    }
    
    /// Checks if the track should be removed due to being lost for too long
    func shouldBeRemoved() -> Bool {
        return state == .lost && framesSinceUpdate > maxFramesToKeep
    }
    
    /// Predicts the new position using motion estimation
    /// - Note: This is a placeholder for Kalman filter integration
    func predict() {
        print("DEBUG: Predicting position for STrack \(trackId)")
        // Will integrate with KalmanFilter in the full implementation
    }
    
    /// Gets the current center point of the bounding box
    var center: CGPoint {
        return CGPoint(x: bbox.midX, y: bbox.midY)
    }
} 