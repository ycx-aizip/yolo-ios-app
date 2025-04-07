// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  STrack.swift
//  YOLO
//
//  A simple track representation for object tracking.
//  This is a placeholder implementation for debugging the app flow.

import Foundation
import UIKit

/// Simple representation of a tracked object
public class STrack {
    /// Unique identifier for this track
    public let trackId: Int
    
    /// Current position of the tracked object (normalized coordinates)
    public var position: (x: CGFloat, y: CGFloat)
    
    /// Flag indicating whether this object has been counted
    public var counted: Bool = false
    
    /// Time-to-live counter for the track (decremented when object not detected)
    public var ttl: Int = 5
    
    /// The most recent detection box associated with this track
    public var lastDetection: Box?
    
    /// Create a new track
    /// - Parameters:
    ///   - trackId: Unique identifier for this track
    ///   - position: Initial position of the tracked object
    ///   - detection: The detection box associated with this track
    public init(trackId: Int, position: (x: CGFloat, y: CGFloat), detection: Box?) {
        self.trackId = trackId
        self.position = position
        self.lastDetection = detection
        print("STrack: Created new track with ID \(trackId) at position (\(position.x), \(position.y))")
    }
    
    /// Update the track with a new detection
    /// - Parameters:
    ///   - newPosition: The new position of the tracked object
    ///   - detection: The new detection box
    public func update(newPosition: (x: CGFloat, y: CGFloat), detection: Box?) {
        self.position = newPosition
        self.lastDetection = detection
        self.ttl = 5  // Reset TTL
        print("STrack: Updated track \(trackId) to position (\(newPosition.x), \(newPosition.y))")
    }
    
    /// Decrease the TTL of the track
    /// - Returns: Whether the track is still alive
    public func decreaseTTL() -> Bool {
        ttl -= 1
        return ttl > 0
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
} 