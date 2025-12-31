// TrackerProtocol.swift
// Protocol abstraction for multi-object tracking algorithms
//
// This protocol defines a common interface for different tracking algorithms
// (e.g., ByteTrack, OC-SORT) to be used interchangeably with the fish counting system.

import Foundation

/// Protocol defining the interface for multi-object tracking algorithms
@MainActor
public protocol TrackerProtocol {
    /// Update tracker with new detections and return updated tracks
    ///
    /// - Parameters:
    ///   - detections: Array of detected bounding boxes
    ///   - scores: Confidence scores for each detection
    ///   - classes: Class labels for each detection
    /// - Returns: Array of tracked objects with persistent IDs
    func update(detections: [Box], scores: [Float], classes: [String]) -> [STrack]
    
    /// Reset tracker state, clearing all tracks and history
    func reset()
    
    /// Get currently active tracks
    ///
    /// - Returns: Array of active tracked objects
    func getActiveTracks() -> [STrack]
}

