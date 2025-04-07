// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  ByteTracker.swift
//  YOLO
//
//  Core tracking algorithm using ByteTrack approach.
//  Adapted from the Python ByteTracker implementation with optimizations for Swift.

import Foundation
import UIKit

/**
 * ByteTracker
 *
 * Maps to Python Implementation:
 * - Primary Correspondence: `BYTETracker` class in `ultralytics/trackers/bytetrack.py`
 * - Core functionality:
 *   - Manages track objects across frames
 *   - Associates new detections with existing tracks
 *   - Handles track lifecycle (create, update, delete)
 *   - Maintains consistent track IDs across frames
 *
 * Implementation Details:
 * - Follows ByteTrack algorithm principles from the BYTE paper
 * - Implements two-stage matching for robust tracking
 * - Uses Kalman filter for motion prediction
 * - Handles track status transitions (activated, lost, removed)
 */
@MainActor
public class ByteTracker {
    // MARK: - Properties
    
    /// Tracks that are currently being tracked
    private var activeTracks: [STrack] = []
    
    /// Tracks that were recently lost and might be recovered
    private var lostTracks: [STrack] = []
    
    /// Tracks that were removed from tracking
    private var removedTracks: [STrack] = []
    
    /// Current frame ID
    private var frameId: Int = 0
    
    /// Shared Kalman filter for track state estimation
    private let kalmanFilter = KalmanFilter()
    
    /// Matching threshold for high confidence detections
    private let highThreshold: Float = 0.6
    
    /// Matching threshold for low confidence detections
    private let lowThreshold: Float = 0.3
    
    /// Max time to keep a track in lost state
    private let maxTimeLost: Int = 30
    
    // MARK: - Initialization
    
    public init() {
        print("ByteTracker: Initialized")
    }
    
    // MARK: - Public Methods
    
    /**
     * update
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `update()` method in BYTETracker class
     * - Core tracking algorithm that processes new detections and updates tracks
     * - Implements first and second stage matching similar to Python version
     * - Handles creation of new tracks and lifecycle management
     */
    public func update(detections: [Box], scores: [Float], classes: [String]) -> [STrack] {
        frameId += 1
        clearResetCounter()
        
        // Convert detections to STrack objects
        var detTrackArr: [STrack] = []
        
        for i in 0..<detections.count {
            let detection = detections[i]
            let score = scores[i]
            let cls = classes[i]
            
            // Get bbox center
            let bbox = detection.xywhn
            let center = (
                x: (bbox.minX + bbox.maxX) / 2,
                y: (bbox.minY + bbox.maxY) / 2
            )
            
            // Create STrack from detection
            let newTrack = STrack(
                trackId: STrack.nextId(), 
                position: center,
                detection: detection,
                score: score,
                cls: cls
            )
            
            detTrackArr.append(newTrack)
        }
        
        // Filter high score detections
        let remainedDetections = detTrackArr
        
        // Predict new locations for active tracks
        let predActiveTracks = activeTracks
        STrack.multiPredict(tracks: predActiveTracks)
        
        // Match with high score detections first
        let (firstMatches, firstUnmatchedTracks, firstUnmatchedDetections) = 
            MatchingUtils.associateFirstStage(
                tracks: predActiveTracks,
                detections: remainedDetections,
                thresholdFirstStage: highThreshold
            )
        
        // Update matched tracks with detections
        for (trackIdx, detIdx) in firstMatches {
            let track = predActiveTracks[trackIdx]
            let detection = remainedDetections[detIdx]
            
            track.update(
                newPosition: detection.position, 
                detection: detection.lastDetection,
                newScore: detection.score,
                frameId: frameId
            )
        }
        
        // Create array of unmatched tracks
        var unmatchedTrackArray: [STrack] = firstUnmatchedTracks.map { predActiveTracks[$0] }
        
        // Handle lost tracks
        let lostTracksCopy = lostTracks
        
        // Predict new locations for lost tracks
        STrack.multiPredict(tracks: lostTracksCopy)
        
        // Match remaining detections with lost tracks
        let (secondMatches, secondUnmatchedDetections) = 
            MatchingUtils.associateSecondStage(
                tracks: lostTracksCopy,
                detections: firstUnmatchedDetections.map { remainedDetections[$0] },
                thresholdSecondStage: lowThreshold
            )
        
        // Reactivate matched lost tracks
        for (trackIdx, detIdx) in secondMatches {
            let lostTrack = lostTracksCopy[trackIdx]
            let detection = remainedDetections[firstUnmatchedDetections[detIdx]]
            
            lostTrack.reactivate(
                newTrack: detection,
                frameId: frameId
            )
        }
        
        // Mark unmatched tracks as lost
        for trackIdx in firstUnmatchedTracks {
            let track = predActiveTracks[trackIdx]
            if !track.decreaseTTL() {
                track.markLost()
            }
        }
        
        // Delete lost tracks that have been lost for too long
        let lostTracksCopyFiltered = lostTracksCopy.filter { track in
            let timeLost = frameId - track.endFrame
            return timeLost < maxTimeLost
        }
        
        for track in lostTracksCopy {
            let timeLost = frameId - track.endFrame
            if timeLost >= maxTimeLost {
                track.markRemoved()
                removedTracks.append(track)
            }
        }
        
        // Create new tracks for unmatched detections
        var newTracks: [STrack] = []
        for detIdx in secondUnmatchedDetections {
            let originalDetIdx = firstUnmatchedDetections[detIdx]
            let detection = remainedDetections[originalDetIdx]
            
            // Only create tracks for high confidence detections
            if detection.score >= 0.5 {
                detection.activate(kalmanFilter: kalmanFilter, frameId: frameId)
                newTracks.append(detection)
            }
        }
        
        // Update tracked tracks and lost tracks
        activeTracks = []
        
        for track in predActiveTracks {
            if track.state == .tracked {
                activeTracks.append(track)
            } else if track.state == .lost {
                lostTracks.append(track)
            }
        }
        
        // Add reactivated lost tracks to active tracks
        for track in lostTracksCopyFiltered {
            if track.state == .tracked {
                activeTracks.append(track)
            }
        }
        
        // Add new tracks to active tracks
        activeTracks.append(contentsOf: newTracks)
        
        // Filter possible duplicate tracks
        let duplicates = MatchingUtils.filterTracks(tracks: activeTracks)
        
        // Remove duplicated tracks
        for idx in duplicates.sorted(by: >) {
            activeTracks.remove(at: idx)
        }
        
        print("ByteTracker: Updated with \(detections.count) detections, now tracking \(activeTracks.count) objects")
        return activeTracks
    }
    
    /// For backward compatibility with existing code
    /// - Parameter detections: Array of detection position and box tuples
    /// - Returns: Array of updated tracks
    @MainActor
    public func update(detections: [(position: (x: CGFloat, y: CGFloat), box: Box)]) -> [STrack] {
        // Convert the old format to the new format
        var boxes: [Box] = []
        var scores: [Float] = []
        var classes: [String] = []
        
        for detection in detections {
            boxes.append(detection.box)
            scores.append(1.0) // Default score
            classes.append(detection.box.cls) // Use class from box
        }
        
        return update(detections: boxes, scores: scores, classes: classes)
    }
    
    /// Reset the tracker, clearing all tracks
    @MainActor
    public func reset() {
        activeResets = 0
        activeTracks = []
        lostTracks = []
        removedTracks = []
        frameId = 0
        STrack.resetId()
        print("ByteTracker: Reset all tracking state")
    }
    
    /// Counter for tracking consecutive reset calls
    private var activeResets: Int = 0
    
    /// Reset frame counter without clearing tracks
    /// This supports scenarios where frames might be skipped
    @MainActor
    public func resetFrame() {
        activeResets += 1
        
        // After 3 consecutive reset calls, perform a full reset
        if activeResets >= 3 {
            reset()
        } else {
            // Just reset frame counter
            frameId = 0
            print("ByteTracker: Reset frame counter, maintaining tracks")
        }
    }
    
    /// Called when update is invoked to clear the reset counter
    private func clearResetCounter() {
        activeResets = 0
    }
    
    /// Get current tracks for backward compatibility
    /// - Returns: Array of current active tracks
    public func getTracks() -> [STrack] {
        return activeTracks
    }
    
    // MARK: - Debug Methods
    
    /// Get current tracking stats
    /// - Returns: Dictionary with tracking statistics
    public func getStats() -> [String: Int] {
        return [
            "active_tracks": activeTracks.count,
            "lost_tracks": lostTracks.count,
            "removed_tracks": removedTracks.count,
            "frame_id": frameId
        ]
    }
} 