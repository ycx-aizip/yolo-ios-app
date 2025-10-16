// from Aizip
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
    
    /// Track history for biasing matching towards prior associations
    private var trackMatchHistory: [Int: Set<Int>] = [:] // Track ID -> Set of previously matched detection IDs
    
    // Potential tracks
    /// Buffer for potential new tracks - stores potential tracks before assigning real IDs
    /// Key: temporary ID, Value: (position, detection, confidence, class, framesObserved, lastUpdatedFrame)
    private var potentialTracks: [Int: (position: (x: CGFloat, y: CGFloat), detection: Box, score: Float, cls: String, frames: Int, lastFrame: Int)] = [:]
    
    /// Counter for temporary IDs
    private var tempIdCounter: Int = 0
    
    /// Estimated camera motion between frames
    private var lastFrameDetectionCenters: [(x: CGFloat, y: CGFloat)] = []
    private var estimatedCameraMotion: (dx: CGFloat, dy: CGFloat) = (0, 0)
    
    // Maximum number of tracks to maintain in each collection to prevent unbounded growth
    private let maxActiveTracks: Int = 100
    private let maxLostTracks: Int = 50
    private let maxRemovedTracks: Int = 50
    private let maxPotentialTracks: Int = 50
    
    // MARK: - Initialization

    /// Initialize ByteTracker
    ///
    /// - Note: ByteTracker accesses configuration dynamically via TrackingUtils static methods
    nonisolated public init() {
        // Configuration accessed via TrackingUtils static methods during update() calls
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
        
        // Clean up collections before processing new detections
        limitCollectionSizes()
        
        // Estimate camera motion if we have previous frame data using helper
        let currentCenters = detections.map { box in
            let bbox = box.xywhn
            return (x: (bbox.minX + bbox.maxX) / 2, y: (bbox.minY + bbox.maxY) / 2)
        }
        estimatedCameraMotion = ByteTrackHelpers.estimateCameraMotion(
            currentCenters: currentCenters,
            previousCenters: lastFrameDetectionCenters,
            expectedDirection: STrack.expectedMovementDirection,
            previousMotion: estimatedCameraMotion
        )
        if !detections.isEmpty {
            lastFrameDetectionCenters = currentCenters
        }
        
        // Removed tracks are no longer needed as they're never reactivated
        // We can just clear them instead of keeping them around
        removedTracks.removeAll(keepingCapacity: true)
        
        // Convert detections to STrack objects - preallocate capacity for better performance
        var detTrackArr: [STrack] = []
        detTrackArr.reserveCapacity(detections.count)
        
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
            // Use a temporary ID for now (will be replaced with real ID if track persists)
            let newTrack = STrack(
                trackId: -1, // Temporary ID will be replaced later
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
        
        // Apply motion compensation to track predictions if significant camera motion detected
        ByteTrackHelpers.applyMotionCompensation(
            to: predActiveTracks,
            motion: estimatedCameraMotion,
            expectedDirection: STrack.expectedMovementDirection
        )
        
        // Match with high score detections first - using the improved matching process
        let (firstMatches, firstUnmatchedTracks, firstUnmatchedDetections) =
            ByteTrackHelpers.improvedAssociateFirstStage(
                tracks: predActiveTracks,
                detections: remainedDetections,
                expectedDirection: STrack.expectedMovementDirection,
                matchHistory: trackMatchHistory,
                matchThreshold: TrackingParameters.highMatchThreshold
            )
        
        // Update matched tracks with detections
        for (trackIdx, detIdx) in firstMatches {
            let track = predActiveTracks[trackIdx]
            let detection = remainedDetections[detIdx]

            // Update movement consistency (ByteTrack-specific)
            ByteTrackHelpers.updateMovementConsistency(
                for: track,
                newPosition: detection.position,
                expectedDirection: STrack.expectedMovementDirection,
                isReactivation: false
            )

            track.update(
                newPosition: detection.position,
                detection: detection.lastDetection,
                newScore: detection.score,
                frameId: frameId
            )

            // Set adaptive TTL based on movement consistency (ByteTrack-specific)
            track.ttl = ByteTrackHelpers.calculateAdaptiveTTL(for: track, isReactivation: false)

            // Update track match history for future matching bias
            ByteTrackHelpers.updateMatchHistory(
                trackId: track.trackId,
                detectionId: detIdx,
                history: &trackMatchHistory
            )
        }
        
        // Create array of unmatched tracks
        var unmatchedTrackArray: [STrack] = firstUnmatchedTracks.map { predActiveTracks[$0] }
        
        // Handle lost tracks
        let lostTracksCopy = lostTracks
        
        // Predict new locations for lost tracks
        STrack.multiPredict(tracks: lostTracksCopy)
        
        // Apply motion compensation to lost tracks too
        ByteTrackHelpers.applyMotionCompensation(
            to: lostTracksCopy,
            motion: estimatedCameraMotion,
            expectedDirection: STrack.expectedMovementDirection
        )
        
        // Match remaining detections with lost tracks - using improved second stage matching
        let (secondMatches, secondUnmatchedDetections) =
            ByteTrackHelpers.improvedAssociateSecondStage(
                tracks: lostTracksCopy,
                detections: firstUnmatchedDetections.map { remainedDetections[$0] },
                expectedDirection: STrack.expectedMovementDirection,
                matchThreshold: TrackingParameters.lowMatchThreshold
            )
        
        // Reactivate matched lost tracks
        for (trackIdx, detIdx) in secondMatches {
            let lostTrack = lostTracksCopy[trackIdx]
            let detection = remainedDetections[firstUnmatchedDetections[detIdx]]

            // Update movement consistency for reactivation (ByteTrack-specific)
            ByteTrackHelpers.updateMovementConsistency(
                for: lostTrack,
                newPosition: detection.position,
                expectedDirection: STrack.expectedMovementDirection,
                isReactivation: true
            )

            lostTrack.reactivate(
                newTrack: detection,
                frameId: frameId
            )

            // Set adaptive TTL for reactivation (ByteTrack-specific)
            lostTrack.ttl = ByteTrackHelpers.calculateAdaptiveTTL(for: lostTrack, isReactivation: true)

            // Update track match history for future matching bias
            ByteTrackHelpers.updateMatchHistory(
                trackId: lostTrack.trackId,
                detectionId: firstUnmatchedDetections[detIdx],
                history: &trackMatchHistory
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
            return timeLost < TrackingParameters.maxTimeLost
        }
        
        for track in lostTracksCopy {
            let timeLost = frameId - track.endFrame
            if timeLost >= TrackingParameters.maxTimeLost {
                track.markRemoved()
                track.cleanup() // Release references
                // We don't need to store removed tracks since they'll never be reactivated
                // Just let them be garbage collected
            }
        }
        
        // Process unmatched detections through potential tracks buffer
        var newTracks: [STrack] = []
        
        // To reduce overhead, only process detections if we're not over capacity
        if activeTracks.count < maxActiveTracks {
            for detIdx in secondUnmatchedDetections {
                let originalDetIdx = firstUnmatchedDetections[detIdx]
                let detection = remainedDetections[originalDetIdx]
                
                if let actualDetection = detection.lastDetection, detection.score >= 0.35 { // Reduced threshold from 0.4
                    // Check if this detection matches any existing potential track
                    var matchedPotentialId: Int? = nil
                    var closestDistance: CGFloat = TrackingParameters.maxMatchingDistance
                    
                    for (tempId, potentialTrack) in potentialTracks {
                        // Compensate for camera motion when calculating distance
                        let adjustedDx = potentialTrack.position.x - detection.position.x + estimatedCameraMotion.dx
                        let adjustedDy = potentialTrack.position.y - detection.position.y + estimatedCameraMotion.dy
                        let distance = sqrt(adjustedDx*adjustedDx + adjustedDy*adjustedDy)
                        
                        if distance < closestDistance {
                            closestDistance = distance
                            matchedPotentialId = tempId
                        }
                    }
                    
                    if let matchedId = matchedPotentialId {
                        // Update existing potential track
                        let currentEntry = potentialTracks[matchedId]!
                        let newFrameCount = currentEntry.frames + 1
                        
                        potentialTracks[matchedId] = (
                            position: detection.position,
                            detection: actualDetection,
                            score: detection.score,
                            cls: detection.cls,
                            frames: newFrameCount,
                            lastFrame: frameId
                        )
                        
                        // If track has been observed for enough frames, create a real track
                        if newFrameCount >= TrackingParameters.requiredFramesForTrack {
                            // Create new track with real ID
                            let newTrack = STrack(
                                trackId: STrack.nextId(),
                                position: detection.position,
                                detection: actualDetection,
                                score: detection.score,
                                cls: detection.cls
                            )
                            
                            // Activate the track
                            newTrack.activate(kalmanFilter: kalmanFilter, frameId: frameId)
                            newTracks.append(newTrack)
                            
                            // Remove from potential tracks
                            potentialTracks.removeValue(forKey: matchedId)
                        }
                    } else if potentialTracks.count < maxPotentialTracks {
                        // Only create new potential track if we're under the limit
                        let tempId = tempIdCounter
                        tempIdCounter += 1
                        
                        potentialTracks[tempId] = (
                            position: detection.position,
                            detection: actualDetection,
                            score: detection.score,
                            cls: detection.cls,
                            frames: 1,
                            lastFrame: frameId
                        )
                    }
                }
            }
        }
        
        // Clean up potential tracks that haven't been updated recently
        let keysToRemove = potentialTracks.keys.filter { key in
            // Remove potential tracks that haven't been updated in several frames
            return frameId - potentialTracks[key]!.lastFrame > TrackingParameters.maxUnmatchedFrames
        }
        
        for key in keysToRemove {
            potentialTracks.removeValue(forKey: key)
        }
        
        // Update tracked tracks and lost tracks
        activeTracks = []
        
        // Only keep tracked tracks, move the rest to lost tracks
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
        
        // Limit collection sizes to ensure we don't exceed maximums
        limitCollectionSizes()
        
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
            scores.append(Float(detection.box.conf))
            classes.append(detection.box.cls)
        }
        
        return update(detections: boxes, scores: scores, classes: classes)
    }
    
    /// Reset the tracker, clearing all tracks
    @MainActor
    public func reset() {
        // Clear all track collections
        activeTracks.removeAll(keepingCapacity: true)
        lostTracks.removeAll(keepingCapacity: true)
        // removedTracks is no longer needed for tracking since we never reactivate removed tracks
        removedTracks.removeAll(keepingCapacity: true)
        potentialTracks.removeAll(keepingCapacity: true)
        trackMatchHistory.removeAll(keepingCapacity: true)
        
        // Reset frame counter
        frameId = 0
        
        // Reset camera motion estimation
        estimatedCameraMotion = (0, 0)
        lastFrameDetectionCenters.removeAll(keepingCapacity: true)
        
        // Reset ID counter in STrack class
        STrack.resetId()
        
        // Reset temporary ID counter
        tempIdCounter = 0
    }
    
    /// Clear the reset counter
    private func clearResetCounter() {
        // After a while, reset the STrack counter
        if frameId % 10000 == 0 {
            STrack.resetId()
        }
    }
    
    /// Limit sizes of all track collections to prevent memory growth
    private func limitCollectionSizes() {
        // Cap active tracks - priority is recent tracks with higher scores
        ByteTrackHelpers.limitCollectionSize(&activeTracks, maxSize: maxActiveTracks) { (a, b) -> Bool in
            if a.endFrame == b.endFrame {
                return a.score > b.score
            }
            return a.endFrame > b.endFrame
        }

        // Cap lost tracks - priority is recent tracks
        ByteTrackHelpers.limitCollectionSize(&lostTracks, maxSize: maxLostTracks) { $0.endFrame > $1.endFrame }

        // We no longer maintain removedTracks since we never reactivate removed tracks

        // Cap potential tracks - priority is tracks seen more times
        if potentialTracks.count > maxPotentialTracks {
            let sortedKeys = potentialTracks.keys.sorted {
                let trackA = potentialTracks[$0]!
                let trackB = potentialTracks[$1]!

                // Sort by frames observed, then by recency
                if trackA.frames == trackB.frames {
                    return trackA.lastFrame > trackB.lastFrame
                }
                return trackA.frames > trackB.frames
            }

            let keysToKeep = sortedKeys.prefix(maxPotentialTracks)
            let keysToRemove = Set(potentialTracks.keys).subtracting(keysToKeep)

            for key in keysToRemove {
                potentialTracks.removeValue(forKey: key)
            }
        }
    }
    
    /// Get current tracks for backward compatibility
    /// - Returns: Array of current active tracks
    public func getTracks() -> [STrack] {
        return activeTracks
    }
    
    // MARK: - Debug Methods
    
    /// Get current tracking stats
    /// - Returns: Dictionary with tracking statistics
    public func getStats() -> [String: Any] {
        return [
            "active_tracks": activeTracks.count,
            "lost_tracks": lostTracks.count,
            // Removed tracks are no longer maintained
            "removed_tracks": 0,
            "potential_tracks": potentialTracks.count,
            "frame_id": frameId,
            "camera_motion": [
                "dx": estimatedCameraMotion.dx,
                "dy": estimatedCameraMotion.dy
            ]
        ]
    }
    
    /// Remove old tracks from the removed tracks list to avoid memory growth
    private func removeOldRemovedTracks() {
        // Since we never reuse IDs and removed tracks are never reactivated,
        // we can just clear the entire removedTracks array
        for track in removedTracks {
            track.cleanup() // Release resources
        }
        removedTracks.removeAll(keepingCapacity: true)
    }
    
    /**
     * Set adaptive Time-To-Live (TTL) values for tracks based on position and density.
     * Fish in the middle of the frame (primary tracking area) get higher TTL values,
     * while fish near the edges get lower TTL values to clean up faster.
     * When there are many tracks, TTL values are reduced to improve performance.
     */
    // private func setAdaptiveTTL(for track: STrack) {
    //     let y = track.position.y
    //     let x = track.position.x
        
    //     // Calculate base TTL values based on position
    //     var baseTTL: Int
        
    //     // Higher TTL for fish in the primary tracking zone (middle of frame)
    //     if y > 0.2 && y < 0.8 && x > 0.1 && x < 0.9 {
    //         baseTTL = 15  // Higher TTL in primary tracking zone
    //     }
    //     // Middle TTL for fish in transition zones
    //     else if y > 0.1 && y < 0.9 && x > 0.05 && x < 0.95 {
    //         baseTTL = 10  // Default TTL in intermediate zones
    //     }
    //     // Lower TTL for fish near edges that may be entering/exiting
    //     else {
    //         baseTTL = 6  // Lower TTL near edges
    //     }
        
    //     // Adjust TTL based on track density
    //     let totalTracks = activeTracks.count + lostTracks.count
    //     // if totalTracks > 100 {
    //     //     baseTTL = max(2, baseTTL - 3)  // Significantly reduce TTL in high density scenarios
    //     // } else if totalTracks > 50 {
    //     //     baseTTL = max(3, baseTTL - 2)  // Moderately reduce TTL in medium density scenarios
    //     // } else if totalTracks > 30 {
    //     //     baseTTL = max(3, baseTTL - 1)  // Slightly reduce TTL in lower medium density scenarios
    //     // }
        
    //     track.ttl = baseTTL
        
    //     // Bonus TTL for fish moving in the expected direction (top to bottom)
    //     if let mean = track.mean, mean.count >= 6, let lastDetection = track.lastDetection {
    //         let vy = mean[5]  // Vertical velocity component
    //         if vy > 0 {  // Moving downward (as expected in fish tunnel)
    //             // In high density scenarios, limit the bonus
    //             if totalTracks > 80 {
    //                 // No bonus in extremely high density scenarios
    //             } else if totalTracks > 50 {
    //                 track.ttl += 1  // Small bonus in high density scenarios
    //             } else {
    //                 track.ttl += 2  // Normal bonus for expected movement direction
    //             }
    //         }
    //     }
    // }
    
}

// MARK: - TrackerProtocol Conformance

extension ByteTracker: TrackerProtocol {
    /// Get currently active tracks
    /// - Returns: Array of active tracked objects
    public func getActiveTracks() -> [STrack] {
        return activeTracks
    }
}
