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
    
    /// Matching threshold for high confidence detections
    private let highThreshold: Float = 0.4  // Reduced from 0.5 to be more lenient
    
    /// Matching threshold for low confidence detections
    private let lowThreshold: Float = 0.15  // Reduced from 0.2 to be more lenient
    
    /// Max time to keep a track in lost state
    private let maxTimeLost: Int = 15  // Reduced from 45 to reduce memory usage
    
    /// Maximum time to remember removed tracks to avoid ID reuse
    private let maxTimeRemembered: Int = 60 // Reduced from 90 to reduce memory usage
    
    /// Buffer for potential new tracks - stores potential tracks before assigning real IDs
    /// Key: temporary ID, Value: (position, detection, confidence, class, framesObserved, lastUpdatedFrame)
    private var potentialTracks: [Int: (position: (x: CGFloat, y: CGFloat), detection: Box, score: Float, cls: String, frames: Int, lastFrame: Int)] = [:]
    
    /// Required frames to consider a potential track as real (to avoid spurious tracks)
    private let requiredFramesForTrack: Int = 2 // Reduced from 4 to be more responsive
    
    /// Counter for temporary IDs
    private var tempIdCounter: Int = 0
    
    /// Maximum matching distance for potential tracks - increased for fast-moving fish
    private let maxMatchingDistance: CGFloat = 0.3  // Increased from 0.2 to better handle camera movement
    
    /// Maximum frames a potential track can be unmatched before removal
    private let maxUnmatchedFrames: Int = 10  // Reduced from 15 to clean up faster
    
    /// Estimated camera motion between frames (simple implementation)
    private var lastFrameDetectionCenters: [(x: CGFloat, y: CGFloat)] = []
    private var estimatedCameraMotion: (dx: CGFloat, dy: CGFloat) = (0, 0)
    
    // Maximum number of tracks to maintain in each collection to prevent unbounded growth
    private let maxActiveTracks: Int = 100
    private let maxLostTracks: Int = 50
    private let maxRemovedTracks: Int = 50
    private let maxPotentialTracks: Int = 50
    
    // MARK: - Initialization
    
    public init() {
        // Empty initializer
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
        
        // Estimate camera motion if we have previous frame data
        estimateCameraMotion(from: detections)
        
        // Clean up very old removed tracks to prevent memory growth
        removeOldRemovedTracks()
        
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
        
        // Save current detection centers for next frame's motion estimation
        if !detTrackArr.isEmpty {
            lastFrameDetectionCenters = detTrackArr.map { $0.position }
        }
        
        // Filter high score detections
        let remainedDetections = detTrackArr
        
        // Predict new locations for active tracks
        let predActiveTracks = activeTracks
        STrack.multiPredict(tracks: predActiveTracks)
        
        // Apply motion compensation to track predictions if significant camera motion detected
        applyMotionCompensation(to: predActiveTracks)
        
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
        
        // Apply motion compensation to lost tracks too
        applyMotionCompensation(to: lostTracksCopy)
        
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
                track.cleanup() // Release references
                removedTracks.append(track)
            }
        }
        
        // Process unmatched detections through potential tracks buffer
        var newTracks: [STrack] = []
        
        // To reduce overhead, only process detections if we're not over capacity
        if activeTracks.count < maxActiveTracks {
            for detIdx in secondUnmatchedDetections {
                let originalDetIdx = firstUnmatchedDetections[detIdx]
                let detection = remainedDetections[originalDetIdx]
                
                if let actualDetection = detection.lastDetection, detection.score >= 0.4 { // Kept threshold at 0.4
                    // Check if this detection matches any existing potential track
                    var matchedPotentialId: Int? = nil
                    var closestDistance: CGFloat = maxMatchingDistance // Using increased distance
                    
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
                        if newFrameCount >= requiredFramesForTrack {
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
            return frameId - potentialTracks[key]!.lastFrame > maxUnmatchedFrames
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
        activeTracks = []
        lostTracks = []
        removedTracks = []
        potentialTracks = [:]
        tempIdCounter = 0
        frameId = 0
        lastFrameDetectionCenters = []
        estimatedCameraMotion = (0, 0)
        STrack.resetId()
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
        if activeTracks.count > maxActiveTracks {
            // Sort by recency and score
            activeTracks.sort { (a, b) -> Bool in
                if a.endFrame == b.endFrame {
                    return a.score > b.score
                }
                return a.endFrame > b.endFrame
            }
            activeTracks = Array(activeTracks.prefix(maxActiveTracks))
        }
        
        // Cap lost tracks - priority is recent tracks
        if lostTracks.count > maxLostTracks {
            lostTracks.sort { $0.endFrame > $1.endFrame }
            lostTracks = Array(lostTracks.prefix(maxLostTracks))
        }
        
        // Cap removed tracks - priority is recent tracks
        if removedTracks.count > maxRemovedTracks {
            removedTracks.sort { $0.endFrame > $1.endFrame }
            removedTracks = Array(removedTracks.prefix(maxRemovedTracks))
        }
        
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
    
    // MARK: - Motion Compensation
    
    /// Estimate camera motion between frames based on detection centers
    private func estimateCameraMotion(from currentDetections: [Box]) {
        // Need at least a few detections in both frames to estimate motion
        if lastFrameDetectionCenters.count < 3 || currentDetections.count < 3 {
            estimatedCameraMotion = (0, 0)
            return
        }
        
        // Calculate centers of current detections
        let currentCenters = currentDetections.map { box in
            let bbox = box.xywhn
            return (x: (bbox.minX + bbox.maxX) / 2, y: (bbox.minY + bbox.maxY) / 2)
        }
        
        // Calculate average shift
        var totalDx: CGFloat = 0
        var totalDy: CGFloat = 0
        var count: Int = 0
        
        // Simple approach: match closest centers between frames
        for prevCenter in lastFrameDetectionCenters {
            var closestDist: CGFloat = 1.0
            var closestCenter: (x: CGFloat, y: CGFloat)? = nil
            
            for currCenter in currentCenters {
                let dx = prevCenter.x - currCenter.x
                let dy = prevCenter.y - currCenter.y
                let dist = sqrt(dx*dx + dy*dy)
                
                if dist < closestDist {
                    closestDist = dist
                    closestCenter = currCenter
                }
            }
            
            // Only use matches that are reasonably close
            if closestDist < 0.2, let closestCenter = closestCenter {
                totalDx += prevCenter.x - closestCenter.x
                totalDy += prevCenter.y - closestCenter.y
                count += 1
            }
        }
        
        if count > 0 {
            // Average motion with decay from previous estimate for smoothness
            let avgDx = totalDx / CGFloat(count)
            let avgDy = totalDy / CGFloat(count)
            
            // Apply smoothing with the previous estimate (exponential moving average)
            estimatedCameraMotion.dx = avgDx * 0.7 + estimatedCameraMotion.dx * 0.3
            estimatedCameraMotion.dy = avgDy * 0.7 + estimatedCameraMotion.dy * 0.3
        } else {
            // Gradually decay the motion estimate if no matches
            estimatedCameraMotion.dx *= 0.8
            estimatedCameraMotion.dy *= 0.8
        }
    }
    
    /// Apply estimated camera motion compensation to track predictions
    private func applyMotionCompensation(to tracks: [STrack]) {
        // Only apply if we have significant motion
        if abs(estimatedCameraMotion.dx) < 0.01 && abs(estimatedCameraMotion.dy) < 0.01 {
            return
        }
        
        for track in tracks {
            // Adjust the track's position by the estimated camera motion
            let newX = track.position.x - estimatedCameraMotion.dx
            let newY = track.position.y - estimatedCameraMotion.dy
            
            // Keep position within normalized bounds [0, 1]
            track.position = (
                x: max(0, min(1, newX)),
                y: max(0, min(1, newY))
            )
        }
    }
    
    // MARK: - Debug Methods
    
    /// Get current tracking stats
    /// - Returns: Dictionary with tracking statistics
    public func getStats() -> [String: Any] {
        return [
            "active_tracks": activeTracks.count,
            "lost_tracks": lostTracks.count,
            "removed_tracks": removedTracks.count,
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
        // Instead of filtering which creates a new array, use removeAll with a condition
        let oldTracks = removedTracks.filter { frameId - $0.endFrame >= maxTimeRemembered }
        for track in oldTracks {
            track.cleanup() // Release resources
        }
        removedTracks.removeAll(where: { frameId - $0.endFrame >= maxTimeRemembered })
    }
} 