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
        
        // Match with high score detections first - using the improved matching process
        let (firstMatches, firstUnmatchedTracks, firstUnmatchedDetections) = 
            improvedAssociateFirstStage(
                tracks: predActiveTracks,
                detections: remainedDetections
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
            
            // Update track match history for future matching bias
            updateMatchHistory(trackId: track.trackId, detectionId: detIdx)
        }
        
        // Create array of unmatched tracks
        var unmatchedTrackArray: [STrack] = firstUnmatchedTracks.map { predActiveTracks[$0] }
        
        // Handle lost tracks
        let lostTracksCopy = lostTracks
        
        // Predict new locations for lost tracks
        STrack.multiPredict(tracks: lostTracksCopy)
        
        // Apply motion compensation to lost tracks too
        applyMotionCompensation(to: lostTracksCopy)
        
        // Match remaining detections with lost tracks - using improved second stage matching
        let (secondMatches, secondUnmatchedDetections) = 
            improvedAssociateSecondStage(
                tracks: lostTracksCopy,
                detections: firstUnmatchedDetections.map { remainedDetections[$0] }
            )
        
        // Reactivate matched lost tracks
        for (trackIdx, detIdx) in secondMatches {
            let lostTrack = lostTracksCopy[trackIdx]
            let detection = remainedDetections[firstUnmatchedDetections[detIdx]]
            
            lostTrack.reactivate(
                newTrack: detection,
                frameId: frameId
            )
            
            // Update track match history for future matching bias
            updateMatchHistory(trackId: lostTrack.trackId, detectionId: firstUnmatchedDetections[detIdx])
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
        var matchingPairs: [(prev: (x: CGFloat, y: CGFloat), curr: (x: CGFloat, y: CGFloat))] = []
        
        // Simple approach: match closest centers between frames
        for prevCenter in lastFrameDetectionCenters {
            var closestDist: CGFloat = 0.4 // Increased from 1.0 to be more selective with matches
            var closestCenter: (x: CGFloat, y: CGFloat)? = nil
            
            for currCenter in currentCenters {
                let dx = prevCenter.x - currCenter.x
                let dy = prevCenter.y - currCenter.y
                let dist = sqrt(dx*dx + dy*dy)
                
                // Determine expected movement based on current counting direction
                var isExpectedMovement = false
                
                switch STrack.expectedMovementDirection {
                case .topToBottom:
                    // For top to bottom, expect downward movement (dy > 0) with limited horizontal motion
                    isExpectedMovement = dy > 0 && abs(dx) < 0.2
                case .bottomToTop:
                    // For bottom to top, expect upward movement (dy < 0) with limited horizontal motion  
                    isExpectedMovement = dy < 0 && abs(dx) < 0.2
                case .leftToRight:
                    // For left to right, expect rightward movement (dx > 0) with limited vertical motion
                    isExpectedMovement = dx > 0 && abs(dy) < 0.2
                case .rightToLeft:
                    // For right to left, expect leftward movement (dx < 0) with limited vertical motion
                    isExpectedMovement = dx < 0 && abs(dy) < 0.2
                }
                
                // Adjust the matching distance based on expected movement pattern
                let adjustedDist = isExpectedMovement ? dist * 0.8 : dist
                
                if adjustedDist < closestDist {
                    closestDist = adjustedDist
                    closestCenter = currCenter
                }
            }
            
            // Only use matches that are reasonably close
            // Lower threshold to 0.15 for more precise matching
            if closestDist < 0.15, let closestCenter = closestCenter {
                totalDx += prevCenter.x - closestCenter.x
                totalDy += prevCenter.y - closestCenter.y
                count += 1
                matchingPairs.append((prev: prevCenter, curr: closestCenter))
            }
        }
        
        // If we have enough matches, compute the motion
        if count >= 3 {
            // Average motion with decay from previous estimate for smoothness
            let avgDx = totalDx / CGFloat(count)
            let avgDy = totalDy / CGFloat(count)
            
            // Apply smoothing with the previous estimate (exponential moving average)
            // Use stronger smoothing factor to reduce jitter
            estimatedCameraMotion.dx = avgDx * 0.6 + estimatedCameraMotion.dx * 0.4
            estimatedCameraMotion.dy = avgDy * 0.6 + estimatedCameraMotion.dy * 0.4
            
            // Limit the maximum camera motion to prevent excessive compensation
            // This helps when a single fish leaves the frame causing false motion estimate
            let maxMotion: CGFloat = 0.05
            estimatedCameraMotion.dx = max(-maxMotion, min(maxMotion, estimatedCameraMotion.dx))
            estimatedCameraMotion.dy = max(-maxMotion, min(maxMotion, estimatedCameraMotion.dy))
        } else {
            // Gradually decay the motion estimate if no matches
            estimatedCameraMotion.dx *= 0.7
            estimatedCameraMotion.dy *= 0.7
        }
    }
    
    /// Apply estimated camera motion compensation to track predictions
    private func applyMotionCompensation(to tracks: [STrack]) {
        // Only apply if we have significant motion
        if abs(estimatedCameraMotion.dx) < 0.005 && abs(estimatedCameraMotion.dy) < 0.005 {
            return
        }
        
        for track in tracks {
            // Get expected movement direction from velocity if available
            var expectedHorizontalMotion: CGFloat = 0
            var expectedVerticalMotion: CGFloat = 0
            if let mean = track.mean, mean.count >= 6 {
                expectedHorizontalMotion = CGFloat(mean[4]) // Horizontal velocity component  
                expectedVerticalMotion = CGFloat(mean[5]) // Vertical velocity component
            }
            
            // Apply motion compensation based on the current counting direction
            var newX: CGFloat
            var newY: CGFloat
            
            switch STrack.expectedMovementDirection {
            case .topToBottom:
                // For top-to-bottom counting:
                // Apply full compensation to horizontal movement (x-axis)
                newX = track.position.x - estimatedCameraMotion.dx
            
                // For vertical position: if fish has downward velocity, apply less compensation
            let verticalCompensationFactor: CGFloat = expectedVerticalMotion > 0 ? 0.7 : 1.0
                newY = track.position.y - (estimatedCameraMotion.dy * verticalCompensationFactor)
                
            case .bottomToTop:
                // For bottom-to-top counting:
                // Apply full compensation to horizontal movement (x-axis)
                newX = track.position.x - estimatedCameraMotion.dx
                
                // For vertical position: if fish has upward velocity, apply less compensation
                let verticalCompensationFactor: CGFloat = expectedVerticalMotion < 0 ? 0.7 : 1.0
                newY = track.position.y - (estimatedCameraMotion.dy * verticalCompensationFactor)
                
            case .leftToRight:
                // For left-to-right counting:
                // For horizontal position: if fish has rightward velocity, apply less compensation
                let horizontalCompensationFactor: CGFloat = expectedHorizontalMotion > 0 ? 0.7 : 1.0
                newX = track.position.x - (estimatedCameraMotion.dx * horizontalCompensationFactor)
                
                // Apply full compensation to vertical movement (y-axis)
                newY = track.position.y - estimatedCameraMotion.dy
                
            case .rightToLeft:
                // For right-to-left counting:
                // For horizontal position: if fish has leftward velocity, apply less compensation
                let horizontalCompensationFactor: CGFloat = expectedHorizontalMotion < 0 ? 0.7 : 1.0
                newX = track.position.x - (estimatedCameraMotion.dx * horizontalCompensationFactor)
                
                // Apply full compensation to vertical movement (y-axis)
                newY = track.position.y - estimatedCameraMotion.dy
            }
            
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
    
    // MARK: - Improved Association Methods
    
    /// Enhanced first stage association that considers track history and adds bias for existing associations
    private func improvedAssociateFirstStage(
        tracks: [STrack],
        detections: [STrack]
    ) -> ([(Int, Int)], [Int], [Int]) {
        if tracks.isEmpty || detections.isEmpty {
            return ([], Array(0..<tracks.count), Array(0..<detections.count))
        }
        
        // Calculate IoU distance matrix
        var dists = MatchingUtils.iouDistance(tracks: tracks, detections: detections)
        
        // Fish-specific enhancements:
        
        // 1. Apply bias for track history - reduce distance for historical associations
        for (i, track) in tracks.enumerated() {
            if let history = trackMatchHistory[track.trackId] {
                for j in 0..<detections.count {
                    // If there was a previous match association, bias the cost lower
                    if history.contains(j) && dists[i][j] > 0.1 {
                        dists[i][j] = max(0.1, dists[i][j] - 0.25) // Stronger bias (0.25 vs 0.2)
                    }
                }
            }
        }
        
        // 2. Add directional bias for fish swimming in the expected direction
        for i in 0..<tracks.count {
            let track = tracks[i]
            // If we have Kalman filter mean data with velocity components
            if let mean = track.mean, mean.count >= 6 {
                // Extract velocity components
                let vx = CGFloat(mean[4])
                let vy = CGFloat(mean[5])
                
                // For each detection
                for j in 0..<detections.count {
                    let detection = detections[j]
                    
                    // Calculate relative movement
                    let dx = detection.position.x - track.position.x
                    let dy = detection.position.y - track.position.y
                    
                    // Check if movement aligns with expected direction based on counting direction
                    var isExpectedMovement = false
                    
                    switch STrack.expectedMovementDirection {
                    case .topToBottom:
                        // Expected movement is downward
                        isExpectedMovement = dy > 0 && vy > 0
                    case .bottomToTop:
                        // Expected movement is upward
                        isExpectedMovement = dy < 0 && vy < 0
                    case .leftToRight:
                        // Expected movement is rightward
                        isExpectedMovement = dx > 0 && vx > 0
                    case .rightToLeft:
                        // Expected movement is leftward
                        isExpectedMovement = dx < 0 && vx < 0
                    }
                    
                    // If the fish is moving in the expected direction, bias matching
                    if isExpectedMovement {
                        // More aggressive bias for expected movement
                        let distReduction: Float = 0.25
                        dists[i][j] = max(0.05, dists[i][j] - distReduction)
                    }
                    
                    // Also consider if movement direction aligns with velocity
                    let velocityAlignment = (dx * vx + dy * vy) / 
                        (sqrt(dx*dx + dy*dy) * sqrt(vx*vx + vy*vy) + 0.0001)
                    
                    if velocityAlignment > 0.5 { // Only favor strong alignment
                        dists[i][j] = max(0.1, dists[i][j] - 0.15)
                    }
                }
            }
        }
        
        // 3. Add spatial proximity bias - if a detection is closer to one track than others,
        // further bias the distance to prevent ID switches between nearby fish
        for j in 0..<detections.count {
            let detection = detections[j]
            
            // Find closest track by position
            var closestTrackIdx = -1
            var closestDist: CGFloat = 1.0
            
            for i in 0..<tracks.count {
                let track = tracks[i]
                let dx = detection.position.x - track.position.x
                let dy = detection.position.y - track.position.y
                let distSq = dx*dx + dy*dy
                
                if distSq < closestDist {
                    closestDist = distSq
                    closestTrackIdx = i
                }
            }
            
            // If we found a significantly closer track, bias the matching
            if closestTrackIdx >= 0 && closestDist < 0.1 { // Threshold for "significantly closer"
                dists[closestTrackIdx][j] = max(0.05, dists[closestTrackIdx][j] - 0.2)
            }
        }
        
        // Run linear assignment with the distance matrix
        let (matchedTrackIndices, matchedDetIndices, unmatchedTrackIndices, unmatchedDetIndices) =
            MatchingUtils.linearAssignment(costMatrix: dists, threshold: TrackingParameters.highMatchThreshold)
        
        // Convert to tuples of (track_idx, det_idx)
        var matches: [(Int, Int)] = []
        for i in 0..<matchedTrackIndices.count {
            matches.append((matchedTrackIndices[i], matchedDetIndices[i]))
        }
        
        return (matches, unmatchedTrackIndices, unmatchedDetIndices)
    }
    
    /// Enhanced second stage association that considers directional movement and is more lenient
    private func improvedAssociateSecondStage(
        tracks: [STrack],
        detections: [STrack]
    ) -> ([(Int, Int)], [Int]) {
        if tracks.isEmpty || detections.isEmpty {
            return ([], Array(0..<detections.count))
        }
        
        // Filter out tracks that are in 'removed' state - they should never be reactivated
        let validTracks = tracks.enumerated().filter { $0.element.state != .removed }
        
        // If no valid tracks remain after filtering, return early
        if validTracks.isEmpty {
            return ([], Array(0..<detections.count))
        }
        
        // Create mapping from new indices to original indices
        let trackIndices = validTracks.map { $0.offset }
        let filteredTracks = validTracks.map { $0.element }
        
        // Calculate position distance matrix
        var dists = MatchingUtils.positionDistance(tracks: filteredTracks, detections: detections)
        
        // Add directional bias for movement in the expected direction
        for i in 0..<filteredTracks.count {
            let track = filteredTracks[i]
            // If we have mean data from Kalman filter
            if let mean = track.mean, mean.count >= 6 {
                // Extract velocity components
                let vx = CGFloat(mean[4])
                let vy = CGFloat(mean[5])
                
                // For each detection
                for j in 0..<detections.count {
                    let detection = detections[j]
                    
                    // Calculate expected direction of movement
                    let dx = detection.position.x - track.position.x
                    let dy = detection.position.y - track.position.y
                    
                    // Check if movement aligns with expected direction based on counting direction
                    var isExpectedMovement = false
                    
                    switch STrack.expectedMovementDirection {
                    case .topToBottom:
                        // Expected movement is downward
                        isExpectedMovement = dy > 0 && vy > 0
                    case .bottomToTop:
                        // Expected movement is upward
                        isExpectedMovement = dy < 0 && vy < 0
                    case .leftToRight:
                        // Expected movement is rightward
                        isExpectedMovement = dx > 0 && vx > 0
                    case .rightToLeft:
                        // Expected movement is leftward
                        isExpectedMovement = dx < 0 && vx < 0
                    }
                    
                    // If movement aligns with velocity (especially in expected direction),
                    // reduce the distance to make matching more likely
                    let velocityAlignment = (dx * vx + dy * vy) / 
                        (sqrt(dx*dx + dy*dy) * sqrt(vx*vx + vy*vy) + 0.0001)
                    
                    if velocityAlignment > 0 || isExpectedMovement {
                        // Reduce distance based on alignment and expected movement
                        let reductionFactor: Float = isExpectedMovement ? 0.4 : 0.2
                        dists[i][j] = max(0.1, dists[i][j] - reductionFactor)
                    }
                }
            }
        }
        
        // Run linear assignment with the distance matrix
        let (matchedTrackIndices, matchedDetIndices, _, unmatchedDetIndices) =
            MatchingUtils.linearAssignment(costMatrix: dists, threshold: TrackingParameters.lowMatchThreshold)
        
        // Convert to tuples of (track_idx, det_idx), mapping back to original indices
        var matches: [(Int, Int)] = []
        for i in 0..<matchedTrackIndices.count {
            let originalTrackIdx = trackIndices[matchedTrackIndices[i]]
            matches.append((originalTrackIdx, matchedDetIndices[i]))
        }
        
        return (matches, unmatchedDetIndices)
    }
    
    /// Update track match history for biasing future associations
    private func updateMatchHistory(trackId: Int, detectionId: Int) {
        // Initialize if needed
        if trackMatchHistory[trackId] == nil {
            trackMatchHistory[trackId] = []
        }
        
        // Add detection ID to history
        trackMatchHistory[trackId]?.insert(detectionId)
        
        // Limit size to prevent unlimited growth
        if trackMatchHistory[trackId]?.count ?? 0 > 10 {
            // This is a simplified approach - in practice you might want to keep the most recent matches
            // For now, we just limit the set size by removing a random element
            if let randomElement = trackMatchHistory[trackId]?.randomElement() {
                trackMatchHistory[trackId]?.remove(randomElement)
            }
        }
        
        // Clean up match history for tracks that no longer exist
        if frameId % 60 == 0 {
            // Create a set of active track IDs for efficient lookup
            let activeTrackIds = Set(activeTracks.map { $0.trackId })
            let lostTrackIds = Set(lostTracks.map { $0.trackId })
            let allTrackIds = activeTrackIds.union(lostTrackIds)
            
            // Remove history for tracks that no longer exist
            let historyKeysToRemove = trackMatchHistory.keys.filter { !allTrackIds.contains($0) }
            for key in historyKeysToRemove {
                trackMatchHistory.removeValue(forKey: key)
            }
        }
    }
} 