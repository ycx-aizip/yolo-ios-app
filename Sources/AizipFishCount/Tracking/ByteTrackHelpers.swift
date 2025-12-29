// ByteTrackHelpers.swift
// Fish-specific tracking optimizations and heuristics for ByteTrack
//
// This file contains helper functions that implement fish-specific improvements
// to the ByteTrack algorithm, including:
// - Camera motion estimation and compensation
// - Directional biasing for expected fish movement
// - Enhanced association with track history and spatial proximity
// - Collection size limiting for memory management
//
// These utilities are extracted from ByteTracker to enable potential reuse
// in other tracking algorithms (e.g., OC-SORT) while keeping ByteTrack's behavior unchanged.

import Foundation
import UIKit

/**
 * ByteTrackHelpers
 *
 * Static utility functions for fish-specific tracking optimizations.
 * These methods implement heuristics tailored for tracking fish in tunnel environments
 * where movement is predominantly in one direction with occasional lateral motion.
 *
 * Key Features:
 * - Camera motion estimation and compensation
 * - Directional bias for expected movement patterns
 * - Track history-based association biasing
 * - Spatial proximity matching to prevent ID switches
 * - Memory-efficient collection management
 */
@MainActor
public struct ByteTrackHelpers {

    // MARK: - Camera Motion Estimation

    /**
     * Estimate camera motion between frames based on detection centers
     *
     * Uses closest-point matching between frames to estimate overall camera movement.
     * Applies directional filtering to focus on expected fish movement patterns.
     *
     * - Parameters:
     *   - currentCenters: Array of detection center positions in current frame
     *   - previousCenters: Array of detection center positions in previous frame
     *   - expectedDirection: Expected movement direction for directional filtering
     *   - previousMotion: Previous frame's estimated motion for smoothing
     * - Returns: Tuple of (dx, dy) representing estimated camera motion
     */
    public static func estimateCameraMotion(
        currentCenters: [(x: CGFloat, y: CGFloat)],
        previousCenters: [(x: CGFloat, y: CGFloat)],
        expectedDirection: CountingDirection?,
        previousMotion: (dx: CGFloat, dy: CGFloat)
    ) -> (dx: CGFloat, dy: CGFloat) {
        // Need at least a few detections in both frames to estimate motion
        if previousCenters.count < 3 || currentCenters.count < 3 {
            return (0, 0)
        }

        // Calculate average shift
        var totalDx: CGFloat = 0
        var totalDy: CGFloat = 0
        var count: Int = 0

        // Simple approach: match closest centers between frames
        for prevCenter in previousCenters {
            var closestDist: CGFloat = 0.4 // Increased from 1.0 to be more selective with matches
            var closestCenter: (x: CGFloat, y: CGFloat)? = nil

            for currCenter in currentCenters {
                let dx = prevCenter.x - currCenter.x
                let dy = prevCenter.y - currCenter.y
                let dist = sqrt(dx*dx + dy*dy)

                // Determine expected movement based on current counting direction
                var isExpectedMovement = false

                if let direction = expectedDirection {
                    switch direction {
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
            }
        }

        // If we have enough matches, compute the motion
        if count >= 3 {
            // Average motion with decay from previous estimate for smoothness
            let avgDx = totalDx / CGFloat(count)
            let avgDy = totalDy / CGFloat(count)

            // Apply smoothing with the previous estimate (exponential moving average)
            // Use stronger smoothing factor to reduce jitter
            var estimatedDx = avgDx * 0.6 + previousMotion.dx * 0.4
            var estimatedDy = avgDy * 0.6 + previousMotion.dy * 0.4

            // Limit the maximum camera motion to prevent excessive compensation
            // This helps when a single fish leaves the frame causing false motion estimate
            let maxMotion: CGFloat = 0.05
            estimatedDx = max(-maxMotion, min(maxMotion, estimatedDx))
            estimatedDy = max(-maxMotion, min(maxMotion, estimatedDy))

            return (estimatedDx, estimatedDy)
        } else {
            // Gradually decay the motion estimate if no matches
            return (previousMotion.dx * 0.7, previousMotion.dy * 0.7)
        }
    }

    // MARK: - Motion Compensation

    /**
     * Apply estimated camera motion compensation to track predictions
     *
     * Adjusts track positions based on estimated camera motion, with direction-specific
     * compensation factors to avoid over-correcting fish that are actually moving.
     *
     * - Parameters:
     *   - tracks: Array of tracks to compensate
     *   - motion: Estimated camera motion (dx, dy)
     *   - expectedDirection: Expected fish movement direction
     */
    public static func applyMotionCompensation(
        to tracks: [STrack],
        motion: (dx: CGFloat, dy: CGFloat),
        expectedDirection: CountingDirection?
    ) {
        // Only apply if we have significant motion
        if abs(motion.dx) < 0.005 && abs(motion.dy) < 0.005 {
            return
        }

        guard let direction = expectedDirection else {
            // No direction specified, apply uniform compensation
            for track in tracks {
                let newX = track.position.x - motion.dx
                let newY = track.position.y - motion.dy
                track.position = (newX, newY)
            }
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

            switch direction {
            case .topToBottom:
                // For top-to-bottom counting:
                // Apply full compensation to horizontal movement (x-axis)
                newX = track.position.x - motion.dx

                // For vertical position: if fish has downward velocity, apply less compensation
                let verticalCompensationFactor: CGFloat = expectedVerticalMotion > 0 ? 0.7 : 1.0
                newY = track.position.y - (motion.dy * verticalCompensationFactor)

            case .bottomToTop:
                // For bottom-to-top counting:
                // Apply full compensation to horizontal movement (x-axis)
                newX = track.position.x - motion.dx

                // For vertical position: if fish has upward velocity, apply less compensation
                let verticalCompensationFactor: CGFloat = expectedVerticalMotion < 0 ? 0.7 : 1.0
                newY = track.position.y - (motion.dy * verticalCompensationFactor)

            case .leftToRight:
                // For left-to-right counting:
                // For horizontal position: if fish has rightward velocity, apply less compensation
                let horizontalCompensationFactor: CGFloat = expectedHorizontalMotion > 0 ? 0.7 : 1.0
                newX = track.position.x - (motion.dx * horizontalCompensationFactor)

                // Apply full compensation to vertical movement (y-axis)
                newY = track.position.y - motion.dy

            case .rightToLeft:
                // For right-to-left counting:
                // For horizontal position: if fish has leftward velocity, apply less compensation
                let horizontalCompensationFactor: CGFloat = expectedHorizontalMotion < 0 ? 0.7 : 1.0
                newX = track.position.x - (motion.dx * horizontalCompensationFactor)

                // Apply full compensation to vertical movement (y-axis)
                newY = track.position.y - motion.dy
            }

            // Update track position
            track.position = (newX, newY)
        }
    }

    // MARK: - Enhanced Association

    /**
     * First-stage association with fish-specific enhancements
     *
     * Implements enhanced matching between active tracks and detections using:
     * 1. Track history bias - favor previous associations
     * 2. Directional bias - favor expected fish movement
     * 3. Velocity alignment - match movement with predicted velocity
     * 4. Spatial proximity - prevent ID switches between nearby fish
     *
     * - Parameters:
     *   - tracks: Array of active tracks to match
     *   - detections: Array of current detections
     *   - expectedDirection: Expected fish movement direction
     *   - matchHistory: Historical track-detection associations
     *   - matchThreshold: IoU threshold for matching
     * - Returns: Tuple of (matches, unmatched_tracks, unmatched_detections)
     */
    public static func improvedAssociateFirstStage(
        tracks: [STrack],
        detections: [STrack],
        expectedDirection: CountingDirection?,
        matchHistory: [Int: Set<Int>],
        matchThreshold: Float
    ) -> (matches: [(Int, Int)], unmatchedTracks: [Int], unmatchedDetections: [Int]) {
        if tracks.isEmpty || detections.isEmpty {
            return ([], Array(0..<tracks.count), Array(0..<detections.count))
        }

        // Calculate IoU distance matrix
        var dists = MatchingUtils.iouDistance(tracks: tracks, detections: detections)

        // Fish-specific enhancements:

        // 1. Apply bias for track history - reduce distance for historical associations
        for (i, track) in tracks.enumerated() {
            if let history = matchHistory[track.trackId] {
                for j in 0..<detections.count {
                    // If there was a previous match association, bias the cost lower
                    if history.contains(j) && dists[i][j] > 0.1 {
                        dists[i][j] = max(0.1, dists[i][j] - 0.25) // Stronger bias (0.25 vs 0.2)
                    }
                }
            }
        }

        // 2. Add directional bias for fish swimming in the expected direction
        if let direction = expectedDirection {
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

                        switch direction {
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
            MatchingUtils.linearAssignment(costMatrix: dists, threshold: matchThreshold)

        // Convert to tuples of (track_idx, det_idx)
        var matches: [(Int, Int)] = []
        for i in 0..<matchedTrackIndices.count {
            matches.append((matchedTrackIndices[i], matchedDetIndices[i]))
        }

        return (matches, unmatchedTrackIndices, unmatchedDetIndices)
    }

    /**
     * Second-stage association for lost tracks
     *
     * More lenient matching for tracks that were recently lost, with
     * directional biasing and velocity alignment consideration.
     *
     * - Parameters:
     *   - tracks: Array of lost tracks to match
     *   - detections: Array of unmatched detections
     *   - expectedDirection: Expected fish movement direction
     *   - matchThreshold: Position distance threshold for matching
     * - Returns: Tuple of (matches, unmatched_detections)
     */
    public static func improvedAssociateSecondStage(
        tracks: [STrack],
        detections: [STrack],
        expectedDirection: CountingDirection?,
        matchThreshold: Float
    ) -> (matches: [(Int, Int)], unmatchedDetections: [Int]) {
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
        if let direction = expectedDirection {
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

                        switch direction {
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
        }

        // Run linear assignment with the distance matrix
        let (matchedTrackIndices, matchedDetIndices, _, unmatchedDetIndices) =
            MatchingUtils.linearAssignment(costMatrix: dists, threshold: matchThreshold)

        // Convert to tuples of (track_idx, det_idx), mapping back to original indices
        var matches: [(Int, Int)] = []
        for i in 0..<matchedTrackIndices.count {
            let originalTrackIdx = trackIndices[matchedTrackIndices[i]]
            matches.append((originalTrackIdx, matchedDetIndices[i]))
        }

        return (matches, unmatchedDetIndices)
    }

    // MARK: - Track History Management

    /**
     * Update track match history for biasing future associations
     *
     * Maintains a limited history of track-detection associations to bias
     * future matching towards consistent pairings.
     *
     * - Parameters:
     *   - trackId: ID of the track
     *   - detectionId: ID of the matched detection
     *   - history: Mutable dictionary of track histories (updated in place)
     *   - maxHistorySize: Maximum number of associations to remember per track
     */
    public static func updateMatchHistory(
        trackId: Int,
        detectionId: Int,
        history: inout [Int: Set<Int>],
        maxHistorySize: Int = 10
    ) {
        // Initialize if needed
        if history[trackId] == nil {
            history[trackId] = []
        }

        // Add detection ID to history
        history[trackId]?.insert(detectionId)

        // Limit size to prevent unlimited growth
        if history[trackId]?.count ?? 0 > maxHistorySize {
            // Remove a random element to limit size
            if let randomElement = history[trackId]?.randomElement() {
                history[trackId]?.remove(randomElement)
            }
        }
    }

    /**
     * Clean up match history for tracks that no longer exist
     *
     * - Parameters:
     *   - history: Mutable dictionary of track histories (updated in place)
     *   - activeTrackIds: Set of currently active track IDs
     *   - lostTrackIds: Set of currently lost track IDs
     */
    public static func cleanupMatchHistory(
        history: inout [Int: Set<Int>],
        activeTrackIds: Set<Int>,
        lostTrackIds: Set<Int>
    ) {
        let allTrackIds = activeTrackIds.union(lostTrackIds)

        // Remove history for tracks that no longer exist
        let historyKeysToRemove = history.keys.filter { !allTrackIds.contains($0) }
        for key in historyKeysToRemove {
            history.removeValue(forKey: key)
        }
    }

    // MARK: - Collection Management

    /**
     * Limit collection size with priority-based pruning
     *
     * Generic utility to limit any collection size while preserving higher-priority elements.
     *
     * - Parameters:
     *   - collection: Collection to limit (modified in place)
     *   - maxSize: Maximum allowed size
     *   - priorityFunc: Function to determine sort order (higher priority first)
     */
    public static func limitCollectionSize<T>(
        _ collection: inout [T],
        maxSize: Int,
        priorityFunc: (T, T) -> Bool
    ) {
        if collection.count > maxSize {
            collection.sort(by: priorityFunc)
            collection = Array(collection.prefix(maxSize))
        }
    }

    // MARK: - Movement Consistency Tracking (ByteTrack-Specific)

    /**
     * Update track's movement consistency after a successful match
     *
     * This is a ByteTrack-specific feature that tracks how consistently a fish
     * moves in the expected direction, used for adaptive TTL assignment.
     *
     * - Parameters:
     *   - track: Track to update
     *   - newPosition: New position from detection
     *   - expectedDirection: Expected movement direction
     *   - isReactivation: Whether this is a reactivation (uses different rates)
     */
    public static func updateMovementConsistency(
        for track: STrack,
        newPosition: (x: CGFloat, y: CGFloat),
        expectedDirection: CountingDirection?,
        isReactivation: Bool
    ) {
        let dx = newPosition.x - track.position.x
        let dy = newPosition.y - track.position.y

        var isExpectedMovement = false

        // Determine if movement matches the expected direction
        switch expectedDirection {
        case .topToBottom:
            let isMovingDown = dy > 0
            isExpectedMovement = isMovingDown && abs(dx) < TrackingParameters.maxHorizontalDeviation
        case .bottomToTop:
            let isMovingUp = dy < 0
            isExpectedMovement = isMovingUp && abs(dx) < TrackingParameters.maxHorizontalDeviation
        case .leftToRight:
            let isMovingRight = dx > 0
            isExpectedMovement = isMovingRight && abs(dy) < TrackingParameters.maxVerticalDeviation
        case .rightToLeft:
            let isMovingLeft = dx < 0
            isExpectedMovement = isMovingLeft && abs(dy) < TrackingParameters.maxVerticalDeviation
        case .none:
            // No expected direction - treat all movement as valid
            isExpectedMovement = true
        }

        // Select appropriate rates based on whether this is reactivation
        let increaseRate = isReactivation ?
            TrackingParameters.reactivationConsistencyIncreaseRate :
            TrackingParameters.consistencyIncreaseRate
        let decreaseRate = isReactivation ?
            TrackingParameters.reactivationConsistencyDecreaseRate :
            TrackingParameters.consistencyDecreaseRate

        // Update consistency score
        if isExpectedMovement {
            track.framesWithExpectedMovement += 1
            track.movementConsistency = min(1.0, track.movementConsistency + Float(increaseRate))
        } else {
            track.movementConsistency = max(0.0, track.movementConsistency - Float(decreaseRate))
        }
    }

    /**
     * Calculate adaptive TTL based on movement consistency
     *
     * ByteTrack uses movement consistency to assign higher TTL values to tracks
     * that move predictably in the expected direction.
     *
     * - Parameters:
     *   - track: Track to calculate TTL for
     *   - isReactivation: Whether this is for a reactivated track (more conservative)
     * - Returns: Recommended TTL value
     */
    public static func calculateAdaptiveTTL(
        for track: STrack,
        isReactivation: Bool
    ) -> Int {
        if isReactivation {
            // More conservative TTL for reactivated tracks
            if track.movementConsistency > 0.7 && track.framesWithExpectedMovement > 5 {
                return TrackingParameters.reactivationHighTTL
            } else if track.movementConsistency > 0.4 {
                return TrackingParameters.reactivationMediumTTL
            } else {
                return TrackingParameters.reactivationLowTTL
            }
        } else {
            // Regular TTL for ongoing tracks
            if track.movementConsistency > 0.7 && track.framesWithExpectedMovement > 5 {
                return TrackingParameters.highConsistencyTTL
            } else if track.movementConsistency > 0.4 {
                return TrackingParameters.mediumConsistencyTTL
            } else {
                return TrackingParameters.lowConsistencyTTL
            }
        }
    }
}
