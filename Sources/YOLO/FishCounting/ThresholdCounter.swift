// ThresholdCounter.swift
// Threshold-based fish counting implementation
//
// This class implements threshold crossing detection for counting fish
// as they move past designated threshold lines. Extracted from TrackingDetector
// to enable pluggable counting strategies.

import Foundation
import CoreGraphics

/// Direction of fish movement
/// These represent the actual movement direction of the fish:
/// - up: Fish moving upward (used with CountingDirection.bottomToTop)
/// - down: Fish moving downward (used with CountingDirection.topToBottom)
/// - left: Fish moving leftward (used with CountingDirection.rightToLeft)
/// - right: Fish moving rightward (used with CountingDirection.leftToRight)
private enum Direction {
    case up
    case down
    case left
    case right
}

/// Threshold-based counting implementation
@MainActor
public class ThresholdCounter: CounterProtocol {
    
    // MARK: - Properties
    
    /// Total count of objects that have crossed the threshold(s)
    private var totalCount: Int = 0
    
    /// Thresholds used for counting (normalized coordinates, 0.0-1.0)
    /// For vertical directions (topToBottom, bottomToTop), these are y-coordinates
    /// For horizontal directions (leftToRight, rightToLeft), these are x-coordinates
    private var thresholds: [CGFloat] = [0.2, 0.4]
    
    /// Map of track IDs to counting status
    private var countedTracks: [Int: Bool] = [:]
    
    /// Current counting direction
    private var countingDirection: CountingDirection = .bottomToTop
    
    /// Direction of fish movement for each track
    private var crossingDirections: [Int: Direction] = [:]
    
    /// Previous positions for each track
    private var previousPositions: [Int: (x: CGFloat, y: CGFloat)] = [:]
    
    /// Map of track positions from 5 frames ago for detecting fast movements
    private var historyPositions: [Int: (x: CGFloat, y: CGFloat)] = [:]
    
    /// Frame counter for periodic cleanup
    private var frameCount: Int = 0
    
    /// Current tracked objects (cached for getTrackingInfo)
    private var cachedTracks: [STrack] = []
    
    // MARK: - Initialization
    
    nonisolated public init() {
        // Initialize with default values
    }
    
    // MARK: - CounterProtocol Implementation
    
    /// Process tracked objects and update counts
    public func processFrame(tracks: [STrack]) -> CountingResult {
        // Cache tracks for getTrackingInfo
        cachedTracks = tracks
        
        var newlyCounted: [Int] = []
        
        // Get the expected direction of movement for the current counting direction
        let expectedDirection = expectedMovementDirection(for: countingDirection)
        
        // Process all tracks each frame
        for track in tracks {
            let trackId = track.trackId
            let currentPos = track.position
            
            // Skip if no previous position
            guard let lastPosition = previousPositions[trackId] else {
                // If no previous position, just store current and continue
                previousPositions[trackId] = currentPos
                continue
            }
            
            // Store current position for next frame
            previousPositions[trackId] = currentPos
            
            // Calculate movement direction
            let dx = currentPos.x - lastPosition.x
            let dy = currentPos.y - lastPosition.y
            
            // Determine actual movement direction
            let actualDirection: Direction
            if abs(dx) > abs(dy) {
                // Horizontal movement is dominant
                actualDirection = dx > 0 ? .right : .left
            } else {
                // Vertical movement is dominant
                actualDirection = dy > 0 ? .down : .up
            }
            
            // Store the track's movement direction
            crossingDirections[trackId] = actualDirection
            
            // Get previous and current coordinates
            let center_y = currentPos.y
            let last_y = lastPosition.y
            let center_x = currentPos.x
            let last_x = lastPosition.x
            
            // Get counted state for this track
            let alreadyCounted = countedTracks[trackId, default: false]
            
            // Convert positions to unified coordinate system for consistent threshold checking
            let currentUnified = UnifiedCoordinateSystem.UnifiedRect(
                x: center_x, y: center_y, width: 0, height: 0
            )
            let lastUnified = UnifiedCoordinateSystem.UnifiedRect(
                x: last_x, y: last_y, width: 0, height: 0
            )
            
            // Convert to counting coordinates based on direction
            let currentCounting = UnifiedCoordinateSystem.toCounting(currentUnified, countingDirection: countingDirection)
            let lastCounting = UnifiedCoordinateSystem.toCounting(lastUnified, countingDirection: countingDirection)
            
            // Use direction-specific threshold crossing logic
            switch countingDirection {
            case .topToBottom:
                // Top to bottom: fish moving downward (Y increasing)
                let current_y = currentCounting.y
                let last_y = lastCounting.y
                
                // Increment count: Check for crossing from above to below threshold
                if !alreadyCounted {
                    for threshold in thresholds {
                        if last_y < threshold && current_y >= threshold {
                            if countObject(trackId: trackId) {
                                newlyCounted.append(trackId)
                            }
                            break // Count only once per threshold crossing event
                        }
                    }
                }
                
                // Decrement count: Check for reverse crossing (only first threshold)
                if let firstThreshold = thresholds.first,
                   last_y > firstThreshold && current_y <= firstThreshold && alreadyCounted {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                    if let trackIndex = cachedTracks.firstIndex(where: { $0.trackId == trackId }) {
                        cachedTracks[trackIndex].counted = false
                    }
                }
                
            case .bottomToTop:
                // Bottom to top: fish moving upward (Y decreasing)
                let current_y = currentCounting.y
                let last_y = lastCounting.y
                
                // Increment count: Check for crossing from below to above threshold
                if !alreadyCounted {
                    for threshold in thresholds {
                        if last_y > threshold && current_y <= threshold {
                            if countObject(trackId: trackId) {
                                newlyCounted.append(trackId)
                            }
                            break // Count only once per threshold crossing event
                        }
                    }
                }
                
                // Decrement count: Check for reverse crossing (only first threshold)
                if let firstThreshold = thresholds.first,
                   last_y < firstThreshold && current_y >= firstThreshold && alreadyCounted {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                    if let trackIndex = cachedTracks.firstIndex(where: { $0.trackId == trackId }) {
                        cachedTracks[trackIndex].counted = false
                    }
                }
                
            case .leftToRight:
                // Left to right: fish moving rightward (X increasing)
                let current_x = currentCounting.x
                let last_x = lastCounting.x
                
                // Increment count: Check for crossing from left to right of threshold
                if !alreadyCounted {
                    for threshold in thresholds {
                        if last_x < threshold && current_x >= threshold {
                            if countObject(trackId: trackId) {
                                newlyCounted.append(trackId)
                            }
                            break // Count only once per threshold crossing event
                        }
                    }
                }
                
                // Decrement count: Check for reverse crossing (only first threshold)
                if let firstThreshold = thresholds.first,
                   last_x > firstThreshold && current_x <= firstThreshold && alreadyCounted {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                    if let trackIndex = cachedTracks.firstIndex(where: { $0.trackId == trackId }) {
                        cachedTracks[trackIndex].counted = false
                    }
                }
                
            case .rightToLeft:
                // Right to left: fish moving leftward (X decreasing)
                let current_x = currentCounting.x
                let last_x = lastCounting.x
                
                // Increment count: Check for crossing from right to left of threshold
                if !alreadyCounted {
                    for threshold in thresholds {
                        if last_x > threshold && current_x <= threshold {
                            if countObject(trackId: trackId) {
                                newlyCounted.append(trackId)
                            }
                            break // Count only once per threshold crossing event
                        }
                    }
                }
                
                // Decrement count: Check for reverse crossing (only first threshold)
                if let firstThreshold = thresholds.first,
                   last_x < firstThreshold && current_x >= firstThreshold && alreadyCounted {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                    if let trackIndex = cachedTracks.firstIndex(where: { $0.trackId == trackId }) {
                        cachedTracks[trackIndex].counted = false
                    }
                }
            }
        }
        
        // Increment frame count for each update
        frameCount += 1
        
        // Clean up old data periodically (every 30 frames)
        if frameCount % 30 == 0 {
            cleanupOldTracks(currentTrackIds: Set(tracks.map { $0.trackId }))
        }
        
        // Return counting result
        return CountingResult(
            totalCount: totalCount,
            newlyCounted: newlyCounted,
            trackingInfo: getTrackingInfo()
        )
    }
    
    /// Reset counter state
    public func resetCount() {
        totalCount = 0
        countedTracks.removeAll(keepingCapacity: true)
        crossingDirections.removeAll(keepingCapacity: true)
        previousPositions.removeAll(keepingCapacity: true)
        historyPositions.removeAll(keepingCapacity: true)
        frameCount = 0
        cachedTracks.removeAll(keepingCapacity: true)
    }
    
    /// Configure counting parameters
    public func configure(thresholds: [CGFloat], direction: CountingDirection) {
        // Ensure thresholds are within valid range (0.0-1.0)
        guard thresholds.count >= 1 else { return }
        let validThresholds = thresholds.map { max(0.0, min(1.0, $0)) }
        self.thresholds = validThresholds
        self.countingDirection = direction
    }
    
    /// Get current total count
    public func getTotalCount() -> Int {
        return totalCount
    }
    
    /// Get tracking information for visualization
    public func getTrackingInfo() -> [(trackId: Int, position: (x: CGFloat, y: CGFloat), isCounted: Bool)] {
        return cachedTracks.map { track in
            (
                trackId: track.trackId,
                position: track.position,
                isCounted: countedTracks[track.trackId, default: false]
            )
        }
    }
    
    // MARK: - Private Helper Methods
    
    /// Count an object
    /// - Parameter trackId: The track ID to count
    /// - Returns: True if the object was newly counted, false if already counted
    private func countObject(trackId: Int) -> Bool {
        // Only count if not already counted
        if countedTracks[trackId] != true {
            totalCount += 1
            countedTracks[trackId] = true
            
            // Mark the track as counted in cached tracks
            if let trackIndex = cachedTracks.firstIndex(where: { $0.trackId == trackId }) {
                cachedTracks[trackIndex].markCounted()
            }
            
            return true
        }
        return false
    }
    
    /// Clean up data for tracks that no longer exist
    /// - Parameter currentTrackIds: Set of currently active track IDs
    private func cleanupOldTracks(currentTrackIds: Set<Int>) {
        // Remove keys for tracks that no longer exist
        let keysToRemove = countedTracks.keys.filter { !currentTrackIds.contains($0) }
        for key in keysToRemove {
            countedTracks.removeValue(forKey: key)
            previousPositions.removeValue(forKey: key)
            historyPositions.removeValue(forKey: key)
            crossingDirections.removeValue(forKey: key)
        }
    }
    
    /// Helper method to get the expected movement direction based on counting direction
    private func expectedMovementDirection(for countingDirection: CountingDirection) -> Direction {
        switch countingDirection {
        case .topToBottom:
            return .down
        case .bottomToTop:
            return .up
        case .leftToRight:
            return .right
        case .rightToLeft:
            return .left
        }
    }
}

