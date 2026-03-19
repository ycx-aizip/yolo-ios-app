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

    // MARK: - Class Configuration (Single Source of Truth for Counting)

    /// Default thresholds for counting (normalized coordinates, 0.0-1.0)
    /// For vertical directions (topToBottom, bottomToTop), these are y-coordinates
    /// For horizontal directions (leftToRight, rightToLeft), these are x-coordinates
    nonisolated(unsafe) public static var defaultThresholds: [CGFloat] = [0.4, 0.6]

    /// Default counting direction (which way fish move through the scene)
    /// Determines threshold interpretation and expected movement direction
    nonisolated(unsafe) public static var defaultCountingDirection: CountingDirection = .bottomToTop

    // MARK: - Instance Properties

    /// Total count of objects that have crossed the threshold(s)
    private var totalCount: Int = 0

    /// Thresholds used for counting (initialized from defaultThresholds)
    private var thresholds: [CGFloat]
    
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
        // Initialize with class default thresholds
        self.thresholds = ThresholdCounter.defaultThresholds
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

        // Periodic pipeline summary every 150 frames (~5s at 30fps)
        if TrackingParameters.debugLogging && frameCount % 150 == 0 {
            let activeTracks = tracks.count
            let countedSoFar = countedTracks.values.filter { $0 }.count
            print("[DIAG] SUMMARY frame=\(frameCount): active_tracks=\(activeTracks) counted_tracks=\(countedSoFar) total_count=\(totalCount) thresh=\(thresholds.map { String(format: "%.2f", $0) })")
        }

        // Clean up old data periodically (every 10 frames ~0.3-0.7 sec at 15-30 FPS)
        // More frequent cleanup prevents dictionary bloat in dense, long recordings
        if frameCount % 10 == 0 {
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

            if TrackingParameters.debugLogging {
                let pos = previousPositions[trackId]
                let posStr = pos.map { String(format: "(%.3f,%.3f)", $0.x, $0.y) } ?? "?"
                print("[DIAG] COUNT: #\(trackId) pos=\(posStr) thresh=\(thresholds.map { String(format: "%.2f", $0) }) total=\(totalCount)")
            }

            return true
        }
        return false
    }
    
    /// Clean up data for tracks that no longer exist
    /// - Parameter currentTrackIds: Set of currently active track IDs
    private func cleanupOldTracks(currentTrackIds: Set<Int>) {
        // previousPositions is the master set — all active tracks have an entry here.
        // countedTracks entries for departed fish are intentionally preserved (see below).
        let keysToRemove = previousPositions.keys.filter { !currentTrackIds.contains($0) }
        for key in keysToRemove {
            let wasCounted = countedTracks[key] == true
            if !wasCounted, let lastPos = previousPositions[key] {
                // Convert to counting coordinate to compare against thresholds
                let unified = UnifiedCoordinateSystem.UnifiedRect(x: lastPos.x, y: lastPos.y, width: 0, height: 0)
                let counting = UnifiedCoordinateSystem.toCounting(unified, countingDirection: countingDirection)
                let coord = (countingDirection == .leftToRight || countingDirection == .rightToLeft) ? counting.x : counting.y
                let threshStr = thresholds.map { String(format: "%.2f", $0) }.joined(separator: ",")

                // Retroactive counting: fish that passed all thresholds but were never caught.
                // This handles fish that entered the frame already past the counting zone
                // (e.g., detected for the first time in the top 40% of screen for bottomToTop).
                // Condition: last known position is beyond all thresholds AND last movement
                // was in the expected direction (prevents counting fish that wandered back).
                let expectedDir = expectedMovementDirection(for: countingDirection)
                let lastDir = crossingDirections[key]
                let passedAllThresholds: Bool
                switch countingDirection {
                case .bottomToTop:
                    passedAllThresholds = lastPos.y < (thresholds.min() ?? 1.0)
                case .topToBottom:
                    passedAllThresholds = lastPos.y > (thresholds.max() ?? 0.0)
                case .leftToRight:
                    passedAllThresholds = lastPos.x > (thresholds.max() ?? 0.0)
                case .rightToLeft:
                    passedAllThresholds = lastPos.x < (thresholds.min() ?? 1.0)
                }

                if passedAllThresholds && lastDir == expectedDir {
                    if TrackingParameters.debugLogging {
                        print("[DIAG] RETRO_COUNT: #\(key) pos=\(String(format: "(%.3f,%.3f)", lastPos.x, lastPos.y)) thresh=[\(threshStr)] dir=\(countingDirection)")
                    }
                    _ = countObject(trackId: key)
                } else if TrackingParameters.debugLogging {
                    print("[DIAG] TRACK_LOST_UNCOUNTED: #\(key) counting_pos=\(String(format: "%.3f", coord)) thresh=[\(threshStr)] dir=\(countingDirection)")
                }
            }
            // IMPORTANT: Never remove countedTracks[id] = true entries.
            // OCSort uses monotonically increasing IDs — a counted ID is never reassigned to a
            // different fish. Preserving the entry prevents double-counting when OCSort temporarily
            // loses and re-confirms the same track ID.
            if !wasCounted {
                countedTracks.removeValue(forKey: key)
            }
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

