// CounterProtocol.swift
// Protocol abstraction for fish counting algorithms
//
// This protocol defines a common interface for different counting strategies
// (e.g., threshold-based, adaptive) to be used interchangeably with the tracking system.

import Foundation
import CoreGraphics

/// Result of counting operation for a frame
public struct CountingResult {
    /// Current total count
    let totalCount: Int
    
    /// Track IDs that were newly counted in this frame
    let newlyCounted: [Int]
    
    /// Tracking information for visualization
    /// Each tuple contains: (trackId, position, isCounted)
    let trackingInfo: [(trackId: Int, position: (x: CGFloat, y: CGFloat), isCounted: Bool)]
}

/// Protocol defining the interface for fish counting algorithms
@MainActor
public protocol CounterProtocol {
    /// Process tracked objects and update counts
    ///
    /// - Parameter tracks: Array of tracked objects to process
    /// - Returns: CountingResult with updated count and tracking information
    func processFrame(tracks: [STrack]) -> CountingResult
    
    /// Reset counter state, clearing all counts and history
    func resetCount()
    
    /// Configure counting parameters
    ///
    /// - Parameters:
    ///   - thresholds: Threshold values for counting (normalized 0.0-1.0)
    ///   - direction: Direction of counting (topToBottom, bottomToTop, etc.)
    func configure(thresholds: [CGFloat], direction: CountingDirection)
    
    /// Get current total count
    ///
    /// - Returns: Total number of objects counted
    func getTotalCount() -> Int
    
    /// Get tracking information for visualization
    ///
    /// - Returns: Array of tuples with trackId, position, and counted status
    func getTrackingInfo() -> [(trackId: Int, position: (x: CGFloat, y: CGFloat), isCounted: Bool)]
}

