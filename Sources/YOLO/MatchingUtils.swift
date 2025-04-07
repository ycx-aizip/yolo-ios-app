// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  MatchingUtils.swift
//  YOLO
//
//  Utility functions for track association and matching.
//  Implements distance calculation and track matching algorithms.

import Foundation
import UIKit

/**
 * MatchingUtils
 *
 * Maps to Python Implementation:
 * - Primary Correspondence: Functions in `ultralytics/trackers/matching.py`
 * - Core functionality:
 *   - Calculates distances between detections and tracks
 *   - Performs optimal assignment for track matching
 *   - Provides utilities for track association
 *
 * Implementation Details:
 * - Implements IoU-based distance calculations similar to Python version
 * - Provides a simplified but effective linear assignment algorithm
 * - Follows the same two-stage matching approach as ByteTrack
 */
/// MatchingUtils provides functions for track association and distance calculations
public class MatchingUtils {
    // MARK: - Distance Calculations
    
    /**
     * calculateIoU
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `iou_batch()` function in `matching.py`
     * - Calculates intersection over union for bounding boxes
     * - Used for determining matching quality between tracks and detections
     */
    /// Calculate IoU between two bounding boxes
    /// - Parameters:
    ///   - box1: First bounding box in normalized coordinates
    ///   - box2: Second bounding box in normalized coordinates
    /// - Returns: IoU value between 0 and 1
    public static func calculateIoU(box1: CGRect, box2: CGRect) -> Float {
        let intersection = box1.intersection(box2)
        
        if intersection.isEmpty {
            return 0.0
        }
        
        let intersectionArea = intersection.width * intersection.height
        let box1Area = box1.width * box1.height
        let box2Area = box2.width * box2.height
        let unionArea = box1Area + box2Area - intersectionArea
        
        return Float(intersectionArea / unionArea)
    }
    
    /**
     * iouDistance
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `iou_distance()` function in `matching.py`
     * - Creates cost matrix based on IoU similarity between boxes
     * - Used as primary distance metric for first-stage matching
     */
    /// Calculate cost matrix using IoU between tracks and detections
    /// - Parameters:
    ///   - tracks: Array of tracks
    ///   - detections: Array of detections
    /// - Returns: Cost matrix as 2D array where lower values indicate better matches
    public static func iouDistance(tracks: [STrack], detections: [STrack]) -> [[Float]] {
        let numTracks = tracks.count
        let numDetections = detections.count
        
        // Initialize cost matrix with max value
        var costMatrix = Array(repeating: Array(repeating: Float(1.0), count: numDetections), count: numTracks)
        
        for i in 0..<numTracks {
            if let trackBox = tracks[i].lastDetection?.xywhn {
                for j in 0..<numDetections {
                    if let detBox = detections[j].lastDetection?.xywhn {
                        let iou = calculateIoU(box1: trackBox, box2: detBox)
                        // 1 - IoU as distance (lower is better)
                        costMatrix[i][j] = 1.0 - iou
                    }
                }
            }
        }
        
        return costMatrix
    }
    
    /**
     * positionDistance
     *
     * Maps to Python Implementation:
     * - Similar to distance calculations in `matching.py`
     * - Uses Euclidean distance between centers as fallback metric
     */
    /// Calculate cost matrix using position distance between tracks and detections
    /// Can be used as fallback when IoU is not available
    /// - Parameters:
    ///   - tracks: Array of tracks
    ///   - detections: Array of detections
    /// - Returns: Cost matrix as 2D array where lower values indicate better matches
    public static func positionDistance(tracks: [STrack], detections: [STrack]) -> [[Float]] {
        let numTracks = tracks.count
        let numDetections = detections.count
        
        // Initialize cost matrix with max value
        var costMatrix = Array(repeating: Array(repeating: Float(2.0), count: numDetections), count: numTracks)
        
        for i in 0..<numTracks {
            let trackPos = tracks[i].position
            for j in 0..<numDetections {
                let detPos = detections[j].position
                // Euclidean distance normalized by 2 (since coords are in [0,1])
                let dx = trackPos.x - detPos.x
                let dy = trackPos.y - detPos.y
                let distance = Float(sqrt(dx * dx + dy * dy))
                // Distance is between 0 and ~1.4 (diagonal of unit square)
                costMatrix[i][j] = min(distance, 2.0)
            }
        }
        
        return costMatrix
    }
    
    // MARK: - Matching Operations
    
    /**
     * linearAssignment
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `linear_assignment()` function in `matching.py`
     * - Simplified version of the Hungarian algorithm used in Python
     * - Optimized for the specific needs of the tracking application
     */
    /// Hungarian algorithm for optimal assignment, minimizing cost
    /// - Parameter costMatrix: Cost matrix where rows are tracks and columns are detections
    /// - Returns: Tuple of (matched track indices, matched detection indices, unmatched track indices, unmatched detection indices)
    public static func linearAssignment(costMatrix: [[Float]], threshold: Float = 0.7) -> (
        [Int], [Int], [Int], [Int]
    ) {
        let rowIndices = Array(0..<costMatrix.count)
        let colIndices = costMatrix.isEmpty ? [] : Array(0..<costMatrix[0].count)
        
        var matches: [(Int, Int)] = []
        var unmatched1: [Int] = []
        var unmatched2: [Int] = []
        
        // Handle empty cases
        if costMatrix.isEmpty || colIndices.isEmpty {
            unmatched1 = rowIndices
            unmatched2 = colIndices
            return ([], [], unmatched1, unmatched2)
        }
        
        // Find matches using greedy approach (for simplicity)
        // This is a simplified version of the Hungarian algorithm
        var usedRows = Set<Int>()
        var usedCols = Set<Int>()
        
        // Convert cost matrix to flat array of tuples (row, col, cost)
        var costs: [(row: Int, col: Int, cost: Float)] = []
        for i in 0..<costMatrix.count {
            for j in 0..<costMatrix[i].count {
                costs.append((row: i, col: j, cost: costMatrix[i][j]))
            }
        }
        
        // Sort by cost (ascending)
        costs.sort { $0.cost < $1.cost }
        
        // Pick best matches
        for cost in costs {
            if cost.cost > threshold {
                break // Stop if cost exceeds threshold
            }
            
            if !usedRows.contains(cost.row) && !usedCols.contains(cost.col) {
                matches.append((cost.row, cost.col))
                usedRows.insert(cost.row)
                usedCols.insert(cost.col)
            }
        }
        
        // If matches are too few, try to be more lenient with threshold
        if matches.count < min(rowIndices.count, colIndices.count) / 3 {
            let lenientThreshold = min(threshold * 1.5, 0.9) // Increase threshold but cap at 0.9
            
            for cost in costs {
                if cost.cost > lenientThreshold || cost.cost <= threshold {
                    continue // Skip if still too high or already processed
                }
                
                if !usedRows.contains(cost.row) && !usedCols.contains(cost.col) {
                    matches.append((cost.row, cost.col))
                    usedRows.insert(cost.row)
                    usedCols.insert(cost.col)
                }
            }
        }
        
        // Find unmatched indices
        for i in rowIndices {
            if !usedRows.contains(i) {
                unmatched1.append(i)
            }
        }
        
        for j in colIndices {
            if !usedCols.contains(j) {
                unmatched2.append(j)
            }
        }
        
        return (
            matches.map { $0.0 },
            matches.map { $0.1 },
            unmatched1,
            unmatched2
        )
    }
    
    /**
     * associateFirstStage
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: First call to `_match_track_detection()` in ByteTracker
     * - First stage matches high-confidence detections with active tracks
     * - Uses IoU as primary matching metric
     */
    /// Perform track association in the first stage, using IoU distance
    /// - Parameters:
    ///   - tracks: Array of tracks
    ///   - detections: Array of detections
    ///   - thresholdFirstStage: Matching threshold for first stage
    /// - Returns: Tuple of matches, unmatched tracks, and unmatched detections
    public static func associateFirstStage(
        tracks: [STrack], 
        detections: [STrack], 
        thresholdFirstStage: Float
    ) -> ([(Int, Int)], [Int], [Int]) {
        if tracks.isEmpty || detections.isEmpty {
            return ([], Array(0..<tracks.count), Array(0..<detections.count))
        }
        
        // Calculate IoU distance matrix
        let dists = iouDistance(tracks: tracks, detections: detections)
        
        // Run linear assignment with the distance matrix
        let (matchedTrackIndices, matchedDetIndices, unmatchedTrackIndices, unmatchedDetIndices) =
            linearAssignment(costMatrix: dists, threshold: thresholdFirstStage)
        
        // Convert to tuples of (track_idx, det_idx)
        var matches: [(Int, Int)] = []
        for i in 0..<matchedTrackIndices.count {
            matches.append((matchedTrackIndices[i], matchedDetIndices[i]))
        }
        
        return (matches, unmatchedTrackIndices, unmatchedDetIndices)
    }
    
    /**
     * associateSecondStage
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: Second call to `_match_track_detection()` in ByteTracker
     * - Matches remaining detections with lost tracks
     * - Uses same metrics but with lower threshold
     */
    /// Perform track association in the second stage, using position distance
    /// - Parameters:
    ///   - tracks: Array of tracks
    ///   - detections: Array of detections
    ///   - thresholdSecondStage: Matching threshold for second stage
    /// - Returns: Tuple of matches and unmatched detections
    public static func associateSecondStage(
        tracks: [STrack], 
        detections: [STrack], 
        thresholdSecondStage: Float
    ) -> ([(Int, Int)], [Int]) {
        if tracks.isEmpty || detections.isEmpty {
            return ([], Array(0..<detections.count))
        }
        
        // Calculate position distance matrix
        let dists = positionDistance(tracks: tracks, detections: detections)
        
        // Run linear assignment with the distance matrix
        let (matchedTrackIndices, matchedDetIndices, _, unmatchedDetIndices) =
            linearAssignment(costMatrix: dists, threshold: thresholdSecondStage)
        
        // Convert to tuples of (track_idx, det_idx)
        var matches: [(Int, Int)] = []
        for i in 0..<matchedTrackIndices.count {
            matches.append((matchedTrackIndices[i], matchedDetIndices[i]))
        }
        
        return (matches, unmatchedDetIndices)
    }
    
    /// Filter overlapping tracks using their scores
    /// - Parameters:
    ///   - tracks: Array of tracks to filter
    ///   - threshold: IoU threshold to consider tracks as overlapping
    /// - Returns: Array of filtered track indices to remove
    public static func filterTracks(tracks: [STrack], threshold: Float = 0.6) -> [Int] {
        if tracks.count <= 1 {
            return []
        }
        
        var toRemove = Set<Int>()
        
        // Sort tracks by score (descending)
        let sortedIndices = tracks.indices.sorted { tracks[$0].score > tracks[$1].score }
        
        for i in 0..<sortedIndices.count {
            if toRemove.contains(sortedIndices[i]) {
                continue
            }
            
            let trackA = tracks[sortedIndices[i]]
            guard let boxA = trackA.lastDetection?.xywhn else {
                continue
            }
            
            for j in i+1..<sortedIndices.count {
                if toRemove.contains(sortedIndices[j]) {
                    continue
                }
                
                let trackB = tracks[sortedIndices[j]]
                guard let boxB = trackB.lastDetection?.xywhn else {
                    continue
                }
                
                // If two tracks overlap significantly, keep the one with higher score
                let iou = calculateIoU(box1: boxA, box2: boxB)
                if iou > threshold {
                    toRemove.insert(sortedIndices[j])
                }
            }
        }
        
        return Array(toRemove)
    }
} 