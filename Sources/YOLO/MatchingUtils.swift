// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  MatchingUtils.swift
//  YOLO
//
//  Utilities for matching detections with existing tracks.
//  This is a placeholder implementation for debugging the app flow.

import Foundation
import UIKit

/// Utilities for matching detections with tracks
public class MatchingUtils {
    /// Calculate the distance between two points
    /// - Parameters:
    ///   - p1: First point
    ///   - p2: Second point
    /// - Returns: Euclidean distance between the points
    public static func distance(p1: (x: CGFloat, y: CGFloat), p2: (x: CGFloat, y: CGFloat)) -> CGFloat {
        return hypot(p1.x - p2.x, p1.y - p2.y)
    }
    
    /// Find the best match for a detection among existing tracks
    /// - Parameters:
    ///   - position: The position to match
    ///   - tracks: Array of existing tracks
    ///   - maxDistance: Maximum allowed distance for a match
    /// - Returns: The index of the best matching track, or nil if no match found
    public static func findBestMatch(position: (x: CGFloat, y: CGFloat), 
                                    tracks: [STrack], 
                                    maxDistance: CGFloat = 0.1) -> Int? {
        var bestTrackIdx: Int? = nil
        var bestDistance = CGFloat.greatestFiniteMagnitude
        
        for (idx, track) in tracks.enumerated() {
            let dist = distance(p1: position, p2: track.position)
            
            if dist < bestDistance && dist < maxDistance {
                bestDistance = dist
                bestTrackIdx = idx
            }
        }
        
        if let idx = bestTrackIdx {
            print("MatchingUtils: Found match for position (\(position.x), \(position.y)) with track \(tracks[idx].trackId) at distance \(bestDistance)")
        } else {
            print("MatchingUtils: No match found for position (\(position.x), \(position.y))")
        }
        
        return bestTrackIdx
    }
} 