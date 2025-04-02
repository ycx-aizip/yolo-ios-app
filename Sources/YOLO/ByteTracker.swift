// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  This file is part of the Ultralytics YOLO Package, implementing ByteTrack algorithm for object tracking.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//
//  The ByteTracker class implements the ByteTrack multi-object tracking algorithm.
//  It associates detections with existing tracks and maintains the status of all tracks.

import Foundation
import CoreGraphics
import Vision

/// ByteTracker class for tracking objects across frames
public class ByteTracker {
    /// Collection of active tracks
    private var tracks: [STrack] = []
    
    /// Next available track ID
    private var nextTrackId: Int = 0
    
    /// Kalman filter for motion prediction
    private let kalmanFilter = KalmanFilter()
    
    /// High detection confidence threshold
    private let highThreshold: Float = 0.5
    
    /// Low detection confidence threshold for track matching
    private let lowThreshold: Float = 0.1
    
    /// IoU threshold for detection-track matching
    private let iouThreshold: Float = 0.5
    
    /// Initialize a new ByteTracker
    public init() {
        print("DEBUG: ByteTracker initialized")
    }
    
    /// Update tracks with new detections
    /// - Parameters:
    ///   - detections: Array of bounding boxes from the detector
    ///   - scores: Array of confidence scores
    ///   - classIds: Array of class IDs
    /// - Returns: Updated array of active tracks
    public func update(detections: [CGRect], scores: [Float], classIds: [Int]) -> [STrack] {
        print("DEBUG: ByteTracker updating with \(detections.count) detections")
        
        // 1. Predict new locations for existing tracks
        predictTracks()
        
        // 2. Separate detections into high-confidence and low-confidence
        var highScoreDetections: [(box: CGRect, score: Float, classId: Int)] = []
        var lowScoreDetections: [(box: CGRect, score: Float, classId: Int)] = []
        
        for i in 0..<detections.count {
            if i < scores.count && i < classIds.count {
                let detection = (box: detections[i], score: scores[i], classId: classIds[i])
                if scores[i] >= highThreshold {
                    highScoreDetections.append(detection)
                } else if scores[i] >= lowThreshold {
                    lowScoreDetections.append(detection)
                }
            }
        }
        
        print("DEBUG: \(highScoreDetections.count) high-confidence detections, \(lowScoreDetections.count) low-confidence detections")
        
        // 3. Match high-score detections to tracks
        let (matchedTrackIndices, matchedDetectionIndices, unmatchedTrackIndices, unmatchedDetectionIndices) = 
            matchDetectionsToTracks(detections: highScoreDetections.map { $0.box })
        
        print("DEBUG: First association - matched: \(matchedTrackIndices.count), unmatched tracks: \(unmatchedTrackIndices.count), unmatched detections: \(unmatchedDetectionIndices.count)")
        
        // 4. Update matched tracks
        for i in 0..<matchedTrackIndices.count {
            if i < matchedDetectionIndices.count {
                let trackIdx = matchedTrackIndices[i]
                let detIdx = matchedDetectionIndices[i]
                if trackIdx < tracks.count && detIdx < highScoreDetections.count {
                    let detection = highScoreDetections[detIdx]
                    tracks[trackIdx].update(bbox: detection.box, score: detection.score)
                }
            }
        }
        
        // 5. Process unmatched tracks
        for idx in unmatchedTrackIndices {
            if idx < tracks.count {
                tracks[idx].markAsLost()
            }
        }
        
        // 6. Process unmatched detections (create new tracks)
        for idx in unmatchedDetectionIndices {
            if idx < highScoreDetections.count {
                let detection = highScoreDetections[idx]
                createNewTrack(bbox: detection.box, score: detection.score, classId: detection.classId)
            }
        }
        
        // 7. Clean up lost tracks
        cleanUpTracks()
        
        // Return active tracks
        let activeTracks = tracks.filter { $0.state == .tracked }
        print("DEBUG: ByteTracker now has \(activeTracks.count) active tracks")
        
        return activeTracks
    }
    
    /// Predict new locations for all tracks
    private func predictTracks() {
        print("DEBUG: Predicting new positions for \(tracks.count) tracks")
        for track in tracks {
            track.predict()
        }
    }
    
    /// Match detections to existing tracks using IoU
    /// - Parameter detections: Array of detection bounding boxes
    /// - Returns: Tuple of (matched track indices, matched detection indices, unmatched track indices, unmatched detection indices)
    private func matchDetectionsToTracks(detections: [CGRect]) -> ([Int], [Int], [Int], [Int]) {
        print("DEBUG: Matching \(detections.count) detections to \(tracks.count) tracks")
        
        // This is a placeholder for the full IoU-based matching algorithm
        // In the full implementation, we would compute IoU between all detections and tracks
        // and use the Hungarian algorithm to find optimal matching
        
        var matchedTrackIndices: [Int] = []
        var matchedDetectionIndices: [Int] = []
        var unmatchedTrackIndices: [Int] = []
        var unmatchedDetectionIndices: [Int] = []
        
        // Simple greedy matching for now
        var usedDetections = Array(repeating: false, count: detections.count)
        
        for (trackIdx, track) in tracks.enumerated() {
            if track.state != .removed {
                var bestIoU: Float = 0
                var bestDetectionIdx: Int = -1
                
                for (detIdx, detection) in detections.enumerated() {
                    if !usedDetections[detIdx] {
                        let iou = calculateIoU(track.bbox, detection)
                        if iou > bestIoU && iou > iouThreshold {
                            bestIoU = iou
                            bestDetectionIdx = detIdx
                        }
                    }
                }
                
                if bestDetectionIdx >= 0 {
                    matchedTrackIndices.append(trackIdx)
                    matchedDetectionIndices.append(bestDetectionIdx)
                    usedDetections[bestDetectionIdx] = true
                } else {
                    unmatchedTrackIndices.append(trackIdx)
                }
            }
        }
        
        // Find unmatched detections
        for (detIdx, used) in usedDetections.enumerated() {
            if !used {
                unmatchedDetectionIndices.append(detIdx)
            }
        }
        
        return (matchedTrackIndices, matchedDetectionIndices, unmatchedTrackIndices, unmatchedDetectionIndices)
    }
    
    /// Create a new track from a detection
    /// - Parameters:
    ///   - bbox: Bounding box
    ///   - score: Confidence score
    ///   - classId: Class ID
    private func createNewTrack(bbox: CGRect, score: Float, classId: Int) {
        let newTrack = STrack(trackId: nextTrackId, bbox: bbox, score: score, classId: classId)
        nextTrackId += 1
        tracks.append(newTrack)
        print("DEBUG: Created new track with ID \(newTrack.trackId)")
    }
    
    /// Remove tracks that have been lost for too long
    private func cleanUpTracks() {
        tracks = tracks.filter { !$0.shouldBeRemoved() }
        print("DEBUG: After cleanup, \(tracks.count) tracks remain")
    }
    
    /// Calculate Intersection over Union between two bounding boxes
    /// - Parameters:
    ///   - box1: First bounding box
    ///   - box2: Second bounding box
    /// - Returns: IoU value (0-1)
    private func calculateIoU(_ box1: CGRect, _ box2: CGRect) -> Float {
        let intersection = box1.intersection(box2)
        if intersection.isEmpty {
            return 0
        }
        
        let unionArea = box1.width * box1.height + box2.width * box2.height - intersection.width * intersection.height
        if unionArea <= 0 {
            return 0
        }
        
        return Float(intersection.width * intersection.height / unionArea)
    }
} 