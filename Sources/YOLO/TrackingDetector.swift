// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  This file is part of the Ultralytics YOLO Package, implementing object tracking functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The TrackingDetector class extends ObjectDetector with tracking capabilities using ByteTrack.
//  It tracks objects across frames, associates detections with existing tracks, and supports
//  threshold crossing detection for counting applications like fish counting. The implementation
//  maintains tracking state for each detected object including position history and counting status.

import Foundation
import UIKit
import Vision

/// Specialized detector for YOLO models with object tracking and counting functionality.
///
/// This class extends the standard ObjectDetector with tracking capabilities,
/// specifically designed for applications like fish counting where objects
/// need to be tracked across frames and counted when crossing defined thresholds.
///
/// - Note: Uses ByteTrack-inspired tracking algorithm to associate detections between frames.
class TrackingDetector: ObjectDetector {
    
    /// The ByteTracker instance used for tracking objects
    private let byteTracker = ByteTracker()
    
    /// Total count of objects that have crossed the threshold(s)
    private var totalCount: Int = 0
    
    /// Thresholds used for counting (normalized y-coordinates, 0.0-1.0)
    private var thresholds: [CGFloat] = [0.3, 0.5]
    
    /// Map of track IDs to counting status
    private var countedTracks: [Int: Bool] = [:]
    
    /// Sets the thresholds for counting
    ///
    /// - Parameter values: Array of normalized y-coordinate values (0.0-1.0)
    func setThresholds(_ values: [CGFloat]) {
        thresholds = values
        print("TrackingDetector: Set thresholds to \(values)")
    }
    
    /// Gets the current count of objects that have crossed the threshold
    ///
    /// - Returns: The total count
    func getCount() -> Int {
        return totalCount
    }
    
    /// Resets the counting state and clears all tracked objects
    func resetCount() {
        totalCount = 0
        countedTracks.removeAll()
        print("TrackingDetector: Reset count")
    }
    
    /// Processes the results from the Vision framework's object detection request.
    ///
    /// This overridden method adds tracking functionality to the standard object detection
    /// process, associating new detections with existing tracks and updating tracking state.
    ///
    /// - Parameters:
    ///   - request: The completed Vision request containing object detection results.
    ///   - error: Any error that occurred during the Vision request.
    override func processObservations(for request: VNRequest, error: Error?) {
        if let results = request.results as? [VNRecognizedObjectObservation] {
            var boxes = [Box]()
            var detections: [(position: (x: CGFloat, y: CGFloat), box: Box)] = []
            
            // Process detections
            for i in 0..<100 {
                if i < results.count && i < self.numItemsThreshold {
                    let prediction = results[i]
                    let invertedBox = CGRect(
                        x: prediction.boundingBox.minX,
                        y: 1 - prediction.boundingBox.maxY,
                        width: prediction.boundingBox.width,
                        height: prediction.boundingBox.height)
                    let imageRect = VNImageRectForNormalizedRect(
                        invertedBox, Int(inputSize.width), Int(inputSize.height))
                    
                    // Extract detection information
                    let label = prediction.labels[0].identifier
                    let index = self.labels.firstIndex(of: label) ?? 0
                    let confidence = prediction.labels[0].confidence
                    let box = Box(
                        index: index, cls: label, conf: confidence, xywh: imageRect, xywhn: invertedBox)
                    
                    boxes.append(box)
                    
                    // Calculate center point - weighted toward the bottom like in Python implementation
                    let centerX = CGFloat(invertedBox.minX + invertedBox.maxX) / 2
                    // Use weighted average to get center closer to the tail (5x weight to bottom)
                    let centerY = CGFloat(invertedBox.minY + invertedBox.maxY * 4) / 5
                    
                    detections.append((position: (x: centerX, y: centerY), box: box))
                }
            }
            
            // Update tracks with new detections using ByteTracker
            let tracks = byteTracker.update(detections: detections)
            
            // Check for threshold crossings
            for track in tracks {
                checkThresholdCrossing(track: track)
            }
            
            // Measure FPS
            if self.t1 < 10.0 {  // valid dt
                self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
            }
            self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
            self.t3 = CACurrentMediaTime()
            
            self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000, fpsRate: 1 / self.t4)
            
            // Add tracking information to the result
            let result = YOLOResult(
                orig_shape: inputSize, 
                boxes: boxes, 
                speed: self.t2, 
                fps: 1 / self.t4, 
                names: labels)
            
            self.currentOnResultsListener?.on(result: result)
        }
    }
    
    /// Checks if an object has crossed any of the defined thresholds
    ///
    /// - Parameter track: The track to check for threshold crossing
    private func checkThresholdCrossing(track: STrack) {
        let trackId = track.trackId
        let currentY = track.position.y
        
        // Get previous counted status or default to false
        let wasCounted = countedTracks[trackId] ?? false
        
        // Check for forward crossing (top to bottom) - count up
        for threshold in thresholds {
            if !wasCounted && track.ttl == 5 { // only check tracks that were just updated
                if currentY >= threshold {
                    totalCount += 1
                    countedTracks[trackId] = true
                    track.markCounted()
                    print("TrackingDetector: Track \(trackId) crossed threshold \(threshold), count: \(totalCount)")
                    break
                }
            }
        }
        
        // Check for reverse crossing (bottom to top) - count down
        // Only using first threshold for reverse counting to avoid double-counting
        if let firstThreshold = thresholds.first {
            if wasCounted && track.ttl == 5 && currentY <= firstThreshold {
                totalCount -= 1
                countedTracks[trackId] = false
                track.markUncounted()
                print("TrackingDetector: Track \(trackId) crossed threshold back, count: \(totalCount)")
            }
        }
    }
    
    /// Gets the tracking information for a detection box
    ///
    /// - Parameters:
    ///   - box: The detection box to check
    /// - Returns: A tuple containing (isTracked, isCounted)
    func getTrackingStatus(for box: Box) -> (isTracked: Bool, isCounted: Bool) {
        // Calculate the center of the box
        let centerX = CGFloat(box.xywhn.minX + box.xywhn.maxX) / 2
        // Use weighted average to get center closer to the tail (5x weight to bottom)
        let centerY = CGFloat(box.xywhn.minY + box.xywhn.maxY * 4) / 5
        
        // Find tracks that match this box
        let position = (x: centerX, y: centerY)
        let tracks = byteTracker.getTracks()
        
        if let idx = MatchingUtils.findBestMatch(position: position, tracks: tracks) {
            let track = tracks[idx]
            return (true, countedTracks[track.trackId] ?? false)
        }
        
        return (false, false)
    }
    
    /// Gets detailed tracking information for a box including the track ID
    ///
    /// - Parameter box: The detection box to get tracking info for
    /// - Returns: A tuple with tracking information or nil if not tracked
    func getTrackInfo(for box: Box) -> (trackId: Int, isCounted: Bool)? {
        // Calculate the center of the box
        let centerX = CGFloat(box.xywhn.minX + box.xywhn.maxX) / 2
        // Use weighted average to get center closer to the tail (5x weight to bottom)
        let centerY = CGFloat(box.xywhn.minY + box.xywhn.maxY * 4) / 5
        
        // Find tracks that match this box
        let position = (x: centerX, y: centerY)
        let tracks = byteTracker.getTracks()
        
        if let idx = MatchingUtils.findBestMatch(position: position, tracks: tracks) {
            let track = tracks[idx]
            return (trackId: track.trackId, isCounted: countedTracks[track.trackId] ?? false)
        }
        
        return nil
    }
    
    /// Checks if a detection box is currently being tracked
    ///
    /// - Parameter box: The detection box to check
    /// - Returns: True if the box is associated with an active track
    func isObjectTracked(box: Box) -> Bool {
        return getTrackingStatus(for: box).isTracked
    }
    
    /// Checks if a detection box has been counted
    ///
    /// - Parameter box: The detection box to check
    /// - Returns: True if the box is associated with a track that has been counted
    func isObjectCounted(box: Box) -> Bool {
        return getTrackingStatus(for: box).isCounted
    }
}
