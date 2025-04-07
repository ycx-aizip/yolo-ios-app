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
    
    /// Represents the current state of tracked objects
    private var trackStates: [Int: TrackState] = [:]
    
    /// Total count of objects that have crossed the threshold(s)
    private var totalCount: Int = 0
    
    /// Thresholds used for counting (normalized y-coordinates, 0.0-1.0)
    private var thresholds: [CGFloat] = [0.3, 0.5]
    
    /// The next available tracking ID
    private var nextTrackId: Int = 0
    
    /// Structure to store the state of a tracked object
    private struct TrackState {
        /// The last known position of the tracked object (normalized coordinates)
        var lastPosition: (x: CGFloat, y: CGFloat)
        
        /// Flag indicating whether this object has been counted
        var counted: Bool = false
        
        /// Time-to-live counter for the track (decremented when object not detected)
        var ttl: Int = 5
        
        /// The most recent detection box associated with this track
        var lastDetection: Box?
    }
    
    /// Sets the thresholds for counting
    ///
    /// - Parameter values: Array of normalized y-coordinate values (0.0-1.0)
    func setThresholds(_ values: [CGFloat]) {
        thresholds = values
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
        trackStates.removeAll()
    }
    
    /// Checks if a detection box's center is close to any of the tracked centers
    ///
    /// - Parameters:
    ///   - box: The detection box to check
    ///   - position: The position to check against tracked objects
    /// - Returns: (isTracked, isCounted, trackId) tuple
    private func getTrackingInfo(for position: (x: CGFloat, y: CGFloat)) -> (isTracked: Bool, isCounted: Bool, trackId: Int?) {
        var bestTrackId: Int? = nil
        var bestDistance = CGFloat.greatestFiniteMagnitude
        
        for (trackId, trackState) in trackStates {
            let distance = hypot(position.x - trackState.lastPosition.x, 
                                position.y - trackState.lastPosition.y)
            
            if distance < bestDistance && distance < 0.1 { // threshold for matching
                bestDistance = distance
                bestTrackId = trackId
            }
        }
        
        if let trackId = bestTrackId {
            return (true, trackStates[trackId]?.counted ?? false, trackId)
        }
        
        return (false, false, nil)
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
            var detections: [(box: Box, center: (x: CGFloat, y: CGFloat))] = []
            
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
                    
                    detections.append((box: box, center: (x: centerX, y: centerY)))
                }
            }
            
            // Update track states and decrease TTL for missing tracks
            for trackId in trackStates.keys {
                trackStates[trackId]?.ttl -= 1
            }
            
            // Remove tracks that have expired TTL
            trackStates = trackStates.filter { $0.value.ttl > 0 }
            
            // Associate detections with existing tracks using simple IoU-based matching
            // In a full implementation, this would use more sophisticated ByteTrack algorithm
            var assignedTracks = Set<Int>()
            var assignedDetections = Set<Int>()
            
            // Simple tracking by IoU overlap
            for (trackId, trackState) in trackStates {
                let trackPoint = CGPoint(x: trackState.lastPosition.x, y: trackState.lastPosition.y)
                
                var bestDetectionIdx = -1
                var bestDistance = CGFloat.greatestFiniteMagnitude
                
                // Find closest detection to this track
                for (idx, detection) in detections.enumerated() {
                    if assignedDetections.contains(idx) {
                        continue
                    }
                    
                    let detectionPoint = CGPoint(x: detection.center.x, y: detection.center.y)
                    let distance = hypot(trackPoint.x - detectionPoint.x, trackPoint.y - detectionPoint.y)
                    
                    if distance < bestDistance {
                        bestDistance = distance
                        bestDetectionIdx = idx
                    }
                }
                
                // If we found a close enough detection, update the track
                if bestDetectionIdx >= 0 && bestDistance < 0.1 { // threshold for matching
                    let detection = detections[bestDetectionIdx]
                    
                    // Update position and reset TTL
                    trackStates[trackId]?.lastPosition = detection.center
                    trackStates[trackId]?.ttl = 5
                    trackStates[trackId]?.lastDetection = detection.box
                    
                    // Check for threshold crossing
                    checkThresholdCrossing(trackId: trackId, 
                                          currentPosition: detection.center)
                    
                    assignedTracks.insert(trackId)
                    assignedDetections.insert(bestDetectionIdx)
                }
            }
            
            // Create new tracks for unassigned detections
            for (idx, detection) in detections.enumerated() {
                if !assignedDetections.contains(idx) {
                    let trackId = nextTrackId
                    nextTrackId += 1
                    
                    trackStates[trackId] = TrackState(
                        lastPosition: detection.center,
                        counted: false,
                        ttl: 5,
                        lastDetection: detection.box
                    )
                }
            }
            
            // Add tracking information to the boxes for visualization
            // We use a side-channel approach to pass tracking information to the UI
            // by creating a custom property in YOLOResult
            
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
    /// - Parameters:
    ///   - trackId: The ID of the track to check
    ///   - currentPosition: The current position of the tracked object
    private func checkThresholdCrossing(trackId: Int, currentPosition: (x: CGFloat, y: CGFloat)) {
        guard let trackState = trackStates[trackId] else { return }
        
        let lastY = trackState.lastPosition.y
        let currentY = currentPosition.y
        
        // Check for forward crossing (top to bottom) - count up
        for threshold in thresholds {
            if !trackState.counted && lastY < threshold && currentY >= threshold {
                totalCount += 1
                trackStates[trackId]?.counted = true
                break
            }
        }
        
        // Check for reverse crossing (bottom to top) - count down
        // Only using first threshold for reverse counting to avoid double-counting
        if let firstThreshold = thresholds.first {
            if trackState.counted && lastY > firstThreshold && currentY <= firstThreshold {
                totalCount -= 1
                trackStates[trackId]?.counted = false
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
        
        // Get tracking info based on position
        let (isTracked, isCounted, _) = getTrackingInfo(for: (x: centerX, y: centerY))
        return (isTracked, isCounted)
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
