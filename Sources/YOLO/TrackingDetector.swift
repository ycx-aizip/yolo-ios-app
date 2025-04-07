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

/**
 * TrackingDetector
 *
 * Maps to Python Implementation:
 * - Primary Correspondence: The main fish counting logic in `counting_demo2.py`
 * - Core functionality:
 *   - Integrates with ByteTracker for object tracking
 *   - Implements threshold crossing detection logic
 *   - Maintains count of objects that cross thresholds
 *   - Updates tracking state for visualization
 *
 * Implementation Details:
 * - Extends ObjectDetector class to maintain compatibility with YOLO app
 * - Implements threshold crossing logic similar to `check_threshold_crossing` in Python
 * - Maintains tracking states similar to `track_states` dictionary in Python
 * - Provides methods to integrate tracking information with visualization
 */
class TrackingDetector: ObjectDetector {
    
    /// The ByteTracker instance used for tracking objects
    @MainActor private let byteTracker = ByteTracker()
    
    /// Total count of objects that have crossed the threshold(s)
    private var totalCount: Int = 0
    
    /// Thresholds used for counting (normalized y-coordinates, 0.0-1.0)
    private var thresholds: [CGFloat] = [0.3, 0.5]
    
    /// Map of track IDs to counting status
    private var countedTracks: [Int: Bool] = [:]
    
    /// The current tracked objects
    private var trackedObjects: [STrack] = []
    
    /// Direction of counting
    public enum CountingDirection {
        case topToBottom
        case bottomToTop
    }
    
    /// Current counting direction
    private var countingDirection: CountingDirection = .topToBottom
    
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
    @MainActor
    func resetCount() {
        totalCount = 0
        countedTracks.removeAll()
        byteTracker.reset()
        trackedObjects.removeAll()
        print("TrackingDetector: Reset count")
    }
    
    /// Sets the counting direction
    ///
    /// - Parameter direction: The direction to count (topToBottom or bottomToTop)
    func setCountingDirection(_ direction: CountingDirection) {
        countingDirection = direction
        print("TrackingDetector: Set counting direction to \(direction)")
    }
    
    /**
     * processObservations
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: Similar to logic in `on_predict_postprocess_end` callback
     * - Takes detection results and passes them to ByteTracker
     * - Updates tracking state and performs counting logic
     */
    override func processObservations(for request: VNRequest, error: Error?) {
        if let results = request.results as? [VNRecognizedObjectObservation] {
            var boxes: [Box] = []
            var detectionBoxes: [Box] = []
            var scores: [Float] = []
            var labels: [String] = []
            
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
                    detectionBoxes.append(box)
                    scores.append(confidence)
                    labels.append(label)
                }
            }
            
            // Update tracks with new detections using enhanced ByteTracker
            Task { @MainActor in
                trackedObjects = byteTracker.update(detections: detectionBoxes, scores: scores, classes: labels)
                // Check for threshold crossings
                updateCounting()
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
                names: self.labels)
            
            self.currentOnResultsListener?.on(result: result)
        }
    }
    
    /**
     * updateCounting
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `check_threshold_crossing` and `check_reverse_threshold_crossing` in counting_demo2.py
     * - Checks if objects have crossed threshold lines
     * - Updates count when objects cross in the designated direction
     * - Prevents double-counting through track state management
     */
    @MainActor
    private func updateCounting() {
        let upperThreshold = thresholds.first ?? 0.3
        let lowerThreshold = thresholds.last ?? 0.5
        
        for track in trackedObjects {
            let trackId = track.trackId
            let y = track.position.y
            
            // Skip tracks that have already been counted
            if countedTracks[trackId] == true {
                continue
            }
            
            // Only consider tracks that are recently updated (high confidence in position)
            guard track.isActivated && track.state == .tracked else {
                continue
            }
            
            switch countingDirection {
            case .topToBottom:
                // Count when crossing from upper to lower threshold
                if y >= lowerThreshold {
                    countObject(trackId: trackId)
                }
                
            case .bottomToTop:
                // Count when crossing from lower to upper threshold
                if y <= upperThreshold {
                    countObject(trackId: trackId)
                }
            }
        }
        
        // Clean up tracking for objects that are no longer visible
        let currentIds = Set(trackedObjects.map { $0.trackId })
        var keysToRemove: [Int] = []
        
        for key in countedTracks.keys {
            if !currentIds.contains(key) {
                keysToRemove.append(key)
            }
        }
        
        for key in keysToRemove {
            countedTracks.removeValue(forKey: key)
        }
    }
    
    /**
     * countObject
     *
     * Maps to Python Implementation:
     * - In Python, this is part of the `check_threshold_crossing` function
     * - Increments count and updates tracking state
     * - Provides visualization feedback
     */
    @MainActor
    private func countObject(trackId: Int) {
        // Only count if not already counted
        if countedTracks[trackId] != true {
            totalCount += 1
            countedTracks[trackId] = true
            
            // Mark the track as counted
            if let trackIndex = trackedObjects.firstIndex(where: { $0.trackId == trackId }) {
                trackedObjects[trackIndex].markCounted()
            }
            
            print("TrackingDetector: Counted object with track ID \(trackId), total count: \(totalCount)")
        }
    }
    
    /// Gets the tracking information for a detection box
    ///
    /// - Parameters:
    ///   - box: The detection box to check
    /// - Returns: A tuple containing (isTracked, isCounted)
    func getTrackingStatus(for box: Box) -> (isTracked: Bool, isCounted: Bool) {
        // Since we're accessing actor-isolated state, we need to use a non-blocking approach
        // For now, return default values unless we're already on the main actor
        guard Thread.isMainThread else {
            return (false, false)
        }
        
        if let trackInfo = getTrackInfo(for: box) {
            return (true, trackInfo.isCounted)
        }
        return (false, false)
    }
    
    /**
     * getTrackInfo
     *
     * Maps to Python Implementation:
     * - Used for visualization, similar to coloring logic in Python's drawing functions
     * - Identifies track ID and counted status for UI rendering
     */
    func getTrackInfo(for box: Box) -> (trackId: Int, isCounted: Bool)? {
        // Since we're accessing actor-isolated state, we need to use a non-blocking approach
        // For now, return nil unless we're already on the main actor
        guard Thread.isMainThread else {
            return nil
        }
        
        // Calculate the center of the box
        let centerX = (box.xywhn.minX + box.xywhn.maxX) / 2
        let centerY = (box.xywhn.minY + box.xywhn.maxY) / 2
        
        // Find tracks that match this box
        var bestMatch: STrack? = nil
        var minDistance: CGFloat = 0.1 // Maximum distance to consider
        
        for track in trackedObjects {
            let dx = track.position.x - centerX
            let dy = track.position.y - centerY
            let distance = sqrt(dx*dx + dy*dy)
            
            if distance < minDistance {
                minDistance = distance
                bestMatch = track
            }
        }
        
        if let track = bestMatch {
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
