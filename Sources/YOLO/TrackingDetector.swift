// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  This file is part of the Ultralytics YOLO Package, implementing object detection with tracking.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//
//  The TrackingDetector class extends ObjectDetector to add object tracking capabilities.
//  It uses ByteTrack to maintain consistent object IDs across frames and supports fish counting.

import Foundation
import UIKit
import Vision

/// Specialized predictor for object detection with tracking capabilities
class TrackingDetector: ObjectDetector {
    /// ByteTracker for object tracking
    private let tracker = ByteTracker()
    
    /// List of tracks from the previous frame
    private var tracks: [STrack] = []
    
    /// Callback for when count changes
    var onCountChanged: ((Int) -> Void)?
    
    /// Fish counting properties
    private var fishCount: Int = 0
    private var fishPassingInfo: [Int: (crossed: Bool, direction: String?)] = [:]
    
    /// Threshold values for horizontal lines (normalized 0-1)
    /// Multiple thresholds for top-down view (default: 0.3, 0.5)
    var thresholds: [CGFloat] = [0.3, 0.5]
    
    // Comment out individual directional thresholds as they're replaced by the thresholds array
    /*
    /// Threshold values (normalized 0-1)
    var upperThreshold: CGFloat = 0.3
    var bottomThreshold: CGFloat = 0.7
    var leftThreshold: CGFloat = 0.3
    var rightThreshold: CGFloat = 0.7
    
    /// Active thresholds based on camera angle
    var activeThresholds: Set<String> = ["upper", "bottom", "left", "right"]
    */
    
    /// Factory method to create a TrackingDetector instance
    /// - Parameters:
    ///   - unwrappedModelURL: URL to the CoreML model
    ///   - isRealTime: Whether the detector will be used for real-time processing
    ///   - completion: Completion handler called with the created detector or an error
    // public static func create(
    //     unwrappedModelURL: URL,
    //     isRealTime: Bool = false,
    //     completion: @escaping (Result<TrackingDetector, Error>) -> Void
    // ) {
    //     // Delegate to ObjectDetector.create() but cast the result to TrackingDetector
    //     ObjectDetector.create(unwrappedModelURL: unwrappedModelURL, isRealTime: isRealTime) { result in
    //         switch result {
    //         case .success(let predictor):
    //             // Create a new TrackingDetector and copy properties from the ObjectDetector
    //             let trackingDetector = TrackingDetector()
                
    //             // Copy necessary properties from the ObjectDetector
    //             trackingDetector.detector = predictor.detector
    //             trackingDetector.visionRequest = predictor.visionRequest
    //             trackingDetector.labels = predictor.labels
    //             trackingDetector.inputSize = predictor.inputSize
    //             trackingDetector.modelInputSize = predictor.modelInputSize
    //             trackingDetector.isModelLoaded = predictor.isModelLoaded
                
    //             print("DEBUG: Created TrackingDetector with \(trackingDetector.labels.count) classes")
                
    //             // Return the TrackingDetector
    //             completion(.success(trackingDetector))
                
    //         case .failure(let error):
    //             completion(.failure(error))
    //         }
    //     }
    // }
    
    /// Override the process observations method to add tracking
    override func processObservations(for request: VNRequest, error: Error?) {
        // First, let the parent class process the basic detections
        super.processObservations(for: request, error: error)
        
        // Then, extract the bounding boxes from the most recent result
        if let onResultsListener = self.currentOnResultsListener {
            // Get the detection boxes from the most recent result
            if let results = request.results as? [VNRecognizedObjectObservation] {
                var boxes: [CGRect] = []
                var scores: [Float] = []
                var classIds: [Int] = []
                
                for i in 0..<min(results.count, 100) {
                    let prediction = results[i]
                    let invertedBox = CGRect(
                        x: prediction.boundingBox.minX, 
                        y: 1 - prediction.boundingBox.maxY,
                        width: prediction.boundingBox.width, 
                        height: prediction.boundingBox.height)
                    let imageRect = VNImageRectForNormalizedRect(
                        invertedBox, Int(inputSize.width), Int(inputSize.height))
                    
                    boxes.append(imageRect)
                    scores.append(prediction.labels[0].confidence)
                    
                    let label = prediction.labels[0].identifier
                    let index = self.labels.firstIndex(of: label) ?? 0
                    classIds.append(index)
                }
                
                // Update tracks with ByteTracker
                if !boxes.isEmpty {
                    self.tracks = tracker.update(detections: boxes, scores: scores, classIds: classIds)
                    print("DEBUG: TrackingDetector has \(tracks.count) active tracks")
                    
                    // Process tracks for fish counting
                    processTracksForCounting()
                }
            }
        }
    }
    
    /// Process tracks for fish counting using the new top-down approach from counting_demo2.py
    private func processTracksForCounting() {
        // Skip if no thresholds defined
        if thresholds.isEmpty {
            return
        }
        
        // For each active track, check if it crossed a threshold
        for track in tracks {
            let trackId = track.trackId
            let trackRect = track.bbox
            
            // Calculate weighted center point (80% toward bottom as in Python implementation)
            // This matches the Python code: center_x, center_y = (x_min + x_max) / 2, (y_min + y_max*4) / 5
            let centerX = trackRect.midX
            let weightedCenterY = (trackRect.minY + trackRect.maxY * 4) / 5 // Weight toward bottom/tail
            
            // Normalize to 0-1 range
            let normalizedCenterX = centerX / inputSize.width
            let normalizedCenterY = weightedCenterY / inputSize.height
            
            // Check if this is a new track
            if fishPassingInfo[trackId] == nil {
                fishPassingInfo[trackId] = (crossed: false, direction: nil)
                continue // Skip first frame for this track since we need previous position
            }
            
            // Get track state and previous position
            let trackState = fishPassingInfo[trackId]!
            let lastY = trackRect.midY // Use previous position from state
            
            // Process forward crossing (increment count) on all thresholds
            var hasChanged = false
            for threshold in thresholds {
                let pixelThreshold = threshold * inputSize.height
                
                // Check if track crossed threshold from top to bottom (forward)
                if !trackState.crossed && lastY < pixelThreshold && weightedCenterY >= pixelThreshold {
                    fishPassingInfo[trackId] = (crossed: true, direction: "down")
                    fishCount += 1
                    hasChanged = true
                    print("DEBUG: Fish #\(trackId) crossed threshold \(threshold), count = \(fishCount)")
                }
            }
            
            // Process reverse crossing (decrement count) only on first threshold
            // This matches Python: check_reverse_threshold_crossing(thresholds[:1])
            if let firstThreshold = thresholds.first {
                let pixelThreshold = firstThreshold * inputSize.height
                
                // Check if track crossed back over first threshold
                if trackState.crossed && lastY > pixelThreshold && weightedCenterY <= pixelThreshold {
                    fishPassingInfo[trackId] = (crossed: false, direction: nil)
                    fishCount = max(0, fishCount - 1) // Prevent negative count
                    hasChanged = true
                    print("DEBUG: Fish #\(trackId) crossed back over threshold \(firstThreshold), count = \(fishCount)")
                }
            }
            
            // Notify count change if needed
            if hasChanged {
                notifyCountChanged()
            }
        }
        
        // Clean up old tracks
        cleanupOldTracks()
    }
    
    /// Clean up tracks that are no longer active
    private func cleanupOldTracks() {
        let activeTrackIds = Set(tracks.map { $0.trackId })
        let inactiveTrackIds = Set(fishPassingInfo.keys).subtracting(activeTrackIds)
        
        for trackId in inactiveTrackIds {
            fishPassingInfo.removeValue(forKey: trackId)
        }
    }
    
    /// Reset the fish count
    func resetCount() {
        fishCount = 0
        fishPassingInfo.removeAll()
        notifyCountChanged()
    }
    
    /// Get the current fish count
    var currentCount: Int {
        return fishCount
    }
    
    /// Update thresholds for fish counting
    func updateThresholds(thresholds: [CGFloat]) {
        self.thresholds = thresholds
        print("DEBUG: Updated thresholds to \(thresholds)")
    }
    
    /// Notify count changed (thread-safe)
    private func notifyCountChanged() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.onCountChanged?(self.fishCount)
        }
    }
    
    /// Configure for camera angle (commented out - keeping for reference)
    /*
    func configureForCameraAngle(_ angle: String) {
        // Reset active thresholds
        activeThresholds.removeAll()
        
        // Set active thresholds based on camera angle
        switch angle {
        case "Front":
            activeThresholds.insert("upper")
            activeThresholds.insert("bottom")
            activeThresholds.insert("left")
            activeThresholds.insert("right")
            
            // Set threshold values to match counting_demo.py
            upperThreshold = 0.2
            bottomThreshold = 0.8
            leftThreshold = 0.2
            rightThreshold = 0.8
            
        case "Top", "Bottom":
            // For Top/Bottom views, only show bottom threshold (matching counting_demo.py)
            activeThresholds.insert("bottom")
            
            // Set threshold values to match counting_demo.py
            bottomThreshold = 0.8
            
        case "Left":
            // For Left view, only show right threshold (matching counting_demo.py)
            activeThresholds.insert("right")
            
            // Set threshold values to match counting_demo.py
            rightThreshold = 0.7
            
        case "Right":
            // For Right view, only show left threshold (matching counting_demo.py)
            activeThresholds.insert("left")
            
            // Set threshold values to match counting_demo.py
            leftThreshold = 0.5
            
        default:
            // Default to all thresholds
            activeThresholds.insert("upper")
            activeThresholds.insert("bottom")
            activeThresholds.insert("left")
            activeThresholds.insert("right")
        }
    }
    */
} 