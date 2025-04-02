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
            // For testing only - convert detected boxes to the format needed by ByteTracker
            print("DEBUG: TrackingDetector processing observations")
            
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
                    print("DEBUG: TrackingDetector updated tracks, now has \(tracks.count) active tracks")
                    
                    // TODO: In the future, we'll add fish counting logic here
                }
            }
        }
    }
} 