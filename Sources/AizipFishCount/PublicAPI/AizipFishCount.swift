//
//  AizipFishCount.swift
//  AizipFishCount
//
//  Main entry point for the AizipFishCount framework
//  This is the only class frontend developers need to import and use
//

import Foundation
import CoreML
import Vision

// MARK: - Main Entry Point

/// AizipFishCount - Main entry point for fish counting functionality
///
/// Usage:
/// ```swift
/// import AizipFishCount
///
/// class MyViewController: UIViewController, FishCountSessionDelegate {
///     private var session: FishCountSession?
///
///     func setup() {
///         do {
///             // Create session
///             session = try AizipFishCount.createSession(
///                 modelName: "FishCount_v12",
///                 delegate: self
///             )
///
///             // Configure counting
///             session?.setThresholds([0.3, 0.7])
///             session?.setCountingDirection(.bottomToTop)
///
///         } catch {
///             print("Failed to create session: \(error)")
///         }
///     }
///
///     func processCameraFrame(_ pixelBuffer: CVPixelBuffer) {
///         session?.processFrame(pixelBuffer)
///     }
///
///     func session(_ session: FishCountSession, didDetect result: DetectionResult) {
///         // Handle detection results
///     }
/// }
/// ```
@MainActor
public class AizipFishCount {

    // MARK: - Session Creation

    /// Create a new fish counting session
    ///
    /// - Parameters:
    ///   - modelName: Name of the CoreML model file (without .mlmodelc extension)
    ///                The model should be located in your app's FishCountModels folder
    ///   - delegate: Optional delegate to receive session callbacks
    ///   - configuration: Optional detection configuration (uses defaults if nil)
    ///
    /// - Returns: A new FishCountSession instance
    ///
    /// - Throws: FishCountError if model cannot be loaded
    ///
    /// - Note: Ensure your CoreML model is added to the app bundle in a folder named "FishCountModels"
    public static func createSession(
        modelName: String,
        delegate: FishCountSessionDelegate? = nil,
        configuration: DetectionConfig? = nil
    ) throws -> FishCountSession {

        // Find the model in the bundle
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") ??
                             Bundle.main.url(forResource: modelName, withExtension: "mlmodel") else {
            throw FishCountError.modelNotFound(modelName)
        }

        // Create the tracking detector synchronously for simplicity
        // We use a semaphore to wait for async model loading
        var loadedDetector: TrackingDetector?
        var loadError: Error?
        let semaphore = DispatchSemaphore(value: 0)

        TrackingDetector.create(unwrappedModelURL: modelURL, isRealTime: true) { result in
            switch result {
            case .success(let detector):
                loadedDetector = detector as? TrackingDetector
            case .failure(let error):
                loadError = error
            }
            semaphore.signal()
        }

        // Wait for model loading to complete
        semaphore.wait()

        // Check if loading succeeded
        guard let detector = loadedDetector else {
            if let error = loadError {
                throw FishCountError.modelNotFound("\(modelName): \(error.localizedDescription)")
            } else {
                throw FishCountError.modelNotFound(modelName)
            }
        }

        // Apply configuration if provided
        if let config = configuration {
            detector.confidenceThreshold = Double(config.confidenceThreshold)
            detector.iouThreshold = Double(config.iouThreshold)
            detector.numItemsThreshold = config.maxDetections
        }

        // Create and return the session adapter
        let session = SessionAdapter(
            modelName: modelName,
            detector: detector,
            delegate: delegate
        )

        return session
    }

    // MARK: - Global Configuration

    /// Update global default configuration for all new sessions
    ///
    /// This affects the initial values for any sessions created after this call
    ///
    /// - Parameters:
    ///   - detectionConfig: Default detection parameters
    ///   - thresholds: Default counting thresholds (0.0-1.0)
    ///   - direction: Default counting direction
    ///
    /// - Note: These defaults are stored in TrackingDetectorConfig and ThresholdCounter
    public static func setGlobalDefaults(
        detectionConfig: DetectionConfig? = nil,
        thresholds: [CGFloat]? = nil,
        direction: CountingDirection? = nil
    ) {
        // Update detection defaults
        if let config = detectionConfig {
            TrackingDetectorConfig.shared.updateDefaults(
                confidenceThreshold: config.confidenceThreshold,
                iouThreshold: config.iouThreshold,
                numItemsThreshold: config.maxDetections
            )
        }

        // Update counting defaults
        if let thresholds = thresholds {
            ThresholdCounter.defaultThresholds = thresholds
        }

        if let direction = direction {
            ThresholdCounter.defaultCountingDirection = direction
        }
    }

    /// Get current global default configuration
    ///
    /// - Returns: Tuple containing current defaults
    public static func getGlobalDefaults() -> (
        detectionConfig: DetectionConfig,
        thresholds: [CGFloat],
        direction: CountingDirection
    ) {
        let detectionConfig = DetectionConfig(
            confidenceThreshold: TrackingDetectorConfig.shared.defaultConfidenceThreshold,
            iouThreshold: TrackingDetectorConfig.shared.defaultIoUThreshold,
            maxDetections: TrackingDetectorConfig.shared.defaultNumItemsThreshold
        )

        return (
            detectionConfig: detectionConfig,
            thresholds: ThresholdCounter.defaultThresholds,
            direction: ThresholdCounter.defaultCountingDirection
        )
    }

    // MARK: - Utility Methods

    /// Check if a CoreML model exists in the app bundle
    ///
    /// - Parameter modelName: Name of the model to check
    /// - Returns: True if the model exists
    public static func isModelAvailable(_ modelName: String) -> Bool {
        // Try to find the compiled model
        if let _ = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") {
            return true
        }

        // Try to find the uncompiled model
        if let _ = Bundle.main.url(forResource: modelName, withExtension: "mlmodel") {
            return true
        }

        return false
    }

    /// List all available CoreML models in the FishCountModels folder
    ///
    /// - Returns: Array of model names (without extensions)
    public static func listAvailableModels() -> [String] {
        var models: [String] = []

        // Get the main bundle
        guard let bundlePath = Bundle.main.resourcePath else {
            return models
        }

        // Look for FishCountModels folder
        let fishCountModelsPath = (bundlePath as NSString).appendingPathComponent("FishCountModels")

        // Check if directory exists
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: fishCountModelsPath, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return models
        }

        // List all .mlmodelc directories
        do {
            let contents = try FileManager.default.contentsOfDirectory(atPath: fishCountModelsPath)
            for item in contents {
                if item.hasSuffix(".mlmodelc") {
                    let modelName = item.replacingOccurrences(of: ".mlmodelc", with: "")
                    models.append(modelName)
                }
            }
        } catch {
            print("AizipFishCount: Error listing models - \(error)")
        }

        return models.sorted()
    }

    /// Get version information for the AizipFishCount framework
    ///
    /// - Returns: Version string
    public static func getVersion() -> String {
        return "1.3.0"
    }
}
