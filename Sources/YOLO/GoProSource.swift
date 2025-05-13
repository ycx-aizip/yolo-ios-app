// from softbank_fishcount_iphone14
//
//  GoProSource.swift
//  YOLO
//
//  Implementation of a frame source that connects to GoPro cameras via WiFi
//  using the OpenGoPro HTTP API and RTSP streaming.

import AVFoundation
import Foundation
// UIKit is not needed for the current implementation

/// GoPro webcam version response structure - simplified version
struct GoProWebcamVersion: Decodable {
    let version: Int
    
    // Optional fields - won't cause decoding to fail if missing
    let max_lens_support: Bool?
    let usb_3_1_compatible: Bool?
}

/// Class for handling GoPro camera as a frame source
class GoProSource: NSObject {
    // Default GoPro IP address when connected via WiFi
    private let goProIP = "10.5.5.9"
    private let goProPort = 8080
    
    // HTTP endpoints
    private let versionEndpoint = "/gopro/webcam/version"
    private let previewEndpoint = "/gopro/webcam/preview"
    private let startEndpoint = "/gopro/webcam/start"
    private let stopEndpoint = "/gopro/webcam/stop"
    private let exitEndpoint = "/gopro/webcam/exit"
    
    // RTSP configuration
    private let rtspPort = 554
    private let rtspPath = "/live"
    
    /// Check if connected to a GoPro camera by requesting webcam version
    /// - Parameter completion: Callback with result (success/failure)
    func checkConnection(completion: @escaping (Result<GoProWebcamVersion, Error>) -> Void) {
        print("GoPro: Starting connection check")
        let urlString = "http://\(goProIP):\(goProPort)\(versionEndpoint)"
        
        guard let url = URL(string: urlString) else {
            print("GoPro: Invalid URL format")
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
        // Create a URL request with a timeout
        var request = URLRequest(url: url)
        request.timeoutInterval = 5.0 // 5 second timeout
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            // Handle network error
            if let error = error {
                print("GoPro: Network error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }
            
            // Check HTTP response
            guard let httpResponse = response as? HTTPURLResponse else {
                print("GoPro: Invalid HTTP response")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response"])))
                }
                return
            }
            
            print("GoPro: HTTP Status Code: \(httpResponse.statusCode)")
            
            // Check status code
            guard httpResponse.statusCode == 200 else {
                print("GoPro: HTTP error status code: \(httpResponse.statusCode)")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP error: \(httpResponse.statusCode)"])))
                }
                return
            }
            
            // Check data
            guard let data = data, !data.isEmpty else {
                print("GoPro: No data received from server")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: 3, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                }
                return
            }
            
            // Try to parse JSON - more lenient approach
            do {
                // First try parsing as a dictionary to check for version field
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    if let versionValue = json["version"] as? Int {
                        print("GoPro: Found version: \(versionValue)")
                        // Create a simplified version object with just what we need
                        let version = GoProWebcamVersion(
                            version: versionValue,
                            max_lens_support: json["max_lens_support"] as? Bool,
                            usb_3_1_compatible: json["usb_3_1_compatible"] as? Bool
                        )
                        print("GoPro: Connection successful")
                        DispatchQueue.main.async {
                            completion(.success(version))
                        }
                        return
                    }
                }
                
                // If that didn't work, try standard decoding
                let decoder = JSONDecoder()
                let version = try decoder.decode(GoProWebcamVersion.self, from: data)
                print("GoPro: Connection successful (standard decoding)")
                DispatchQueue.main.async {
                    completion(.success(version))
                }
            } catch {
                // Log detailed error information
                if let dataString = String(data: data, encoding: .utf8) {
                    // If we got ANY response data, consider it a success
                    if !dataString.isEmpty {
                        // Create a minimal version object with default values
                        let defaultVersion = GoProWebcamVersion(
                            version: 1,  // Default version
                            max_lens_support: nil,
                            usb_3_1_compatible: nil
                        )
                        print("GoPro: Connection successful (with default values)")
                        DispatchQueue.main.async {
                            completion(.success(defaultVersion))
                        }
                        return
                    }
                }
                
                print("GoPro: Connection failed - \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
        
        task.resume()
    }
    
    /// Enter webcam preview mode
    /// - Parameter completion: Callback with result (success/failure)
    func enterWebcamPreview(completion: @escaping (Result<Void, Error>) -> Void) {
        print("GoPro: Entering webcam preview mode")
        let urlString = "http://\(goProIP):\(goProPort)\(previewEndpoint)"
        
        guard let url = URL(string: urlString) else {
            print("GoPro: Invalid preview URL format")
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid preview URL"])))
            return
        }
        
        // Create a URL request with a timeout
        var request = URLRequest(url: url)
        request.timeoutInterval = 5.0 // 5 second timeout
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            // Handle network error
            if let error = error {
                print("GoPro: Preview network error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }
            
            // Check HTTP response
            guard let httpResponse = response as? HTTPURLResponse else {
                print("GoPro: Invalid HTTP response for preview")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response"])))
                }
                return
            }
            
            print("GoPro: Preview HTTP Status Code: \(httpResponse.statusCode)")
            
            // Check status code
            guard httpResponse.statusCode == 200 else {
                print("GoPro: Preview HTTP error status code: \(httpResponse.statusCode)")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP error: \(httpResponse.statusCode)"])))
                }
                return
            }
            
            // Success - empty response is expected
            print("GoPro: Preview mode enabled successfully")
            DispatchQueue.main.async {
                completion(.success(()))
            }
        }
        
        task.resume()
    }
    
    /// Start webcam streaming
    /// - Parameter completion: Callback with result (success/failure)
    func startWebcam(completion: @escaping (Result<Void, Error>) -> Void) {
        print("GoPro: Starting webcam stream")
        // Construct URL with query parameters
        let queryParams = "res=12&fov=0&port=8556&protocol=RTSP"
        let urlString = "http://\(goProIP):\(goProPort)\(startEndpoint)?\(queryParams)"
        
        guard let url = URL(string: urlString) else {
            print("GoPro: Invalid start URL format")
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid start URL"])))
            return
        }
        
        // Create a URL request with a timeout
        var request = URLRequest(url: url)
        request.timeoutInterval = 5.0 // 5 second timeout
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            // Handle network error
            if let error = error {
                print("GoPro: Start webcam network error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }
            
            // Check HTTP response
            guard let httpResponse = response as? HTTPURLResponse else {
                print("GoPro: Invalid HTTP response for start")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response"])))
                }
                return
            }
            
            print("GoPro: Start HTTP Status Code: \(httpResponse.statusCode)")
            
            // Check status code
            guard httpResponse.statusCode == 200 else {
                print("GoPro: Start HTTP error status code: \(httpResponse.statusCode)")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP error: \(httpResponse.statusCode)"])))
                }
                return
            }
            
            // Success - empty response is expected
            print("GoPro: Webcam started successfully")
            DispatchQueue.main.async {
                completion(.success(()))
            }
        }
        
        task.resume()
    }
    
    /// Stop webcam streaming
    /// - Parameter completion: Callback with result (success/failure)
    func stopWebcam(completion: @escaping (Result<Void, Error>) -> Void) {
        print("GoPro: Stopping webcam stream")
        let urlString = "http://\(goProIP):\(goProPort)\(stopEndpoint)"
        
        guard let url = URL(string: urlString) else {
            print("GoPro: Invalid stop URL format")
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid stop URL"])))
            return
        }
        
        // Create a URL request with a timeout
        var request = URLRequest(url: url)
        request.timeoutInterval = 5.0 // 5 second timeout
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            // Handle network error
            if let error = error {
                print("GoPro: Stop webcam network error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }
            
            // Check HTTP response
            guard let httpResponse = response as? HTTPURLResponse else {
                print("GoPro: Invalid HTTP response for stop")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response"])))
                }
                return
            }
            
            print("GoPro: Stop HTTP Status Code: \(httpResponse.statusCode)")
            
            // Check status code
            guard httpResponse.statusCode == 200 else {
                print("GoPro: Stop HTTP error status code: \(httpResponse.statusCode)")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP error: \(httpResponse.statusCode)"])))
                }
                return
            }
            
            // Success - empty response is expected
            print("GoPro: Webcam stopped successfully")
            DispatchQueue.main.async {
                completion(.success(()))
            }
        }
        
        task.resume()
    }
    
    /// Exit webcam mode
    /// - Parameter completion: Callback with result (success/failure)
    func exitWebcam(completion: @escaping (Result<Void, Error>) -> Void) {
        print("GoPro: Exiting webcam mode")
        let urlString = "http://\(goProIP):\(goProPort)\(exitEndpoint)"
        
        guard let url = URL(string: urlString) else {
            print("GoPro: Invalid exit URL format")
            completion(.failure(NSError(domain: "GoProSource", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid exit URL"])))
            return
        }
        
        // Create a URL request with a timeout
        var request = URLRequest(url: url)
        request.timeoutInterval = 5.0 // 5 second timeout
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            // Handle network error
            if let error = error {
                print("GoPro: Exit webcam network error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
                return
            }
            
            // Check HTTP response
            guard let httpResponse = response as? HTTPURLResponse else {
                print("GoPro: Invalid HTTP response for exit")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response"])))
                }
                return
            }
            
            print("GoPro: Exit HTTP Status Code: \(httpResponse.statusCode)")
            
            // Check status code
            guard httpResponse.statusCode == 200 else {
                print("GoPro: Exit HTTP error status code: \(httpResponse.statusCode)")
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "GoProSource", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP error: \(httpResponse.statusCode)"])))
                }
                return
            }
            
            // Success - empty response is expected
            print("GoPro: Webcam exited successfully")
            DispatchQueue.main.async {
                completion(.success(()))
            }
        }
        
        task.resume()
    }
    
    /// Gracefully exit webcam mode with both stop and exit commands
    /// - Parameter completion: Callback with result (success/failure)
    func gracefulWebcamExit(completion: @escaping (Result<Void, Error>) -> Void) {
        print("GoPro: Starting graceful webcam exit")
        
        // First stop the webcam
        stopWebcam { [weak self] stopResult in
            guard let self = self else { return }
            
            switch stopResult {
            case .success:
                // Then exit webcam mode
                self.exitWebcam { exitResult in
                    switch exitResult {
                    case .success:
                        print("GoPro: Graceful exit completed successfully")
                        completion(.success(()))
                    case .failure(let error):
                        print("GoPro: Exit command failed: \(error.localizedDescription)")
                        completion(.failure(error))
                    }
                }
            case .failure(let error):
                print("GoPro: Stop command failed: \(error.localizedDescription)")
                // Try exit anyway as a best effort
                self.exitWebcam { exitResult in
                    switch exitResult {
                    case .success:
                        // If exit succeeds, still report the stop error
                        completion(.failure(error))
                    case .failure(let exitError):
                        // Report both errors
                        let combinedError = NSError(
                            domain: "GoProSource",
                            code: 3,
                            userInfo: [
                                NSLocalizedDescriptionKey: "Multiple errors: Stop - \(error.localizedDescription), Exit - \(exitError.localizedDescription)"
                            ]
                        )
                        completion(.failure(combinedError))
                    }
                }
            }
        }
    }
}
