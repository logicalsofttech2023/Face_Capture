import React, { useEffect, useRef, useState } from 'react';
import vision from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3';

const App = () => {
  const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

  // Refs
  const webcamRef = useRef(null);
  const outputCanvasRef = useRef(null);

  // State
  const [faceLandmarker, setFaceLandmarker] = useState(null);
  const [webcamRunning, setWebcamRunning] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [error, setError] = useState(null);
  const [webcamError, setWebcamError] = useState(null);
  const [fps, setFps] = useState(0);
  const [measurements, setMeasurements] = useState(null);

  // Performance optimization
  const lastFrameTimeRef = useRef(0);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(0);
  const minFrameInterval = 100;

  // Initialize face landmarker
  useEffect(() => {
    const createFaceLandmarker = async () => {
      setIsModelLoading(true);
      setError(null);
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        const landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
          },
          outputFaceBlendshapes: false,
          runningMode: "VIDEO",
          numFaces: 1
        });
        setFaceLandmarker(landmarker);
        setIsModelLoading(false);
      } catch (error) {
        console.error("Error creating face landmarker:", error);
        setError(`Failed to load face detection model: ${error.message}`);
        setIsModelLoading(false);
      }
    };

    createFaceLandmarker();
  }, []);

  // Calculate measurements from landmarks
  const calculateMeasurements = (landmarks, canvas) => {
    if (!landmarks || landmarks.length === 0) return null;
    
    const landmark = landmarks[0];
    
    // Convert normalized coordinates to pixel coordinates
    const toPixels = (point) => {
      return {
        x: point.x * canvas.width,
        y: point.y * canvas.height
      };
    };
    
    // Calculate distance between two points in mm
    // We need a reference measurement to convert pixels to mm
    // Using inter-pupillary distance as reference (approx 63mm for adults)
    const leftPupil = toPixels(landmark[468]); // Left eye center
    const rightPupil = toPixels(landmark[473]); // Right eye center
    
    const pupilDistancePx = Math.sqrt(
      Math.pow(rightPupil.x - leftPupil.x, 2) + 
      Math.pow(rightPupil.y - leftPupil.y, 2)
    );
    
    // Assume average PD of 63mm for conversion
    const pxToMm = 63 / pupilDistancePx;
    
    // Calculate PD (Pupillary Distance)
    const pd = pupilDistancePx * pxToMm;
    
    // Calculate NPD (Naso-Pupillary Distance)
    const noseTip = toPixels(landmark[4]); // Nose tip
    const leftNpd = Math.sqrt(
      Math.pow(leftPupil.x - noseTip.x, 2) + 
      Math.pow(leftPupil.y - noseTip.y, 2)
    ) * pxToMm;
    
    const rightNpd = Math.sqrt(
      Math.pow(rightPupil.x - noseTip.x, 2) + 
      Math.pow(rightPupil.y - noseTip.y, 2)
    ) * pxToMm;
    
    // Calculate eye opening height
    const leftEyeTop = toPixels(landmark[159]); // Left eye top
    const leftEyeBottom = toPixels(landmark[145]); // Left eye bottom
    const leftEyeHeight = Math.abs(leftEyeTop.y - leftEyeBottom.y) * pxToMm;
    
    const rightEyeTop = toPixels(landmark[386]); // Right eye top
    const rightEyeBottom = toPixels(landmark[374]); // Right eye bottom
    const rightEyeHeight = Math.abs(rightEyeTop.y - rightEyeBottom.y) * pxToMm;
    
    // Calculate pupil height (relative to eye corners)
    const leftEyeInner = toPixels(landmark[133]); // Left eye inner corner
    const leftEyeOuter = toPixels(landmark[33]); // Left eye outer corner
    const leftPupilHeight = Math.abs(leftPupil.y - (leftEyeInner.y + leftEyeOuter.y) / 2) * pxToMm;
    
    const rightEyeInner = toPixels(landmark[362]); // Right eye inner corner
    const rightEyeOuter = toPixels(landmark[263]); // Right eye outer corner
    const rightPupilHeight = Math.abs(rightPupil.y - (rightEyeInner.y + rightEyeOuter.y) / 2) * pxToMm;
    
    // Determine face shape (simplified)
    const jawWidth = Math.sqrt(
      Math.pow(toPixels(landmark[234]).x - toPixels(landmark[454]).x, 2)
    ) * pxToMm;
    
    const faceHeight = Math.sqrt(
      Math.pow(toPixels(landmark[10]).y - toPixels(landmark[152]).y, 2)
    ) * pxToMm;
    
    const faceRatio = jawWidth / faceHeight;
    let faceShape = "Oval";
    
    if (faceRatio > 0.85) faceShape = "Round";
    if (faceRatio < 0.75) faceShape = "Long";
    
    return {
      pd: pd.toFixed(1),
      npd: {
        left: leftNpd.toFixed(1),
        right: rightNpd.toFixed(1)
      },
      eyeHeight: {
        left: leftEyeHeight.toFixed(1),
        right: rightEyeHeight.toFixed(1)
      },
      pupilHeight: {
        left: leftPupilHeight.toFixed(1),
        right: rightPupilHeight.toFixed(1),
        combined: ((leftPupilHeight + rightPupilHeight) / 2).toFixed(1)
      },
      faceShape
    };
  };

  // Toggle webcam
  const toggleWebcam = async () => {
    if (!faceLandmarker) {
      setError("Face measurement model not loaded yet.");
      return;
    }

    if (webcamRunning) {
      // Stop webcam
      setWebcamRunning(false);
      setWebcamError(null);
      setMeasurements(null);
      if (webcamRef.current && webcamRef.current.srcObject) {
        const tracks = webcamRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        webcamRef.current.srcObject = null;
        webcamRef.current.pause();
      }
    } else {
      // Start webcam
      setWebcamRunning(true);
      setWebcamError(null);

      const constraints = {
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 }
        }
      };
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (webcamRef.current) {
          if (webcamRef.current.srcObject) {
            webcamRef.current.srcObject.getTracks().forEach(track => track.stop());
          }
          webcamRef.current.srcObject = stream;

          await new Promise((resolve, reject) => {
            webcamRef.current.onloadedmetadata = () => resolve();
            webcamRef.current.onerror = () => reject(new Error("Failed to load webcam metadata."));
          });

          await webcamRef.current.play();
        } else {
          throw new Error("Webcam reference not available.");
        }
      } catch (error) {
        console.error("Error accessing webcam:", error);
        setWebcamError(`Failed to access webcam: ${error.message}`);
        setWebcamRunning(false);
      }
    }
  };

  // Webcam prediction with throttling
  const predictWebcam = async () => {
    if (!webcamRunning || !webcamRef.current || !outputCanvasRef.current || !faceLandmarker) {
      setWebcamError("Webcam, canvas, or face measurement model not ready.");
      return;
    }

    const currentTime = performance.now();
    if (currentTime - lastFrameTimeRef.current < minFrameInterval) {
      requestAnimationFrame(predictWebcam);
      return;
    }
    lastFrameTimeRef.current = currentTime;

    // Update FPS counter
    frameCountRef.current += 1;
    if (currentTime - lastFpsUpdateRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastFpsUpdateRef.current = currentTime;
    }

    const video = webcamRef.current;
    const canvas = outputCanvasRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0 || video.paused || video.ended) {
      setWebcamError("Video stream not ready or paused. Waiting for webcam to load...");
      requestAnimationFrame(predictWebcam);
      return;
    }

    try {
      // Set video and canvas dimensions
      const radio = video.videoHeight / video.videoWidth;
      const videoWidth = 640;
      video.style.width = `${videoWidth}px`;
      video.style.height = `${videoWidth * radio}px`;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.style.width = `${videoWidth}px`;
      canvas.style.height = `${videoWidth * radio}px`;

      // Detect face landmarks
      const startTimeMs = performance.now();
      const results = await faceLandmarker.detectForVideo(video, startTimeMs);
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error("Failed to get canvas 2D context.");
      }

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw landmarks if face detected
      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const drawingUtils = new DrawingUtils(ctx);
        
        // Calculate and update measurements
        const newMeasurements = calculateMeasurements(results.faceLandmarks, canvas);
        if (newMeasurements) {
          setMeasurements(newMeasurements);
        }

        // Draw face landmarks with minimal styling for measurement purposes
        for (const landmarks of results.faceLandmarks) {
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_TESSELATION,
            { color: "#C0C0C030", lineWidth: 1 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
            { color: "#4285f4", lineWidth: 2 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
            { color: "#4285f4", lineWidth: 2 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
            { color: "#34a853", lineWidth: 2 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
            { color: "#34a853", lineWidth: 2 }
          );
          
          // Draw measurement points
          drawingUtils.drawCircle(landmarks[468], { color: "#ea4335", radius: 3 }); // Left pupil
          drawingUtils.drawCircle(landmarks[473], { color: "#ea4335", radius: 3 }); // Right pupil
          drawingUtils.drawCircle(landmarks[4], { color: "#fbbc05", radius: 3 }); // Nose tip
        }
      } else {
        setWebcamError("No face detected. Please position your face in the frame.");
        setMeasurements(null);
      }
    } catch (error) {
      console.error("Error during face measurement:", error);
      setWebcamError(`Measurement error: ${error.message}`);
    }

    if (webcamRunning) {
      requestAnimationFrame(predictWebcam);
    }
  };

  // Start webcam prediction when video is loaded
  useEffect(() => {
    let animationFrameId;
    if (webcamRunning && webcamRef.current) {
      const onLoadedData = () => {
        if (webcamRunning) {
          animationFrameId = requestAnimationFrame(predictWebcam);
        }
      };
      webcamRef.current.addEventListener('loadeddata', onLoadedData);
      return () => {
        if (webcamRef.current) {
          webcamRef.current.removeEventListener('loadeddata', onLoadedData);
        }
        if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);
        }
      };
    }
  }, [webcamRunning]);

  // Check if webcam is supported
  const hasGetUserMedia = () => {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Face Measurement Analysis</h1>
        <p>Get precise facial measurements using AI technology</p>
      </header>

      <main className="app-main">
        {/* Error Display */}
        {error && (
          <div className="error-message">
            <span><strong>Error: </strong>{error}</span>
            <button onClick={() => setError(null)}>×</button>
          </div>
        )}

        {isModelLoading ? (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Loading measurement model...</p>
          </div>
        ) : (
          <section className="measurement-section">
            <div className="webcam-container">
              <div className="webcam-controls">
                <button
                  className={`webcam-toggle ${webcamRunning ? 'active' : ''}`}
                  onClick={toggleWebcam}
                  disabled={!hasGetUserMedia()}
                >
                  {webcamRunning ? (
                    <>
                      <span className="icon">●</span> Stop Measurement
                    </>
                  ) : (
                    <>
                      <span className="icon">▶</span> Start Measurement
                    </>
                  )}
                </button>
                
                {webcamRunning && (
                  <div className="fps-counter">
                    FPS: {fps}
                  </div>
                )}
              </div>

              {webcamError && (
                <div className="error-message">
                  <span><strong>Webcam Error: </strong>{webcamError}</span>
                  <button onClick={() => setWebcamError(null)}>×</button>
                </div>
              )}

              <div className="video-wrapper">
                <video
                  ref={webcamRef}
                  autoPlay
                  playsInline
                  muted
                  className="webcam-feed"
                  style={{ display: webcamRunning ? 'block' : 'none' }}
                ></video>
                <canvas
                  ref={outputCanvasRef}
                  className="measurement-canvas"
                ></canvas>

                {!webcamRunning && (
                  <div className="webcam-placeholder">
                    <div className="placeholder-icon">👤</div>
                    <p>Webcam is disabled</p>
                    <p>Click "Start Measurement" to begin</p>
                  </div>
                )}
              </div>

              {!hasGetUserMedia() && (
                <p className="browser-warning">
                  Your browser does not support webcam access. Please try Chrome, Firefox, or Edge.
                </p>
              )}
            </div>

            {measurements && (
              <div className="measurements-results">
                <h2>Facial Measurements</h2>
                <div className="measurements-grid">
                  <div className="measurement-card">
                    <h3>Pupillary Distance (PD)</h3>
                    <div className="measurement-value">{measurements.pd} mm</div>
                    <p className="measurement-desc">Distance between pupils</p>
                  </div>
                  
                  <div className="measurement-card">
                    <h3>Naso-Pupillary Distance (NPD)</h3>
                    <div className="measurement-subvalues">
                      <div>
                        <span className="label">Left Eye:</span>
                        <span className="value">{measurements.npd.left} mm</span>
                      </div>
                      <div>
                        <span className="label">Right Eye:</span>
                        <span className="value">{measurements.npd.right} mm</span>
                      </div>
                    </div>
                    <p className="measurement-desc">Distance from nose to each pupil</p>
                  </div>
                  
                  <div className="measurement-card">
                    <h3>Eye Opening Height</h3>
                    <div className="measurement-subvalues">
                      <div>
                        <span className="label">Left Eye:</span>
                        <span className="value">{measurements.eyeHeight.left} mm</span>
                      </div>
                      <div>
                        <span className="label">Right Eye:</span>
                        <span className="value">{measurements.eyeHeight.right} mm</span>
                      </div>
                    </div>
                    <p className="measurement-desc">Vertical opening of eyes</p>
                  </div>
                  
                  <div className="measurement-card">
                    <h3>Pupil Height</h3>
                    <div className="measurement-subvalues">
                      <div>
                        <span className="label">Left Eye:</span>
                        <span className="value">{measurements.pupilHeight.left} mm</span>
                      </div>
                      <div>
                        <span className="label">Right Eye:</span>
                        <span className="value">{measurements.pupilHeight.right} mm</span>
                      </div>
                      <div>
                        <span className="label">Combined:</span>
                        <span className="value">{measurements.pupilHeight.combined} mm</span>
                      </div>
                    </div>
                    <p className="measurement-desc">Vertical position of pupils</p>
                  </div>
                  
                  <div className="measurement-card">
                    <h3>Face Shape</h3>
                    <div className="measurement-value shape">{measurements.faceShape}</div>
                    <p className="measurement-desc">Classification based on proportions</p>
                  </div>
                </div>
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>Note: Measurements are approximations. For precise measurements, consult a professional.</p>
      </footer>

      <style jsx>{`
        .app-container {
          min-height: 100vh;
          background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .app-header {
          text-align: center;
          padding: 2rem 1rem;
          background: white;
          box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .app-header h1 {
          margin: 0;
          color: #2c3e50;
          font-size: 2.5rem;
        }
        
        .app-header p {
          margin: 0.5rem 0 0;
          color: #7f8c8d;
          font-size: 1.1rem;
        }
        
        .app-main {
          max-width: 1200px;
          margin: 0 auto;
          padding: 2rem 1rem;
        }
        
        .loading-container {
          text-align: center;
          padding: 3rem;
        }
        
        .loading-spinner {
          width: 50px;
          height: 50px;
          border: 5px solid #e3e3e3;
          border-top: 5px solid #3498db;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 0 auto 1rem;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .webcam-container {
          background: white;
          border-radius: 12px;
          padding: 1.5rem;
          box-shadow: 0 4px 20px rgba(0,0,0,0.1);
          margin-bottom: 2rem;
        }
        
        .webcam-controls {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }
        
        .webcam-toggle {
          background: #4285f4;
          color: white;
          border: none;
          padding: 0.75rem 1.5rem;
          border-radius: 50px;
          font-weight: 600;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          transition: all 0.2s;
        }
        
        .webcam-toggle:hover {
          background: #3367d6;
        }
        
        .webcam-toggle.active {
          background: #ea4335;
        }
        
        .webcam-toggle:disabled {
          background: #ccc;
          cursor: not-allowed;
        }
        
        .fps-counter {
          background: #f1f1f1;
          padding: 0.5rem 1rem;
          border-radius: 20px;
          font-size: 0.9rem;
          color: #666;
        }
        
        .video-wrapper {
          position: relative;
          width: 100%;
          height: 480px;
          background: #000;
          border-radius: 8px;
          overflow: hidden;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .webcam-feed, .measurement-canvas {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          object-fit: cover;
        }
        
        .measurement-canvas {
          z-index: 10;
        }
        
        .webcam-placeholder {
          text-align: center;
          color: #bbb;
          z-index: 1;
        }
        
        .placeholder-icon {
          font-size: 4rem;
          margin-bottom: 1rem;
        }
        
        .browser-warning {
          color: #e74c3c;
          margin-top: 1rem;
          text-align: center;
        }
        
        .error-message {
          background: #ffebee;
          color: #c62828;
          padding: 1rem;
          border-radius: 8px;
          margin-bottom: 1.5rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        .error-message button {
          background: none;
          border: none;
          color: #c62828;
          font-size: 1.5rem;
          cursor: pointer;
          padding: 0;
          width: 30px;
          height: 30px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .measurements-results {
          background: white;
          border-radius: 12px;
          padding: 2rem;
          box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .measurements-results h2 {
          margin-top: 0;
          color: #2c3e50;
          text-align: center;
          margin-bottom: 2rem;
        }
        
        .measurements-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1.5rem;
        }
        
        .measurement-card {
          background: #f8f9fa;
          border-radius: 10px;
          padding: 1.5rem;
          text-align: center;
          border-left: 4px solid #4285f4;
        }
        
        .measurement-card h3 {
          margin-top: 0;
          color: #2c3e50;
          font-size: 1.1rem;
        }
        
        .measurement-value {
          font-size: 2rem;
          font-weight: bold;
          color: #4285f4;
          margin: 1rem 0;
        }
        
        .measurement-value.shape {
          font-size: 1.5rem;
          text-transform: uppercase;
          letter-spacing: 1px;
          color: #34a853;
        }
        
        .measurement-subvalues {
          margin: 1rem 0;
        }
        
        .measurement-subvalues div {
          display: flex;
          justify-content: space-between;
          margin-bottom: 0.5rem;
        }
        
        .measurement-subvalues .label {
          color: #7f8c8d;
        }
        
        .measurement-subvalues .value {
          font-weight: bold;
          color: #4285f4;
        }
        
        .measurement-desc {
          color: #7f8c8d;
          font-size: 0.9rem;
          margin: 0;
        }
        
        .app-footer {
          text-align: center;
          padding: 1.5rem;
          color: #7f8c8d;
          font-size: 0.9rem;
          background: white;
          border-top: 1px solid #eee;
        }
        
        @media (max-width: 768px) {
          .app-header h1 {
            font-size: 2rem;
          }
          
          .video-wrapper {
            height: 360px;
          }
          
          .measurements-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

export default App;