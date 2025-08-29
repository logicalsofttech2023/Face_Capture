import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const App = () => {
  const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

  // Refs
  const webcamRef = useRef(null);
  const outputCanvasRef = useRef(null);
  const containerRef = useRef(null);

  // State
  const [faceLandmarker, setFaceLandmarker] = useState(null);
  const [webcamRunning, setWebcamRunning] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [error, setError] = useState(null);
  const [webcamError, setWebcamError] = useState(null);
  const [fps, setFps] = useState(0);
  const [measurements, setMeasurements] = useState(null);
  const [finalMeasurements, setFinalMeasurements] = useState(null);
  const [isCaptured, setIsCaptured] = useState(false);
  const [appState, setAppState] = useState("instructions"); // instructions, measuring, results
  const [distanceStatus, setDistanceStatus] = useState("checking"); // checking, tooClose, tooFar, optimal

  // Performance optimization
  const lastFrameTimeRef = useRef(0);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(0);
  const minFrameInterval = 100;
  const optimalFaceHeightPx = useRef(0);

  // Initialize face landmarker
  useEffect(() => {
    const createFaceLandmarker = async () => {
      setIsModelLoading(true);
      setError(null);
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        const landmarker = await FaceLandmarker.createFromOptions(
          filesetResolver,
          {
            baseOptions: {
              modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
              delegate: "GPU",
            },
            outputFaceBlendshapes: false,
            runningMode: "VIDEO",
            numFaces: 1,
          }
        );
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
        y: point.y * canvas.height,
      };
    };

    // Get more accurate eye center points
    const leftEyeCenter = {
      x:
        (landmark[33].x +
          landmark[133].x +
          landmark[157].x +
          landmark[158].x +
          landmark[159].x +
          landmark[160].x +
          landmark[161].x +
          landmark[246].x) /
        8,
      y:
        (landmark[33].y +
          landmark[133].y +
          landmark[157].y +
          landmark[158].y +
          landmark[159].y +
          landmark[160].y +
          landmark[161].y +
          landmark[246].y) /
        8,
    };

    const rightEyeCenter = {
      x:
        (landmark[263].x +
          landmark[362].x +
          landmark[373].x +
          landmark[374].x +
          landmark[380].x +
          landmark[381].x +
          landmark[382].x +
          landmark[466].x) /
        8,
      y:
        (landmark[263].y +
          landmark[362].y +
          landmark[373].y +
          landmark[374].y +
          landmark[380].y +
          landmark[381].y +
          landmark[382].y +
          landmark[466].y) /
        8,
    };

    // Convert to pixels
    const leftPupil = toPixels(leftEyeCenter);
    const rightPupil = toPixels(rightEyeCenter);

    // Calculate face height using more stable reference points
    const foreheadTop = toPixels(landmark[10]); // Forehead top
    const chinBottom = toPixels(landmark[152]); // Chin bottom

    // Calculate face height in pixels using more accurate vertical measurement
    const faceHeightPx = Math.abs(chinBottom.y - foreheadTop.y);

    // Use interpupillary distance as reference (more stable than face height)
    // Average PD is ~62mm for adults, use this for calibration
    const averagePDMm = 62;
    const pupilDistancePx = Math.sqrt(
      Math.pow(rightPupil.x - leftPupil.x, 2) +
        Math.pow(rightPupil.y - leftPupil.y, 2)
    );

    // Calculate pxToMm ratio using PD as reference (more accurate for PD measurement)
    let pxToMm = averagePDMm / pupilDistancePx;

    // Recalculate PD with calibrated ratio
    const pd = pupilDistancePx * pxToMm;

    // Calculate face width for additional validation
    const faceLeft = toPixels(landmark[234]); // Left face edge
    const faceRight = toPixels(landmark[454]); // Right face edge
    const faceWidthPx = Math.abs(faceRight.x - faceLeft.x);

    // Validate measurements - face width should be reasonable (typically 1.3-1.5x face height)
    const faceWidthMm = faceWidthPx * pxToMm;
    const faceHeightMm = faceHeightPx * pxToMm;

    // If measurements seem unreasonable, fall back to face height reference
    if (
      faceWidthMm < 100 ||
      faceWidthMm > 200 ||
      faceHeightMm < 150 ||
      faceHeightMm > 250
    ) {
      // Use face height as reference instead
      const averageFaceHeightMm = 190;
      pxToMm = averageFaceHeightMm / faceHeightPx;
    }

    // Calculate NPD (Naso-Pupillary Distance)
    const noseTip = toPixels(landmark[4]); // Nose tip
    const leftNpd =
      Math.sqrt(
        Math.pow(leftPupil.x - noseTip.x, 2) +
          Math.pow(leftPupil.y - noseTip.y, 2)
      ) * pxToMm;

    const rightNpd =
      Math.sqrt(
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
    const leftPupilHeight =
      Math.abs(leftPupil.y - (leftEyeInner.y + leftEyeOuter.y) / 2) * pxToMm;

    const rightEyeInner = toPixels(landmark[362]); // Right eye inner corner
    const rightEyeOuter = toPixels(landmark[263]); // Right eye outer corner
    const rightPupilHeight =
      Math.abs(rightPupil.y - (rightEyeInner.y + rightEyeOuter.y) / 2) * pxToMm;

    // Enhanced face shape detection
    const jawLeft = toPixels(landmark[234]);
    const jawRight = toPixels(landmark[454]);
    const jawlinePoints = Array.from({ length: 17 }, (_, i) =>
      toPixels(landmark[234 + i])
    );

    const faceWidth = Math.sqrt(Math.pow(jawRight.x - jawLeft.x, 2)) * pxToMm;
    const faceHeight =
      Math.sqrt(Math.pow(chinBottom.y - foreheadTop.y, 2)) * pxToMm;

    // Get cheekbone width
    const leftCheek = toPixels(landmark[123]);
    const rightCheek = toPixels(landmark[352]);
    const cheekboneWidth =
      Math.sqrt(Math.pow(rightCheek.x - leftCheek.x, 2)) * pxToMm;

    // Get forehead width
    const leftTemple = toPixels(landmark[21]);
    const rightTemple = toPixels(landmark[251]);
    const foreheadWidth =
      Math.sqrt(Math.pow(rightTemple.x - leftTemple.x, 2)) * pxToMm;

    // Get jawline angles for better shape detection
    const jawAngleLeft =
      (Math.atan2(
        jawlinePoints[0].y - jawlinePoints[4].y,
        jawlinePoints[0].x - jawlinePoints[4].x
      ) *
        180) /
      Math.PI;

    const jawAngleRight =
      (Math.atan2(
        jawlinePoints[16].y - jawlinePoints[12].y,
        jawlinePoints[16].x - jawlinePoints[12].x
      ) *
        180) /
      Math.PI;

    // Calculate ratios for face shape detection
    const faceRatio = faceWidth / faceHeight;
    const cheekboneJawRatio = cheekboneWidth / faceWidth;
    const foreheadJawRatio = foreheadWidth / faceWidth;

    // Enhanced face shape classification with more accurate criteria
    let faceShape = "Oval";

    if (
      faceRatio > 0.85 &&
      Math.abs(jawAngleLeft) < 120 &&
      Math.abs(jawAngleRight) < 120
    ) {
      faceShape = "Round";
    } else if (faceRatio < 0.75 && cheekboneJawRatio > 0.95) {
      faceShape = "Long";
    } else if (
      Math.abs(faceWidth - cheekboneWidth) < 5 &&
      Math.abs(faceWidth - foreheadWidth) < 5
    ) {
      faceShape = "Square";
    } else if (
      foreheadJawRatio > 1.1 &&
      cheekboneJawRatio > 1.05 &&
      faceRatio > 0.8
    ) {
      faceShape = "Heart";
    } else if (
      cheekboneJawRatio > 1.05 &&
      foreheadJawRatio < 0.95 &&
      faceRatio < 0.85
    ) {
      faceShape = "Diamond";
    } else if (Math.abs(jawAngleLeft) > 130 || Math.abs(jawAngleRight) > 130) {
      faceShape = "Triangle";
    }

    // Check distance based on face height in pixels
    let distanceStatus = "checking";
    if (faceHeightPx > 0) {
      if (!optimalFaceHeightPx.current) {
        // Set optimal face height based on canvas height (around 60% of canvas height)
        optimalFaceHeightPx.current = canvas.height * 0.6;
      }

      if (faceHeightPx > optimalFaceHeightPx.current * 1.2) {
        distanceStatus = "tooClose";
      } else if (faceHeightPx < optimalFaceHeightPx.current * 0.8) {
        distanceStatus = "tooFar";
      } else {
        distanceStatus = "optimal";
      }
    }

    // Validate measurements
    const validateMeasurement = (value, min, max) => {
      return value >= min && value <= max;
    };

    const isValid =
      validateMeasurement(pd, 50, 80) &&
      validateMeasurement(leftNpd, 20, 40) &&
      validateMeasurement(rightNpd, 20, 40);
    distanceStatus === "optimal";

    if (!isValid) {
      console.warn("Measurements outside expected ranges:", {
        pd,
        leftNpd,
        rightNpd,
      });
    }

    return {
      pd: pd.toFixed(1),
      npd: {
        left: leftNpd.toFixed(1),
        right: rightNpd.toFixed(1),
      },
      eyeHeight: {
        left: leftEyeHeight.toFixed(1),
        right: rightEyeHeight.toFixed(1),
      },
      pupilHeight: {
        left: leftPupilHeight.toFixed(1),
        right: rightPupilHeight.toFixed(1),
        combined: ((leftPupilHeight + rightPupilHeight) / 2).toFixed(1),
      },
      faceShape,
      faceWidth: faceWidth.toFixed(1),
      faceLength: faceHeight.toFixed(1),
      faceHeightPx,
      distanceStatus,
      isValid,
    };
  };

  // Capture final measurements
  const captureMeasurements = () => {
    if (measurements && distanceStatus === "optimal") {
      setFinalMeasurements(measurements);
      setIsCaptured(true);
      setAppState("results");
    } else {
      setWebcamError(
        "Cannot capture measurements. Please ensure your face is properly detected and positioned at the optimal distance."
      );
    }
  };

  // Update the resetCapture function
  const resetCapture = () => {
    // Stop webcam if it's running
    if (webcamRunning) {
      setWebcamRunning(false);
      if (webcamRef.current && webcamRef.current.srcObject) {
        const tracks = webcamRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
        webcamRef.current.srcObject = null;
      }
    }

    setFinalMeasurements(null);
    setIsCaptured(false);
    setAppState("instructions");
    setDistanceStatus("checking");
    setMeasurements(null);
    setWebcamError(null);
  };

  // Update the startMeasurement function
  const startMeasurement = async () => {
    setAppState("measuring");
    // Small delay to ensure state updates before starting webcam
    await new Promise((resolve) => setTimeout(resolve, 100));
    toggleWebcam();
  };

  // Update the toggleWebcam function to handle state transitions better
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
        tracks.forEach((track) => track.stop());
        webcamRef.current.srcObject = null;
      }
    } else {
      // Start webcam
      setWebcamRunning(true);
      setWebcamError(null);

      const constraints = {
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 },
        },
      };

      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (webcamRef.current) {
          if (webcamRef.current.srcObject) {
            webcamRef.current.srcObject
              .getTracks()
              .forEach((track) => track.stop());
          }
          webcamRef.current.srcObject = stream;

          await new Promise((resolve, reject) => {
            webcamRef.current.onloadedmetadata = () => resolve();
            webcamRef.current.onerror = () =>
              reject(new Error("Failed to load webcam metadata."));
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

  // Update the distance status
  useEffect(() => {
    if (measurements && measurements.distanceStatus) {
      setDistanceStatus(measurements.distanceStatus);
    }
  }, [measurements]);

  // Webcam prediction with throttling
  const predictWebcam = async () => {
    if (
      !webcamRunning ||
      !webcamRef.current ||
      !outputCanvasRef.current ||
      !faceLandmarker
    ) {
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

    if (
      video.videoWidth === 0 ||
      video.videoHeight === 0 ||
      video.paused ||
      video.ended
    ) {
      setWebcamError(
        "Video stream not ready or paused. Waiting for webcam to load..."
      );
      requestAnimationFrame(predictWebcam);
      return;
    }

    try {
      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Detect face landmarks
      const startTimeMs = performance.now();
      const results = await faceLandmarker.detectForVideo(video, startTimeMs);
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        throw new Error("Failed to get canvas 2D context.");
      }

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw landmarks if face detected
      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const drawingUtils = new DrawingUtils(ctx);

        // Clear previous error if face is detected
        setWebcamError(null);

        // Calculate and update measurements
        const newMeasurements = calculateMeasurements(
          results.faceLandmarks,
          canvas
        );
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

          // Draw measurement points using canvas directly instead of DrawingUtils
          // to avoid the "drawCircle is not a function" error
          const drawPoint = (point, color, size) => {
            const x = point.x * canvas.width;
            const y = point.y * canvas.height;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI);
            ctx.fill();
          };

          // Get more accurate eye center points for drawing
          const leftEyeCenter = {
            x:
              (landmarks[33].x +
                landmarks[133].x +
                landmarks[157].x +
                landmarks[158].x +
                landmarks[159].x +
                landmarks[160].x +
                landmarks[161].x +
                landmarks[246].x) /
              8,
            y:
              (landmarks[33].y +
                landmarks[133].y +
                landmarks[157].y +
                landmarks[158].y +
                landmarks[159].y +
                landmarks[160].y +
                landmarks[161].y +
                landmarks[246].y) /
              8,
          };

          const rightEyeCenter = {
            x:
              (landmarks[263].x +
                landmarks[362].x +
                landmarks[373].x +
                landmarks[374].x +
                landmarks[380].x +
                landmarks[381].x +
                landmarks[382].x +
                landmarks[466].x) /
              8,
            y:
              (landmarks[263].y +
                landmarks[362].y +
                landmarks[373].y +
                landmarks[374].y +
                landmarks[380].y +
                landmarks[381].y +
                landmarks[382].y +
                landmarks[466].y) /
              8,
          };

          drawPoint(leftEyeCenter, "#ea4335", 3); // Left pupil
          drawPoint(rightEyeCenter, "#ea4335", 3); // Right pupil
          drawPoint(landmarks[4], "#fbbc05", 3); // Nose tip
        }
      } else {
        setWebcamError(
          "No face detected. Please position your face in the frame and ensure good lighting."
        );
        setMeasurements(null);
        setDistanceStatus("checking");
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
      webcamRef.current.addEventListener("loadeddata", onLoadedData);
      return () => {
        if (webcamRef.current) {
          webcamRef.current.removeEventListener("loadeddata", onLoadedData);
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

  // Render different screens based on app state
  const renderInstructionsScreen = () => (
    <div className="screen instructions-screen">
      <div className="screen-content">
        <h2>Getting Accurate Measurements</h2>
        <div className="instruction-steps">
          <div className="instruction-step">
            <div className="step-icon">💡</div>
            <div className="step-content">
              <h3>Good Lighting</h3>
              <p>Make sure your face is well-lit without harsh shadows</p>
            </div>
          </div>
          <div className="instruction-step">
            <div className="step-icon">👓</div>
            <div className="step-content">
              <h3>Remove Glasses</h3>
              <p>Take off glasses for more accurate measurements</p>
            </div>
          </div>
          <div className="instruction-step">
            <div className="step-icon">🎯</div>
            <div className="step-content">
              <h3>Position Your Face</h3>
              <p>Look straight at the camera with a neutral expression</p>
            </div>
          </div>
          <div className="instruction-step">
            <div className="step-icon">📏</div>
            <div className="step-content">
              <h3>Optimal Distance</h3>
              <p>Position yourself about 50-60cm from the camera</p>
            </div>
          </div>
        </div>
        <button className="primary-button" onClick={startMeasurement}>
          Start Measurement
        </button>
      </div>
    </div>
  );

  const renderMeasuringScreen = () => (
    <div className="screen measuring-screen">
      <div className="webcam-container">
        <div className="webcam-controls">
          <button className="webcam-toggle active" onClick={toggleWebcam}>
            <span className="icon">●</span> Stop Measurement
          </button>

          <div className="fps-counter">FPS: {fps}</div>
        </div>

        {webcamError && (
          <div className="error-message">
            <span>
              <strong>Webcam Error: </strong>
              {webcamError}
            </span>
            <button onClick={() => setWebcamError(null)}>×</button>
          </div>
        )}

        <div className="video-wrapper" ref={containerRef}>
          <video
            ref={webcamRef}
            autoPlay
            playsInline
            muted
            className="webcam-feed"
            style={{
              display: webcamRunning ? "block" : "none",
              transform: "scaleX(-1)", // Mirror the webcam feed
            }}
          ></video>
          <canvas
            ref={outputCanvasRef}
            className="measurement-canvas"
            style={{ transform: "scaleX(-1)" }} // Mirror the canvas
          ></canvas>

          {!webcamRunning && (
            <div className="webcam-placeholder">
              <div className="placeholder-icon">👤</div>
              <p>Webcam is disabled</p>
              <p>Click "Start Measurement" to begin</p>
            </div>
          )}
        </div>

        <div className="distance-feedback">
          {distanceStatus === "checking" && (
            <div className="feedback checking">
              <div className="feedback-icon">🔍</div>
              <p>Looking for your face...</p>
            </div>
          )}
          {distanceStatus === "tooClose" && (
            <div className="feedback too-close">
              <div className="feedback-icon">👆</div>
              <p>Move slightly farther from the camera</p>
            </div>
          )}
          {distanceStatus === "tooFar" && (
            <div className="feedback too-far">
              <div className="feedback-icon">👇</div>
              <p>Move slightly closer to the camera</p>
            </div>
          )}
          {distanceStatus === "optimal" && (
            <div className="feedback optimal">
              <div className="feedback-icon">✅</div>
              <p>Perfect distance! Ready to capture</p>
            </div>
          )}
        </div>

        <div className="capture-controls">
          <button
            className="capture-button"
            onClick={captureMeasurements}
            disabled={!measurements || distanceStatus !== "optimal"}
          >
            <span className="icon">📷</span> Capture Measurements
          </button>
        </div>
      </div>
    </div>
  );

  const renderResultsScreen = () => (
    <div className="screen results-screen">
      <div className="screen-content">
        <h2>Your Facial Measurements</h2>
        <div className="measurements-grid">
          <div className="measurement-card">
            <h3>Pupillary Distance (PD)</h3>
            <div className="measurement-value">{finalMeasurements.pd} mm</div>
            <p className="measurement-desc">Distance between pupils</p>
          </div>

          <div className="measurement-card">
            <h3>Naso-Pupillary Distance (NPD)</h3>
            <div className="measurement-subvalues">
              <div>
                <span className="label">Left Eye:</span>
                <span className="value">{finalMeasurements.npd.left} mm</span>
              </div>
              <div>
                <span className="label">Right Eye:</span>
                <span className="value">{finalMeasurements.npd.right} mm</span>
              </div>
            </div>
            <p className="measurement-desc">Distance from nose to each pupil</p>
          </div>

          <div className="measurement-card">
            <h3>Eye Opening Height</h3>
            <div className="measurement-subvalues">
              <div>
                <span className="label">Left Eye:</span>
                <span className="value">
                  {finalMeasurements.eyeHeight.left} mm
                </span>
              </div>
              <div>
                <span className="label">Right Eye:</span>
                <span className="value">
                  {finalMeasurements.eyeHeight.right} mm
                </span>
              </div>
            </div>
            <p className="measurement-desc">Vertical opening of eyes</p>
          </div>

          <div className="measurement-card">
            <h3>Pupil Height</h3>
            <div className="measurement-subvalues">
              <div>
                <span className="label">Left Eye:</span>
                <span className="value">
                  {finalMeasurements.pupilHeight.left} mm
                </span>
              </div>
              <div>
                <span className="label">Right Eye:</span>
                <span className="value">
                  {finalMeasurements.pupilHeight.right} mm
                </span>
              </div>
              <div>
                <span className="label">Combined:</span>
                <span className="value">
                  {finalMeasurements.pupilHeight.combined} mm
                </span>
              </div>
            </div>
            <p className="measurement-desc">Vertical position of pupils</p>
          </div>

          <div className="measurement-card">
            <h3>Face Dimensions</h3>
            <div className="measurement-subvalues">
              <div>
                <span className="label">Width:</span>
                <span className="value">{finalMeasurements.faceWidth} mm</span>
              </div>
              <div>
                <span className="label">Length:</span>
                <span className="value">{finalMeasurements.faceLength} mm</span>
              </div>
            </div>
            <p className="measurement-desc">Basic face measurements</p>
          </div>

          <div className="measurement-card">
            <h3>Face Shape</h3>
            <div className="measurement-value shape">
              {finalMeasurements.faceShape}
            </div>
            <p className="measurement-desc">
              Classification based on proportions
            </p>
          </div>
        </div>

        <div className="results-actions">
          <button className="primary-button" onClick={resetCapture}>
            <span className="icon">↺</span> Measure Again
          </button>
          <button className="secondary-button">
            <span className="icon">📧</span> Email Results
          </button>
        </div>
      </div>
    </div>
  );

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
            <span>
              <strong>Error: </strong>
              {error}
            </span>
            <button onClick={() => setError(null)}>×</button>
          </div>
        )}

        {isModelLoading ? (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Loading measurement model...</p>
          </div>
        ) : (
          <>
            {appState === "instructions" && renderInstructionsScreen()}
            {appState === "measuring" && renderMeasuringScreen()}
            {appState === "results" && renderResultsScreen()}
          </>
        )}
      </main>

      <footer className="app-footer">
        <p>
          Note: Measurements are approximations. For precise measurements,
          consult a professional.
        </p>
      </footer>
    </div>
  );
};

export default App;
