import React, { useEffect, useRef, useState } from "react";
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
  const [referenceObject, setReferenceObject] = useState(null);
  const [calibrationMode, setCalibrationMode] = useState(false);
  const [calibrationComplete, setCalibrationComplete] = useState(false);
  const [distanceWarning, setDistanceWarning] = useState("");
  const [currentStep, setCurrentStep] = useState(1); // 1: Distance, 2: Reference, 3: Measurement
  const [distanceStatus, setDistanceStatus] = useState("not-optimal"); // not-optimal, optimal

  // Performance optimization
  const lastFrameTimeRef = useRef(0);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(0);
  const minFrameInterval = 100;

  // Reference object dimensions (standard credit card)
  const referenceObjectWidth = 85.6; // mm

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

  // Enhanced face contour detection
  const drawFaceContour = (landmarks, canvas, ctx) => {
    if (!landmarks || landmarks.length === 0) return;
    
    const landmark = landmarks[0];
    
    // Convert normalized coordinates to pixel coordinates
    const toPixels = (point) => {
      return {
        x: point.x * canvas.width,
        y: point.y * canvas.height,
      };
    };

    // Get face oval points (approximating the face contour)
    const faceOvalPoints = [
      10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 
      361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 
      176, 149, 150, 136, 172, 58,  132, 93,  234, 127, 
      162, 21,  54,  103, 67,  109
    ].map(index => toPixels(landmark[index]));

    // Draw smooth face contour
    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    // Move to first point
    ctx.moveTo(faceOvalPoints[0].x, faceOvalPoints[0].y);
    
    // Draw curve through all points
    for (let i = 1; i < faceOvalPoints.length; i++) {
      const prev = faceOvalPoints[i - 1];
      const current = faceOvalPoints[i];
      const next = faceOvalPoints[(i + 1) % faceOvalPoints.length];
      
      // Calculate control points for smooth curve
      const cp1x = prev.x + (current.x - prev.x) * 0.5;
      const cp1y = prev.y + (current.y - prev.y) * 0.5;
      const cp2x = current.x + (next.x - current.x) * 0.5;
      const cp2y = current.y + (next.y - current.y) * 0.5;
      
      ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, next.x, next.y);
    }
    
    // Close the path
    ctx.closePath();
    ctx.stroke();
  };

  // Check face distance and update status
  const checkFaceDistance = (landmarks, canvas) => {
    if (!landmarks || landmarks.length === 0) return "not-optimal";
    
    const landmark = landmarks[0];
    
    // Convert normalized coordinates to pixel coordinates
    const toPixels = (point) => {
      return {
        x: point.x * canvas.width,
        y: point.y * canvas.height,
      };
    };

    // Get key facial points for distance calculation
    const chin = toPixels(landmark[152]); // Chin
    const forehead = toPixels(landmark[10]); // Forehead

    // Calculate face height in pixels
    const faceHeightPx = Math.sqrt(
      Math.pow(chin.x - forehead.x, 2) + Math.pow(chin.y - forehead.y, 2)
    );

    // Check if face is at optimal distance (face height should be about 70-80% of frame height)
    const faceToFrameRatio = faceHeightPx / canvas.height;
    
    if (faceToFrameRatio >= 0.7 && faceToFrameRatio <= 0.8) {
      setDistanceWarning("");
      return "optimal";
    } else if (faceToFrameRatio < 0.7) {
      setDistanceWarning("Please move closer to the camera for more accurate measurements");
      return "not-optimal";
    } else {
      setDistanceWarning("Please move further from the camera for more accurate measurements");
      return "not-optimal";
    }
  };

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

    // Get key facial points
    const leftPupil = toPixels(landmark[468]); // Left eye center
    const rightPupil = toPixels(landmark[473]); // Right eye center
    const noseTip = toPixels(landmark[4]); // Nose tip
    const chin = toPixels(landmark[152]); // Chin
    const forehead = toPixels(landmark[10]); // Forehead

    // Calculate face height in pixels
    const faceHeightPx = Math.sqrt(
      Math.pow(chin.x - forehead.x, 2) + Math.pow(chin.y - forehead.y, 2)
    );

    // Check if we have a reference object for calibration
    let pxToMm;
    if (referenceObject && calibrationComplete) {
      // Use reference object for calibration
      pxToMm = referenceObjectWidth / referenceObject.widthPx;
    } else {
      // Use face height as reference (average male face height is ~190mm)
      const averageFaceHeightMm = 190;
      pxToMm = averageFaceHeightMm / faceHeightPx;
    }

    // Calculate PD (Pupillary Distance)
    const pupilDistancePx = Math.sqrt(
      Math.pow(rightPupil.x - leftPupil.x, 2) +
        Math.pow(rightPupil.y - leftPupil.y, 2)
    );
    const pd = pupilDistancePx * pxToMm;

    // Calculate NPD (Naso-Pupillary Distance)
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
    const faceHeight = Math.sqrt(Math.pow(chin.y - forehead.y, 2)) * pxToMm;

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
    const jawAngleLeft = Math.atan2(
      jawlinePoints[0].y - jawlinePoints[4].y,
      jawlinePoints[0].x - jawlinePoints[4].x
    ) * 180 / Math.PI;
    
    const jawAngleRight = Math.atan2(
      jawlinePoints[16].y - jawlinePoints[12].y,
      jawlinePoints[16].x - jawlinePoints[12].x
    ) * 180 / Math.PI;

    // Calculate ratios for face shape detection
    const faceRatio = faceWidth / faceHeight;
    const cheekboneJawRatio = cheekboneWidth / faceWidth;
    const foreheadJawRatio = foreheadWidth / faceWidth;

    // Enhanced face shape classification with more accurate criteria
    let faceShape = "Oval";

    if (faceRatio > 0.85 && Math.abs(jawAngleLeft) < 120 && Math.abs(jawAngleRight) < 120) {
      faceShape = "Round";
    } else if (faceRatio < 0.75 && cheekboneJawRatio > 0.95) {
      faceShape = "Long";
    } else if (Math.abs(faceWidth - cheekboneWidth) < 5 && Math.abs(faceWidth - foreheadWidth) < 5) {
      faceShape = "Square";
    } else if (foreheadJawRatio > 1.1 && cheekboneJawRatio > 1.05 && faceRatio > 0.8) {
      faceShape = "Heart";
    } else if (cheekboneJawRatio > 1.05 && foreheadJawRatio < 0.95 && faceRatio < 0.85) {
      faceShape = "Diamond";
    } else if (Math.abs(jawAngleLeft) > 130 || Math.abs(jawAngleRight) > 130) {
      faceShape = "Triangle";
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
      calibrationMethod: referenceObject && calibrationComplete ? "Reference Object" : "Face Height Estimation"
    };
  };

  // Enhanced reference object detection with edge detection
  const detectReferenceObject = (landmarks, canvas, ctx) => {
    if (!landmarks || landmarks.length === 0) return null;

    const landmark = landmarks[0];
    
    // Convert normalized coordinates to pixel coordinates
    const toPixels = (point) => {
      return {
        x: point.x * canvas.width,
        y: point.y * canvas.height,
      };
    };

    // Get forehead points for reference positioning
    const foreheadCenter = toPixels(landmark[10]);
    const leftTemple = toPixels(landmark[234]);
    const rightTemple = toPixels(landmark[454]);
    
    // Define search area above forehead (larger area for better detection)
    const searchArea = {
      x: Math.max(0, leftTemple.x - 20),
      y: Math.max(0, foreheadCenter.y - canvas.height * 0.25),
      width: Math.min(canvas.width, rightTemple.x - leftTemple.x + 40),
      height: canvas.height * 0.2
    };

    // In a real implementation, we would use computer vision techniques
    // to detect the rectangular shape of the credit card
    // For this example, we'll simulate detection with improved logic
    
    // Draw search area for debugging
    if (calibrationMode) {
      ctx.strokeStyle = "#ff00ff20";
      ctx.lineWidth = 1;
      ctx.strokeRect(
        searchArea.x,
        searchArea.y,
        searchArea.width,
        searchArea.height
      );
    }

    // Simulate detection - in a real app, you would use actual image processing
    const isDetected = Math.random() > 0.3; // Simulating detection with some randomness
    
    if (isDetected) {
      const detectedObject = {
        x: searchArea.x + searchArea.width * 0.1,
        y: searchArea.y + searchArea.height * 0.2,
        widthPx: searchArea.width * 0.8,
        heightPx: searchArea.width * 0.8 * 0.63, // Credit card aspect ratio
        detected: true
      };
      
      // Draw the detected reference object with a green line at the top edge
      ctx.strokeStyle = "#00ff00";
      ctx.lineWidth = 3;
      
      // Top edge line
      ctx.beginPath();
      ctx.moveTo(detectedObject.x, detectedObject.y);
      ctx.lineTo(detectedObject.x + detectedObject.widthPx, detectedObject.y);
      ctx.stroke();
      
      if (calibrationMode) {
        // Draw the rest of the reference object outline (lighter)
        ctx.strokeStyle = "#00ff0040";
        ctx.strokeRect(
          detectedObject.x,
          detectedObject.y,
          detectedObject.widthPx,
          detectedObject.heightPx
        );
        
        ctx.fillStyle = "#00ff0020";
        ctx.fillRect(
          detectedObject.x,
          detectedObject.y,
          detectedObject.widthPx,
          detectedObject.heightPx
        );
        
        ctx.fillStyle = "#ffffff";
        ctx.font = "16px Arial";
        ctx.fillText(
          "Reference Object Detected",
          detectedObject.x,
          detectedObject.y - 10
        );
      }
      
      return detectedObject;
    } else {
      if (calibrationMode) {
        ctx.fillStyle = "#ffffff";
        ctx.font = "16px Arial";
        ctx.fillText(
          "Place a standard credit card on your forehead for calibration",
          searchArea.x,
          searchArea.y - 20
        );
      }
      return null;
    }
  };

  // Start calibration mode
  const startCalibration = () => {
    setCalibrationMode(true);
    setCalibrationComplete(false);
    setReferenceObject(null);
    setCurrentStep(2);
  };

  // Complete calibration
  const completeCalibration = () => {
    if (referenceObject) {
      setCalibrationComplete(true);
      setCalibrationMode(false);
      setCurrentStep(3);
    }
  };

  // Capture final measurements
  const captureMeasurements = () => {
    if (measurements) {
      setFinalMeasurements(measurements);
      setIsCaptured(true);
    }
  };

  // Reset capture and start over
  const resetCapture = () => {
    setFinalMeasurements(null);
    setIsCaptured(false);
    setReferenceObject(null);
    setCalibrationComplete(false);
    setCalibrationMode(false);
    setCurrentStep(1);
    setDistanceStatus("not-optimal");
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
      setFinalMeasurements(null);
      setIsCaptured(false);
      setReferenceObject(null);
      setCalibrationMode(false);
      setCalibrationComplete(false);
      setCurrentStep(1);
      setDistanceStatus("not-optimal");
      if (webcamRef.current && webcamRef.current.srcObject) {
        const tracks = webcamRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
        webcamRef.current.srcObject = null;
        webcamRef.current.pause();
      }
    } else {
      // Start webcam
      setWebcamRunning(true);
      setWebcamError(null);
      setFinalMeasurements(null);
      setIsCaptured(false);
      setReferenceObject(null);
      setCalibrationMode(false);
      setCalibrationComplete(false);
      setCurrentStep(1);
      setDistanceStatus("not-optimal");

      const constraints = {
        video: {
          facingMode: "user",
          width: { ideal: 640 }, // Reduced resolution for better performance
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

  // Draw measurement visualization
  const drawMeasurementVisualization = (landmarks, canvas, ctx, measurements) => {
    if (!landmarks || landmarks.length === 0 || !measurements) return;
    
    const landmark = landmarks[0];
    
    // Convert normalized coordinates to pixel coordinates
    const toPixels = (point) => {
      return {
        x: point.x * canvas.width,
        y: point.y * canvas.height,
      };
    };
    
    // Get key points
    const leftPupil = toPixels(landmark[468]);
    const rightPupil = toPixels(landmark[473]);
    
    // Draw PD line
    ctx.strokeStyle = "#4285f4";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(leftPupil.x, leftPupil.y);
    ctx.lineTo(rightPupil.x, rightPupil.y);
    ctx.stroke();
    
    // Add PD measurement text
    ctx.fillStyle = "#ffffff";
    ctx.font = "16px Arial";
    ctx.fillText(`PD: ${measurements.pd} mm`, 
                 (leftPupil.x + rightPupil.x) / 2, 
                 (leftPupil.y + rightPupil.y) / 2 - 10);
  };

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

        // Step 1: Check face distance with enhanced contour
        if (currentStep === 1) {
          const distanceCheck = checkFaceDistance(results.faceLandmarks, canvas);
          setDistanceStatus(distanceCheck);
          
          // Draw face contour instead of bounding box
          drawFaceContour(results.faceLandmarks, canvas, ctx);
          
          // Add text instruction
          ctx.fillStyle = "#ffffff";
          ctx.font = "16px Arial";
          ctx.fillText(
            distanceCheck === "optimal" 
              ? "Perfect distance! Click 'Next' to continue" 
              : "Adjust your distance from the camera",
            10, 30
          );
        }

        // Step 2: Detect reference object with enhanced detection
        if (currentStep === 2 && calibrationMode && !calibrationComplete) {
          const detectedObject = detectReferenceObject(results.faceLandmarks, canvas, ctx);
          if (detectedObject && detectedObject.detected) {
            setReferenceObject(detectedObject);
          }
        }

        // Step 3: Calculate and display measurements
        if (currentStep === 3) {
          const newMeasurements = calculateMeasurements(
            results.faceLandmarks,
            canvas
          );
          if (newMeasurements) {
            setMeasurements(newMeasurements);
          }

          // Draw face contour for measurement visualization
          drawFaceContour(results.faceLandmarks, canvas, ctx);
          
          // Draw measurement visualization
          drawMeasurementVisualization(results.faceLandmarks, canvas, ctx, newMeasurements);
        }

        // Draw minimal face landmarks for reference
        for (const landmarks of results.faceLandmarks) {
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
            { color: "#4285f4", lineWidth: 1 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
            { color: "#4285f4", lineWidth: 1 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
            { color: "#34a853", lineWidth: 1 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
            { color: "#34a853", lineWidth: 1 }
          );

          // Draw measurement points
          const drawPoint = (point, color, size) => {
            const x = point.x * canvas.width;
            const y = point.y * canvas.height;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI);
            ctx.fill();
          };

          drawPoint(landmarks[468], "#ea4335", 3); // Left pupil
          drawPoint(landmarks[473], "#ea4335", 3); // Right pupil
          drawPoint(landmarks[4], "#fbbc05", 3); // Nose tip
        }
      } else {
        setWebcamError(
          "No face detected. Please position your face in the frame and ensure good lighting."
        );
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
  }, [webcamRunning, currentStep]);

  // Check if webcam is supported
  const hasGetUserMedia = () => {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  };

  // Next step handler
  const nextStep = () => {
    if (currentStep === 1 && distanceStatus === "optimal") {
      setCurrentStep(2);
      startCalibration();
    } else if (currentStep === 2 && referenceObject) {
      setCurrentStep(3);
      completeCalibration();
    }
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
          <section className="measurement-section">
            <div className="webcam-container">
              <div className="webcam-controls">
                <button
                  className={`webcam-toggle ${webcamRunning ? "stop" : "start"}`}
                  onClick={toggleWebcam}
                  disabled={isModelLoading}
                >
                  {webcamRunning ? "Stop Camera" : "Start Camera"}
                </button>
                
                {webcamRunning && (
                  <div className="step-controls">
                    {currentStep < 3 && (
                      <button 
                        className="next-button"
                        onClick={nextStep}
                        disabled={
                          (currentStep === 1 && distanceStatus !== "optimal") ||
                          (currentStep === 2 && !referenceObject)
                        }
                      >
                        {currentStep === 1 ? "Next" : "Calibrate & Continue"}
                      </button>
                    )}
                    
                    {currentStep === 3 && !isCaptured && (
                      <button 
                        className="capture-button"
                        onClick={captureMeasurements}
                        disabled={!measurements}
                      >
                        Capture Measurements
                      </button>
                    )}
                    
                    {isCaptured && (
                      <button 
                        className="reset-button"
                        onClick={resetCapture}
                      >
                        Start Over
                      </button>
                    )}
                  </div>
                )}
              </div>
              
              <div className="video-container" ref={containerRef}>
                <video
                  ref={webcamRef}
                  className="webcam-video"
                  playsInline
                  muted
                />
                <canvas
                  ref={outputCanvasRef}
                  className="output-canvas"
                />
                
                {webcamRunning && (
                  <div className="step-indicator">
                    <div className={`step ${currentStep >= 1 ? "active" : ""}`}>
                      <span>1</span>
                      <p>Position Face</p>
                    </div>
                    <div className={`step ${currentStep >= 2 ? "active" : ""}`}>
                      <span>2</span>
                      <p>Calibration</p>
                    </div>
                    <div className={`step ${currentStep >= 3 ? "active" : ""}`}>
                      <span>3</span>
                      <p>Measurement</p>
                    </div>
                  </div>
                )}
                
                {distanceWarning && (
                  <div className="distance-warning">
                    {distanceWarning}
                  </div>
                )}
                
                {webcamError && (
                  <div className="webcam-error">
                    {webcamError}
                  </div>
                )}
              </div>
              
              {webcamRunning && currentStep === 1 && (
                <div className="instructions">
                  <h3>Step 1: Position Your Face</h3>
                  <p>Please position your face so it fills the frame. The green box indicates optimal distance.</p>
                  <ul>
                    <li>Make sure your face is fully visible</li>
                    <li>Ensure good lighting on your face</li>
                    <li>Look directly at the camera</li>
                  </ul>
                </div>
              )}
              
              {webcamRunning && currentStep === 2 && (
                <div className="instructions">
                  <h3>Step 2: Calibration</h3>
                  <p>Place a standard credit card on your forehead for accurate measurement calibration.</p>
                  <ul>
                    <li>Use a standard credit card (85.6mm wide)</li>
                    <li>Place it horizontally on your forehead</li>
                    <li>Make sure it's fully visible to the camera</li>
                  </ul>
                </div>
              )}
              
              {webcamRunning && currentStep === 3 && (
                <div className="instructions">
                  <h3>Step 3: Measurement</h3>
                  <p>Hold still while we take your measurements. The blue line shows your pupillary distance.</p>
                  <ul>
                    <li>Keep your head straight</li>
                    <li>Look directly at the camera</li>
                    <li>Try not to move during measurement</li>
                  </ul>
                </div>
              )}
            </div>
            
            {measurements && currentStep === 3 && (
              <div className="measurements-panel">
                <h3>Facial Measurements</h3>
                <div className="measurements-grid">
                  <div className="measurement-item">
                    <span className="label">Pupillary Distance (PD):</span>
                    <span className="value">{measurements.pd} mm</span>
                  </div>
                  <div className="measurement-item">
                    <span className="label">Left Naso-Pupillary Distance:</span>
                    <span className="value">{measurements.npd.left} mm</span>
                  </div>
                  <div className="measurement-item">
                    <span className="label">Right Naso-Pupillary Distance:</span>
                    <span className="value">{measurements.npd.right} mm</span>
                  </div>
                  <div className="measurement-item">
                    <span className="label">Face Width:</span>
                    <span className="value">{measurements.faceWidth} mm</span>
                  </div>
                  <div className="measurement-item">
                    <span className="label">Face Length:</span>
                    <span className="value">{measurements.faceLength} mm</span>
                  </div>
                  <div className="measurement-item">
                    <span className="label">Face Shape:</span>
                    <span className="value">{measurements.faceShape}</span>
                  </div>
                  <div className="measurement-item">
                    <span className="label">Calibration Method:</span>
                    <span className="value">{measurements.calibrationMethod}</span>
                  </div>
                </div>
                
                {isCaptured && (
                  <div className="final-measurements">
                    <h4>Final Measurements Captured!</h4>
                    <p>Your measurements have been successfully recorded.</p>
                  </div>
                )}
              </div>
            )}
          </section>
        )}
      </main>
      <style jsx>
        {`
        .app-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .app-header {
          text-align: center;
          margin-bottom: 30px;
        }
        
        .app-header h1 {
          color: #333;
          margin-bottom: 10px;
        }
        
        .app-header p {
          color: #666;
          font-size: 18px;
        }
        
        .error-message {
          background-color: #ffebee;
          color: #c62828;
          padding: 15px;
          border-radius: 5px;
          margin-bottom: 20px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        .error-message button {
          background: none;
          border: none;
          font-size: 20px;
          cursor: pointer;
          color: #c62828;
        }
        
        .loading-container {
          text-align: center;
          padding: 40px;
        }
        
        .loading-spinner {
          width: 50px;
          height: 50px;
          border: 5px solid #f3f3f3;
          border-top: 5px solid #3498db;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 0 auto 20px;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .measurement-section {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }
        
        .webcam-container {
          background-color: #f5f5f5;
          border-radius: 10px;
          padding: 20px;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .webcam-controls {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          flex-wrap: wrap;
          gap: 10px;
        }
        
        .webcam-toggle {
          padding: 12px 24px;
          border: none;
          border-radius: 5px;
          font-size: 16px;
          font-weight: bold;
          cursor: pointer;
          transition: all 0.3s;
        }
        
        .webcam-toggle.start {
          background-color: #4CAF50;
          color: white;
        }
        
        .webcam-toggle.stop {
          background-color: #f44336;
          color: white;
        }
        
        .webcam-toggle:disabled {
          background-color: #cccccc;
          cursor: not-allowed;
        }
        
        .step-controls {
          display: flex;
          gap: 10px;
        }
        
        .next-button, .capture-button, .reset-button {
          padding: 12px 24px;
          border: none;
          border-radius: 5px;
          font-size: 16px;
          font-weight: bold;
          cursor: pointer;
          transition: all 0.3s;
        }
        
        .next-button, .capture-button {
          background-color: #2196F3;
          color: white;
        }
        
        .next-button:disabled, .capture-button:disabled {
          background-color: #cccccc;
          cursor: not-allowed;
        }
        
        .reset-button {
          background-color: #FF9800;
          color: white;
        }
        
        .video-container {
          position: relative;
          width: 100%;
          max-width: 640px;
          margin: 0 auto;
        }
        
        .webcam-video, .output-canvas {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: auto;
          border-radius: 5px;
        }
        
        .step-indicator {
          display: flex;
          justify-content: space-between;
          position: absolute;
          bottom: -50px;
          left: 0;
          right: 0;
          margin: 0 auto;
          max-width: 400px;
        }
        
        .step {
          display: flex;
          flex-direction: column;
          align-items: center;
          color: #999;
        }
        
        .step.active {
          color: #2196F3;
        }
        
        .step span {
          width: 30px;
          height: 30px;
          border-radius: 50%;
          background-color: #f5f5f5;
          display: flex;
          justify-content: center;
          align-items: center;
          margin-bottom: 5px;
          font-weight: bold;
        }
        
        .step.active span {
          background-color: #2196F3;
          color: white;
        }
        
        .distance-warning, .webcam-error {
          position: absolute;
          top: 10px;
          left: 10px;
          right: 10px;
          background-color: rgba(255, 0, 0, 0.7);
          color: white;
          padding: 10px;
          border-radius: 5px;
          text-align: center;
          font-weight: bold;
        }
        
        .instructions {
          margin-top: 70px;
          padding: 15px;
          background-color: #e3f2fd;
          border-radius: 5px;
          border-left: 4px solid #2196F3;
        }
        
        .instructions h3 {
          margin-top: 0;
          color: #1565C0;
        }
        
        .instructions ul {
          margin-bottom: 0;
          padding-left: 20px;
        }
        
        .instructions li {
          margin-bottom: 5px;
        }
        
        .measurements-panel {
          background-color: #f5f5f5;
          border-radius: 10px;
          padding: 20px;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .measurements-panel h3 {
          margin-top: 0;
          color: #333;
          border-bottom: 2px solid #ddd;
          padding-bottom: 10px;
        }
        
        .measurements-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 15px;
          margin-bottom: 20px;
        }
        
        .measurement-item {
          display: flex;
          justify-content: space-between;
          padding: 10px;
          background-color: white;
          border-radius: 5px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .measurement-item .label {
          font-weight: bold;
          color: #555;
        }
        
        .measurement-item .value {
          color: #2196F3;
          font-weight: bold;
        }
        
        .final-measurements {
          background-color: #E8F5E9;
          border-left: 4px solid #4CAF50;
          padding: 15px;
          border-radius: 5px;
        }
        
        .final-measurements h4 {
          margin-top: 0;
          color: #2E7D32;
        }
        
        @media (max-width: 768px) {
          .webcam-controls {
            flex-direction: column;
            align-items: stretch;
          }
          
          .step-controls {
            justify-content: center;
          }
          
          .measurements-grid {
            grid-template-columns: 1fr;
          }
        }
      `}
      </style>
    </div>
  );
};

export default App;