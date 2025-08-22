import React, { useEffect, useRef, useState } from 'react';
import vision from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3';

const App = () => {
  const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

  // Refs
  const imageRef = useRef(null);
  const webcamRef = useRef(null);
  const outputCanvasRef = useRef(null);
  const instructionsRef = useRef(null);

  // State
  const [faceLandmarker, setFaceLandmarker] = useState(null);
  const [runningMode, setRunningMode] = useState('IMAGE');
  const [webcamRunning, setWebcamRunning] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [error, setError] = useState(null);
  const [webcamError, setWebcamError] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [currentStep, setCurrentStep] = useState(1);
  const [referenceObject, setReferenceObject] = useState(null);
  const [hasReference, setHasReference] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);

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
          outputFaceBlendshapes: true,
          runningMode: runningMode,
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

  // Handle image capture from webcam
  const captureImage = () => {
    if (!webcamRef.current) return;
    
    const video = webcamRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const imageData = canvas.toDataURL('image/png');
    setCapturedImage(imageData);
    setWebcamRunning(false);
    
    // Stop webcam
    if (webcamRef.current && webcamRef.current.srcObject) {
      const tracks = webcamRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      webcamRef.current.srcObject = null;
    }
    
    setCurrentStep(4); // Move to analysis step
  };

  // Toggle webcam for capturing reference image
  const toggleWebcam = async () => {
    if (!faceLandmarker) {
      setError("Face landmark detector not loaded yet.");
      return;
    }

    if (webcamRunning) {
      // Stop webcam
      setWebcamRunning(false);
      setWebcamError(null);
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
          setCurrentStep(3); // Move to capture step
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

  // Analyze the captured image
  const analyzeImage = async () => {
    if (!faceLandmarker || !capturedImage) {
      setError("Face landmarker or image not available!");
      return;
    }

    try {
      // Create image element from captured data
      const img = new Image();
      img.src = capturedImage;
      
      await new Promise((resolve) => {
        img.onload = resolve;
      });

      // Remove any existing canvas
      const existingCanvas = document.querySelector('.canvas');
      if (existingCanvas) {
        existingCanvas.remove();
      }

      // Create new canvas for drawing landmarks
      const canvas = document.createElement('canvas');
      canvas.setAttribute('class', 'canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.style.position = 'absolute';
      canvas.style.left = '0px';
      canvas.style.top = '0px';
      canvas.style.width = `${img.width}px`;
      canvas.style.height = `${img.height}px`;

      const container = document.createElement('div');
      container.style.position = 'relative';
      container.style.display = 'inline-block';
      container.appendChild(img);
      container.appendChild(canvas);
      
      const resultsContainer = document.getElementById('results-container');
      resultsContainer.innerHTML = '';
      resultsContainer.appendChild(container);
      
      const ctx = canvas.getContext('2d');
      const drawingUtils = new DrawingUtils(ctx);

      // Detect face landmarks
      const faceLandmarkerResult = faceLandmarker.detect(img);

      // Draw landmarks
      if (faceLandmarkerResult.faceLandmarks) {
        for (const landmarks of faceLandmarkerResult.faceLandmarks) {
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_TESSELATION,
            { color: "#C0C0C070", lineWidth: 1 }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
            { color: "#FF3030" }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
            { color: "#FF3030" }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
            { color: "#30FF30" }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
            { color: "#30FF30" }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
            { color: "#E0E0E0" }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LIPS,
            { color: "#E0E0E0" }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
            { color: "#FF3030" }
          );
          drawingUtils.drawConnectors(
            landmarks,
            FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
            { color: "#30FF30" }
          );
        }

        // Calculate measurements
        const measurements = calculateMeasurements(faceLandmarkerResult.faceLandmarks[0], img);
        setAnalysisResults(measurements);
        setCurrentStep(5); // Move to results step
      } else {
        setError("No face detected in the image. Please try again.");
      }
    } catch (error) {
      console.error("Error analyzing image:", error);
      setError(`Failed to analyze image: ${error.message}`);
    }
  };

  // Calculate all required measurements
  const calculateMeasurements = (landmarks, image) => {
    if (!hasReference) {
      return { error: "Reference object not detected" };
    }

    // Get key landmarks
    const leftEyeInner = landmarks[33];  // Left eye inner corner
    const leftEyeOuter = landmarks[133]; // Left eye outer corner
    const rightEyeInner = landmarks[362]; // Right eye inner corner
    const rightEyeOuter = landmarks[263]; // Right eye outer corner
    const leftEyeTop = landmarks[159];    // Left eye top
    const leftEyeBottom = landmarks[145]; // Left eye bottom
    const rightEyeTop = landmarks[386];   // Right eye top
    const rightEyeBottom = landmarks[374]; // Right eye bottom
    const noseTip = landmarks[1];         // Nose tip
    const leftPupil = landmarks[468];     // Left pupil center
    const rightPupil = landmarks[473];    // Right pupil center
    const chin = landmarks[152];          // Chin point

    // Calculate pixel distances
    const leftEyeHeightPx = Math.abs(leftEyeTop.y * image.height - leftEyeBottom.y * image.height);
    const rightEyeHeightPx = Math.abs(rightEyeTop.y * image.height - rightEyeBottom.y * image.height);
    
    const leftNpdPx = calculateDistance(
      noseTip.x * image.width, noseTip.y * image.height,
      leftPupil.x * image.width, leftPupil.y * image.height
    );
    
    const rightNpdPx = calculateDistance(
      noseTip.x * image.width, noseTip.y * image.height,
      rightPupil.x * image.width, rightPupil.y * image.height
    );
    
    const pdPx = calculateDistance(
      leftPupil.x * image.width, leftPupil.y * image.height,
      rightPupil.x * image.width, rightPupil.y * image.height
    );
    
    const leftPupilHeightPx = Math.abs(leftPupil.y * image.height - chin.y * image.height);
    const rightPupilHeightPx = Math.abs(rightPupil.y * image.height - chin.y * image.height);
    const combinedPupilHeightPx = (leftPupilHeightPx + rightPupilHeightPx) / 2;

    // Convert to mm using reference object
    const pxToMm = referenceObject.actualSize / referenceObject.pixelSize;
    
    const leftEyeHeightMm = (leftEyeHeightPx * pxToMm).toFixed(1);
    const rightEyeHeightMm = (rightEyeHeightPx * pxToMm).toFixed(1);
    const leftNpdMm = (leftNpdPx * pxToMm).toFixed(1);
    const rightNpdMm = (rightNpdPx * pxToMm).toFixed(1);
    const pdMm = (pdPx * pxToMm).toFixed(1);
    const leftPupilHeightMm = (leftPupilHeightPx * pxToMm).toFixed(1);
    const rightPupilHeightMm = (rightPupilHeightPx * pxToMm).toFixed(1);
    const combinedPupilHeightMm = (combinedPupilHeightPx * pxToMm).toFixed(1);

    // Determine face shape (simplified)
    const faceWidthPx = calculateDistance(
      landmarks[234].x * image.width, landmarks[234].y * image.height, // Right side
      landmarks[454].x * image.width, landmarks[454].y * image.height  // Left side
    );
    
    const faceHeightPx = calculateDistance(
      landmarks[10].x * image.width, landmarks[10].y * image.height,   // Forehead
      chin.x * image.width, chin.y * image.height                      // Chin
    );
    
    const faceRatio = faceWidthPx / faceHeightPx;
    let faceShape = "Oval";
    
    if (faceRatio > 0.9) faceShape = "Round";
    else if (faceRatio < 0.7) faceShape = "Long";
    else if (landmarks[152].y - landmarks[10].y > faceWidthPx * 1.2) faceShape = "Square";

    return {
      "Eye Opening Height": {
        "Left Eye": `${leftEyeHeightMm} mm`,
        "Right Eye": `${rightEyeHeightMm} mm`
      },
      "Face Shape": faceShape,
      "Naso-Pupillary Distance (NPD)": {
        "Left Eye": `${leftNpdMm} mm`,
        "Right Eye": `${rightNpdMm} mm`
      },
      "Pupil Height": {
        "Combined": `${combinedPupilHeightMm} mm`,
        "Left Eye": `${leftPupilHeightMm} mm`,
        "Right Eye": `${rightPupilHeightMm} mm`
      },
      "Pupillary Distance (PD)": `${pdMm} mm`
    };
  };

  // Calculate distance between two points
  const calculateDistance = (x1, y1, x2, y2) => {
    return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
  };

  // Handle reference object selection
  const handleReferenceSelect = (object) => {
    setReferenceObject(object);
    setHasReference(true);
    setCurrentStep(2); // Move to positioning step
  };

  // Check if webcam is supported
  const hasGetUserMedia = () => {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  };

  // Reset the process
  const resetProcess = () => {
    setCurrentStep(1);
    setCapturedImage(null);
    setAnalysisResults(null);
    setHasReference(false);
    setReferenceObject(null);
    setWebcamRunning(false);
    
    if (webcamRef.current && webcamRef.current.srcObject) {
      const tracks = webcamRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      webcamRef.current.srcObject = null;
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif', maxWidth: '1000px', margin: '0 auto' }}>
      <h1 style={{ textAlign: 'center', color: '#2c3e50' }}>Ophthalmic Face Analysis</h1>
      
      {/* Progress Indicator */}
      <div style={{ display: 'flex', justifyContent: 'space-between', margin: '20px 0', position: 'relative' }}>
        <div style={{ 
          position: 'absolute', 
          top: '50%', 
          height: '2px', 
          backgroundColor: '#e0e0e0', 
          width: '100%', 
          zIndex: 1 
        }}></div>
        {[1, 2, 3, 4, 5].map(step => (
          <div key={step} style={{
            width: '30px',
            height: '30px',
            borderRadius: '50%',
            backgroundColor: currentStep >= step ? '#4285f4' : '#e0e0e0',
            color: 'white',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            zIndex: 2,
            fontWeight: 'bold'
          }}>
            {step}
          </div>
        ))}
      </div>
      
      <div style={{ textAlign: 'center', marginBottom: '20px', fontSize: '14px' }}>
        <span style={{ color: currentStep >= 1 ? '#4285f4' : '#aaa' }}>Reference</span> • 
        <span style={{ color: currentStep >= 2 ? '#4285f4' : '#aaa' }}> Positioning</span> • 
        <span style={{ color: currentStep >= 3 ? '#4285f4' : '#aaa' }}> Capture</span> • 
        <span style={{ color: currentStep >= 4 ? '#4285f4' : '#aaa' }}> Analysis</span> • 
        <span style={{ color: currentStep >= 5 ? '#4285f4' : '#aaa' }}> Results</span>
      </div>

      {/* Error Display */}
      {error && (
        <div style={{
          padding: '10px',
          backgroundColor: '#ffebee',
          color: '#c62828',
          border: '1px solid #ef9a9a',
          borderRadius: '4px',
          marginBottom: '15px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <span><strong>Error: </strong>{error}</span>
          <button
            onClick={() => setError(null)}
            style={{ background: 'none', border: 'none', color: '#c62828', cursor: 'pointer' }}
          >
            ×
          </button>
        </div>
      )}

      {isModelLoading ? (
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <p>Loading facial analysis model...</p>
          <div style={{ width: '100%', height: '4px', backgroundColor: '#e0e0e0', borderRadius: '2px', margin: '20px auto' }}>
            <div style={{
              width: '100%',
              height: '100%',
              backgroundColor: '#4285f4',
              borderRadius: '2px',
              animation: 'loading 1.5s infinite ease-in-out'
            }}></div>
          </div>
        </div>
      ) : (
        <div>
          {/* Step 1: Reference Object Selection */}
          {currentStep === 1 && (
            <div style={{ textAlign: 'center' }}>
              <h2>Step 1: Select Reference Object</h2>
              <p>Please place a reference object of known size on your forehead or near your face.</p>
              <p>This will be used to convert pixel measurements to millimeters.</p>
              
              <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', margin: '30px 0' }}>
                <div 
                  style={{ 
                    border: referenceObject?.name === 'Credit Card' ? '2px solid #4285f4' : '1px solid #ddd', 
                    borderRadius: '8px', 
                    padding: '15px', 
                    cursor: 'pointer',
                    width: '150px'
                  }}
                  onClick={() => handleReferenceSelect({ name: 'Credit Card', actualSize: 85.6, pixelSize: 0 })}
                >
                  <div style={{ height: '90px', backgroundColor: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '5px' }}>
                    <div style={{ width: '80px', height: '50px', backgroundColor: '#ddd', borderRadius: '5px' }}></div>
                  </div>
                  <p style={{ marginTop: '10px' }}>Credit Card</p>
                  <p style={{ fontSize: '12px', color: '#666' }}>85.6 mm wide</p>
                </div>
                
                <div 
                  style={{ 
                    border: referenceObject?.name === 'Coin' ? '2px solid #4285f4' : '1px solid #ddd', 
                    borderRadius: '8px', 
                    padding: '15px', 
                    cursor: 'pointer',
                    width: '150px'
                  }}
                  onClick={() => handleReferenceSelect({ name: 'Coin', actualSize: 24.26, pixelSize: 0 })}
                >
                  <div style={{ height: '90px', backgroundColor: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '5px' }}>
                    <div style={{ width: '40px', height: '40px', backgroundColor: '#ddd', borderRadius: '50%' }}></div>
                  </div>
                  <p style={{ marginTop: '10px' }}>Quarter</p>
                  <p style={{ fontSize: '12px', color: '#666' }}>24.26 mm diameter</p>
                </div>
                
                <div 
                  style={{ 
                    border: referenceObject?.name === 'Custom' ? '2px solid #4285f4' : '1px solid #ddd', 
                    borderRadius: '8px', 
                    padding: '15px', 
                    cursor: 'pointer',
                    width: '150px'
                  }}
                  onClick={() => {
                    const size = prompt("Enter the size of your reference object in millimeters:");
                    if (size && !isNaN(size)) {
                      handleReferenceSelect({ name: 'Custom', actualSize: parseFloat(size), pixelSize: 0 });
                    }
                  }}
                >
                  <div style={{ height: '90px', backgroundColor: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '5px' }}>
                    <div style={{ fontSize: '30px' }}>+</div>
                  </div>
                  <p style={{ marginTop: '10px' }}>Custom</p>
                  <p style={{ fontSize: '12px', color: '#666' }}>Specify size</p>
                </div>
              </div>
            </div>
          )}

          {/* Step 2: Positioning Instructions */}
          {currentStep === 2 && (
            <div style={{ textAlign: 'center' }}>
              <h2>Step 2: Proper Positioning</h2>
              <div ref={instructionsRef} style={{ 
                backgroundColor: '#e8f4f8', 
                padding: '20px', 
                borderRadius: '8px', 
                margin: '20px 0',
                textAlign: 'left'
              }}>
                <h3 style={{ color: '#2c3e50', marginTop: 0 }}>Important Instructions:</h3>
                <ul style={{ lineHeight: '1.6' }}>
                  <li>Position your face straight, looking directly at the camera</li>
                  <li>Keep your head upright with no tilt or rotation</li>
                  <li>Ensure good lighting with both eyes clearly visible</li>
                  <li>Place the <strong>{referenceObject.name}</strong> on your forehead or near your head</li>
                  <li>Remove glasses or any obstructions</li>
                  <li>Keep a neutral expression with eyes open</li>
                </ul>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'center', gap: '15px', marginTop: '30px' }}>
                <button 
                  onClick={() => setCurrentStep(1)}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#f5f5f5',
                    color: '#333',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Back
                </button>
                <button 
                  onClick={toggleWebcam}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#4285f4',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Open Camera
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Capture Image */}
          {currentStep === 3 && (
            <div style={{ textAlign: 'center' }}>
              <h2>Step 3: Capture Image</h2>
              <p>Position yourself according to the instructions and click "Capture" when ready.</p>
              
              <div id="liveView" className="videoView" style={{ position: 'relative', width: '640px', height: '480px', margin: '0 auto' }}>
                <div style={{
                  position: 'relative',
                  width: '100%',
                  height: '100%',
                  backgroundColor: '#f5f5f5',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  overflow: 'hidden',
                  borderRadius: '8px'
                }}>
                  <video
                    ref={webcamRef}
                    autoPlay
                    playsInline
                    muted
                    style={{
                      position: 'absolute',
                      left: 0,
                      top: 0,
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      display: webcamRunning ? 'block' : 'none'
                    }}
                  ></video>

                  {!webcamRunning && (
                    <div style={{ textAlign: 'center', color: '#9e9e9e' }}>
                      <p>Webcam is disabled</p>
                      <p>Click "Enable Webcam" to start</p>
                    </div>
                  )}
                </div>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'center', gap: '15px', marginTop: '20px' }}>
                <button 
                  onClick={() => {
                    setWebcamRunning(false);
                    setCurrentStep(2);
                  }}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#f5f5f5',
                    color: '#333',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Back
                </button>
                <button 
                  onClick={captureImage}
                  disabled={!webcamRunning}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#4285f4',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    opacity: webcamRunning ? 1 : 0.5
                  }}
                >
                  Capture Image
                </button>
              </div>
            </div>
          )}

          {/* Step 4: Analysis */}
          {currentStep === 4 && (
            <div style={{ textAlign: 'center' }}>
              <h2>Step 4: Analyzing Image</h2>
              <p>We're analyzing your facial features. This may take a few moments.</p>
              
              <div style={{ width: '100%', margin: '30px 0' }}>
                <img 
                  src={capturedImage} 
                  alt="Captured" 
                  style={{ 
                    maxWidth: '100%', 
                    maxHeight: '400px', 
                    border: '1px solid #ddd', 
                    borderRadius: '8px' 
                  }} 
                />
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'center', gap: '15px' }}>
                <button 
                  onClick={() => {
                    setCapturedImage(null);
                    setCurrentStep(3);
                    toggleWebcam();
                  }}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#f5f5f5',
                    color: '#333',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Retake
                </button>
                <button 
                  onClick={analyzeImage}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#4285f4',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Analyze Image
                </button>
              </div>
            </div>
          )}

          {/* Step 5: Results */}
          {currentStep === 5 && analysisResults && (
            <div style={{ textAlign: 'center' }}>
              <h2>Step 5: Analysis Results</h2>
              
              <div id="results-container" style={{ margin: '20px 0' }}></div>
              
              {analysisResults.error ? (
                <div style={{ 
                  padding: '15px', 
                  backgroundColor: '#ffebee', 
                  color: '#c62828', 
                  borderRadius: '4px', 
                  margin: '20px 0' 
                }}>
                  <strong>Error:</strong> {analysisResults.error}
                </div>
              ) : (
                <div style={{ 
                  backgroundColor: '#f9f9f9', 
                  padding: '20px', 
                  borderRadius: '8px', 
                  marginTop: '20px',
                  textAlign: 'left'
                }}>
                  <h3 style={{ color: '#2c3e50', marginTop: 0 }}>Measurement Results:</h3>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                    <div style={{ padding: '10px', backgroundColor: 'white', borderRadius: '4px' }}>
                      <h4 style={{ margin: '0 0 10px 0', color: '#4285f4' }}>Eye Opening Height</h4>
                      <p>Left Eye: {analysisResults["Eye Opening Height"]["Left Eye"]}</p>
                      <p>Right Eye: {analysisResults["Eye Opening Height"]["Right Eye"]}</p>
                    </div>
                    
                    <div style={{ padding: '10px', backgroundColor: 'white', borderRadius: '4px' }}>
                      <h4 style={{ margin: '0 0 10px 0', color: '#4285f4' }}>Face Shape</h4>
                      <p>{analysisResults["Face Shape"]}</p>
                    </div>
                    
                    <div style={{ padding: '10px', backgroundColor: 'white', borderRadius: '4px' }}>
                      <h4 style={{ margin: '0 0 10px 0', color: '#4285f4' }}>Naso-Pupillary Distance</h4>
                      <p>Left Eye: {analysisResults["Naso-Pupillary Distance (NPD)"]["Left Eye"]}</p>
                      <p>Right Eye: {analysisResults["Naso-Pupillary Distance (NPD)"]["Right Eye"]}</p>
                    </div>
                    
                    <div style={{ padding: '10px', backgroundColor: 'white', borderRadius: '4px' }}>
                      <h4 style={{ margin: '0 0 10px 0', color: '#4285f4' }}>Pupil Height</h4>
                      <p>Combined: {analysisResults["Pupil Height"]["Combined"]}</p>
                      <p>Left Eye: {analysisResults["Pupil Height"]["Left Eye"]}</p>
                      <p>Right Eye: {analysisResults["Pupil Height"]["Right Eye"]}</p>
                    </div>
                    
                    <div style={{ padding: '10px', backgroundColor: 'white', borderRadius: '4px', gridColumn: '1 / -1' }}>
                      <h4 style={{ margin: '0 0 10px 0', color: '#4285f4' }}>Pupillary Distance</h4>
                      <p>{analysisResults["Pupillary Distance (PD)"]}</p>
                    </div>
                  </div>
                </div>
              )}
              
              <button 
                onClick={resetProcess}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#4285f4',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  marginTop: '20px'
                }}
              >
                Start New Analysis
              </button>
            </div>
          )}
        </div>
      )}

      <style>
        {`
          @keyframes loading {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
          }
        `}
      </style>
    </div>
  );
};

export default App;