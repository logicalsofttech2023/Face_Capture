import React, { useEffect, useRef, useState } from 'react';
import vision from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3';

const App = () => {
  const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;
  
  // Refs
  const imageRef = useRef(null);
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const outputCanvasRef = useRef(null);
  
  // State
  const [faceLandmarker, setFaceLandmarker] = useState(null);
  const [runningMode, setRunningMode] = useState('IMAGE');
  const [webcamRunning, setWebcamRunning] = useState(false);
  const [imageBlendShapes, setImageBlendShapes] = useState([]);
  const [videoBlendShapes, setVideoBlendShapes] = useState([]);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [error, setError] = useState(null);
  const [webcamError, setWebcamError] = useState(null);

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

  // Handle image click for detection
  const handleImageClick = async () => {
    if (!faceLandmarker || !imageRef.current) {
      setError("Wait for faceLandmarker to load before clicking!");
      return;
    }

    try {
      if (runningMode === "VIDEO") {
        await faceLandmarker.setOptions({ runningMode: "IMAGE" });
        setRunningMode("IMAGE");
      }

      // Remove any existing canvas
      const existingCanvas = document.querySelector('.canvas');
      if (existingCanvas) {
        existingCanvas.remove();
      }

      // Create new canvas for drawing landmarks
      const canvas = document.createElement('canvas');
      canvas.setAttribute('class', 'canvas');
      canvas.setAttribute('width', imageRef.current.naturalWidth + 'px');
      canvas.setAttribute('height', imageRef.current.naturalHeight + 'px');
      canvas.style.left = '0px';
      canvas.style.top = '0px';
      canvas.style.width = `${imageRef.current.width}px`;
      canvas.style.height = `${imageRef.current.height}px`;
      canvas.style.position = 'absolute';
      
      imageRef.current.parentNode.style.position = 'relative';
      imageRef.current.parentNode.appendChild(canvas);
      const ctx = canvas.getContext('2d');
      const drawingUtils = new DrawingUtils(ctx);

      // Detect face landmarks
      const faceLandmarkerResult = faceLandmarker.detect(imageRef.current);
      
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
      }

      // Update blend shapes
      if (faceLandmarkerResult.faceBlendshapes && faceLandmarkerResult.faceBlendshapes.length > 0) {
        setImageBlendShapes(faceLandmarkerResult.faceBlendshapes[0].categories);
      }
    } catch (error) {
      console.error("Error detecting face landmarks:", error);
      setError(`Failed to detect face landmarks: ${error.message}`);
    }
  };

  // Toggle webcam
  const toggleWebcam = async () => {
    if (!faceLandmarker) {
      setError("Wait! Face landmark detector not loaded yet.");
      return;
    }

    if (webcamRunning) {
      setWebcamRunning(false);
      setWebcamError(null);
      // Stop webcam stream
      if (webcamRef.current && webcamRef.current.srcObject) {
        const tracks = webcamRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    } else {
      setWebcamRunning(true);
      setWebcamError(null);
      
      // Get webcam access
      const constraints = { video: true };
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        webcamRef.current.srcObject = stream;
        webcamRef.current.addEventListener('loadeddata', predictWebcam);
      } catch (error) {
        console.error("Error accessing webcam:", error);
        setWebcamError(`Failed to access webcam: ${error.message}`);
        setWebcamRunning(false);
      }
    }
  };

  // Webcam prediction
  const predictWebcam = async () => {
    if (!webcamRunning || !webcamRef.current || !outputCanvasRef.current || !faceLandmarker) return;

    const video = webcamRef.current;
    const canvas = outputCanvasRef.current;
    
    // Check if video is ready
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      requestAnimationFrame(predictWebcam);
      return;
    }
    
    try {
      const radio = video.videoHeight / video.videoWidth;
      const videoWidth = 480;

      // Set video and canvas dimensions
      video.style.width = videoWidth + "px";
      video.style.height = videoWidth * radio + "px";
      canvas.style.width = videoWidth + "px";
      canvas.style.height = videoWidth * radio + "px";
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Set running mode to VIDEO if needed
      if (runningMode === "IMAGE") {
        await faceLandmarker.setOptions({ runningMode: "VIDEO" });
        setRunningMode("VIDEO");
      }

      // Detect face landmarks
      let startTimeMs = performance.now();
      const results = faceLandmarker.detectForVideo(video, startTimeMs);
      const ctx = canvas.getContext('2d');
      const drawingUtils = new DrawingUtils(ctx);

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw landmarks
      if (results.faceLandmarks) {
        for (const landmarks of results.faceLandmarks) {
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
      }

      // Update blend shapes
      if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
        setVideoBlendShapes(results.faceBlendshapes[0].categories);
      }
    } catch (error) {
      console.error("Error during webcam prediction:", error);
      setWebcamError(`Webcam error: ${error.message}`);
      setWebcamRunning(false);
    }

    // Call this function again to keep predicting when the browser is ready
    if (webcamRunning) {
      requestAnimationFrame(predictWebcam);
    }
  };

  // Check if webcam is supported
  const hasGetUserMedia = () => {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  };

  // Render blend shapes
  const renderBlendShapes = (blendShapes) => {
    return (
      <ul className="blend-shapes-list">
        {blendShapes.map((shape, index) => (
          <li key={index} className="blend-shapes-item">
            <span className="blend-shapes-label">
              {shape.displayName || shape.categoryName}
            </span>
            <span 
              className="blend-shapes-value" 
              style={{ width: `calc(${Number(shape.score) * 100}% - 120px)` }}
            >
              {Number(shape.score).toFixed(4)}
            </span>
          </li>
        ))}
      </ul>
    );
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Face Landmark Detection</h1>
      
      {/* Error Display */}
      {error && (
        <div style={{ 
          padding: '10px', 
          backgroundColor: '#ffebee', 
          color: '#c62828', 
          border: '1px solid #ef9a9a',
          borderRadius: '4px',
          marginBottom: '15px'
        }}>
          <strong>Error: </strong>{error}
          <button 
            onClick={() => setError(null)}
            style={{ marginLeft: '10px', background: 'none', border: 'none', color: '#c62828', cursor: 'pointer' }}
          >
            ×
          </button>
        </div>
      )}

      {isModelLoading ? (
        <div>
          <p>Loading model...</p>
          <div style={{ width: '100%', height: '4px', backgroundColor: '#e0e0e0', borderRadius: '2px' }}>
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
        <section>
          <h2>Demo: Detecting Images</h2>
          <p><b>Click on the image below</b> to see the key landmarks of the face.</p>

          <div className="detectOnClick" style={{ position: 'relative', display: 'inline-block' }}>
            <img 
              ref={imageRef}
              src="https://storage.googleapis.com/mediapipe-assets/portrait.jpg" 
              width="480" 
              crossOrigin="anonymous" 
              title="Click to get detection!" 
              onClick={handleImageClick}
              alt="Face for detection"
              style={{ cursor: 'pointer', display: 'block' }}
            />
          </div>
          
          {imageBlendShapes.length > 0 && (
            <div className="blend-shapes">
              <h3>Detected Facial Expressions:</h3>
              {renderBlendShapes(imageBlendShapes)}
            </div>
          )}

          <h2>Demo: Webcam Continuous Face Landmarks Detection</h2>
          <p>
            Hold your face in front of your webcam to get real-time face landmarker detection.
            <br />
            Click <b>enable webcam</b> below and grant access to the webcam if prompted.
          </p>

          {webcamError && (
            <div style={{ 
              padding: '10px', 
              backgroundColor: '#ffebee', 
              color: '#c62828', 
              border: '1px solid #ef9a9a',
              borderRadius: '4px',
              marginBottom: '15px'
            }}>
              <strong>Webcam Error: </strong>{webcamError}
              <button 
                onClick={() => setWebcamError(null)}
                style={{ marginLeft: '10px', background: 'none', border: 'none', color: '#c62828', cursor: 'pointer' }}
              >
                ×
              </button>
            </div>
          )}

          <div id="liveView" className="videoView">
            <button 
              className="mdc-button mdc-button--raised" 
              onClick={toggleWebcam}
              disabled={!hasGetUserMedia()}
              style={{
                padding: '10px 15px',
                backgroundColor: webcamRunning ? '#f44336' : '#4285f4',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                marginBottom: '10px'
              }}
            >
              {webcamRunning ? 'DISABLE WEBCAM' : 'ENABLE WEBCAM'}
            </button>
            
            {!hasGetUserMedia() && (
              <p style={{ color: '#f44336' }}>
                Your browser does not support webcam access. Please try Chrome, Firefox, or Edge.
              </p>
            )}
            
            <div style={{ position: 'relative', width: '480px', height: '360px', backgroundColor: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <video 
                ref={webcamRef}
                style={{ position: 'absolute', left: 0, top: 0, display: webcamRunning ? 'block' : 'none' }}
                autoPlay 
                playsInline
              ></video>
              <canvas 
                ref={outputCanvasRef}
                className="output_canvas" 
                style={{ position: 'absolute', left: '0px', top: '0px', zIndex: 10 }}
              ></canvas>
              
              {!webcamRunning && (
                <div style={{ textAlign: 'center', color: '#9e9e9e' }}>
                  <p>Webcam is disabled</p>
                  <p>Click "Enable Webcam" to start</p>
                </div>
              )}
            </div>
          </div>
          
          {videoBlendShapes.length > 0 && (
            <div className="blend-shapes">
              <h3>Real-time Facial Expressions:</h3>
              {renderBlendShapes(videoBlendShapes)}
            </div>
          )}
        </section>
      )}
      
      <style>
        {`
          .blend-shapes-list {
            list-style: none;
            padding: 0;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            margin-top: 10px;
          }
          
          .blend-shapes-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            border-bottom: 1px solid #eeeeee;
          }
          
          .blend-shapes-item:last-child {
            border-bottom: none;
          }
          
          .blend-shapes-label {
            width: 120px;
            font-weight: bold;
          }
          
          .blend-shapes-value {
            background-color: #e3f2fd;
            padding: 2px 8px;
            border-radius: 4px;
            text-align: right;
          }
          
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