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

  // Initialize face landmarker
  useEffect(() => {
    const createFaceLandmarker = async () => {
      setIsModelLoading(true);
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
        setIsModelLoading(false);
      }
    };

    createFaceLandmarker();
  }, []);

  // Handle image click for detection
  const handleImageClick = async () => {
    if (!faceLandmarker || !imageRef.current) {
      console.log("Wait for faceLandmarker to load before clicking!");
      return;
    }

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
  };

  // Toggle webcam
  const toggleWebcam = async () => {
    if (!faceLandmarker) {
      console.log("Wait! faceLandmarker not loaded yet.");
      return;
    }

    if (webcamRunning) {
      setWebcamRunning(false);
      // Stop webcam stream
      if (webcamRef.current && webcamRef.current.srcObject) {
        const tracks = webcamRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    } else {
      setWebcamRunning(true);
      
      // Get webcam access
      const constraints = { video: true };
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        webcamRef.current.srcObject = stream;
        webcamRef.current.addEventListener('loadeddata', predictWebcam);
      } catch (error) {
        console.error("Error accessing webcam:", error);
      }
    }
  };

  // Webcam prediction
  const predictWebcam = async () => {
    if (!webcamRef.current || !outputCanvasRef.current || !faceLandmarker) return;

    const video = webcamRef.current;
    const canvas = outputCanvasRef.current;
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
    <div>
      <h1>Face landmark detection using the MediaPipe FaceLandmarker task</h1>

      {isModelLoading ? (
        <p>Loading model...</p>
      ) : (
        <section>
          <h2>Demo: Detecting Images</h2>
          <p><b>Click on the image below</b> to see the key landmarks of the face.</p>

          <div className="detectOnClick">
            <img 
              ref={imageRef}
              src="https://storage.googleapis.com/mediapipe-assets/portrait.jpg" 
              width="100%" 
              crossOrigin="anonymous" 
              title="Click to get detection!" 
              onClick={handleImageClick}
              alt="Face for detection"
              style={{ cursor: 'pointer' }}
            />
          </div>
          <div className="blend-shapes">
            {imageBlendShapes.length > 0 && renderBlendShapes(imageBlendShapes)}
          </div>

          <h2>Demo: Webcam continuous face landmarks detection</h2>
          <p>
            Hold your face in front of your webcam to get real-time face landmarker detection.
            <br />
            Click <b>enable webcam</b> below and grant access to the webcam if prompted.
          </p>

          <div id="liveView" className="videoView">
            <button 
              className="mdc-button mdc-button--raised" 
              onClick={toggleWebcam}
              disabled={!hasGetUserMedia()}
            >
              <span className="mdc-button__ripple"></span>
              <span className="mdc-button__label">
                {webcamRunning ? 'DISABLE WEBCAM' : 'ENABLE WEBCAM'}
              </span>
            </button>
            <div style={{ position: 'relative' }}>
              <video 
                ref={webcamRef}
                style={{ position: 'absolute' }}
                autoPlay 
                playsInline
                hidden={!webcamRunning}
              ></video>
              <canvas 
                ref={outputCanvasRef}
                className="output_canvas" 
                style={{ position: 'absolute', left: '0px', top: '0px' }}
              ></canvas>
            </div>
          </div>
          <div className="blend-shapes">
            {videoBlendShapes.length > 0 && renderBlendShapes(videoBlendShapes)}
          </div>
        </section>
      )}
    </div>
  );
};

export default App;