import React, { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const [landmarks, setLandmarks] = useState([]);
  const [error, setError] = useState("");
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);

  useEffect(() => {
    const initFaceLandmarker = async () => {
      try {
        setError("Loading MediaPipe vision tasks...");
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );

        setError("Loading face landmark model...");
        faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(
          vision,
          {
            baseOptions: {
              modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
              delegate: "GPU"
            },
            runningMode: "VIDEO",
            numFaces: 1,
            outputFaceBlendshapes: true
          }
        );
        
        setIsModelLoaded(true);
        setError("");
      } catch (err) {
        setError(`Failed to initialize face landmarker: ${err.message}`);
        console.error(err);
      }
    };

    initFaceLandmarker();
  }, []);

  const startCamera = async () => {
    try {
      setError("Requesting camera access...");
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 500, height: 400 } 
      });
      
      videoRef.current.srcObject = stream;
      setIsCameraOn(true);
      setError("");
      
      // Wait for video to load metadata before starting detection
      videoRef.current.onloadedmetadata = () => {
        detectFace();
      };
    } catch (err) {
      setError(`Camera error: ${err.message}`);
      console.error(err);
    }
  };

  const detectFace = async () => {
    if (!faceLandmarkerRef.current || !videoRef.current) return;

    const ctx = canvasRef.current.getContext("2d");
    
    try {
      // Update canvas dimensions to match video
      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;
      
      const results = await faceLandmarkerRef.current.detectForVideo(
        videoRef.current,
        performance.now()
      );

      // Clear canvas and draw video frame
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      ctx.drawImage(
        videoRef.current, 
        0, 0, 
        canvasRef.current.width, 
        canvasRef.current.height
      );

      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        setLandmarks(results.faceLandmarks[0]);

        const drawingUtils = new DrawingUtils(ctx);
        for (const landmark of results.faceLandmarks) {
          drawingUtils.drawConnectors(
            landmark,
            FaceLandmarker.FACE_LANDMARKS_TESSELATION,
            { color: "#C0C0C070", lineWidth: 1 }
          );
          drawingUtils.drawConnectors(
            landmark,
            FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
            { color: "#30FF30" }
          );
          drawingUtils.drawConnectors(
            landmark,
            FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
            { color: "#FF3030" }
          );
          drawingUtils.drawConnectors(
            landmark,
            FaceLandmarker.FACE_LANDMARKS_LIPS,
            { color: "#E0E0E0" }
          );
          drawingUtils.drawConnectors(
            landmark,
            FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
            { color: "#FFFFFF" }
          );
        }
      } else {
        setLandmarks([]);
      }
    } catch (err) {
      setError(`Detection error: ${err.message}`);
      console.error(err);
    }

    // Continue detection
    if (isCameraOn) {
      requestAnimationFrame(detectFace);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      const tracks = stream.getTracks();
      
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    
    setIsCameraOn(false);
    setLandmarks([]);
    
    // Clear canvas
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h2>Face Landmark Detection</h2>
      
      {/* Error Display */}
      {error && (
        <div style={{
          color: "red", 
          backgroundColor: "#ffeeee", 
          padding: "10px", 
          borderRadius: "5px",
          marginBottom: "15px"
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}
      
      {/* Camera Controls */}
      <div style={{ marginBottom: "15px" }}>
        {!isCameraOn ? (
          <button 
            onClick={startCamera}
            disabled={!isModelLoaded}
            style={{
              padding: "10px 15px",
              backgroundColor: isModelLoaded ? "#4CAF50" : "#cccccc",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: isModelLoaded ? "pointer" : "not-allowed"
            }}
          >
            {isModelLoaded ? "Start Camera" : "Loading Model..."}
          </button>
        ) : (
          <button 
            onClick={stopCamera}
            style={{
              padding: "10px 15px",
              backgroundColor: "#f44336",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer"
            }}
          >
            Stop Camera
          </button>
        )}
      </div>

      {/* Video and Canvas */}
      <div style={{ position: "relative", width: 500, height: 400, marginBottom: "20px" }}>
        <video
          ref={videoRef}
          width="500"
          height="400"
          style={{ 
            position: "absolute", 
            top: 0, 
            left: 0,
            transform: "scaleX(-1)" // Mirror the video for more natural experience
          }}
        />
        <canvas
          ref={canvasRef}
          width="500"
          height="400"
          style={{ 
            position: "absolute", 
            top: 0, 
            left: 0,
            transform: "scaleX(-1)" // Mirror the canvas to match video
          }}
        />
        
        {!isCameraOn && (
          <div style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            backgroundColor: "#f0f0f0",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#666",
            border: "1px solid #ccc"
          }}>
            Camera is off. Click "Start Camera" to begin.
          </div>
        )}
      </div>

      {/* Landmarks Display */}
      <div style={{ marginTop: "20px", maxHeight: "300px", overflowY: "scroll" }}>
        <h3>Detected Landmarks: {landmarks.length > 0 ? `${landmarks.length} points` : 'None'}</h3>
        {landmarks.length > 0 && (
          <pre
            style={{ fontSize: "12px", background: "#f4f4f4", padding: "10px" }}
          >
            {JSON.stringify(landmarks, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}

export default App;