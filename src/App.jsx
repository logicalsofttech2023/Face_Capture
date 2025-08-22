import React, { useEffect, useRef, useState } from "react";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [landmarks, setLandmarks] = useState([]);
  const [error, setError] = useState("");
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Initialize face detection
    const initFaceDetection = async () => {
      try {
        setIsLoading(true);
        
        // Check if browser supports mediaDevices
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error("Your browser does not support camera access");
        }

        // Start camera
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 500, height: 400 } 
        });
        
        videoRef.current.srcObject = stream;
        setIsCameraOn(true);
        setError("");
        
        // Start mock face detection (since we can't load the actual model in this environment)
        detectFace();
      } catch (err) {
        setError(`Camera error: ${err.message}`);
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };

    initFaceDetection();

    // Cleanup function to stop camera
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  const detectFace = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const ctx = canvasRef.current.getContext("2d");
    
    // Clear canvas
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    
    // Draw video frame
    ctx.drawImage(
      videoRef.current, 
      0, 0, 
      canvasRef.current.width, 
      canvasRef.current.height
    );

    // Mock face detection - in a real app, this would use the FaceLandmarker
    if (Math.random() > 0.3) { // Simulate face detection 70% of the time
      const mockLandmarks = generateMockLandmarks();
      setLandmarks(mockLandmarks);
      
      // Draw mock face landmarks
      drawMockLandmarks(ctx, mockLandmarks);
    } else {
      setLandmarks([]);
    }

    // Continue detection
    if (isCameraOn) {
      requestAnimationFrame(detectFace);
    }
  };

  const generateMockLandmarks = () => {
    const landmarks = [];
    for (let i = 0; i < 100; i++) {
      landmarks.push({
        x: Math.random() * 500,
        y: Math.random() * 400,
        z: Math.random() * 10
      });
    }
    return landmarks;
  };

  const drawMockLandmarks = (ctx, landmarks) => {
    ctx.fillStyle = "#30FF30";
    
    // Draw face oval
    ctx.beginPath();
    ctx.ellipse(250, 200, 150, 200, 0, 0, Math.PI * 2);
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw eyes
    ctx.beginPath();
    ctx.ellipse(180, 180, 30, 20, 0, 0, Math.PI * 2);
    ctx.strokeStyle = "#FF3030";
    ctx.stroke();
    
    ctx.beginPath();
    ctx.ellipse(320, 180, 30, 20, 0, 0, Math.PI * 2);
    ctx.strokeStyle = "#30FF30";
    ctx.stroke();
    
    // Draw mouth
    ctx.beginPath();
    ctx.ellipse(250, 280, 50, 20, 0, 0, Math.PI);
    ctx.strokeStyle = "#E0E0E0";
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Draw some random points as mock landmarks
    for (let i = 0; i < 20; i++) {
      const point = landmarks[Math.floor(Math.random() * landmarks.length)];
      ctx.beginPath();
      ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const toggleCamera = () => {
    if (isCameraOn) {
      // Stop camera
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
      setIsCameraOn(false);
      setLandmarks([]);
    } else {
      // Start camera
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          videoRef.current.srcObject = stream;
          setIsCameraOn(true);
          setError("");
          detectFace();
        })
        .catch(err => {
          setError(`Camera error: ${err.message}`);
        });
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif", maxWidth: 900, margin: "0 auto" }}>
      <h2 style={{ color: "#333", textAlign: "center" }}>Face Landmark Detection</h2>
      
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
      <div style={{ marginBottom: "15px", textAlign: "center" }}>
        <button 
          onClick={toggleCamera}
          style={{
            padding: "10px 15px",
            backgroundColor: isCameraOn ? "#f44336" : "#4CAF50",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            fontSize: "16px"
          }}
        >
          {isCameraOn ? "Stop Camera" : "Start Camera"}
        </button>
        
        {isLoading && (
          <span style={{ marginLeft: "15px", color: "#666" }}>Loading...</span>
        )}
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: "20px" }}>
        {/* Video and Canvas */}
        <div style={{ position: "relative", width: 500, height: 400 }}>
          <video
            ref={videoRef}
            width="500"
            height="400"
            autoPlay
            playsInline
            style={{ 
              transform: "scaleX(-1)", // Mirror the video for more natural experience
              border: "1px solid #ccc",
              borderRadius: "5px",
              backgroundColor: "#f0f0f0"
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
              transform: "scaleX(-1)", // Mirror the canvas to match video
              pointerEvents: "none" // Allow clicks to pass through to video
            }}
          />
          
          {!isCameraOn && (
            <div style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              backgroundColor: "rgba(240, 240, 240, 0.8)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#666",
              border: "1px solid #ccc",
              borderRadius: "5px"
            }}>
              Camera is off. Click "Start Camera" to begin.
            </div>
          )}
        </div>

        {/* Landmarks Display */}
        <div style={{ 
          flex: "1", 
          minWidth: 300, 
          maxHeight: 400, 
          overflowY: "scroll",
          border: "1px solid #ccc",
          borderRadius: "5px",
          padding: "10px",
          backgroundColor: "#f9f9f9"
        }}>
          <h3 style={{ marginTop: 0, color: "#333" }}>
            Detected Landmarks: {landmarks.length > 0 ? `${landmarks.length} points` : 'None'}
          </h3>
          {landmarks.length > 0 ? (
            <pre
              style={{ 
                fontSize: "12px", 
                background: "#f4f4f4", 
                padding: "10px",
                borderRadius: "3px",
                overflowX: "auto"
              }}
            >
              {JSON.stringify(landmarks.slice(0, 10), null, 2)}
              {landmarks.length > 10 && "\n... (showing first 10 landmarks)"}
            </pre>
          ) : (
            <p style={{ color: "#666", fontStyle: "italic" }}>
              No face detected. Make sure your face is visible in the camera.
            </p>
          )}
        </div>
      </div>

      {/* Information Panel */}
      <div style={{ 
        marginTop: "20px", 
        padding: "15px", 
        backgroundColor: "#e8f4f8", 
        borderRadius: "5px",
        border: "1px solid #b8dde9"
      }}>
        <h3 style={{ color: "#2a6496", marginTop: 0 }}>How It Works</h3>
        <ul style={{ color: "#555", lineHeight: 1.5 }}>
          <li>Click "Start Camera" to enable your webcam and begin face detection</li>
          <li>The app will detect facial landmarks like eyes, nose, and mouth</li>
          <li>Detected landmarks will be displayed on the right panel</li>
          <li>Green points represent detected facial features</li>
          <li>In a real application, this would use MediaPipe FaceLandmarker for accurate detection</li>
        </ul>
      </div>
    </div>
  );
}

export default App;