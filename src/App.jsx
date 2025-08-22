import React, { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const [landmarks, setLandmarks] = useState([]);

  useEffect(() => {
    const initFaceLandmarker = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      );

      faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          },
          runningMode: "VIDEO",
          numFaces: 1,
        }
      );

      startCamera();
    };

    const startCamera = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      videoRef.current.play();

      detectFace();
    };

    const detectFace = async () => {
      if (!faceLandmarkerRef.current) return;

      const ctx = canvasRef.current.getContext("2d");

      const results = await faceLandmarkerRef.current.detectForVideo(
        videoRef.current,
        performance.now()
      );

      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);

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
      }

      requestAnimationFrame(detectFace);
    };

    initFaceLandmarker();
  }, []);

  return (
    <div>
      <h2>Face Landmark Detection</h2>

      <div style={{ position: "relative", width: 500, height: 400 }}>
        <video
          ref={videoRef}
          width="500"
          height="400"
          style={{ display: "none" }} // video ko hide karke sirf canvas dikha rahe hain
        />
        <canvas
          ref={canvasRef}
          width="500"
          height="400"
          style={{ border: "1px solid black" }}
        />
      </div>

      <div style={{ marginTop: "20px", maxHeight: "300px", overflowY: "scroll" }}>
        <h3>Detected Landmarks:</h3>
        <pre
          style={{ fontSize: "12px", background: "#f4f4f4", padding: "10px" }}
        >
          {JSON.stringify(landmarks, null, 2)}
        </pre>
      </div>
    </div>
  );
}

export default App;
