import React, { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

function App() {
  const videoRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const [landmarks, setLandmarks] = useState([]);

  useEffect(() => {
    const initFaceLandmarker = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
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

      const results = await faceLandmarkerRef.current.detectForVideo(
        videoRef.current,
        performance.now()
      );

      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        // Face ke landmarks ko state me save kar do
        setLandmarks(results.faceLandmarks[0]);
      }

      requestAnimationFrame(detectFace);
    };

    initFaceLandmarker();
  }, []);

  return (
    <div>
      <h2>Face Landmark Detection</h2>

      <video
        ref={videoRef}
        width="500"
        height="400"
        style={{ border: "1px solid black" }}
      />

      <div
        style={{ marginTop: "20px", maxHeight: "300px", overflowY: "scroll" }}
      >
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
