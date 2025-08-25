import React, { useRef, useEffect, useState } from 'react';
import { ObjectDetector, FilesetResolver } from '@mediapipe/tasks-vision';
import './ObjectDetection.css';

const ObjectDetection = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const objectDetectorRef = useRef(null);
  const [isWebcamEnabled, setIsWebcamEnabled] = useState(false);
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [selectedObject, setSelectedObject] = useState(null);
  const [isModelLoading, setIsModelLoading] = useState(true);

  // Initialize the object detector
  useEffect(() => {
    const initializeObjectDetector = async () => {
      try {
        setIsModelLoading(true);
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
        );
        objectDetectorRef.current = await ObjectDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
            delegate: "GPU"
          },
          scoreThreshold: 0.5,
          runningMode: "VIDEO"
        });
        setIsModelLoading(false);
        console.log("Object detector initialized");
      } catch (error) {
        console.error("Error initializing object detector:", error);
        setIsModelLoading(false);
      }
    };

    initializeObjectDetector();

    return () => {
      if (objectDetectorRef.current) {
        objectDetectorRef.current.close();
      }
    };
  }, []);

  // Enable webcam and start detection
  const enableWebcam = async () => {
    if (!objectDetectorRef.current) {
      alert("Object Detector is still loading. Please try again.");
      return;
    }

    // Check if webcam access is supported
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("Webcam access is not supported by your browser");
      return;
    }

    // Get usermedia parameters
    const constraints = {
      video: true
    };

    // Activate the webcam stream
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      webcamRef.current.srcObject = stream;
      setIsWebcamEnabled(true);
      
      // Start prediction when video is loaded
      webcamRef.current.addEventListener('loadeddata', predictWebcam);
    } catch (err) {
      console.error("Error accessing webcam:", err);
      alert("Error accessing webcam: " + err.message);
    }
  };

  // Prediction loop for webcam
  const predictWebcam = async () => {
    if (!objectDetectorRef.current || !webcamRef.current || !canvasRef.current) return;

    const video = webcamRef.current;
    const canvas = canvasRef.current;
    const canvasCtx = canvas.getContext('2d');
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Now let's start detecting the stream
    let startTimeMs = performance.now();
    const detections = objectDetectorRef.current.detectForVideo(video, startTimeMs);
    
    // Clear previous drawings
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Process and display detections
    if (detections.detections && detections.detections.length > 0) {
      const objects = [];
      
      detections.detections.forEach((detection, index) => {
        const category = detection.categories[0];
        const boundingBox = detection.boundingBox;
        
        // Draw bounding box
        canvasCtx.strokeStyle = '#00FF00';
        canvasCtx.lineWidth = 2;
        canvasCtx.strokeRect(
          boundingBox.originX, 
          boundingBox.originY, 
          boundingBox.width, 
          boundingBox.height
        );
        
        // Draw label background
        canvasCtx.fillStyle = '#00FF00';
        const text = `${category.categoryName} - ${Math.round(category.score * 100)}%`;
        const textWidth = canvasCtx.measureText(text).width;
        canvasCtx.fillRect(
          boundingBox.originX - 1,
          boundingBox.originY - 20,
          textWidth + 4,
          20
        );
        
        // Draw text
        canvasCtx.fillStyle = '#000000';
        canvasCtx.font = '14px Arial';
        canvasCtx.fillText(text, boundingBox.originX, boundingBox.originY - 5);
        
        // Store object data
        objects.push({
          id: index,
          name: category.categoryName,
          confidence: Math.round(category.score * 100),
          width: boundingBox.width,
          height: boundingBox.height,
          x: boundingBox.originX,
          y: boundingBox.originY
        });
      });
      
      setDetectedObjects(objects);
    } else {
      setDetectedObjects([]);
    }

    // Call this function again to keep predicting when the browser is ready
    if (isWebcamEnabled) {
      requestAnimationFrame(predictWebcam);
    }
  };

  // Handle object selection
  const handleObjectSelect = (object) => {
    setSelectedObject(object);
  };

  return (
    <div className="object-detection-container">
      <h1>Object Detection with MediaPipe</h1>
      <p>Hold objects up to your webcam to detect them and measure their width in pixels.</p>
      
      {isModelLoading && <div className="loading">Loading object detection model...</div>}
      
      <div className="webcam-container">
        <video ref={webcamRef} className="webcam" autoPlay playsInline />
        <canvas ref={canvasRef} className="canvas" />
        
        {!isWebcamEnabled && (
          <button 
            onClick={enableWebcam} 
            disabled={isModelLoading}
            className="enable-webcam-btn"
          >
            {isModelLoading ? 'Loading Model...' : 'Enable Webcam'}
          </button>
        )}
      </div>
      
      <div className="results-container">
        <h2>Detected Objects</h2>
        
        {detectedObjects.length > 0 ? (
          <div className="objects-list">
            {detectedObjects.map((object) => (
              <div 
                key={object.id} 
                className={`object-item ${selectedObject?.id === object.id ? 'selected' : ''}`}
                onClick={() => handleObjectSelect(object)}
              >
                <div className="object-name">{object.name}</div>
                <div className="object-confidence">{object.confidence}% confidence</div>
                <div className="object-dimensions">
                  Width: {Math.round(object.width)}px, Height: {Math.round(object.height)}px
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p>No objects detected. Point your webcam at objects to detect them.</p>
        )}
        
        {selectedObject && (
          <div className="selected-object-info">
            <h3>Selected Object: {selectedObject.name}</h3>
            <p>Width: <strong>{Math.round(selectedObject.width)} pixels</strong></p>
            <p>Height: {Math.round(selectedObject.height)} pixels</p>
            <p>Confidence: {selectedObject.confidence}%</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ObjectDetection;