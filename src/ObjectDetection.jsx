import React, { useRef, useEffect, useState } from 'react';
import './ObjectDetection.css';

const ObjectDetection = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isWebcamEnabled, setIsWebcamEnabled] = useState(false);
  const [objectWidth, setObjectWidth] = useState(null);
  const [isOpenCVLoaded, setIsOpenCVLoaded] = useState(false);
  const [processing, setProcessing] = useState(false);

  // Load OpenCV.js
  useEffect(() => {
    // Check if OpenCV is already loaded
    if (window.cv) {
      setIsOpenCVLoaded(true);
      return;
    }

    // Function to load OpenCV
    const loadOpenCV = () => {
      return new Promise((resolve, reject) => {
        // Check if OpenCV is already loaded
        if (window.cv) {
          resolve();
          return;
        }

        // Create script element
        const script = document.createElement('script');
        script.src = 'https://docs.opencv.org/4.x/opencv.js';
        script.onload = () => {
          // Wait for OpenCV to initialize
          const checkCV = setInterval(() => {
            if (window.cv && window.cv.Mat) {
              clearInterval(checkCV);
              setIsOpenCVLoaded(true);
              resolve();
            }
          }, 50);
        };
        script.onerror = reject;
        document.body.appendChild(script);
      });
    };

    loadOpenCV().catch(error => {
      console.error('Failed to load OpenCV:', error);
    });
  }, []);

  // Enable webcam
  const enableWebcam = async () => {
    if (!isOpenCVLoaded) {
      alert('OpenCV is still loading. Please wait.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      setIsWebcamEnabled(true);
      
      // Start processing after a short delay to allow video to initialize
      setTimeout(() => {
        setProcessing(true);
        processVideo();
      }, 500);
    } catch (err) {
      console.error('Error accessing webcam:', err);
      alert('Error accessing webcam: ' + err.message);
    }
  };

  // Process video frames
  const processVideo = () => {
    if (!videoRef.current || !canvasRef.current || !processing) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get image data from canvas
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Create OpenCV Mat object
    const src = new window.cv.Mat(imageData.height, imageData.width, window.cv.CV_8UC4);
    src.data.set(imageData.data);

    // Convert to grayscale
    const gray = new window.cv.Mat();
    window.cv.cvtColor(src, gray, window.cv.COLOR_RGBA2GRAY);

    // Apply Gaussian blur to reduce noise
    const blurred = new window.cv.Mat();
    window.cv.GaussianBlur(gray, blurred, new window.cv.Size(5, 5), 0, 0, window.cv.BORDER_DEFAULT);

    // Apply threshold to create binary image
    const thresholded = new window.cv.Mat();
    window.cv.threshold(blurred, thresholded, 0, 255, window.cv.THRESH_BINARY + window.cv.THRESH_OTSU);

    // Find contours
    const contours = new window.cv.MatVector();
    const hierarchy = new window.cv.Mat();
    window.cv.findContours(thresholded, contours, hierarchy, window.cv.RETR_EXTERNAL, window.cv.CHAIN_APPROX_SIMPLE);

    // Find the largest contour (likely the object in hand)
    let maxArea = 0;
    let maxContour = null;
    
    for (let i = 0; i < contours.size(); ++i) {
      const contour = contours.get(i);
      const area = window.cv.contourArea(contour);
      
      if (area > maxArea && area > 1000) { // Filter out small contours
        maxArea = area;
        maxContour = contour;
      }
    }

    // If a significant contour is found
    if (maxContour) {
      // Get bounding rectangle
      const rect = window.cv.boundingRect(maxContour);
      
      // Draw rectangle on canvas
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 2;
      ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
      
      // Draw width text
      ctx.fillStyle = '#00ff00';
      ctx.font = '16px Arial';
      ctx.fillText(`Width: ${rect.width}px`, rect.x, rect.y - 5);
      
      // Update state with width
      setObjectWidth(rect.width);
    }

    // Clean up
    src.delete();
    gray.delete();
    blurred.delete();
    thresholded.delete();
    contours.delete();
    hierarchy.delete();

    // Continue processing
    if (processing) {
      requestAnimationFrame(processVideo);
    }
  };

  // Stop processing
  const stopProcessing = () => {
    setProcessing(false);
    setIsWebcamEnabled(false);
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }
  };

  return (
    <div className="object-detection-container">
      <h1>Object Width Detection with OpenCV.js</h1>
      <p>Hold an object in your hand to detect its width in pixels.</p>
      
      <div className="webcam-container">
        <video 
          ref={videoRef} 
          className="webcam" 
          autoPlay 
          playsInline 
          style={{ display: isWebcamEnabled ? 'block' : 'none' }} 
        />
        <canvas 
          ref={canvasRef} 
          className="canvas" 
          style={{ display: isWebcamEnabled ? 'block' : 'none' }} 
        />
        
        {!isWebcamEnabled && (
          <button 
            onClick={enableWebcam} 
            disabled={!isOpenCVLoaded}
            className="enable-webcam-btn"
          >
            {isOpenCVLoaded ? 'Enable Webcam' : 'Loading OpenCV...'}
          </button>
        )}
        
        {isWebcamEnabled && (
          <button onClick={stopProcessing} className="stop-webcam-btn">
            Stop Webcam
          </button>
        )}
      </div>
      
      <div className="results-container">
        <h2>Detection Results</h2>
        {objectWidth !== null ? (
          <div className="width-display">
            <h3>Object Width:</h3>
            <div className="width-value">{objectWidth} pixels</div>
          </div>
        ) : (
          <p>No object detected. Hold an object in front of the camera.</p>
        )}
        
        <div className="tips">
          <h3>Tips for better detection:</h3>
          <ul>
            <li>Use a solid-colored background</li>
            <li>Ensure good lighting</li>
            <li>Hold the object with contrast to your hand</li>
            <li>Keep the object still for a moment</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ObjectDetection;