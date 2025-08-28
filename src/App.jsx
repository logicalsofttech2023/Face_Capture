import React, { useEffect, useState } from "react";
import "./App.css";

// Import eye images (you'll need to add these to your project)
// import coverLeftEye from "./assets/cover-left-eye.png";
// import coverRightEye from "./assets/cover-right-eye.png";

const App = () => {
  const [testState, setTestState] = useState("notStarted"); // notStarted, instructions, inProgress, rightEyeInstructions, finished
  const [activeEye, setActiveEye] = useState(null); // 'left' or 'right'
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [testData, setTestData] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const API_BASE_URL = "http://localhost:6005/api/user";
  const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY4ODhiMTIwNTI4Y2ZjMzQ2MTMzMzc2OCIsInBob25lIjoiOTg3NjU0MzIxMCIsInJvbGUiOiJ1c2VyIiwiaWF0IjoxNzU2MzcwMDE4LCJleHAiOjE3NTY5NzQ4MTh9.uGwZeZPTXp0AF6KcZ8pW4AkFyRcKWro38ixPCHoj-3s";

  // Function to start the test
  const startTest = async (eye) => {
    setLoading(true);
    setMessage("");
    try {
      const response = await fetch(`${API_BASE_URL}/startTest`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ activeEye: eye }),
      });

      const data = await response.json();
      
      if (data.success) {
        setActiveEye(eye);
        setTestState("inProgress");
        setTestData(data.data);
        getQuestion(); // Get the first question
      } else {
        setMessage(data.message);
      }
    } catch (error) {
      setMessage("Error starting test: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Function to get the current question
  const getQuestion = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/getQuestion`, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await response.json();
      
      if (data.success) {
        if (data.finished) {
          if (activeEye === "left") {
            // Left eye finished, show right eye instructions
            setTestState("rightEyeInstructions");
          } else {
            setTestState("finished");
            getResults();
          }
        } else {
          // Use the mock function since the API doesn't return the question data
          const mockQuestion = getQuestionForSize(data.currentStep);
          setCurrentQuestion({...mockQuestion, step: data.currentStep});
        }
      } else {
        setMessage(data.message);
      }
    } catch (error) {
      setMessage("Error fetching question: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Function to submit an answer
  const submitAnswer = async (selectedOption, correct) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/submitAnswer`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          selectedOption,
          correct,
          size: currentQuestion.size,
        }),
      });

      const data = await response.json();
      
      if (data.success) {
        if (data.eyeFinished) {
          if (activeEye === "left") {
            // Left eye finished, show right eye instructions
            setTestState("rightEyeInstructions");
          } else {
            setTestState("finished");
            getResults();
          }
        } else {
          getQuestion(); // Get the next question
        }
      } else {
        setMessage(data.message);
      }
    } catch (error) {
      setMessage("Error submitting answer: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Function to get results
  const getResults = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/getResult`, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await response.json();
      
      if (data.success) {
        if (data.completed) {
          setResults(data.result);
        } else {
          setMessage("Test not completed yet");
        }
      } else {
        setMessage(data.message);
      }
    } catch (error) {
      setMessage("Error fetching results: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Function to get question based on size (mock implementation)
  const getQuestionForSize = (step) => {
    // This should match your backend implementation
    const questions = [
      { size: "7", options: ["E", "F", "P", "T", "O"] },
      { size: "6", options: ["T", "O", "Z", "L", "E"] },
      { size: "5", options: ["L", "P", "E", "D", "F"] },
      { size: "4", options: ["E", "D", "F", "C", "Z"] },
      { size: "3", options: ["F", "E", "L", "O", "D"] },
      { size: "2", options: ["D", "E", "F", "P", "T"] },
      { size: "1", options: ["F", "P", "T", "O", "Z"] },
    ];
    
    return questions[step - 1] || questions[0];
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Visual Acuity Test</h1>
        <p>Measure your vision accuracy from the comfort of your home</p>
      </header>
      
      {message && (
        <div className={`message ${message.includes("Error") ? "error" : "info"}`}>
          {message}
          <button onClick={() => setMessage("")} className="close-btn">×</button>
        </div>
      )}
      
      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p>Processing...</p>
        </div>
      )}
      
      {testState === "notStarted" && (
        <div className="test-intro">
          <div className="instructions-card">
            <h2>Welcome to the Visual Acuity Test</h2>
            <p>This test will help determine the clarity of your vision. Please follow these instructions carefully:</p>
            
            <div className="instructions-list">
              <div className="instruction-item">
                <div className="instruction-icon">👁️</div>
                <div className="instruction-text">
                  <h3>Test One Eye at a Time</h3>
                  <p>We'll test each eye separately. You'll need to cover one eye while testing the other.</p>
                </div>
              </div>
              
              <div className="instruction-item">
                <div className="instruction-icon">📏</div>
                <div className="instruction-text">
                  <h3>Maintain Proper Distance</h3>
                  <p>Sit approximately 3-4 feet away from your screen for accurate results.</p>
                </div>
              </div>
              
              <div className="instruction-item">
                <div className="instruction-icon">💡</div>
                <div className="instruction-text">
                  <h3>Ensure Good Lighting</h3>
                  <p>Make sure your room is well-lit without glare on your screen.</p>
                </div>
              </div>
              
              <div className="instruction-item">
                <div className="instruction-icon">🔍</div>
                <div className="instruction-text">
                  <h3>Identify Letters</h3>
                  <p>You'll be shown letters of decreasing size. Select the letter you see.</p>
                </div>
              </div>
            </div>
            
            <div className="action-buttons">
              <button 
                className="primary-button"
                onClick={() => setTestState("eyeSelection")}
              >
                Start Test
              </button>
            </div>
          </div>
        </div>
      )}
      
      {testState === "eyeSelection" && (
        <div className="eye-selection">
          <div className="selection-card">
            <h2>Select Which Eye to Test First</h2>
            <p>We recommend testing your weaker eye first if you know which one it is.</p>
            
            <div className="eye-options">
              <div className="eye-option" onClick={() => setTestState("leftEyeInstructions")}>
                <div className="eye-icon">👁️</div>
                <h3>Left Eye</h3>
                <p>Cover your right eye</p>
              </div>
              
              <div className="eye-option" onClick={() => setTestState("rightEyeInstructions")}>
                <div className="eye-icon">👁️</div>
                <h3>Right Eye</h3>
                <p>Cover your left eye</p>
              </div>
            </div>
            
            <button 
              className="secondary-button"
              onClick={() => setTestState("notStarted")}
            >
              Back to Instructions
            </button>
          </div>
        </div>
      )}
      
      {testState === "leftEyeInstructions" && (
        <div className="test-intro">
          <div className="instructions-card">
            <h2>Left Eye Test</h2>
            <p>Please cover your RIGHT eye as shown in the image below</p>
            
            <div className="eye-cover-demo">
              <div className="cover-image">
                {/* Replace with actual image import */}
                <div className="demo-image placeholder">
                  <div className="face-outline">
                    <div className="left-eye open"></div>
                    <div className="right-eye covered">
                      <div className="cover-hand">✋</div>
                    </div>
                  </div>
                </div>
                <p>Cover your RIGHT eye</p>
              </div>
            </div>
            
            <div className="action-buttons">
              <button 
                className="primary-button"
                onClick={() => startTest("left")}
              >
                Start Left Eye Test
              </button>
              <button 
                className="secondary-button"
                onClick={() => setTestState("eyeSelection")}
              >
                Back
              </button>
            </div>
          </div>
        </div>
      )}
      
      {testState === "rightEyeInstructions" && activeEye === "left" && (
        <div className="test-intro">
          <div className="instructions-card">
            <h2>Left Eye Test Completed</h2>
            <p>Great job! You've completed the left eye test. Now it's time to test your right eye.</p>
            
            <div className="eye-cover-demo">
              <div className="cover-image">
                {/* Replace with actual image import */}
                <div className="demo-image placeholder">
                  <div className="face-outline">
                    <div className="left-eye covered">
                      <div className="cover-hand">✋</div>
                    </div>
                    <div className="right-eye open"></div>
                  </div>
                </div>
                <p>Cover your LEFT eye</p>
              </div>
            </div>
            
            <div className="action-buttons">
              <button 
                className="primary-button"
                onClick={() => startTest("right")}
              >
                Start Right Eye Test
              </button>
            </div>
          </div>
        </div>
      )}
      
      {testState === "rightEyeInstructions" && !activeEye && (
        <div className="test-intro">
          <div className="instructions-card">
            <h2>Right Eye Test</h2>
            <p>Please cover your LEFT eye as shown in the image below</p>
            
            <div className="eye-cover-demo">
              <div className="cover-image">
                {/* Replace with actual image import */}
                <div className="demo-image placeholder">
                  <div className="face-outline">
                    <div className="left-eye covered">
                      <div className="cover-hand">✋</div>
                    </div>
                    <div className="right-eye open"></div>
                  </div>
                </div>
                <p>Cover your LEFT eye</p>
              </div>
            </div>
            
            <div className="action-buttons">
              <button 
                className="primary-button"
                onClick={() => startTest("right")}
              >
                Start Right Eye Test
              </button>
              <button 
                className="secondary-button"
                onClick={() => setTestState("eyeSelection")}
              >
                Back
              </button>
            </div>
          </div>
        </div>
      )}
      
      {testState === "inProgress" && currentQuestion && (
        <div className="test-interface">
          <div className="test-header">
            <div className="progress-indicator">
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{width: `${(currentQuestion.step / 7) * 100}%`}}
                ></div>
              </div>
              <span>Step {currentQuestion.step} of 7</span>
            </div>
            
            <div className="current-eye">
              Testing your <span className="eye-name">{activeEye}</span> eye
            </div>
          </div>
          
          <div className="question-container">
            <h2>What letter do you see?</h2>
            
            <div className="letter-card">
              <div className={`letter-display size-${currentQuestion.size}`}>
                <div className="letter-placeholder">
                  {currentQuestion.options[0]}
                </div>
              </div>
            </div>
            
            <div className="options-grid">
              {currentQuestion.options.map((option, index) => (
                <button 
                  key={index} 
                  className="option-button"
                  onClick={() => submitAnswer(option, true)}
                  disabled={loading}
                >
                  {option}
                </button>
              ))}
            </div>
            
            <button 
              className="secondary-button"
              onClick={() => submitAnswer("", false)} 
              disabled={loading}
            >
              I can't read this
            </button>
          </div>
        </div>
      )}
      
      {testState === "finished" && results && (
        <div className="results-container">
          <div className="results-card">
            <div className="results-header">
              <h2>Test Results</h2>
              <p>Your visual acuity scores</p>
            </div>
            
            <div className="results-content">
              <div className="vision-scores">
                <div className="score-card left-eye">
                  <h3>Left Eye</h3>
                  <div className="score-value">{results.leftEye || "N/A"}</div>
                  <div className="score-label">Visual Acuity</div>
                  {results.interpretation && (
                    <div className="score-interpretation">
                      <p className="status">{results.interpretation.leftEye.status}</p>
                      <p className="note">{results.interpretation.leftEye.note}</p>
                    </div>
                  )}
                </div>
                
                <div className="score-card right-eye">
                  <h3>Right Eye</h3>
                  <div className="score-value">{results.rightEye || "N/A"}</div>
                  <div className="score-label">Visual Acuity</div>
                  {results.interpretation && (
                    <div className="score-interpretation">
                      <p className="status">{results.interpretation.rightEye.status}</p>
                      <p className="note">{results.interpretation.rightEye.note}</p>
                    </div>
                  )}
                </div>
              </div>
              
              {results.interpretation && (
                <div className="results-interpretation">
                  <h3>Overall Assessment</h3>
                  <p className="conclusion">{results.interpretation.overall.conclusion}</p>
                  <p className="recommendation">{results.interpretation.overall.recommendation}</p>
                </div>
              )}
            </div>
            
            <div className="results-actions">
              <button 
                className="primary-button"
                onClick={() => {
                  setTestState("notStarted");
                  setResults(null);
                }}
              >
                Take Test Again
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;