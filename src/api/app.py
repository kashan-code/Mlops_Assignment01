import logging
import os

import requests
from flask import Flask, jsonify, render_template_string, request

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:5000")

# HTML template for Academic Stress Prediction Interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Academic Stress Level Predictor</title>
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            max-width: 900px; 
            margin: 0 auto; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 { 
            color: #4a5568; 
            text-align: center; 
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .form-group { 
            margin-bottom: 20px; 
        }
        label { 
            display: block; 
            margin-bottom: 8px; 
            font-weight: 600;
            color: #2d3748;
        }
        input, select { 
            width: 100%; 
            padding: 12px; 
            margin-bottom: 10px; 
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus, select:focus {
            border-color: #667eea;
            outline: none;
        }
        button { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 15px 30px; 
            border: none; 
            cursor: pointer;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            width: 100%;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        .result { 
            background: #f7fafc; 
            padding: 20px; 
            margin-top: 30px; 
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }
        .stress-level {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .stress-1 { color: #38a169; } /* Green - Low stress */
        .stress-2 { color: #68d391; } /* Light green */
        .stress-3 { color: #f6ad55; } /* Orange - Moderate */
        .stress-4 { color: #fc8181; } /* Light red */
        .stress-5 { color: #e53e3e; } /* Red - High stress */
        .confidence {
            background: #edf2f7;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .info-section {
            background: #e6fffa;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            border-left: 5px solid #38b2ac;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎓 Academic Stress Level Predictor</h1>
        
        <div class="info-section">
            <h3>📊 About This Tool</h3>
            <p>This AI model predicts academic stress levels based on various factors affecting student life. 
            It uses machine learning to analyze patterns in student responses and provide insights into stress management.</p>
        </div>

        <form id="predictionForm">
            <div class="form-group">
                <label for="academic_stage">🎯 Academic Stage:</label>
                <select id="academic_stage" name="academic_stage" required>
                    <option value="">Select your academic stage</option>
                    <option value="high school">High School</option>
                    <option value="undergraduate">Undergraduate</option>
                    <option value="graduate">Graduate/Postgraduate</option>
                </select>
            </div>

            <div class="form-group">
                <label for="peer_pressure">👥 Peer Pressure (1-5 scale):</label>
                <select id="peer_pressure" name="peer_pressure" required>
                    <option value="">Rate peer pressure level</option>
                    <option value="1">1 - Very Low</option>
                    <option value="2">2 - Low</option>
                    <option value="3">3 - Moderate</option>
                    <option value="4">4 - High</option>
                    <option value="5">5 - Very High</option>
                </select>
            </div>

            <div class="form-group">
                <label for="family_pressure">🏠 Family Pressure (1-5 scale):</label>
                <select id="family_pressure" name="family_pressure" required>
                    <option value="">Rate family pressure level</option>
                    <option value="1">1 - Very Low</option>
                    <option value="2">2 - Low</option>
                    <option value="3">3 - Moderate</option>
                    <option value="4">4 - High</option>
                    <option value="5">5 - Very High</option>
                </select>
            </div>

            <div class="form-group">
                <label for="study_environment">📚 Study Environment:</label>
                <select id="study_environment" name="study_environment" required>
                    <option value="">Select your study environment</option>
                    <option value="Peaceful">Peaceful</option>
                    <option value="disrupted">Disrupted</option>
                    <option value="moderate">Moderate</option>
                </select>
            </div>

            <div class="form-group">
                <label for="coping_strategy">🧠 Coping Strategy:</label>
                <select id="coping_strategy" name="coping_strategy" required>
                    <option value="">Select your coping strategy</option>
                    <option value="Analyze the situation and handle it with intellect">Analytical/Intellectual approach</option>
                    <option value="Emotional breakdown (crying a lot)">Emotional response</option>
                    <option value="Physical exercise">Physical exercise</option>
                </select>
            </div>

            <div class="form-group">
                <label for="bad_habits">🚭 Bad Habits (smoking, drinking regularly):</label>
                <select id="bad_habits" name="bad_habits" required>
                    <option value="">Do you have bad habits?</option>
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>

            <div class="form-group">
                <label for="academic_competition">🏆 Academic Competition (1-5 scale):</label>
                <select id="academic_competition" name="academic_competition" required>
                    <option value="">Rate academic competition level</option>
                    <option value="1">1 - Very Low</option>
                    <option value="2">2 - Low</option>
                    <option value="3">3 - Moderate</option>
                    <option value="4">4 - High</option>
                    <option value="5">5 - Very High</option>
                </select>
            </div>

            <button type="submit">🔮 Predict Stress Level</button>
        </form>

        <div id="result" class="result" style="display: none;"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData);
            
            // Convert numeric fields
            data.peer_pressure = parseInt(data.peer_pressure);
            data.family_pressure = parseInt(data.family_pressure);
            data.academic_competition = parseInt(data.academic_competition);
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    const stressLevel = result.predicted_stress_level;
                    const confidence = (result.confidence * 100).toFixed(1);
                    
                    document.getElementById('result').innerHTML = `
                        <h3>📊 Prediction Results</h3>
                        <div class="stress-level stress-${stressLevel}">
                            Predicted Stress Level: ${stressLevel}/5
                        </div>
                        <div class="confidence">
                            <strong>Confidence:</strong> ${confidence}%
                        </div>
                        <div style="margin: 15px 0;">
                            <strong>Interpretation:</strong><br>
                            ${result.interpretation}
                        </div>
                        <div style="margin-top: 15px;">
                            <strong>💡 Stress Management Tips:</strong><br>
                            ${getStressTips(stressLevel)}
                        </div>
                    `;
                } else {
                    document.getElementById('result').innerHTML = 
                        `<h3 style="color: red;">❌ Error: ${result.error}</h3>`;
                }
                
                document.getElementById('result').style.display = 'block';
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    `<h3 style="color: red;">❌ Error: ${error.message}</h3>`;
                document.getElementById('result').style.display = 'block';
            }
        });
        
        function getStressTips(stressLevel) {
            const tips = {
                1: "Great job managing your stress! Keep up the good work with regular breaks and self-care.",
                2: "You're doing well! Consider maintaining your current coping strategies.",
                3: "Moderate stress is normal. Try relaxation techniques, time management, and regular exercise.",
                4: "High stress detected. Consider talking to a counselor, improving study habits, and practicing mindfulness.",
                5: "Very high stress levels. Please consider seeking professional support and implementing stress reduction strategies immediately."
            };
            return tips[stressLevel] || "Focus on healthy coping strategies and seek support when needed.";
        }
    </script>
</body>
</html>
"""

@app.route("/health")
def health():
    """Health check endpoint"""
    return (
        jsonify({"status": "healthy", "service": "academic_stress_web_interface"}),
        200,
    )


@app.route("/")
def index():
    """Serve the main page"""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Proxy prediction request to model service"""
    try:
        data = request.get_json()

        # Forward request to model service
        response = requests.post(f"{MODEL_SERVICE_URL}/predict", json=data)

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Model service error"}), 500

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
