from flask import Flask, request, jsonify, render_template_string, render_template
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prediction_pipeline import PredictionPipeline
from data_ingestion import DataIngestion
from dashboard_service import get_dashboard_service
from kpi_service import get_kpi_metrics

app = Flask(__name__)

model_loaded = False
prediction_pipeline = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IMDB Movie Sentiment Analysis</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            min-height: 100vh; 
            padding: 20px; 
        }
        .container { max-width: 800px; margin: 0 auto; }
        
        /* Hero Section */
        .hero { 
            background: white; 
            border-radius: 20px; 
            padding: 60px 40px; 
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 30px;
        }
        .hero h1 {
            color: #667eea;
            font-size: 2.8rem;
            margin-bottom: 20px;
            font-weight: 700;
        }
        .hero p {
            color: #666;
            font-size: 1.1rem;
            line-height: 1.8;
            max-width: 600px;
            margin: 0 auto;
        }
        
        /* Input Form */
        .input-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .input-form { display: flex; gap: 15px; flex-wrap: wrap; }
        .input-field {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        .input-field:focus {
            outline: none;
            border-color: #667eea;
        }
        .predict-btn {
            padding: 15px 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .predict-btn:hover { transform: translateY(-2px); }
        .predict-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        
        /* Prediction Result */
        .result-card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-top: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            display: none;
        }
        .result-header {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        .sentiment-badge {
            padding: 10px 25px;
            border-radius: 25px;
            font-size: 1.2rem;
            font-weight: 600;
        }
        .sentiment-badge.positive { background: #e8f5e9; color: #2e7d32; }
        .sentiment-badge.negative { background: #ffebee; color: #c62828; }
        .sentiment-badge.neutral { background: #fff3e0; color: #ef6c00; }
        
        .confidence-section { text-align: center; }
        .confidence-label { color: #666; font-size: 0.9rem; margin-bottom: 8px; }
        .confidence-value { 
            font-size: 2rem; 
            font-weight: 700; 
            color: #667eea; 
        }
        
        .prob-bar {
            background: #f0f0f0;
            border-radius: 8px;
            height: 25px;
            margin: 8px 0;
            overflow: hidden;
        }
        .prob-fill {
            height: 100%;
            border-radius: 8px;
            transition: width 0.5s ease;
        }
        .prob-fill.positive { background: linear-gradient(90deg, #43e97b, #38f9d7); }
        .prob-fill.negative { background: linear-gradient(90deg, #f5576c, #f093fb); }
        .prob-fill.neutral { background: linear-gradient(90deg, #4facfe, #00f2fe); }
        
        .prob-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: #666;
            margin-top: 5px;
        }
        
        .text-display {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            color: #333;
            font-style: italic;
        }
        
        /* Nav */
        .nav-bar {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        .nav-btn {
            padding: 10px 25px;
            background: rgba(255,255,255,0.2);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 500;
            transition: background 0.3s;
        }
        .nav-btn:hover { background: rgba(255,255,255,0.3); }
        
        /* Loading */
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
            font-size: 1.1rem;
        }
        
        /* Error */
        .error-msg {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="nav-bar">
            <a href="/" class="nav-btn">Home</a>
            <a href="/dashboard-ui" class="nav-btn">Dashboard</a>
            <a href="/kpi-ui" class="nav-btn">KPI</a>
        </div>
        
        <div class="hero">
            <h1>🎬 IMDB Movie Sentiment Analysis</h1>
            <p>Analyze movie reviews using BERT Based ML for sentiment analysis. 
            Get insights into audience opinions by classifying reviews into Positive, Negative, or Neutral.</p>
        </div>
        
        <div class="input-card">
            <form id="predictForm" class="input-form">
                <input type="text" id="textInput" class="input-field" 
                       placeholder="Enter a movie review to analyze..." required>
                <button type="submit" class="predict-btn">Analyze</button>
            </form>
            <div id="loading" class="loading" style="display: none;">Analyzing your review...</div>
            <div id="error" class="error-msg" style="display: none;"></div>
        </div>
        
        <div id="result" class="result-card">
            <div class="text-display" id="result-text"></div>
            <div class="result-header">
                <span class="sentiment-badge" id="sentiment-badge">Positive</span>
            </div>
            <div class="confidence-section">
                <div class="confidence-label">Confidence Score</div>
                <div class="confidence-value" id="confidence-value">85%</div>
            </div>
            <div style="margin-top: 25px;">
                <div class="prob-label">
                    <span>Positive</span>
                    <span id="prob-positive">0%</span>
                </div>
                <div class="prob-bar"><div class="prob-fill positive" id="bar-positive" style="width: 0%"></div></div>
                
                <div class="prob-label">
                    <span>Negative</span>
                    <span id="prob-negative">0%</span>
                </div>
                <div class="prob-bar"><div class="prob-fill negative" id="bar-negative" style="width: 0%"></div></div>
                
                <div class="prob-label">
                    <span>Neutral</span>
                    <span id="prob-neutral">0%</span>
                </div>
                <div class="prob-bar"><div class="prob-fill neutral" id="bar-neutral" style="width: 0%"></div></div>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = document.getElementById('textInput').value.trim();
            if (!text) return;
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.querySelector('.predict-btn').disabled = true;
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });
                const data = await response.json();
                
                if (data.error) throw new Error(data.error);
                
                document.getElementById('result-text').textContent = '"' + text + '"';
                
                const badge = document.getElementById('sentiment-badge');
                badge.textContent = data.sentiment;
                badge.className = 'sentiment-badge ' + data.sentiment.toLowerCase();
                
                const probs = data.probabilities;
                const maxProb = Math.max(probs.Positive, probs.Negative, probs.Neutral);
                document.getElementById('confidence-value').textContent = (maxProb * 100).toFixed(1) + '%';
                
                document.getElementById('prob-positive').textContent = (probs.Positive * 100).toFixed(1) + '%';
                document.getElementById('bar-positive').style.width = (probs.Positive * 100) + '%';
                
                document.getElementById('prob-negative').textContent = (probs.Negative * 100).toFixed(1) + '%';
                document.getElementById('bar-negative').style.width = (probs.Negative * 100) + '%';
                
                document.getElementById('prob-neutral').textContent = (probs.Neutral * 100).toFixed(1) + '%';
                document.getElementById('bar-neutral').style.width = (probs.Neutral * 100) + '%';
                
                document.getElementById('result').style.display = 'block';
            } catch (error) {
                document.getElementById('error').style.display = 'block';
                document.getElementById('error').textContent = 'Error: ' + error.message;
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.querySelector('.predict-btn').disabled = false;
            }
        });
    </script>
</body>
</html>
"""


def init_model():
    global model_loaded, prediction_pipeline
    try:
        prediction_pipeline = PredictionPipeline()
        prediction_pipeline.load_model()
        model_loaded = True
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model - {e}")
        print("Please run training pipeline first: python src/training_pipeline.py")


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    global model_loaded, prediction_pipeline
    
    if not model_loaded:
        init_model()
    
    if not model_loaded or prediction_pipeline is None:
        return jsonify({
            'error': 'Model not loaded. Please run training pipeline first.',
            'sentiment': 'Unknown',
            'probabilities': {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        }), 500
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request'}), 400
    
    text = data['text']
    
    try:
        result = prediction_pipeline.predict(text)
        return jsonify({
            'text': result['text'],
            'sentiment': result['sentiment'],
            'probabilities': result['probabilities']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard', methods=['GET'])
def dashboard():
    try:
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        dashboard_service = get_dashboard_service()
        data = dashboard_service.get_dashboard_data(force_refresh=force_refresh)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/kpi', methods=['GET'])
def kpi():
    try:
        metrics = get_kpi_metrics()
        return jsonify({
            'success': True,
            'data': metrics
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/dashboard-ui')
def dashboard_ui():
    return render_template('dashboard.html')


@app.route('/kpi-ui')
def kpi_ui():
    return render_template('kpi.html')


@app.route('/artifacts/eda/<path:filename>')
def serve_eda_image(filename):
    from flask import send_from_directory
    import os
    eda_path = os.path.join(os.path.dirname(__file__), 'artifacts', 'eda')
    return send_from_directory(eda_path, filename)


if __name__ == '__main__':
    init_model()
    app.run(debug=True, port=5000)