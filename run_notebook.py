import subprocess
import sys

# Try to execute the notebook using nbconvert
try:
    result = subprocess.run([
        sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook',
        '--execute', '--stdout', 'sentiment_analysis_ml.ipynb'
    ], capture_output=True, text=True, cwd=r"C:\Users\nayak\OneDrive\Desktop\new2")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
except Exception as e:
    print(f"Error: {e}")
    # Alternative: use Python execution directly
    print("Trying alternative method...")