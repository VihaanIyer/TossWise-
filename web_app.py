"""
Web application for uploading bin layout images and selecting locations
"""
import os
import json
import time
import subprocess
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from gemini_classifier import TrashClassifier
from bin_layout_analyzer import BinLayoutAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Store running process PID
running_process_pid = None

# Create uploads directory if it doesn't exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Location options
LOCATIONS = {
    'atlanta_ga_usa': 'Atlanta, GA, USA',
    'budapest_hungary': 'Budapest, Hungary',
    'hong_kong': 'Hong Kong, Hong Kong',
    'singapore': 'Singapore, Singapore'
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html', locations=LOCATIONS)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and location selection"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        location = request.form.get('location')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not location or location not in LOCATIONS:
            return jsonify({'error': 'Invalid location selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, WEBP'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize classifier and analyzer
        print(f"Initializing classifier for location: {LOCATIONS[location]}")
        classifier = TrashClassifier()
        
        if not getattr(classifier, 'supports_vision', False):
            return jsonify({'error': 'Gemini vision model not available'}), 500
        
        analyzer = BinLayoutAnalyzer(classifier)
        
        # Analyze the uploaded image
        print(f"Analyzing bin layout from uploaded image...")
        image = Image.open(filepath)
        result = analyzer._analyze_image(image)
        
        # Add location metadata
        result['location'] = location
        result['location_name'] = LOCATIONS[location]
        
        # Save location-specific bin layout
        location_file = f"bin_layout_{location}.json"
        location_path = Path(location_file)
        location_path.write_text(json.dumps(result, indent=2))
        
        # Also update the main bin_layout_metadata.json (for backward compatibility)
        main_path = Path("bin_layout_metadata.json")
        main_path.write_text(json.dumps(result, indent=2))
        
        print(f"✅ Bin layout saved to {location_file} and bin_layout_metadata.json")
        print(f"📝 To use this configuration in main.py, set environment variable:")
        print(f"   export BIN_LOCATION={location}")
        print(f"   Or update .env file with: BIN_LOCATION={location}")
        
        return jsonify({
            'success': True,
            'message': f'Bin layout analyzed and saved for {LOCATIONS[location]}',
            'bins': result.get('bins', []),
            'scene': result.get('scene', ''),
            'location': location,
            'location_name': LOCATIONS[location],
            'location_file': location_file,
            'raw_response': result.get('raw_response', '')  # Include for editing
        })
        
    except Exception as e:
        print(f"Error processing upload: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/locations', methods=['GET'])
def get_locations():
    """Get list of available locations"""
    return jsonify({'locations': LOCATIONS})

@app.route('/layout/<location>', methods=['GET'])
def get_layout(location):
    """Get bin layout for a specific location"""
    if location not in LOCATIONS:
        return jsonify({'error': 'Invalid location'}), 404
    
    location_file = f"bin_layout_{location}.json"
    location_path = Path(location_file)
    
    if not location_path.exists():
        return jsonify({'error': 'Layout not found for this location'}), 404
    
    try:
        with open(location_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'Error reading layout: {str(e)}'}), 500

@app.route('/setup', methods=['POST'])
def setup_bins():
    """Finalize bin setup and notify main software to reload"""
    try:
        data = request.json
        location = data.get('location')
        bins = data.get('bins', [])
        scene = data.get('scene', '')
        
        if not location or location not in LOCATIONS:
            return jsonify({'error': 'Invalid location'}), 400
        
        if not bins:
            return jsonify({'error': 'No bins provided'}), 400
        
        # Save final configuration
        location_file = f"bin_layout_{location}.json"
        result = {
            'bins': bins,
            'location': location,
            'location_name': LOCATIONS[location],
            'scene': scene,
            'raw_response': data.get('raw_response', '')
        }
        
        location_path = Path(location_file)
        location_path.write_text(json.dumps(result, indent=2))
        
        # Also update main metadata file
        main_path = Path("bin_layout_metadata.json")
        main_path.write_text(json.dumps(result, indent=2))
        
        # Create reload signal file to notify main software
        signal_path = Path("reload_signal.txt")
        signal_path.write_text(str(time.time()))
        
        print(f"✅ Bin layout saved to {location_file}")
        print(f"📡 Reload signal sent to main software")
        print(f"📝 Main software will reload with {len(bins)} bins")
        
        return jsonify({
            'success': True,
            'message': f'Bin layout saved! Main software will reload with {len(bins)} bins.',
            'location': location,
            'location_name': LOCATIONS[location],
            'bins_count': len(bins)
        })
        
    except Exception as e:
        print(f"Error in setup: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error saving configuration: {str(e)}'}), 500

@app.route('/start', methods=['POST'])
def start_main_system():
    """Start the main.py system"""
    try:
        data = request.json
        location = data.get('location')
        
        if not location or location not in LOCATIONS:
            return jsonify({'error': 'Invalid location'}), 400
        
        # Check if main.py is already running
        # Simple check: look for main.py process
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'python.*main.py'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return jsonify({
                    'success': False,
                    'message': 'Main system is already running!',
                    'already_running': True
                })
        except Exception:
            # pgrep might not be available on all systems, continue anyway
            pass
        
        # Get the directory where main.py is located
        script_dir = Path(__file__).parent.absolute()
        main_py_path = script_dir / 'main.py'
        
        if not main_py_path.exists():
            return jsonify({'error': 'main.py not found'}), 404
        
        # Set environment variables
        env = os.environ.copy()
        env['BIN_LOCATION'] = location
        
        # Start main.py in a new process
        # Use subprocess.Popen to run in background
        try:
            # Determine Python executable
            python_exe = sys.executable
            
            # Start the process
            # Note: We don't capture stdout/stderr so OpenCV windows can open properly
            # The output will appear in the terminal where web_app.py is running
            if sys.platform == 'darwin':  # macOS
                # On macOS, use open to run in a new terminal window if needed
                process = subprocess.Popen(
                    [python_exe, str(main_py_path)],
                    cwd=str(script_dir),
                    env=env,
                    start_new_session=True
                )
            else:
                # On Linux/Windows, run directly
                process = subprocess.Popen(
                    [python_exe, str(main_py_path)],
                    cwd=str(script_dir),
                    env=env,
                    stdout=None,  # Don't capture - let it print to terminal
                    stderr=None,  # Don't capture - let it print to terminal
                    start_new_session=True  # Detach from parent process
                )
            
            print(f"✅ Started main.py (PID: {process.pid}) with BIN_LOCATION={location}")
            
            # Store the PID globally so we can stop it later
            global running_process_pid
            running_process_pid = process.pid
            
            return jsonify({
                'success': True,
                'message': f'Main system started! Camera will open shortly.',
                'pid': process.pid,
                'location': location
            })
            
        except Exception as e:
            print(f"Error starting main.py: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error starting main system: {str(e)}'}), 500
        
    except Exception as e:
        print(f"Error in start: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/testing')
def testing_page():
    """Show testing in progress page"""
    return render_template('testing.html')

@app.route('/stop', methods=['POST'])
def stop_main_system():
    """Stop the main.py system"""
    global running_process_pid
    try:
        if running_process_pid is None:
            # Try to find the process
            try:
                result = subprocess.run(
                    ['pgrep', '-f', 'python.*main.py'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    pids = result.stdout.strip().split('\n')
                    if pids:
                        running_process_pid = int(pids[0])
            except Exception:
                pass
        
        if running_process_pid:
            try:
                # Try graceful termination first
                os.kill(running_process_pid, 15)  # SIGTERM
                time.sleep(1)
                # Check if still running, force kill if needed
                try:
                    os.kill(running_process_pid, 0)  # Check if process exists
                    os.kill(running_process_pid, 9)  # SIGKILL
                except ProcessLookupError:
                    pass  # Process already terminated
                
                print(f"✅ Stopped main.py (PID: {running_process_pid})")
                running_process_pid = None
                return jsonify({
                    'success': True,
                    'message': 'Main system stopped successfully.'
                })
            except ProcessLookupError:
                running_process_pid = None
                return jsonify({
                    'success': True,
                    'message': 'Main system was not running.'
                })
            except Exception as e:
                print(f"Error stopping main.py: {e}")
                return jsonify({'error': f'Error stopping system: {str(e)}'}), 500
        else:
            return jsonify({
                'success': False,
                'message': 'No running process found.'
            })
    except Exception as e:
        print(f"Error in stop: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    import socket
    
    def find_free_port(start_port=8080):
        """Find a free port starting from start_port"""
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return start_port  # Fallback
    
    print("Starting bin layout web app...")
    print("Available locations:", list(LOCATIONS.values()))
    port = int(os.environ.get('PORT', find_free_port(8080)))
    
    # Get local IP address for network access
    def get_local_ip():
        """Get the local IP address of this machine"""
        try:
            # Connect to a remote address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # Google DNS
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"
    
    local_ip = get_local_ip()
    print(f"\n{'='*60}")
    print(f"🌐 Web App Accessible On:")
    print(f"   Local:  http://localhost:{port}")
    print(f"   Network: http://{local_ip}:{port}")
    print(f"{'='*60}")
    print(f"\n📱 To access from iPad:")
    print(f"   1. Make sure iPad is on the same Wi-Fi network")
    print(f"   2. Open Safari on iPad")
    print(f"   3. Go to: http://{local_ip}:{port}")
    print(f"{'='*60}\n")
    
    app.run(debug=True, host='0.0.0.0', port=port)

