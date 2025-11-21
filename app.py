import os
import time
import cv2
import base64
import numpy as np
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['IMAGE_DIR'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../images'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables for video stream and search results
current_frame = None
search_results = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def resize_image(img, max_dim):
    """Resize image if larger than max_dim."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img

def match_features(des1, des2, matcher, ratio_threshold=0.7):
    """Feature matching with Lowe's ratio test."""
    if des1 is None or des2 is None:
        return []
    try:
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        return [m for m, n in raw_matches if m.distance < ratio_threshold * n.distance]
    except:
        return []

def image_to_base64(img):
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def run_feature_search(query_image_data, search_id, use_sift=False):
    """Run feature search in background thread with real-time updates."""
    global current_frame
    try:
        # Decode base64 image
        header, encoded = query_image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        query_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if query_img is None:
            search_results[search_id] = {'status': 'error', 'message': 'Could not decode query image'}
            return
        
        # Keep original color for visualization
        query_img_resized = resize_image(query_img, 800)
        query_img_gray = cv2.cvtColor(query_img_resized, cv2.COLOR_BGR2GRAY)
        
        # Initialize ORB detector
        detector_orb = cv2.ORB_create(nfeatures=500)
        matcher_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        query_kp_orb, query_des_orb = detector_orb.detectAndCompute(query_img_gray, None)
        
        if query_des_orb is None:
            search_results[search_id] = {'status': 'error', 'message': 'No features detected in query image'}
            return
        
        # Get all image files
        image_files = [os.path.join(app.config['IMAGE_DIR'], f) for f in os.listdir(app.config['IMAGE_DIR'])
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        search_results[search_id] = {
            'status': 'processing',
            'total': len(image_files),
            'processed': 0
        }
        
        # Progress callback for real-time visualization
        def update_frame(img):
            global current_frame
            ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                current_frame = buffer.tobytes()
        
        # Process images with real-time updates
        top_matches = []
        for idx, filepath in enumerate(image_files):
            img = cv2.imread(filepath)
            if img is None:
                continue
            
            img_resized = resize_image(img, 800)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            kp, des = detector_orb.detectAndCompute(img_gray, None)
            
            if des is not None:
                matches = match_features(query_des_orb, des, matcher_orb, 0.75)
                score = len(matches)
                
                # Only keep matches with score >= 8
                if score >= 8:
                    # Create visualization for real-time display (in color) - center query image
                    h1, w1 = query_img_resized.shape[:2]
                    h2, w2 = img_resized.shape[:2]
                    max_height = max(h1, h2)
                    vis = np.zeros((max_height, w1 + w2, 3), dtype=np.uint8)
                    
                    # Place query image (centered vertically)
                    y_offset_query = (max_height - h1) // 2
                    vis[y_offset_query:y_offset_query + h1, :w1] = query_img_resized
                    
                    # Place matched image (centered vertically)
                    y_offset_match = (max_height - h2) // 2
                    vis[y_offset_match:y_offset_match + h2, w1:w1+w2] = img_resized
                    
                    # Draw matches with green lines (adjust for centering)
                    for m in matches[:20]:
                        pt1 = (int(query_kp_orb[m.queryIdx].pt[0]), int(query_kp_orb[m.queryIdx].pt[1]) + y_offset_query)
                        pt2 = (int(kp[m.trainIdx].pt[0] + w1), int(kp[m.trainIdx].pt[1]) + y_offset_match)
                        cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
                    
                    update_frame(vis)
                    
                    top_matches.append((score, filepath, os.path.basename(filepath), img_resized))
                    top_matches.sort(key=lambda x: x[0], reverse=True)
                    top_matches = top_matches[:15]
            
            search_results[search_id]['processed'] = idx + 1
            time.sleep(0.003)  # Reduced delay for higher FPS
        
        # Final results
        if top_matches:
            best_score, best_path, best_name, best_img = top_matches[0]
            
            # Create final visualization (in color) - center images without white padding
            final_img_gray = cv2.cvtColor(best_img, cv2.COLOR_BGR2GRAY)
            final_kp, final_des = detector_orb.detectAndCompute(final_img_gray, None)
            final_matches = match_features(query_des_orb, final_des, matcher_orb, 0.75)
            
            # Get dimensions
            h1, w1 = query_img_resized.shape[:2]
            h2, w2 = best_img.shape[:2]
            
            # Calculate max height and create visualization with black background
            max_height = max(h1, h2)
            final_vis = np.zeros((max_height, w1 + w2, 3), dtype=np.uint8)
            
            # Place query image (centered vertically)
            y_offset_query = (max_height - h1) // 2
            final_vis[y_offset_query:y_offset_query + h1, :w1] = query_img_resized
            
            # Place matched image (centered vertically)
            y_offset_match = (max_height - h2) // 2
            final_vis[y_offset_match:y_offset_match + h2, w1:w1+w2] = best_img
            
            # Draw matches (adjust y coordinates for centering)
            for m in final_matches[:30]:
                pt1_x = int(query_kp_orb[m.queryIdx].pt[0])
                pt1_y = int(query_kp_orb[m.queryIdx].pt[1]) + y_offset_query
                pt2_x = int(final_kp[m.trainIdx].pt[0] + w1)
                pt2_y = int(final_kp[m.trainIdx].pt[1]) + y_offset_match
                cv2.line(final_vis, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 1)
            
            search_results[search_id] = {
                'status': 'completed',
                'best_match': {
                    'filename': best_name,
                    'score': best_score,
                    'match_image': image_to_base64(final_vis)
                },
                'top_matches': [
                    {'filename': name, 'score': sc, 'image': image_to_base64(img)}
                    for sc, _, name, img in top_matches[:10]
                ]
            }
        else:
            search_results[search_id] = {
                'status': 'completed',
                'best_match': None,
                'top_matches': [],
                'message': 'No results found (all matches below threshold of 8)'
            }
    except Exception as e:
        search_results[search_id] = {'status': 'error', 'message': str(e)}
        print(f"Search error: {e}")

def gen_frames():
    """Generate frames for video feed."""
    global current_frame
    while True:
        if current_frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
        time.sleep(0.03)  # ~33 FPS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({
            'success': True,
            'filename': filename,
            'url': f'/uploads/{filename}'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/search_features', methods=['POST'])
def search_features():
    """Start feature search on extracted image region."""
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    # Generate unique search ID
    search_id = f"search_{int(time.time() * 1000)}"
    
    # Get use_sift parameter (default False now)
    use_sift = data.get('use_sift', False)
    
    # Start search in background thread
    thread = threading.Thread(target=run_feature_search, args=(data['image'], search_id, use_sift))
    thread.daemon = True
    thread.start()
    
    return jsonify({'search_id': search_id})

@app.route('/search_status/<search_id>')
def search_status(search_id):
    """Get current status of feature search."""
    if search_id not in search_results:
        return jsonify({'status': 'not_found'}), 404
    
    return jsonify(search_results[search_id])

@app.route('/video_feed')
def video_feed():
    """Stream real-time visualization."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
