import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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

def draw_matches_white_padding(img1, kp1, img2, kp2, matches):
    """Draw matches with white padding."""
    h1, w1, h2, w2 = img1.shape[:2] + img2.shape[:2]
    max_height = max(h1, h2)
    is_color1, is_color2 = len(img1.shape) == 3, len(img2.shape) == 3
    
    # Create padded images
    def pad_image(img, h, is_color):
        if is_color:
            padded = np.ones((max_height, img.shape[1], 3), dtype=np.uint8) * 255
        else:
            padded = np.ones((max_height, img.shape[1]), dtype=np.uint8) * 255
        y_offset = (max_height - h) // 2
        padded[y_offset:y_offset+h, :] = img
        return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR) if not is_color else padded, y_offset
    
    padded_img1, y_offset1 = pad_image(img1, h1, is_color1)
    padded_img2, y_offset2 = pad_image(img2, h2, is_color2)
    
    # Adjust keypoints
    adjusted_kp1 = [cv2.KeyPoint(kp.pt[0], kp.pt[1] + y_offset1, kp.size) for kp in kp1]
    adjusted_kp2 = [cv2.KeyPoint(kp.pt[0], kp.pt[1] + y_offset2, kp.size) for kp in kp2]
    
    return cv2.drawMatches(padded_img1, adjusted_kp1, padded_img2, adjusted_kp2, 
                          matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def process_orb(filepath, query_des, detector, matcher, use_color=False, query_img=None, query_kp=None, progress_callback=None):
    """Stage 1: Fast ORB filtering."""
    filename = os.path.basename(filepath)
    img = cv2.imread(filepath)
    if img is None:
        return filename, 0, None, None, None, None
    
    if use_color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_resized = resize_image(img, 800)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY) if use_color else img_resized
    
    kp, des = detector.detectAndCompute(img_gray, None)
    matches = match_features(query_des, des, matcher, 0.75)
    
    if progress_callback and query_img is not None and query_kp is not None:
        try:
            # Create a visualization
            h1, w1 = query_img.shape[:2]
            h2, w2 = img_resized.shape[:2]
            vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            
            # Convert to BGR for display if needed (since we read as RGB/Gray)
            q_disp = cv2.cvtColor(query_img, cv2.COLOR_GRAY2BGR) if len(query_img.shape) == 2 else cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
            t_disp = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR) if len(img_resized.shape) == 2 else cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            
            vis[:h1, :w1] = q_disp
            vis[:h2, w1:w1+w2] = t_disp
            
            # Draw matches
            for m in matches:
                pt1 = (int(query_kp[m.queryIdx].pt[0]), int(query_kp[m.queryIdx].pt[1]))
                pt2 = (int(kp[m.trainIdx].pt[0] + w1), int(kp[m.trainIdx].pt[1]))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
            
            progress_callback(vis)
        except Exception as e:
            print(f"Visualization error: {e}")

    return filename, len(matches), filepath, img_resized, kp, matches

def process_sift(filepath, query_des, detector, matcher, use_color=False, query_img=None, query_kp=None, progress_callback=None):
    """Stage 2: Accurate SIFT refinement."""
    filename = os.path.basename(filepath)
    img = cv2.imread(filepath)
    if img is None:
        return filename, 0, None, None, None
    
    if use_color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_resized = resize_image(img, 1000)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY) if use_color else img_resized
    
    kp, des = detector.detectAndCompute(img_gray, None)
    matches = match_features(query_des, des, matcher, 0.7)

    if progress_callback and query_img is not None and query_kp is not None:
        try:
            # Create a visualization
            h1, w1 = query_img.shape[:2]
            h2, w2 = img_resized.shape[:2]
            vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            
            q_disp = cv2.cvtColor(query_img, cv2.COLOR_GRAY2BGR) if len(query_img.shape) == 2 else cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
            t_disp = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR) if len(img_resized.shape) == 2 else cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            
            vis[:h1, :w1] = q_disp
            vis[:h2, w1:w1+w2] = t_disp
            
            for m in matches:
                pt1 = (int(query_kp[m.queryIdx].pt[0]), int(query_kp[m.queryIdx].pt[1]))
                pt2 = (int(kp[m.trainIdx].pt[0] + w1), int(kp[m.trainIdx].pt[1]))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
            
            progress_callback(vis)
        except Exception as e:
            print(f"Visualization error: {e}")
    
    return filename, len(matches), img_resized, kp, matches

def search_image(query_path, image_dir, output_path, top_k=15, use_sift=True, use_color=True, progress_callback=None):
    """
    Executes the hybrid search and returns the results.
    output_path: Path to save the result visualization image.
    Returns: dict with search results and stats.
    """
    if not os.path.exists(query_path) or not os.path.exists(image_dir):
        return {"error": "Path not found"}

    # Read query image
    query_img = cv2.imread(query_path)
    if query_img is None:
        return {"error": "Could not read query image"}
    
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB) if use_color else cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    
    start_time = time.time()

    # Prepare query images
    query_orb = resize_image(cv2.cvtColor(query_img.copy(), cv2.COLOR_RGB2GRAY) if use_color else query_img.copy(), 800)
    query_sift = resize_image(cv2.cvtColor(query_img.copy(), cv2.COLOR_RGB2GRAY) if use_color else query_img.copy(), 1000)

    # Stage 1: ORB filtering
    detector_orb = cv2.ORB_create(nfeatures=500)
    matcher_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    query_kp_orb, query_des_orb = detector_orb.detectAndCompute(query_orb, None)

    # Get image files
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                   and os.path.abspath(os.path.join(image_dir, f)) != os.path.abspath(query_path)]
    
    # Process ORB
    orb_results = []
    # Always use parallel processing for speed
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_orb, fp, query_des_orb, detector_orb, matcher_orb, use_color, query_orb, query_kp_orb, progress_callback): fp for fp in image_files}
        orb_results = [(future.result()[0], future.result()[1], futures[future]) for future in as_completed(futures)]
    
    # Get top candidates
    orb_results.sort(key=lambda x: x[1], reverse=True)
    top_candidates = orb_results[:top_k]
    
    stage1_time = time.time() - start_time
    
    final_name, final_orb_score, final_path = top_candidates[0]
    
    # Stage 2: SIFT verification (only if enabled)
    sift_results = []
    if use_sift and len(top_candidates) > 1:
        detector_sift = cv2.SIFT_create(nfeatures=1000)
        matcher_sift = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        query_kp_sift, query_des_sift = detector_sift.detectAndCompute(query_sift, None)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_sift, filepath, query_des_sift, detector_sift, matcher_sift, use_color, query_sift, query_kp_sift, progress_callback): filename
                      for filename, _, filepath in top_candidates[1:]}
            sift_results = sorted([future.result() for future in as_completed(futures)], key=lambda x: x[1], reverse=True)
        
    # Visualization
    final_img_color = cv2.imread(final_path)
    final_img_color = cv2.cvtColor(final_img_color, cv2.COLOR_BGR2RGB)
    final_img_color = resize_image(final_img_color, 1000)
    
    query_img_color = cv2.imread(query_path)
    query_img_color = cv2.cvtColor(query_img_color, cv2.COLOR_BGR2RGB)
    query_img_color = resize_image(query_img_color, 1000)
    
    if use_sift:
        final_img = cv2.imread(final_path, cv2.IMREAD_GRAYSCALE)
        final_img = resize_image(final_img, 1000)
        detector_sift = cv2.SIFT_create(nfeatures=1000)
        matcher_sift = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        query_kp_sift, query_des_sift = detector_sift.detectAndCompute(query_sift, None)
        final_kp, final_des = detector_sift.detectAndCompute(final_img, None)
        final_matches = match_features(query_des_sift, final_des, matcher_sift, 0.7)
        match_type = "SIFT"
        query_kp_final = query_kp_sift
    else:
        final_img = cv2.imread(final_path, cv2.IMREAD_GRAYSCALE)
        final_img = resize_image(final_img, 800)
        final_kp, final_des = detector_orb.detectAndCompute(final_img, None)
        final_matches = match_features(query_des_orb, final_des, matcher_orb, 0.75)
        match_type = "ORB"
        query_kp_final = query_kp_orb

    total_time = time.time() - start_time
    
    # Draw result
    h1, w1, h2, w2 = query_img_color.shape[:2] + final_img_color.shape[:2]
    max_height = max(h1, h2)
    padded_img1, padded_img2 = np.ones((max_height, w1, 3), dtype=np.uint8) * 255, np.ones((max_height, w2, 3), dtype=np.uint8) * 255
    y_offset1, y_offset2 = (max_height - h1) // 2, (max_height - h2) // 2
    padded_img1[y_offset1:y_offset1+h1, :], padded_img2[y_offset2:y_offset2+h2, :] = query_img_color, final_img_color
    
    adjusted_kp1 = [cv2.KeyPoint(kp.pt[0], kp.pt[1] + y_offset1, kp.size) for kp in query_kp_final]
    adjusted_kp2 = [cv2.KeyPoint(kp.pt[0], kp.pt[1] + y_offset2, kp.size) for kp in final_kp]
    
    result_img = cv2.drawMatches(padded_img1, adjusted_kp1, padded_img2, adjusted_kp2, 
                                 final_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    
    return {
        "best_match": final_name,
        "score": len(final_matches),
        "match_type": match_type,
        "total_time": total_time,
        "orb_candidates": [{"name": n, "score": s} for n, s, _ in top_candidates],
        "sift_candidates": [{"name": n, "score": s} for n, s, _, _, _ in sift_results] if use_sift else []
    }
