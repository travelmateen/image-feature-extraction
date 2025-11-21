// Global variables
let currentImage = null;
let isDrawing = false;
let drawingPath = [];
let bboxPath = {};
let canvas, drawCanvas, bboxCanvas, extractedCanvas;
let ctx, drawCtx, bboxCtx, extractedCtx;
let imageLoaded = false;
let currentSearchId = null;
let searchPollInterval = null;

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    setupCanvas();

    // Auto-load Breaking Bad.jpg
    autoLoadDefaultImage();
});

function autoLoadDefaultImage() {
    const defaultImagePath = '/static/Breaking Bad.jpg';

    fetch(defaultImagePath)
        .then(response => {
            if (response.ok) {
                return response.blob();
            }
            throw new Error('Default image not found');
        })
        .then(blob => {
            const file = new File([blob], 'Breaking Bad.jpg', { type: 'image/jpeg' });
            handleFile(file);
        })
        .catch(error => {
            console.log('Default image not available:', error);
        });
}

function setupEventListeners() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const processBtn = document.getElementById('processBtn');

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Process button - select entire image and start search
    processBtn.addEventListener('click', () => {
        if (!imageLoaded) return;

        // Select entire image
        bboxPath = {
            x: 0,
            y: 0,
            width: canvas.width,
            height: canvas.height
        };
        drawBoundingBox(bboxPath);
        extractBboxRegion(bboxPath);
    });
}

function setupCanvas() {
    canvas = document.getElementById('imageCanvas');
    drawCanvas = document.getElementById('drawCanvas');
    bboxCanvas = document.getElementById('bboxCanvas');
    extractedCanvas = document.getElementById('extractedCanvas');

    ctx = canvas.getContext('2d');
    drawCtx = drawCanvas.getContext('2d');
    bboxCtx = bboxCanvas.getContext('2d');
    extractedCtx = extractedCanvas.getContext('2d');

    // Attach events to bboxCanvas (top layer)
    bboxCanvas.addEventListener('mousedown', startDrawing);
    bboxCanvas.addEventListener('mousemove', draw);
    bboxCanvas.addEventListener('mouseup', stopDrawing);
    bboxCanvas.addEventListener('mouseleave', stopDrawing);

    bboxCanvas.style.cursor = 'crosshair';
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }

    clearDrawing();
    imageLoaded = false;

    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                loadImage(data.url);
                document.getElementById('fileName').textContent = file.name;
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error uploading image');
        });
}

function loadImage(url) {
    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
        currentImage = img;

        const maxWidth = window.innerWidth * 0.8;
        const maxHeight = window.innerHeight * 0.5;
        let width = img.width;
        let height = img.height;

        const scale = Math.min(maxWidth / width, maxHeight / height, 1);
        width = width * scale;
        height = height * scale;

        canvas.width = width;
        canvas.height = height;
        drawCanvas.width = width;
        drawCanvas.height = height;
        bboxCanvas.width = width;
        bboxCanvas.height = height;

        ctx.drawImage(img, 0, 0, width, height);

        // Set container height to match canvas
        const canvasContainer = document.getElementById('canvasContainer');
        canvasContainer.style.height = height + 'px';
        canvasContainer.style.minHeight = height + 'px';

        document.getElementById('canvasContainer').style.display = 'block';
        document.getElementById('optionsSection').style.display = 'flex';
        document.getElementById('hintText').style.display = 'block';

        imageLoaded = true;
    };

    img.onerror = () => alert('Error loading image');
    img.src = url;
}

function startDrawing(e) {
    if (!imageLoaded) return;

    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    bboxCtx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
    drawingPath = [];
    bboxPath = {};

    isDrawing = true;
    const rect = bboxCanvas.getBoundingClientRect();

    // Account for zoom (65%)
    const zoom = 0.65;
    const x = (e.clientX - rect.left) / zoom;
    const y = (e.clientY - rect.top) / zoom;

    drawingPath = [{ x, y }];
    drawCtx.beginPath();
    drawCtx.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;

    const rect = bboxCanvas.getBoundingClientRect();

    // Account for zoom (65%)
    const zoom = 0.65;
    const x = (e.clientX - rect.left) / zoom;
    const y = (e.clientY - rect.top) / zoom;

    drawingPath.push({ x, y });

    drawCtx.lineTo(x, y);
    drawCtx.strokeStyle = '#ff00ff';
    drawCtx.lineWidth = 3;
    drawCtx.lineCap = 'round';
    drawCtx.lineJoin = 'round';
    drawCtx.stroke();
}

function stopDrawing() {
    if (!isDrawing) return;
    isDrawing = false;

    // Check if it's just a click (select whole image)
    if (drawingPath.length === 1 ||
        (drawingPath.length === 2 &&
            Math.abs(drawingPath[0].x - drawingPath[drawingPath.length - 1].x) < 5 &&
            Math.abs(drawingPath[0].y - drawingPath[drawingPath.length - 1].y) < 5)) {
        bboxPath = {
            x: 0,
            y: 0,
            width: canvas.width,
            height: canvas.height
        };
        drawBoundingBox(bboxPath);
        extractBboxRegion(bboxPath);
        drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    } else if (drawingPath.length > 2) {
        createBoundingBox();
    }
}

function createBoundingBox() {
    const xs = drawingPath.map(p => p.x);
    const ys = drawingPath.map(p => p.y);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const padding = 10;
    bboxPath = {
        x: Math.max(0, minX - padding),
        y: Math.max(0, minY - padding),
        width: Math.min(canvas.width - Math.max(0, minX - padding), maxX - minX + padding * 2),
        height: Math.min(canvas.height - Math.max(0, minY - padding), maxY - minY + padding * 2)
    };

    drawBoundingBox(bboxPath);
    extractBboxRegion(bboxPath);
}

function drawBoundingBox(bbox) {
    bboxCtx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);

    // Darken outside bbox with rounded corners
    const cornerRadius = Math.min(bbox.width, bbox.height) * 0.08;

    bboxCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    bboxCtx.fillRect(0, 0, bboxCanvas.width, bboxCanvas.height);

    // Clear bbox area with rounded corners
    bboxCtx.globalCompositeOperation = 'destination-out';
    bboxCtx.beginPath();
    bboxCtx.moveTo(bbox.x + cornerRadius, bbox.y);
    bboxCtx.lineTo(bbox.x + bbox.width - cornerRadius, bbox.y);
    bboxCtx.quadraticCurveTo(bbox.x + bbox.width, bbox.y, bbox.x + bbox.width, bbox.y + cornerRadius);
    bboxCtx.lineTo(bbox.x + bbox.width, bbox.y + bbox.height - cornerRadius);
    bboxCtx.quadraticCurveTo(bbox.x + bbox.width, bbox.y + bbox.height, bbox.x + bbox.width - cornerRadius, bbox.y + bbox.height);
    bboxCtx.lineTo(bbox.x + cornerRadius, bbox.y + bbox.height);
    bboxCtx.quadraticCurveTo(bbox.x, bbox.y + bbox.height, bbox.x, bbox.y + bbox.height - cornerRadius);
    bboxCtx.lineTo(bbox.x, bbox.y + cornerRadius);
    bboxCtx.quadraticCurveTo(bbox.x, bbox.y, bbox.x + cornerRadius, bbox.y);
    bboxCtx.closePath();
    bboxCtx.fill();
    bboxCtx.globalCompositeOperation = 'source-over';

    // Draw long corner indicators (extending inward)
    const cornerLength = Math.min(bbox.width, bbox.height) * 0.25; // Longer corners
    const cornerThickness = Math.max(4, Math.min(bbox.width, bbox.height) * 0.015); // Thicker

    bboxCtx.strokeStyle = '#ffffff';
    bboxCtx.lineWidth = cornerThickness;
    bboxCtx.lineCap = 'round';
    bboxCtx.lineJoin = 'round';

    // Top-left corner with rounded edge
    bboxCtx.beginPath();
    bboxCtx.moveTo(bbox.x, bbox.y + cornerLength);
    bboxCtx.lineTo(bbox.x, bbox.y + cornerRadius);
    bboxCtx.quadraticCurveTo(bbox.x, bbox.y, bbox.x + cornerRadius, bbox.y);
    bboxCtx.lineTo(bbox.x + cornerLength, bbox.y);
    bboxCtx.stroke();

    // Top-right corner with rounded edge
    bboxCtx.beginPath();
    bboxCtx.moveTo(bbox.x + bbox.width - cornerLength, bbox.y);
    bboxCtx.lineTo(bbox.x + bbox.width - cornerRadius, bbox.y);
    bboxCtx.quadraticCurveTo(bbox.x + bbox.width, bbox.y, bbox.x + bbox.width, bbox.y + cornerRadius);
    bboxCtx.lineTo(bbox.x + bbox.width, bbox.y + cornerLength);
    bboxCtx.stroke();

    // Bottom-right corner with rounded edge
    bboxCtx.beginPath();
    bboxCtx.moveTo(bbox.x + bbox.width, bbox.y + bbox.height - cornerLength);
    bboxCtx.lineTo(bbox.x + bbox.width, bbox.y + bbox.height - cornerRadius);
    bboxCtx.quadraticCurveTo(bbox.x + bbox.width, bbox.y + bbox.height, bbox.x + bbox.width - cornerRadius, bbox.y + bbox.height);
    bboxCtx.lineTo(bbox.x + bbox.width - cornerLength, bbox.y + bbox.height);
    bboxCtx.stroke();

    // Bottom-left corner with rounded edge
    bboxCtx.beginPath();
    bboxCtx.moveTo(bbox.x + cornerLength, bbox.y + bbox.height);
    bboxCtx.lineTo(bbox.x + cornerRadius, bbox.y + bbox.height);
    bboxCtx.quadraticCurveTo(bbox.x, bbox.y + bbox.height, bbox.x, bbox.y + bbox.height - cornerRadius);
    bboxCtx.lineTo(bbox.x, bbox.y + bbox.height - cornerLength);
    bboxCtx.stroke();
}

function extractBboxRegion(bbox) {
    if (!bbox.width || !bbox.height) return;

    // Show extracted region
    document.getElementById('extractedCard').style.display = 'block';

    const maxWidth = 300;
    const aspectRatio = bbox.height / bbox.width;
    let extractedWidth = Math.min(maxWidth, bbox.width);
    let extractedHeight = extractedWidth * aspectRatio;

    extractedCanvas.width = extractedWidth;
    extractedCanvas.height = extractedHeight;

    const imageData = ctx.getImageData(bbox.x, bbox.y, bbox.width, bbox.height);
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = bbox.width;
    tempCanvas.height = bbox.height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.putImageData(imageData, 0, 0);

    extractedCtx.drawImage(tempCanvas, 0, 0, extractedWidth, extractedHeight);

    // Start feature search
    startFeatureSearch(tempCanvas);
}

function clearDrawing() {
    if (drawCtx) drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    if (bboxCtx) bboxCtx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
    drawingPath = [];
    bboxPath = {};
    document.getElementById('extractedCard').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    stopFeatureSearch();
}

function startFeatureSearch(extractedCanvas) {
    stopFeatureSearch();

    const imageData = extractedCanvas.toDataURL('image/jpeg', 0.9);
    const useSift = document.getElementById('useSift').checked;
    const disableViz = document.getElementById('disableViz').checked;

    // Show loading
    document.getElementById('loading').style.display = 'block';

    // Show video feed if visualization is enabled
    if (!disableViz) {
        document.getElementById('video-container').style.display = 'block';
        document.getElementById('video-stream').src = `/video_feed?t=${new Date().getTime()}`;
    }

    fetch('/search_features', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image: imageData,
            use_sift: useSift
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.search_id) {
                currentSearchId = data.search_id;
                pollSearchResults();
            } else {
                console.error('Failed to start search:', data.error);
                document.getElementById('loading').style.display = 'none';
            }
        })
        .catch(error => {
            console.error('Error starting search:', error);
            document.getElementById('loading').style.display = 'none';
        });
}

function stopFeatureSearch() {
    if (searchPollInterval) {
        clearInterval(searchPollInterval);
        searchPollInterval = null;
    }
    currentSearchId = null;
}

function pollSearchResults() {
    if (!currentSearchId) return;

    searchPollInterval = setInterval(() => {
        fetch(`/search_status/${currentSearchId}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'completed') {
                    stopFeatureSearch();
                    displayFinalResults(data);
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('video-container').style.display = 'none';
                } else if (data.status === 'error') {
                    stopFeatureSearch();
                    console.error('Search error:', data.message);
                    alert('Search error: ' + data.message);
                    document.getElementById('loading').style.display = 'none';
                }
            })
            .catch(error => {
                console.error('Error polling search:', error);
                stopFeatureSearch();
                document.getElementById('loading').style.display = 'none';
            });
    }, 200);
}

function displayFinalResults(data) {
    if (!data.best_match) {
        const message = data.message || 'No matches found';
        alert(message);
        return;
    }

    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('bestMatchImage').src = data.best_match.match_image;
    document.getElementById('bestMatchName').textContent = data.best_match.filename;
    document.getElementById('bestMatchScore').textContent = `${data.best_match.score} matches`;

    // Display top matches as image grid (4 per row, 3 by default)
    const topMatchesGrid = document.getElementById('topMatchesGrid');
    topMatchesGrid.innerHTML = '';

    if (data.top_matches && data.top_matches.length > 0) {
        // Show first 3 by default
        const matchesToShow = data.top_matches.slice(0, 3);

        matchesToShow.forEach((match, index) => {
            const matchDiv = document.createElement('div');
            matchDiv.className = 'match-grid-item';
            matchDiv.innerHTML = `
                <img src="${match.image}" alt="${match.filename}" class="match-thumbnail">
                <div class="match-info-overlay">
                    <span class="match-rank">#${index + 1}</span>
                    <span class="match-name">${match.filename}</span>
                    <span class="match-score">${match.score} matches</span>
                </div>
            `;
            topMatchesGrid.appendChild(matchDiv);
        });
    }
}
