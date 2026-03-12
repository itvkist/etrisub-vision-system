// static/js/frames.js
const MAX_RESULTS = 6;
let results = [];

// Get camera ID from the data attribute
const cameraData = document.getElementById('camera-data');
const CAMERA_ID = cameraData.getAttribute('data-camera-id');

function refreshAnalysis() {
    const content = document.getElementById('analysis-content');
    content.innerHTML = '<div class="loading">Processing new frame...</div>';

    fetch(`/process_frame/${CAMERA_ID}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }

            if (data.image_results && data.image_results.length > 0) {
                data.image_results.forEach((img, index) => {
                    // console.log(data.captions[index]['<ATTRIBUTES>'])
                    results.unshift({
                        timestamp: data.timestamp,
                        image: img,
                        caption: data.caption_results[index]['<DETAILED_CAPTION>'] || 
                                'No attributes detected',
                        // attributes: data.captions[index]['<DETAILED_CAPTION>'] || []
                    });
                });

                if (results.length > MAX_RESULTS) {
                    results = results.slice(0, MAX_RESULTS);
                }

                const resultsHtml = results.map((result, index) => `
                    <div class="result-item">
                        <div class="result-details">
                            <p><strong>Timestamp:</strong> 
                                ${new Date(result.timestamp).toLocaleString()}</p>
                            <div class="result-row">
                                <div class="result-image">
                                    <img loading="lazy" 
                                        src="data:image/jpeg;base64,${result.image}" 
                                        alt="Result Image">
                                </div>
                                <div class="result-caption">
                                    <p>Captions: ${ result.caption}</p>
                                    <!--
                                    <div class="result-caption">
                                    <p>Classes: ${result.attributes}</p>
                                    -->
                                </div>
                            </div>
                        </div>
                    </div>
                `).join('');

                content.innerHTML = `
                    <div class="results-container">
                        ${resultsHtml}
                    </div>
                `;
            } else {
                if (results.length === 0) {
                    content.innerHTML = '<div class="loading">Waiting for analysis results...</div>';
                }
            }
        })
        .catch(error => {
            content.innerHTML = `
                <div class="error">Error loading analysis: ${error.message}</div>
            `;
            console.error('Error:', error);
        });
}

document.addEventListener('DOMContentLoaded', function() {
    refreshAnalysis();
    setInterval(refreshAnalysis, 50000);
});