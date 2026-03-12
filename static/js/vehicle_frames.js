// static/js/frames.js
const MAX_RESULTS = 20;
let results = [];

// Get camera ID from the data attribute
const cameraData = document.getElementById('camera-data');
const CAMERA_ID = cameraData.getAttribute('data-camera-id');

function refreshVehicleAnalysis() {
    const content = document.getElementById('analysis-vehicle-content');
    content.innerHTML = '<div class="loading">Processing new frame...</div>';

    fetch(`/process_frame/${CAMERA_ID}/vehicle`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }

            console.log("Received data:", data);

            if (data.image_results && data.image_results.length > 0) {
                console.log("Got vehicle!")
                data.image_results.forEach((img, index) => {
                    results.unshift({
                        timestamp: data.timestamp,
                        image: img,
                        indiv_att: data.attributes_everycar[index]
                    });
                });

                if (results.length > MAX_RESULTS) {
                    results = results.slice(0, MAX_RESULTS);
                }

                const resultsHtml = results.map((result, index) => {
                    const indivAttText = Array.isArray(result.indiv_att)
                        ? result.indiv_att.filter(Boolean).join(", ")
                        : result.indiv_att || "N/A";

                    return `
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
                                        <p><strong>Attributes:</strong> ${indivAttText}</p>
                                            ${result.caption ? `<p><strong>Caption:</strong> ${result.caption}</p>` : ""}
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');

                content.innerHTML = `
                    <div class="results-container">
                        ${resultsHtml}
                    </div>
                `;
            } else {
                //console.log("No vehicle detected.")
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

refreshVehicleAnalysis();
    setInterval(refreshVehicleAnalysis, 5000);