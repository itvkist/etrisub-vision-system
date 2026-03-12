function performSearch() {
    const searchTerm = document.getElementById('searchInput').value;
    const resultsContainer = document.getElementById('searchResults');
    resultsContainer.innerHTML = '<div class="loading">Searching...</div>';

    if (!searchTerm.trim()) {
        resultsContainer.innerHTML = '<div class="result-item">Please enter a search term</div>';
        return;
    }

    fetch(`/search_results?q=${encodeURIComponent(searchTerm)}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(results => {
            if (!Array.isArray(results)) {
                results = [];
            }

            if (results.length === 0) {
                resultsContainer.innerHTML = '<div class="result-item">No results found</div>';
                return;
            }

            const resultsHtml = results.map(result => `
                <div class="result-item">
                    <div class="result-row">
                        <div class="result-image">
                            <img src="data:image/jpeg;base64,${result.image}" alt="Result Image">
                        </div>
                        <div class="result-details">
                            <p><strong>Timestamp:</strong> ${new Date(result.timestamp).toLocaleString()}</p>
                            <p><strong>Description (EN):</strong> ${result.filtered_content}</p>
                            <p><strong>Mô tả (VI):</strong> ${result.filtered_content_vi || 'Not available'}</p>
                            ${result.attributes ? `
                                <div class="attributes-section">
                                    <h4>Detected Attributes:</h4>
                                    ${formatAttributes(result.attributes)}
                                </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `).join('');

            resultsContainer.innerHTML = resultsHtml;
        })
        .catch(error => {
            resultsContainer.innerHTML = `
                <div class="error">Error performing search: ${error.message}</div>
            `;
            console.error('Error:', error);
        });
}

function formatAttributes(attributes) {
    let html = '';
    for (const [category, items] of Object.entries(attributes)) {
        if (items.length > 0) {
            html += `<p><strong>${category}:</strong> `;
            html += items.map(item => {
                let text = item.type;
                if (item.color) {
                    text = `${item.color} ${text}`;
                }
                return text;
            }).join(', ');
            html += '</p>';
        }
    }
    return html || '<p>No attributes detected</p>';
}