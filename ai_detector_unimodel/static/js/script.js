// Character count and input validation
document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('textInput');
    const charCount = document.getElementById('charCount');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (textInput) {
        textInput.addEventListener('input', function() {
            const count = this.value.length;
            charCount.textContent = count;
            
            // Enable/disable button based on text length
            if (analyzeBtn) {
                analyzeBtn.disabled = count < 10;
            }
        });
    }
});

function analyzeText() {
    const textInput = document.getElementById('textInput');
    const text = textInput.value.trim();
    
    if (text.length < 10) {
        showResult("Human Written", ["Text is too short (minimum 10 characters required)"]);
        return;
    }
    
    // Show loading, hide results and errors
    showLoading();
    hideResults();
    hideError();
    
    // Disable button during analysis
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
    }
    
    // Send request to server
    fetch('/analyze-text', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        hideLoading();
        
        // Re-enable button
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Check AI or Human';
        }
        
        if (data.error) {
            showError(data.error);
        } else {
            showResult(data.result, data.details);
        }
    })
    .catch(error => {
        hideLoading();
        
        // Re-enable button
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Check AI or Human';
        }
        
        showError('An error occurred while analyzing the text. Please try again.');
        console.error('Error:', error);
    });
}

function showLoading() {
    const loadingElement = document.getElementById('loading');
    if (loadingElement) {
        loadingElement.style.display = 'block';
    }
}

function hideLoading() {
    const loadingElement = document.getElementById('loading');
    if (loadingElement) {
        loadingElement.style.display = 'none';
    }
}

function showError(message) {
    const errorSection = document.getElementById('errorSection');
    if (errorSection) {
        errorSection.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.5em;">⚠️</span>
                <div>
                    <strong>Error:</strong> ${message}
                </div>
            </div>
        `;
        errorSection.style.display = 'block';
        
        // Scroll to error
        errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

function hideError() {
    const errorSection = document.getElementById('errorSection');
    if (errorSection) {
        errorSection.style.display = 'none';
    }
}

function hideResults() {
    const resultSection = document.getElementById('resultSection');
    if (resultSection) {
        resultSection.style.display = 'none';
    }
}

function showResult(result, details) {
    const resultSection = document.getElementById('resultSection');
    const resultContent = document.getElementById('resultContent');
    
    if (!resultSection || !resultContent) {
        showError('Result display elements not found');
        return;
    }
    
    // Determine result styling
    let resultClass = '';
    let resultIcon = '';
    let resultColor = '';
    
    if (result.includes('AI')) {
        resultClass = 'ai-result';
        resultIcon = '🤖';
        resultColor = '#f44336';
    } else {
        resultClass = 'human-result';
        resultIcon = '👤';
        resultColor = '#4CAF50';
    }
    
    // Create details list
    let detailsHtml = '';
    if (details && details.length > 0) {
        detailsHtml = `
            <div style="margin-top: 20px; text-align: left;">
                <h4 style="margin-bottom: 10px; color: #666;">Analysis Details:</h4>
                <ul style="list-style: none; padding: 0;">
                    ${details.map(detail => `
                        <li style="padding: 8px 0; border-bottom: 1px solid #eee; display: flex; align-items: center;">
                            <span style="color: ${resultColor}; margin-right: 10px;">•</span>
                            ${detail}
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
    }
    
    // Create result HTML
    resultContent.innerHTML = `
        <div class="result-card ${resultClass}">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="font-size: 3em; margin-bottom: 10px;">${resultIcon}</div>
                <h3 style="color: ${resultColor}; font-size: 2em; margin: 0;">${result}</h3>
            </div>
            ${detailsHtml}
        </div>
    `;
    
    // Show result section
    resultSection.style.display = 'block';
    
    // Scroll to results smoothly
    resultSection.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'start'
    });
}

// Add keyboard shortcut (Ctrl+Enter to analyze)
document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('textInput');
    if (textInput) {
        textInput.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                analyzeText();
            }
        });
    }
});

// Add paste event handler for better UX
document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('textInput');
    if (textInput) {
        textInput.addEventListener('paste', function(e) {
            // Let the paste happen first, then update character count
            setTimeout(() => {
                const count = this.value.length;
                const charCount = document.getElementById('charCount');
                if (charCount) {
                    charCount.textContent = count;
                }
                
                const analyzeBtn = document.getElementById('analyzeBtn');
                if (analyzeBtn) {
                    analyzeBtn.disabled = count < 10;
                }
            }, 0);
        });
    }
});

// Auto-resize textarea
document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('textInput');
    if (textInput) {
        textInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        
        // Initial resize
        setTimeout(() => {
            textInput.style.height = 'auto';
            textInput.style.height = (textInput.scrollHeight) + 'px';
        }, 100);
    }
});

// Add clear text functionality
function clearText() {
    const textInput = document.getElementById('textInput');
    const charCount = document.getElementById('charCount');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (textInput) {
        textInput.value = '';
        textInput.style.height = '200px';
    }
    
    if (charCount) {
        charCount.textContent = '0';
    }
    
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
    }
    
    hideResults();
    hideError();
}

// Add this HTML button to your analyze.html if you want clear functionality:
// <button onclick="clearText()" class="clear-btn" style="margin-left: 10px;">Clear</button>

// Utility function to copy text to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        // Show copy success message
        const copyMsg = document.createElement('div');
        copyMsg.textContent = 'Text copied to clipboard!';
        copyMsg.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            z-index: 1000;
        `;
        document.body.appendChild(copyMsg);
        
        setTimeout(() => {
            document.body.removeChild(copyMsg);
        }, 2000);
    }).catch(function() {
        showError('Failed to copy text to clipboard');
    });
}

// Add click to copy functionality for results
document.addEventListener('click', function(e) {
    if (e.target.closest('.result-card')) {
        const textInput = document.getElementById('textInput');
        if (textInput && textInput.value.trim()) {
            copyToClipboard(textInput.value.trim());
        }
    }
});

// Prevent form submission on Enter key in textarea
document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('textInput');
    if (textInput) {
        textInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.ctrlKey) {
                e.stopPropagation();
            }
        });
    }
});
// Text Correction Functions
function correctText() {
    const textInput = document.getElementById('textInput');
    const text = textInput.value.trim();
    
    if (text.length < 3) {
        showCorrectionError("Please enter at least 3 characters to correct");
        return;
    }
    
    showCorrectionLoading();
    hideCorrectionResult();
    hideCorrectionError();
    
    const correctBtn = document.getElementById('correctBtn');
    if (correctBtn) {
        correctBtn.disabled = true;
        correctBtn.innerHTML = '⏳ Correcting...';
    }
    
    fetch('/correct-text', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network error occurred');
        }
        return response.json();
    })
    .then(data => {
        hideCorrectionLoading();
        
        if (correctBtn) {
            correctBtn.disabled = false;
            correctBtn.innerHTML = '🛠️ Text Corrector';
        }
        
        if (data.error) {
            showCorrectionError(data.error);
        } else {
            showCorrectionResult(data.corrected_text, data.original_text);
        }
    })
    .catch(error => {
        hideCorrectionLoading();
        
        if (correctBtn) {
            correctBtn.disabled = false;
            correctBtn.innerHTML = '🛠️ Text Corrector';
        }
        
        showCorrectionError('Connection error. Please try again.');
        console.error('Error:', error);
    });
}

function showCorrectionLoading() {
    const loading = document.getElementById('correctionLoading');
    if (loading) loading.style.display = 'block';
}

function hideCorrectionLoading() {
    const loading = document.getElementById('correctionLoading');
    if (loading) loading.style.display = 'none';
}

function showCorrectionResult(correctedText, originalText) {
    const resultDiv = document.getElementById('correctionResult');
    const correctedTextDiv = document.getElementById('correctedText');
    
    if (resultDiv && correctedTextDiv) {
        correctedTextDiv.innerHTML = correctedText.replace(/\n/g, '<br>');
        resultDiv.style.display = 'block';
        resultDiv.scrollIntoView({ behavior: 'smooth' });
    }
}

function hideCorrectionResult() {
    const resultDiv = document.getElementById('correctionResult');
    if (resultDiv) resultDiv.style.display = 'none';
}

function showCorrectionError(message) {
    // You can create a separate error div for corrections or use the main one
    showError(message);
}

function hideCorrectionError() {
    hideError();
}

function copyCorrectedText() {
    const correctedTextDiv = document.getElementById('correctedText');
    if (correctedTextDiv) {
        const textToCopy = correctedTextDiv.innerText || correctedTextDiv.textContent;
        copyToClipboard(textToCopy);
    }
}