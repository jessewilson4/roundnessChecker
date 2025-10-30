// Batch Processing Frontend JavaScript

let uploadedFile = null;
let uploadedColumns = [];
let currentBatchId = null;
let statusPollInterval = null;
let currentPage = 1;
const itemsPerPage = 50;
let allKeywords = [];
let filteredKeywords = [];

// ============= FILE UPLOAD =============

document.getElementById('fileInput').addEventListener('change', async function(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    document.getElementById('fileName').textContent = `Loading ${file.name}...`;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/batch/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('fileName').textContent = `✓ Loaded: ${file.name} (${data.row_count} rows)`;
            
            // Store file object and columns
            uploadedFile = file;
            uploadedColumns = data.columns;
            
            // Populate column dropdown
            const columnSelect = document.getElementById('columnSelect');
            columnSelect.innerHTML = '<option value="">Select column...</option>';
            data.columns.forEach(col => {
                const option = document.createElement('option');
                option.value = col;
                option.textContent = col;
                columnSelect.appendChild(option);
            });
            
            // Generate default batch name
            const now = new Date();
            const defaultName = `Batch ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`;
            document.getElementById('batchName').value = defaultName;
            
            // Show config section
            document.getElementById('configSection').style.display = 'block';
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error uploading file: ' + error.message);
    }
});

// ============= START BATCH =============

document.getElementById('startBatchBtn').addEventListener('click', async function() {
    const batchName = document.getElementById('batchName').value.trim();
    const columnName = document.getElementById('columnSelect').value;
    const imagesPerKeyword = document.getElementById('imagesPerKeyword').value;
    
    if (!batchName) {
        alert('Please enter a batch name');
        return;
    }
    
    if (!columnName) {
        alert('Please select a keyword column');
        return;
    }
    
    if (!uploadedFile) {
        alert('Please upload a file first');
        return;
    }
    
    // Create batch by re-uploading file with metadata
    try {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        formData.append('batch_name', batchName);
        formData.append('column_name', columnName);
        formData.append('images_per_keyword', imagesPerKeyword);
        
        const response = await fetch('/batch/create', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentBatchId = data.batch_id;
            
            // Start processing
            const startResponse = await fetch(`/batch/${currentBatchId}/start`, {
                method: 'POST'
            });
            
            const startData = await startResponse.json();
            
            if (startData.success) {
                // Hide upload section, show processing
                document.getElementById('configSection').style.display = 'none';
                document.getElementById('processingSection').style.display = 'block';
                document.getElementById('keywordsSection').style.display = 'block';
                
                // Start polling for status
                startStatusPolling();
            } else {
                alert('Error starting batch: ' + startData.error);
            }
        } else {
            alert('Error creating batch: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
});

// ============= STATUS POLLING =============

function startStatusPolling() {
    // Poll every 10 seconds
    statusPollInterval = setInterval(updateStatus, 10000);
    updateStatus(); // Initial call
}

function stopStatusPolling() {
    if (statusPollInterval) {
        clearInterval(statusPollInterval);
        statusPollInterval = null;
    }
}

async function updateStatus() {
    if (!currentBatchId) return;
    
    try {
        const response = await fetch(`/batch/${currentBatchId}/status`);
        const data = await response.json();
        
        if (data.success) {
            const batch = data.batch;
            allKeywords = data.keywords;
            
            // Update progress
            const progressPercent = batch.progress_percent;
            document.getElementById('progressFill').style.width = progressPercent + '%';
            document.getElementById('overallProgress').textContent = 
                `${batch.completed_keywords} / ${batch.total_keywords} keywords`;
            
            // Update current keyword
            const currentKeyword = allKeywords[batch.current_keyword_index];
            if (currentKeyword && batch.status === 'processing') {
                document.getElementById('currentKeyword').textContent = 
                    `Processing: "${currentKeyword.keyword}"`;
            } else if (batch.status === 'complete') {
                document.getElementById('currentKeyword').textContent = '✓ Batch Complete!';
                stopStatusPolling();
                
                // Show export button
                showExportButton();
            } else if (batch.status === 'paused') {
                document.getElementById('currentKeyword').textContent = '⏸ Paused';
            }
            
            // Update keywords table
            applyFilters();
            updateKeywordsTable();
            
            // Update total keywords count
            document.getElementById('totalKeywords').textContent = allKeywords.length;
        }
    } catch (error) {
        console.error('Error fetching status:', error);
    }
}

// ============= PAUSE/RESUME/STOP =============

document.getElementById('pauseBtn').addEventListener('click', async function() {
    try {
        const response = await fetch(`/batch/${currentBatchId}/pause`, {method: 'POST'});
        const data = await response.json();
        if (data.success) {
            document.getElementById('pauseBtn').style.display = 'none';
            document.getElementById('resumeBtn').style.display = 'inline-block';
        }
    } catch (error) {
        alert('Error pausing: ' + error.message);
    }
});

document.getElementById('resumeBtn').addEventListener('click', async function() {
    try {
        const response = await fetch(`/batch/${currentBatchId}/resume`, {method: 'POST'});
        const data = await response.json();
        if (data.success) {
            document.getElementById('resumeBtn').style.display = 'none';
            document.getElementById('pauseBtn').style.display = 'inline-block';
        }
    } catch (error) {
        alert('Error resuming: ' + error.message);
    }
});

document.getElementById('stopBtn').addEventListener('click', async function() {
    if (!confirm('Are you sure you want to stop this batch? Progress will be saved.')) return;
    
    try {
        const response = await fetch(`/batch/${currentBatchId}/stop`, {method: 'POST'});
        const data = await response.json();
        if (data.success) {
            stopStatusPolling();
            alert('Batch stopped. Progress has been saved.');
        }
    } catch (error) {
        alert('Error stopping: ' + error.message);
    }
});

// ============= KEYWORDS TABLE =============

function applyFilters() {
    const searchTerm = document.getElementById('searchKeywords').value.toLowerCase();
    const statusFilter = document.getElementById('statusFilter').value;
    
    filteredKeywords = allKeywords.filter(kw => {
        const matchesSearch = kw.keyword.toLowerCase().includes(searchTerm);
        const matchesStatus = statusFilter === 'all' || kw.status === statusFilter;
        return matchesSearch && matchesStatus;
    });
    
    currentPage = 1; // Reset to first page
}

function updateKeywordsTable() {
    const tbody = document.getElementById('keywordsTableBody');
    tbody.innerHTML = '';
    
    // Calculate pagination
    const startIdx = (currentPage - 1) * itemsPerPage;
    const endIdx = Math.min(startIdx + itemsPerPage, filteredKeywords.length);
    const pageKeywords = filteredKeywords.slice(startIdx, endIdx);
    
    // Populate table
    pageKeywords.forEach(kw => {
        const row = document.createElement('tr');
        
        const statusClass = `status-${kw.status}`;
        const statusText = kw.status.charAt(0).toUpperCase() + kw.status.slice(1);
        
        let statusDisplay = `<span class="status-badge ${statusClass}">${statusText}</span>`;
        if (kw.status === 'incomplete') {
            statusDisplay = `<span class="status-badge ${statusClass}">Incomplete (${kw.images} images)</span>`;
        } else if (kw.status === 'failed') {
            statusDisplay = `<span class="status-badge ${statusClass}">Failed (0 images)</span>`;
        }
        
        let actionsHTML = '';
        if (kw.search_id) {
            actionsHTML = `<a href="/search/${kw.search_id}" class="btn-secondary" style="text-decoration: none; padding: 6px 12px; font-size: 0.9rem;">View Results</a>`;
        }
        
        row.innerHTML = `
            <td>${kw.keyword}</td>
            <td>${statusDisplay}</td>
            <td>${kw.images}</td>
            <td>${kw.avg_score}</td>
            <td>${actionsHTML}</td>
        `;
        
        tbody.appendChild(row);
    });
    
    // Update pagination
    updatePagination();
}

function updatePagination() {
    const pagination = document.getElementById('pagination');
    pagination.innerHTML = '';
    
    const totalPages = Math.ceil(filteredKeywords.length / itemsPerPage);
    
    if (totalPages <= 1) return;
    
    // Previous button
    const prevBtn = document.createElement('button');
    prevBtn.textContent = '← Previous';
    prevBtn.disabled = currentPage === 1;
    prevBtn.addEventListener('click', () => {
        currentPage--;
        updateKeywordsTable();
    });
    pagination.appendChild(prevBtn);
    
    // Page numbers
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);
    
    for (let i = startPage; i <= endPage; i++) {
        const pageBtn = document.createElement('button');
        pageBtn.textContent = i;
        pageBtn.classList.toggle('active', i === currentPage);
        pageBtn.addEventListener('click', () => {
            currentPage = i;
            updateKeywordsTable();
        });
        pagination.appendChild(pageBtn);
    }
    
    // Next button
    const nextBtn = document.createElement('button');
    nextBtn.textContent = 'Next →';
    nextBtn.disabled = currentPage === totalPages;
    nextBtn.addEventListener('click', () => {
        currentPage++;
        updateKeywordsTable();
    });
    pagination.appendChild(nextBtn);
}

// ============= FILTERS =============

document.getElementById('searchKeywords').addEventListener('input', function() {
    applyFilters();
    updateKeywordsTable();
});

document.getElementById('statusFilter').addEventListener('change', function() {
    applyFilters();
    updateKeywordsTable();
});

// ============= EXPORT =============

function showExportButton() {
    const processingSection = document.getElementById('processingSection');
    
    // Check if button already exists
    if (document.getElementById('exportBatchBtn')) return;
    
    const exportBtn = document.createElement('a');
    exportBtn.id = 'exportBatchBtn';
    exportBtn.href = `/batch/${currentBatchId}/export.csv`;
    exportBtn.className = 'btn-primary';
    exportBtn.textContent = '📊 Export Batch Results (CSV)';
    exportBtn.style.marginTop = '20px';
    exportBtn.style.display = 'inline-block';
    exportBtn.style.textDecoration = 'none';
    
    processingSection.appendChild(exportBtn);
}