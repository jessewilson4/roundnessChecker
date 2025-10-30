/**
 * Image Roundness Analyzer v2 - JavaScript
 * Handles charts, 4-panel visualization modal, and interactions
 */

// Initialize charts on page load
document.addEventListener('DOMContentLoaded', function() {
    if (typeof chartData !== 'undefined') {
        initializeCharts();
    }
    initializeModal();
});

// Chart Initialization
function initializeCharts() {
    // Bar Chart - Top 10
    const barCtx = document.getElementById('barChart');
    if (barCtx) {
        new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: chartData.bar_chart.labels,
                datasets: [
                    {
                        label: 'Composite Roundness',
                        data: chartData.bar_chart.composite,
                        backgroundColor: 'rgba(37, 99, 235, 0.8)',
                        borderColor: 'rgba(37, 99, 235, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Circularity',
                        data: chartData.bar_chart.circularity,
                        backgroundColor: 'rgba(16, 185, 129, 0.8)',
                        borderColor: 'rgba(16, 185, 129, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Aspect Ratio',
                        data: chartData.bar_chart.aspect_ratio,
                        backgroundColor: 'rgba(245, 158, 11, 0.8)',
                        borderColor: 'rgba(245, 158, 11, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Eccentricity',
                        data: chartData.bar_chart.eccentricity,
                        backgroundColor: 'rgba(168, 85, 247, 0.8)',
                        borderColor: 'rgba(168, 85, 247, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Score (%)'
                        }
                    }
                }
            }
        });
    }

    // Histogram - Distribution
    const histCtx = document.getElementById('histogramChart');
    if (histCtx) {
        new Chart(histCtx, {
            type: 'bar',
            data: {
                labels: chartData.histogram.bins,
                datasets: [{
                    label: 'Number of Objects',
                    data: chartData.histogram.counts,
                    backgroundColor: 'rgba(37, 99, 235, 0.7)',
                    borderColor: 'rgba(37, 99, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Frequency'
                        },
                        ticks: {
                            stepSize: 1
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Composite Roundness (%)'
                        }
                    }
                }
            }
        });
    }

    // Metrics Comparison Chart
    const scatterCtx = document.getElementById('scatterChart');
    if (scatterCtx && chartData.box_plots) {
        const metrics = ['circularity', 'aspect_ratio', 'eccentricity', 'solidity', 'convexity', 'composite'];
        const labels = ['Circularity', 'Aspect Ratio', 'Eccentricity', 'Solidity', 'Convexity', 'Composite'];
        const colors = [
            'rgba(16, 185, 129, 0.8)',
            'rgba(245, 158, 11, 0.8)', 
            'rgba(168, 85, 247, 0.8)',
            'rgba(236, 72, 153, 0.8)',
            'rgba(59, 130, 246, 0.8)',
            'rgba(37, 99, 235, 0.8)'
        ];
        
        // Get median values for each metric
        const medianValues = metrics.map(metric => chartData.box_plots[metric].median);
        const minValues = metrics.map(metric => chartData.box_plots[metric].min);
        const maxValues = metrics.map(metric => chartData.box_plots[metric].max);

        new Chart(scatterCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Median',
                    data: medianValues,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('0.8', '1')),
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Median Values of Shape Metrics',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const idx = context.dataIndex;
                                const metric = metrics[idx];
                                const stats = chartData.box_plots[metric];
                                return [
                                    'Median: ' + stats.median.toFixed(1) + '%',
                                    'Range: ' + stats.min.toFixed(1) + '% - ' + stats.max.toFixed(1) + '%',
                                    'Q1: ' + stats.q1.toFixed(1) + '%',
                                    'Q3: ' + stats.q3.toFixed(1) + '%'
                                ];
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Percentage (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Metric'
                        }
                    }
                }
            }
        });
    }
}

// Modal Initialization
function initializeModal() {
    const modal = document.getElementById('detailsModal');
    if (!modal) return;

    const closeBtn = modal.querySelector('.close');
    
    if (closeBtn) {
        closeBtn.onclick = function() {
            modal.style.display = 'none';
        };
    }
    
    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    };
}

// Show 5-Panel Analysis Pipeline
function showDetails(pexelsId) {
    if (typeof allResults === 'undefined') return;
    
    // Find result by pexels_id (handle both string and number comparison)
    const result = allResults.find(r => String(r.pexels_id) === String(pexelsId));
    if (!result) {
        console.error('Result not found:', pexelsId, 'in', allResults.map(r => r.pexels_id));
        return;
    }
    
    const modal = document.getElementById('detailsModal');
    const modalBody = document.getElementById('modalBody');
    
    // Build 5-panel visualization
    const detailsHTML = `
        <h2>Analysis Pipeline for Rank #${result.rank}</h2>
        <p class="photographer-credit">
            Photo by <a href="${result.photographer_url}" target="_blank">${result.photographer}</a> on ${result.source ? result.source.toUpperCase() : 'Pexels'}
        </p>
        
        <div class="pipeline-grid" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
            <div class="pipeline-panel">
                <h4>1. Original Image</h4>
                <img src="/image/${result.viz_paths.original}" alt="Original">
            </div>
            
            <div class="pipeline-panel">
                <h4>2. Grayscale</h4>
                <img src="/image/${result.viz_paths.grayscale}" alt="Grayscale">
            </div>
            
            <div class="pipeline-panel">
                <h4>3. Raw Canny Edges</h4>
                <img src="/image/${result.viz_paths.edges_raw}" alt="Raw Edges">
                <p style="font-size: 0.85rem;">Before morphological closing</p>
            </div>
            
            <div class="pipeline-panel">
                <h4>4. Closed Edges</h4>
                <img src="/image/${result.viz_paths.edges_closed}" alt="Closed Edges">
                <p style="font-size: 0.85rem;">After aggressive closing (15x15 kernel)</p>
            </div>
            
            <div class="pipeline-panel" style="grid-column: span 2;">
                <h4>5. Final Contour</h4>
                <img src="/image/${result.viz_paths.contour}" alt="Contour">
                <p style="font-size: 0.85rem;">Largest contour (green)</p>
            </div>
        </div>
        
        <div class="modal-metrics" style="margin-top: 30px; background: #f8fafc; padding: 20px; border-radius: 8px;">
            <h3>Roundness Metrics</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 15px;">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #2563eb;">
                        ${(result.composite * 100).toFixed(1)}%
                    </div>
                    <div style="color: #64748b; margin-top: 5px;">Composite</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #10b981;">
                        ${(result.circularity * 100).toFixed(1)}%
                    </div>
                    <div style="color: #64748b; margin-top: 5px;">Circularity</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #f59e0b;">
                        ${(result.aspect_ratio * 100).toFixed(1)}%
                    </div>
                    <div style="color: #64748b; margin-top: 5px;">Aspect Ratio</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #a855f7;">
                        ${(result.eccentricity * 100).toFixed(1)}%
                    </div>
                    <div style="color: #64748b; margin-top: 5px;">Eccentricity</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #ec4899;">
                        ${(result.solidity * 100).toFixed(1)}%
                    </div>
                    <div style="color: #64748b; margin-top: 5px;">Solidity</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #3b82f6;">
                        ${(result.convexity * 100).toFixed(1)}%
                    </div>
                    <div style="color: #64748b; margin-top: 5px;">Convexity</div>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #e2e8f0;">
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div>
                        <strong>Area:</strong> ${result.area.toFixed(0)} pixels²
                    </div>
                    <div>
                        <strong>Perimeter:</strong> ${result.perimeter.toFixed(1)} pixels
                    </div>
                    <div>
                        <strong>Rank:</strong> #${result.rank} of ${allResults.length}
                    </div>
                    <div>
                        <strong>Formula:</strong> 30%C + 25%AR + 20%E + 15%S + 10%Cv
                    </div>
                </div>
            </div>
        </div>
    `;
    
    modalBody.innerHTML = detailsHTML;
    modal.style.display = 'block';
    
    // Re-attach close handler (in case it wasn't attached initially)
    const closeBtn = modal.querySelector('.close');
    if (closeBtn) {
        closeBtn.onclick = function() {
            modal.style.display = 'none';
        };
    }
}

// Interpretation Logic
// Smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});