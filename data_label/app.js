class TranscriptLabeler {
    constructor() {
        this.transcriptIds = [];
        this.currentIndex = 0;
        this.labels = {};
        this.transcripts = {};
        this.init();
    }

    async init() {
        try {
            await this.loadTranscriptIds();
            await this.loadCurrentTranscript();
            this.updateUI();
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to load transcripts. Please make sure the transcript files are accessible.');
        }
    }

    async loadTranscriptIds() {
        try {
            // Since we can't directly read directory contents in browser,
            // we'll generate a list of transcript IDs from the known structure
            // This would normally come from a server API
            const response = await fetch('transcript-ids.json');
            if (response.ok) {
                this.transcriptIds = await response.json();
            } else {
                // Fallback: use the transcripts we know exist from the file structure
                this.transcriptIds = await this.generateTranscriptIds();
            }
        } catch (error) {
            console.error('Error loading transcript IDs:', error);
            this.transcriptIds = await this.generateTranscriptIds();
        }
    }

    async generateTranscriptIds() {
        // For demonstration, we'll create a list based on the known structure
        // In a real implementation, this would come from the server
        return [
            "00031634-197b-4088-bebe-29e162eb705a",
            "0004a2b3-6976-4061-87e3-8c70b0db3ed9",
            "00147336-d97b-460b-b02a-bc899cad057c",
            "00235a77-8795-49ba-b81f-cbdf4b0cf6e5",
            "0040b233-ea4c-4c9f-8207-04718a13d7b3",
            "004a744d-b7aa-43d4-a4d7-e96217732863",
            "00590ae3-14aa-48bb-a138-3f739e899f91",
            "0068c065-f7b7-417a-9941-784c36b90cbf",
            "008951a3-8c03-4a9c-b8d0-343cccdd881e",
            "00a06731-901f-48bd-8a96-b4e414188d97",
            "d8182e11-4717-4a1a-8eaf-42382c0af7e1",
            "d6613eb1-bac1-48da-90a3-ca79dc72af19",
            "d62392cf-92cc-4204-ab15-eed6379cfaf7",
            "d60671fa-2d55-4a21-9bc7-4b1a324a719d",
            "d5236025-0541-40a9-ba88-a20a74711330"
        ];
    }

    async loadCurrentTranscript() {
        if (this.currentIndex >= this.transcriptIds.length) {
            this.showCompletion();
            return;
        }

        const transcriptId = this.transcriptIds[this.currentIndex];
        
        if (!this.transcripts[transcriptId]) {
            try {
                const response = await fetch(`../transcripts/${transcriptId}/transcript.json`);
                if (response.ok) {
                    this.transcripts[transcriptId] = await response.json();
                } else {
                    throw new Error('Failed to load transcript');
                }
            } catch (error) {
                console.error(`Error loading transcript ${transcriptId}:`, error);
                // Create a placeholder for failed loads
                this.transcripts[transcriptId] = [{
                    timestamp: new Date().toISOString(),
                    speaker: 'system',
                    content: `Error: Could not load transcript ${transcriptId}. Please check if the file exists.`
                }];
            }
        }
    }

    updateUI() {
        if (this.currentIndex >= this.transcriptIds.length) {
            this.showCompletion();
            return;
        }

        const transcriptId = this.transcriptIds[this.currentIndex];
        const transcript = this.transcripts[transcriptId];
        const progress = ((this.currentIndex + 1) / this.transcriptIds.length) * 100;

        // Update progress
        document.getElementById('progress-text').textContent = 
            `Transcript ${this.currentIndex + 1} of ${this.transcriptIds.length}`;
        document.getElementById('progress-fill').style.width = `${progress}%`;

        // Update main content
        const mainContent = document.getElementById('main-content');
        mainContent.innerHTML = `
            <div class="transcript-container">
                <div class="transcript-id">ID: ${transcriptId}</div>
                <div class="conversation">
                    ${this.renderConversation(transcript)}
                </div>
                <div class="button-container">
                    <button class="label-btn btn-machine" onclick="labeler.labelTranscript('machine')">
                        🤖 Machine
                    </button>
                    <button class="label-btn btn-human" onclick="labeler.labelTranscript('human')">
                        👤 Human
                    </button>
                    <button class="label-btn btn-discard" onclick="labeler.labelTranscript('discard')">
                        🗑️ Discard
                    </button>
                    <button class="label-btn btn-not-sure" onclick="labeler.labelTranscript('not sure')">
                        ❓ Not Sure
                    </button>
                </div>
                ${this.labels[transcriptId] ? `<div style="text-align: center; margin-top: 20px; color: #666; font-style: italic;">Current label: ${this.labels[transcriptId]}</div>` : ''}
            </div>
        `;

        // Update navigation buttons
        document.getElementById('prev-btn').disabled = this.currentIndex === 0;
        document.getElementById('next-btn').disabled = this.currentIndex >= this.transcriptIds.length - 1;
    }

    renderConversation(transcript) {
        if (!Array.isArray(transcript) || transcript.length === 0) {
            return '<div style="text-align: center; color: #666; padding: 20px;">No conversation data available</div>';
        }

        return transcript.map(message => `
            <div class="message ${message.speaker}">
                <div class="message-meta">${message.speaker} • ${new Date(message.timestamp).toLocaleString()}</div>
                <div class="message-content">${this.escapeHtml(message.content)}</div>
            </div>
        `).join('');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    labelTranscript(label) {
        const transcriptId = this.transcriptIds[this.currentIndex];
        this.labels[transcriptId] = label;
        
        // Auto-advance to next transcript
        setTimeout(() => {
            this.nextTranscript();
        }, 500);
    }

    async nextTranscript() {
        if (this.currentIndex < this.transcriptIds.length - 1) {
            this.currentIndex++;
            await this.loadCurrentTranscript();
            this.updateUI();
        } else {
            this.showCompletion();
        }
    }

    async previousTranscript() {
        if (this.currentIndex > 0) {
            this.currentIndex--;
            await this.loadCurrentTranscript();
            this.updateUI();
        }
    }

    showCompletion() {
        const mainContent = document.getElementById('main-content');
        const labeledCount = Object.keys(this.labels).length;
        
        mainContent.innerHTML = `
            <div class="completion-message">
                <h2>🎉 Labeling Complete!</h2>
                <p>You have labeled ${labeledCount} out of ${this.transcriptIds.length} transcripts.</p>
                <p>Click "Export CSV" to download your labels.</p>
            </div>
        `;

        // Update progress to 100%
        document.getElementById('progress-text').textContent = `Complete! ${labeledCount}/${this.transcriptIds.length} transcripts labeled`;
        document.getElementById('progress-fill').style.width = '100%';
    }

    exportCSV() {
        const csvContent = this.generateCSV();
        this.downloadCSV(csvContent, 'transcript_labels.csv');
    }

    generateCSV() {
        const headers = ['Call_ID', 'Label'];
        const rows = [headers];

        for (const transcriptId of this.transcriptIds) {
            const label = this.labels[transcriptId] || 'unlabeled';
            rows.push([transcriptId, label]);
        }

        return rows.map(row => 
            row.map(field => `"${field}"`).join(',')
        ).join('\n');
    }

    downloadCSV(content, filename) {
        const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        
        if (link.download !== undefined) {
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    showError(message) {
        const mainContent = document.getElementById('main-content');
        mainContent.innerHTML = `
            <div class="completion-message">
                <h2>❌ Error</h2>
                <p>${message}</p>
            </div>
        `;
    }
}

// Navigation functions for buttons
function nextTranscript() {
    if (window.labeler) {
        window.labeler.nextTranscript();
    }
}

function previousTranscript() {
    if (window.labeler) {
        window.labeler.previousTranscript();
    }
}

function exportCSV() {
    if (window.labeler) {
        window.labeler.exportCSV();
    }
}

// Initialize the labeler when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.labeler = new TranscriptLabeler();
});

// Keyboard shortcuts
document.addEventListener('keydown', (event) => {
    if (!window.labeler) return;

    switch(event.key) {
        case '1':
            event.preventDefault();
            window.labeler.labelTranscript('machine');
            break;
        case '2':
            event.preventDefault();
            window.labeler.labelTranscript('human');
            break;
        case '3':
            event.preventDefault();
            window.labeler.labelTranscript('discard');
            break;
        case '4':
            event.preventDefault();
            window.labeler.labelTranscript('not sure');
            break;
        case 'ArrowRight':
            event.preventDefault();
            window.labeler.nextTranscript();
            break;
        case 'ArrowLeft':
            event.preventDefault();
            window.labeler.previousTranscript();
            break;
    }
});