// Tab Switching Logic for Video Feeds
document.addEventListener('DOMContentLoaded', function() {
    console.log("EmoSense AI Frontend Loaded");
    const fusionFeed = document.getElementById('fusion-feed');
    const faceFeed = document.getElementById('face-feed');
    
    // Initial State
    fusionFeed.src = "/video_feed?mode=fusion";

    const tabEls = document.querySelectorAll('button[data-bs-toggle="tab"]');
    tabEls.forEach(tab => {
        tab.addEventListener('shown.bs.tab', function (event) {
            const targetId = event.target.getAttribute('data-bs-target');
            console.log("Switched to tab:", targetId);
            
            // Stop all feeds first
            fusionFeed.src = "";
            faceFeed.src = "";

            if (targetId === '#dashboard') {
                fusionFeed.src = "/video_feed?mode=fusion";
            } else if (targetId === '#face-mod') {
                faceFeed.src = "/video_feed?mode=face";
            } else if (targetId === '#logs-mod') {
                loadLogs();
            }
        });
    });
});

// Text Analysis
async function analyzeText() {
    const text = document.getElementById('text-input').value;
    if (!text) {
        alert("Please enter some text.");
        return;
    }

    const resDiv = document.getElementById('text-result');
    const disp = document.getElementById('text-emotion-display');
    const bar = document.getElementById('text-conf-bar');

    // Reset UI
    resDiv.classList.add('d-none');
    disp.textContent = "Analyzing...";
    disp.className = "alert alert-warning";

    try {
        const response = await fetch('/api/text_emotion', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: text})
        });
        
        if (!response.ok) {
            throw new Error(`Server Error: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        resDiv.classList.remove('d-none');
        disp.className = "alert alert-success";
        disp.textContent = `Emotion: ${data.emotion.toUpperCase()}`;
        const confPercent = Math.round(data.confidence * 100);
        bar.style.width = `${confPercent}%`;
        bar.textContent = `${confPercent}%`;
        
    } catch (e) {
        console.error(e);
        alert("Error analyzing text: " + e.message);
    }
}

// Voice Upload
async function uploadAudio() {
    const fileInput = document.getElementById('audio-file');
    const file = fileInput.files[0];
    if (!file) {
        alert("Please select a file");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    processVoice(formData);
}

// Voice Recording
let audioContext;
let mediaStreamSource;
let recorder;
const recordBtn = document.getElementById('record-btn');
const recordStatus = document.getElementById('record-status');

if (recordBtn) {
    recordBtn.addEventListener('click', async () => {
        if (recordBtn.classList.contains('btn-danger')) {
            startRecording();
        } else {
            stopRecording();
        }
    });
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        mediaStreamSource = audioContext.createMediaStreamSource(stream);
        
        // Create a ScriptProcessorNode (bufferSize, inputChannels, outputChannels)
        recorder = audioContext.createScriptProcessor(4096, 1, 1);
        
        window.leftChannel = [];

        recorder.onaudioprocess = function(e) {
            let samples = e.inputBuffer.getChannelData(0);
            // Clone the samples to save them
            window.leftChannel.push(new Float32Array(samples));
        }

        mediaStreamSource.connect(recorder);
        recorder.connect(audioContext.destination);

        // UI Updates
        recordBtn.classList.add('btn-warning');
        recordBtn.classList.remove('btn-danger');
        recordBtn.classList.add('recording-pulse'); // Add pulse animation
        recordStatus.textContent = "Recording... Click to Stop";
        
        window.currentStream = stream;
        window.sampleRate = audioContext.sampleRate;
        
        // Start Visualizer
        drawVisualizer();

    } catch (err) {
        console.error("Error accessing microphone:", err);
        alert("Could not access microphone. Ensure you are using HTTPS or localhost.");
    }
}

function stopRecording() {
    // Stop recording
    if(recorder) recorder.disconnect();
    if(mediaStreamSource) mediaStreamSource.disconnect();
    if(window.currentStream) window.currentStream.getTracks().forEach(track => track.stop());
    
    // Stop Visualizer
    cancelAnimationFrame(window.visualizerFrame);
    
    // Process Audio
    const leftChannel = window.leftChannel;
    if (!leftChannel || leftChannel.length === 0) return;

    // Flatten
    let recordingLength = 0;
    for(let i=0; i < leftChannel.length; i++){
        recordingLength += leftChannel[i].length;
    }
    
    let result = new Float32Array(recordingLength);
    let offset = 0;
    for(let i=0; i < leftChannel.length; i++){
        result.set(leftChannel[i], offset);
        offset += leftChannel[i].length;
    }
    
    // Encode to WAV
    const wavBlob = encodeWAV(result, window.sampleRate);
    
    const formData = new FormData();
    formData.append('file', wavBlob, 'recording.wav');
    
    // UI Updates
    recordBtn.classList.add('btn-danger');
    recordBtn.classList.remove('btn-warning');
    recordBtn.classList.remove('recording-pulse'); // Remove pulse
    recordStatus.textContent = "Processing...";

    processVoice(formData);
}

function drawVisualizer() {
    const canvas = document.getElementById('visualizer');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // We need an analyser node for visualization
    if (!window.analyser) {
        window.analyser = audioContext.createAnalyser();
        window.analyser.fftSize = 256;
        mediaStreamSource.connect(window.analyser);
    }
    
    const bufferLength = window.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function render() {
        window.visualizerFrame = requestAnimationFrame(render);
        window.analyser.getByteFrequencyData(dataArray);
        
        ctx.clearRect(0, 0, width, height);
        
        const barWidth = (width / bufferLength) * 2.5;
        let barHeight;
        let x = 0;
        
        for(let i = 0; i < bufferLength; i++) {
            barHeight = dataArray[i] / 2;
            
            ctx.fillStyle = `rgb(${barHeight + 100}, 50, 255)`;
            ctx.fillRect(x, height - barHeight, barWidth, barHeight);
            
            x += barWidth + 1;
        }
    }
    render();
}

function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    
    // fmt sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); // 16-bit

    // data sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // Write PCM samples
    floatTo16BitPCM(view, 44, samples);

    return new Blob([view], { type: 'audio/wav' });
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

async function processVoice(formData) {
    const resDiv = document.getElementById('voice-result');
    const disp = document.getElementById('voice-emotion-display');
    
    resDiv.classList.add('d-none');

    try {
        const response = await fetch('/api/voice_emotion', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server Error: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        resDiv.classList.remove('d-none');
        disp.textContent = `Emotion: ${data.emotion.toUpperCase()} (Confidence: ${(data.confidence * 100).toFixed(1)}%)`;
        
    } catch (e) {
        console.error(e);
        alert("Error processing audio: " + e.message);
    }
}

// Logs
async function loadLogs() {
    const tbody = document.getElementById('logs-table-body');
    tbody.innerHTML = '<tr><td colspan="6" class="text-center">Loading...</td></tr>';

    try {
        const response = await fetch('/api/logs');
        if (!response.ok) throw new Error("Failed to fetch logs");
        
        const logs = await response.json();

        tbody.innerHTML = '';
        if (logs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No logs found yet. Try analyzing some text or voice!</td></tr>';
            return;
        }

        logs.forEach(log => {
            const row = `
                <tr>
                    <td>${log.id}</td>
                    <td>${new Date(log.timestamp).toLocaleString()}</td>
                    <td><span class="badge bg-secondary">${log.modality}</span></td>
                    <td><span class="badge bg-primary">${log.predicted_emotion}</span></td>
                    <td>${(log.confidence * 100).toFixed(1)}%</td>
                    <td class="text-truncate" style="max-width: 150px;" title="${log.input_summary || ''}">${log.input_summary || '-'}</td>
                </tr>
            `;
            tbody.innerHTML += row;
        });
    } catch (e) {
        console.error(e);
        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-danger">Error loading logs. Check console.</td></tr>';
    }
}
