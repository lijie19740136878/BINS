document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const status = document.getElementById('status');
    const transcription = document.getElementById('transcription');

    let recognition;
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
    } else if ('SpeechRecognition' in window) {
        recognition = new SpeechRecognition();
    } else {
        alert("Your browser does not support speech recognition.");
    }

    if (recognition) {
        recognition.continuous = true;
        recognition.interimResults = true;

        recognition.onstart = () => {
            status.textContent = 'Listening...';
            startBtn.disabled = true;
            stopBtn.disabled = false;
        };

        recognition.onend = () => {
            status.textContent = 'Click "Start Recording" to begin';
            startBtn.disabled = false;
            stopBtn.disabled = true;
        };

        recognition.onresult = (event) => {
            let interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    transcription.value += event.results[i][0].transcript + '\n';
                    saveTranscription(event.results[i][0].transcript);
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            transcription.value += interimTranscript;
        };

        startBtn.addEventListener('click', () => {
            recognition.start();
        });

        stopBtn.addEventListener('click', () => {
            recognition.stop();
        });
    }
});

function saveTranscription(text) {
    const data = new Blob([text + '\n'], { type: 'text/plain' });
    const url = window.URL.createObjectURL(data);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = 'recognized_text.txt';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
}
