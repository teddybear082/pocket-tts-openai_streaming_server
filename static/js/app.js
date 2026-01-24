document.addEventListener('DOMContentLoaded', async () => {
    const voiceSelect = document.getElementById('voice-select');
    const customVoiceGroup = document.getElementById('custom-voice-group');
    const generateBtn = document.getElementById('generate-btn');
    const textInput = document.getElementById('text-input');
    const voiceFile = document.getElementById('voice-file');
    const outputSection = document.getElementById('output-section');
    const audioPlayer = document.getElementById('audio-player');
    const downloadBtn = document.getElementById('download-btn');
    const streamToggle = document.getElementById('stream-toggle');

    // 1. Load Voices
    async function loadVoices() {
        try {
            const res = await fetch('/v1/voices');
            const data = await res.json();

            // Clear existing except custom
            voiceSelect.innerHTML = '<option value="custom">Custom (Upload .wav, .mp3, .flac)...</option>';

            if (data.data) {
                // Add built-ins first
                data.data.forEach(voice => {
                    const opt = document.createElement('option');
                    opt.value = voice.id;
                    opt.textContent = voice.name || voice.id;
                    voiceSelect.insertBefore(opt, voiceSelect.firstChild); // Prepend
                });

                // Select first by default
                if (voiceSelect.options.length > 0) {
                    voiceSelect.selectedIndex = 0;
                }
            }
        } catch (e) {
            console.error("Failed to list voices:", e);
        }
    }

    await loadVoices();

    // 2. Handle Voice Selection Change
    voiceSelect.addEventListener('change', (e) => {
        if (e.target.value === 'custom') {
            // Setup for Custom Path
            document.querySelector('#custom-voice-group label').textContent = "Absolute Path to Audio File:";
            voiceFile.type = 'text';
            voiceFile.placeholder = "C:\\path\\to\\voice.wav";
            customVoiceGroup.classList.remove('hidden');
        } else {
            customVoiceGroup.classList.add('hidden');
        }
    });

    // 3. Generate Logic
    generateBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) return alert("Please enter text");

        const stream = streamToggle.checked;

        let voice = voiceSelect.value;
        if (voice === 'custom') {
            voice = voiceFile.value.trim();
            if (!voice) return alert("Please enter the path to the voice file.");
        }

        generateBtn.classList.add('loading');
        generateBtn.disabled = true;
        outputSection.classList.remove('active');

        try {
            const response = await fetch('/v1/audio/speech', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: 'pocket-tts',
                    input: text,
                    voice: voice,
                    response_format: 'wav',
                    stream: stream
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || response.statusText);
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            audioPlayer.src = url;
            downloadBtn.href = url;
            downloadBtn.download = "generated_speech.wav";

            audioPlayer.play();
            outputSection.classList.add('active');

        } catch (e) {
            alert("Error generating speech: " + e.message);
        } finally {
            generateBtn.classList.remove('loading');
            generateBtn.disabled = false;
        }
    });
});
