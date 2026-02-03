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
	const speedSlider = document.getElementById('speed-slider');
	const speedValue = document.getElementById('speed-value');

	// Update speed label
	speedSlider.addEventListener('input', (e) => {
		speedValue.textContent = e.target.value;
	});

	// 1. Load Voices
	async function loadVoices() {
		try {
			const res = await fetch('/v1/voices');
			const data = await res.json();

			// Clear existing
			voiceSelect.innerHTML = '';

			if (data.data) {
				// Add voices in order (built-in first, then custom)
				data.data.forEach((voice) => {
					const opt = document.createElement('option');
					opt.value = voice.id;
					opt.textContent = voice.name || voice.id;
					voiceSelect.appendChild(opt);
				});

				// Add custom upload option at the end
				const customOpt = document.createElement('option');
				customOpt.value = 'custom';
				customOpt.textContent = 'Custom (Upload .wav, .mp3, .flac)...';
				voiceSelect.appendChild(customOpt);

				// Select first (alba) by default
				if (voiceSelect.options.length > 0) {
					voiceSelect.selectedIndex = 0;
				}
			}
		} catch (e) {
			console.error('Failed to list voices:', e);
		}
	}

	await loadVoices();

	// 2. Handle Voice Selection Change (hide custom option in Docker mode)
	const isDocker = window.POCKET_TTS_CONFIG?.isDocker || false;

	// Remove the "Custom" option when running in Docker (paths won't work)
	if (isDocker) {
		const customOption = voiceSelect.querySelector('option[value="custom"]');
		if (customOption) customOption.remove();
	}

	voiceSelect.addEventListener('change', (e) => {
		if (e.target.value === 'custom' && !isDocker) {
			// Setup for Custom Path (only for native/exe mode)
			document.querySelector('#custom-voice-group label').textContent =
				'Absolute Path to Audio File:';
			voiceFile.type = 'text';
			voiceFile.placeholder = 'C:\\path\\to\\voice.wav';
			customVoiceGroup.classList.remove('hidden');
		} else {
			customVoiceGroup.classList.add('hidden');
		}
	});

	// 3. Generate Logic
	generateBtn.addEventListener('click', async () => {
		const text = textInput.value.trim();
		if (!text) return alert('Please enter text');

		const stream = streamToggle.checked;
		const speed = parseFloat(speedSlider.value);

		let voice = voiceSelect.value;
		if (voice === 'custom') {
			voice = voiceFile.value.trim();
			if (!voice) return alert('Please enter the path to the voice file.');
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
					stream: stream,
					speed: speed,
				}),
			});

			if (!response.ok) {
				const err = await response.json();
				throw new Error(err.error || response.statusText);
			}

			const blob = await response.blob();
			const url = URL.createObjectURL(blob);
			audioPlayer.src = url;
			downloadBtn.href = url;
			downloadBtn.download = 'generated_speech.wav';

			audioPlayer.play();
			outputSection.classList.add('active');
		} catch (e) {
			alert('Error generating speech: ' + e.message);
		} finally {
			generateBtn.classList.remove('loading');
			generateBtn.disabled = false;
		}
	});
});
