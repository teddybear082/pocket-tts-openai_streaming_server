document.addEventListener('DOMContentLoaded', async () => {
	const voiceInput = document.getElementById('voice-input');
	const voiceList = document.getElementById('voice-list');
	const voiceClearBtn = document.getElementById('voice-clear-btn');
	const customVoiceGroup = document.getElementById('custom-voice-group');
	const generateBtn = document.getElementById('generate-btn');
	const textInput = document.getElementById('text-input');
	const voiceFile = document.getElementById('voice-file');
	const outputSection = document.getElementById('output-section');
	const audioPlayer = document.getElementById('audio-player');
	const downloadBtn = document.getElementById('download-btn');
	const streamToggle = document.getElementById('stream-toggle');
	const formatSelect = document.getElementById('format-select');

	let availableVoices = [];
	let selectedVoiceId = null; // The actual value used for generation

	// Format & Streaming Logic
	function updateStreamingAvailability() {
		const fmt = formatSelect.value;
		// Server only supports streaming for PCM and WAV currently
		const supportsStreaming = ['wav', 'pcm'].includes(fmt);
		const infoLabel = document.getElementById('format-info');

		if (supportsStreaming) {
			streamToggle.disabled = false;
			streamToggle.parentElement.title = '';

			if (fmt === 'pcm') {
				infoLabel.textContent =
					"Streaming is available for Raw PCM. Note: This format creates a specialized raw stream that will not play in the browser's audio player.";
			} else {
				// WAV
				infoLabel.textContent =
					'Streaming is available for WAV. The server streams audio chunks for lower latency.';
			}
		} else {
			streamToggle.disabled = true;
			streamToggle.checked = false;
			streamToggle.parentElement.title =
				'Streaming is only available for WAV and PCM formats';

			if (fmt === 'mp3') {
				infoLabel.textContent =
					'Streaming is not available for MP3 (Server limitation). A full file will be generated and played.';
			} else {
				infoLabel.textContent = 'Streaming is not available for this format.';
			}
		}
	}

	formatSelect.addEventListener('change', updateStreamingAvailability);
	// Initialize state
	updateStreamingAvailability();

	// 1. Load Voices
	async function loadVoices() {
		try {
			const res = await fetch('/v1/voices');
			const data = await res.json();
			availableVoices = [];

			if (data.data) {
				data.data.forEach((voice) => {
					availableVoices.push({
						id: voice.id,
						label: voice.name || voice.id,
						display: voice.name || voice.id, // For search
						type: voice.type || 'builtin',
					});
				});

				// Custom option
				availableVoices.push({
					id: 'custom',
					label: 'Custom Voice',
					display: 'Custom (Upload .wav, .mp3, .flac)...',
					type: 'manual',
				});

				// Default selection: Prefer first non-custom voice
				const defaultVoice = availableVoices.find((v) => v.id !== 'custom');
				if (defaultVoice) {
					selectVoice(defaultVoice.id, false);
				}
			}
		} catch (e) {
			console.error('Failed to list voices:', e);
		}
	}

	// 2. Core Search & Selection Logic

	function selectVoice(id, closeList = true) {
		const voice = availableVoices.find((v) => v.id === id);
		if (!voice) return;

		selectedVoiceId = voice.id;
		voiceInput.value = voice.label; // Display nice name

		// Update ID Display helper
		const idDisplay = document.getElementById('voice-id-display');
		if (idDisplay) {
			if (id !== 'custom') {
				const idSpan = idDisplay.querySelector('.voice-id-text');
				if (idSpan) {
					// Clean extension from ID for cleaner display/copying
					const cleanId = voice.id.replace(
						/\.(wav|mp3|flac|safetensors)$/i,
						'',
					);
					idSpan.textContent = cleanId;
				}
				idDisplay.classList.remove('hidden');
			} else {
				idDisplay.classList.add('hidden');
			}
		}

		// Handle UI state
		voiceClearBtn.disabled = false;
		if (closeList) hideVoiceList();

		// Handle Custom
		if (id === 'custom') {
			const isDocker = window.POCKET_TTS_CONFIG?.isDocker || false;
			if (isDocker) {
				alert('Custom voices are not available in Docker mode.');
				// Fallback to the first non-custom voice if available
				const fallbackVoice = availableVoices.find((v) => v.id !== 'custom');
				if (fallbackVoice) {
					selectVoice(fallbackVoice.id || '');
				} else {
					// No valid fallback; clear selection and hide custom UI
					selectedVoiceId = null;
					voiceInput.value = '';
					voiceClearBtn.disabled = true;
					customVoiceGroup.classList.add('hidden');
				}
				return;
			}
			customVoiceGroup.classList.remove('hidden');
			document.querySelector('#custom-voice-group label').textContent =
				'Absolute Path to Audio File:';
			voiceFile.type = 'text';
			voiceFile.placeholder = 'C:\\path\\to\\voice.wav';
		} else {
			customVoiceGroup.classList.add('hidden');
		}
	}

	function renderVoiceList(filterText = '') {
		const normalizedFilter = filterText.trim().toLowerCase();
		const fragment = document.createDocumentFragment();

		let matchCount = 0;
		let firstMatchId = null;

		const isDocker = window.POCKET_TTS_CONFIG?.isDocker || false;
		const filtered = availableVoices.filter((v) => {
			if (v.id === 'custom' && isDocker) return false;
			if (!normalizedFilter) return true;
			return (
				v.id.toLowerCase().includes(normalizedFilter) ||
				v.label.toLowerCase().includes(normalizedFilter) ||
				(v.display && v.display.toLowerCase().includes(normalizedFilter))
			);
		});

		voiceList.innerHTML = '';

		if (filtered.length === 0) {
			const emptyItem = document.createElement('li');
			emptyItem.className = 'voice-list-empty';
			emptyItem.textContent = 'No matching voices';
			voiceList.appendChild(emptyItem);
		} else {
			filtered.forEach((voice) => {
				matchCount++;
				if (matchCount === 1) firstMatchId = voice.id;

				const item = document.createElement('li');
				const btn = document.createElement('button');
				btn.type = 'button';
				btn.className = 'voice-list-item';
				btn.dataset.voiceId = voice.id;

				// Better content structure
				const infoDiv = document.createElement('div');
				infoDiv.className = 'voice-info';

				const nameSpan = document.createElement('span');
				nameSpan.className = 'voice-name';
				nameSpan.textContent = voice.display || voice.label;

				const subSpan = document.createElement('span');
				subSpan.className = 'voice-sub';
				if (voice.id === 'custom') {
					subSpan.textContent = ''; // No ID for the upload button itself
				} else {
					subSpan.textContent = voice.id;
				}

				infoDiv.appendChild(nameSpan);
				if (subSpan.textContent) infoDiv.appendChild(subSpan);

				const badgeSpan = document.createElement('span');
				badgeSpan.className = 'voice-badge';

				// Format badge text: "builtin" -> "Default", "custom" -> "Custom"
				let badgeText = 'Default';
				if (voice.type === 'custom') badgeText = 'Custom';
				if (voice.type === 'manual') badgeText = 'Upload';

				badgeSpan.textContent = badgeText;

				// Add specific class for styling if needed
				badgeSpan.classList.add(
					voice.type === 'builtin' ? 'badge-builtin' : 'badge-custom',
				);

				btn.appendChild(infoDiv);
				btn.appendChild(badgeSpan);

				item.appendChild(btn);
				fragment.appendChild(item);
			});
			voiceList.appendChild(fragment);
		}

		return { count: matchCount, firstId: firstMatchId };
	}

	function showVoiceList() {
		voiceList.classList.add('show');
		renderVoiceList(
			voiceInput.value === getSelectedVoiceLabel() ? '' : voiceInput.value,
		);
	}

	function hideVoiceList() {
		// Small delay to allow click events to propagate
		setTimeout(() => {
			voiceList.classList.remove('show');
		}, 150);
	}

	function getSelectedVoiceLabel() {
		const v = availableVoices.find((v) => v.id === selectedVoiceId);
		return v ? v.label : '';
	}

	// Smart Input Handling
	voiceInput.addEventListener('focus', () => {
		// On focus, if the input value matches the current selection, wipe it to allow fresh search?
		// Or keep it? Standard combobox keeps it but selects text.
		// Let's select text so user can type over immediately.
		voiceInput.select();
		showVoiceList();
	});

	voiceInput.addEventListener('input', () => {
		voiceClearBtn.disabled = voiceInput.value.length === 0;
		// If user types, we conceptually deselect until they pick or we auto-match
		// But strictly clearing selectedVoiceId might be annoying if they just made a typo.
		// Let's keep selectedVoiceId as fallback, but filter.
		renderVoiceList(voiceInput.value);
		voiceList.classList.add('show');
	});

	voiceInput.addEventListener('keydown', (e) => {
		if (e.key === 'Escape') {
			voiceInput.value = getSelectedVoiceLabel();
			hideVoiceList();
			voiceInput.blur();
		} else if (e.key === 'Enter') {
			e.preventDefault();
			// Auto-select if 1 result
			const { count, firstId } = renderVoiceList(voiceInput.value);
			if (count === 1 && firstId) {
				selectVoice(firstId);
				voiceInput.blur();
			} else if (count > 0 && firstId) {
				// If multiple, maybe select first? Or do nothing?
				// User asked: "If I filter so much that there is just 1 result, I still have to select it"
				// implies standard Enter behavior triggers selection of top result usually.
				selectVoice(firstId);
				voiceInput.blur();
			}
		}
	});

	// Handle Blur: Auto-select if logic dictates
	voiceInput.addEventListener('blur', () => {
		// Delay logic slightly to allow Click to happen first
		setTimeout(() => {
			if (!document.activeElement.classList.contains('voice-list-item')) {
				// Validate: Is text a partial match for exactly one voice?
				const val = voiceInput.value.trim();
				if (!val) {
					// Cleared -> maybe clear selection? Or revert?
					// Let's revert to last selected for safety unless user explicitly cleared?
					// If they cleared, they probably want to clear.
					// But we need a voice to generate?
					// Let's revert if empty.
					voiceInput.value = getSelectedVoiceLabel();
					hideVoiceList();
					return;
				}

				// If the text matches the currently selected label, do nothing
				if (val === getSelectedVoiceLabel()) {
					hideVoiceList();
					return;
				}

				// Try to find a match
				// 1. Exact Name Match
				const exact = availableVoices.find(
					(v) =>
						v.label.toLowerCase() === val.toLowerCase() ||
						v.id.toLowerCase() === val.toLowerCase(),
				);
				if (exact) {
					selectVoice(exact.id);
				} else {
					// 2. Single Filter Match
					const { count, firstId } = renderVoiceList(val);
					if (count === 1) {
						selectVoice(firstId);
					} else {
						// 3. No clean match (0 or >1). Revert to last valid.
						// User said "get an error because the value from the search field is taken"
						// So passing the raw text is bad. We must force valid selection.
						voiceInput.value = getSelectedVoiceLabel();
					}
				}
				hideVoiceList();
			}
		}, 200);
	});

	// List Click Handling
	voiceList.addEventListener('mousedown', (e) => {
		// Use mousedown to trigger before blur
		const btn = e.target.closest('.voice-list-item');
		if (btn) {
			const id = btn.dataset.voiceId;
			selectVoice(id);
		}
	});

	voiceClearBtn.addEventListener('mousedown', (e) => {
		e.preventDefault(); // Prevent blur on input
		selectedVoiceId = null;
		voiceInput.value = '';
		voiceInput.focus();
		renderVoiceList('');
		showVoiceList();
		voiceClearBtn.disabled = true;

		const idDisplay = document.getElementById('voice-id-display');
		if (idDisplay) idDisplay.classList.add('hidden');
	});

	// Copy Button Logic
	const copyBtn = document.getElementById('voice-id-copy-btn');
	if (copyBtn) {
		copyBtn.addEventListener('click', async () => {
			const idText = document.querySelector('.voice-id-text')?.textContent;
			if (idText) {
				try {
					await navigator.clipboard.writeText(idText);
					const originalHTML = copyBtn.innerHTML;
					// Show checkmark
					copyBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#2ea043" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
					copyBtn.classList.add('copied');

					setTimeout(() => {
						copyBtn.innerHTML = originalHTML;
						copyBtn.classList.remove('copied');
					}, 1500);
				} catch (err) {
					console.error('Failed to copy: ', err);
					// Fallback for non-secure contexts (optional but good for localhost sometimes)
					const input = document.createElement('textarea');
					input.value = idText;
					document.body.appendChild(input);
					input.select();
					document.execCommand('copy');
					document.body.removeChild(input);
				}
			}
		});
	}

	// 3. Generate Logic
	generateBtn.addEventListener('click', async () => {
		const text = textInput.value.trim();
		if (!text) return alert('Please enter text');

		// Use the ID, not the Input Value
		let voice = selectedVoiceId;

		// Fallback: If for some reason ID is null but text exists (shouldn't happen with our blur logic), try to resolve
		if (!voice) {
			// Try to find by name from input
			const val = voiceInput.value.trim();
			const isDocker = window.POCKET_TTS_CONFIG?.isDocker || false;
			const match = availableVoices.find(
				(v) =>
					(v.label === val || v.id === val) &&
					// In Docker mode, do not allow resolving the special "custom" voice
					!(isDocker && v.id === 'custom'),
			);
			if (match) voice = match.id;
		}

		if (!voice) return alert('Please choose a valid voice from the list');

		const isDocker = window.POCKET_TTS_CONFIG?.isDocker || false;
		if (isDocker && voice === 'custom') {
			return alert(
				'The custom voice is not available in Docker mode. Please choose another voice.',
			);
		}

		if (voice === 'custom') {
			voice = voiceFile.value.trim();
			if (!voice) return alert('Please enter the path to the voice file.');
		}

		// ... rest of generation logic ...
		const stream = streamToggle.checked;
		const fmt = formatSelect.value;

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
					response_format: fmt,
					stream: stream,
				}),
			});

			if (!response.ok) {
				const err = await response.json();
				throw new Error(err.error || response.statusText);
			}

			// Currently we always fetch the full blob and play it once ready.
			// The `stream` flag is still sent to the server, but client playback
			// uses a single blob path for robustness.
			const blob = await response.blob();
			const url = URL.createObjectURL(blob);
			audioPlayer.src = url;
			downloadBtn.href = url;
			downloadBtn.download = `generated_speech.${fmt}`;

			// PCM raw audio usually won't play in standard <audio> elements
			if (fmt !== 'pcm') {
				audioPlayer
					.play()
					.catch((e) => console.warn('Auto-play blocked or failed:', e));
			}
			outputSection.classList.add('active');
		} catch (e) {
			alert('Error generating speech: ' + e.message);
		} finally {
			generateBtn.classList.remove('loading');
			generateBtn.disabled = false;
		}
	});

	// Initial load
	await loadVoices();
});
