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
	const speedSlider = document.getElementById('speed-slider');
	const speedValue = document.getElementById('speed-value');
	const statsRow = document.getElementById('stats-row');
	const statLatency = document.getElementById('stat-latency');
	const statDuration = document.getElementById('stat-duration');
	const statRtf = document.getElementById('stat-rtf');

	let availableVoices = [];
	let selectedVoiceId = null;
	let generateStartTime = 0;
	let currentStatsId = null;

	// Format & Streaming Logic
	function updateStreamingAvailability() {
		const fmt = formatSelect.value;
		const infoLabel = document.getElementById('format-info');

		// All formats support streaming now — the server encodes on the fly via
		// ffmpeg. PCM is raw bytes that won't play in the <audio> element.
		streamToggle.disabled = false;
		streamToggle.parentElement.title = '';

		if (fmt === 'pcm') {
			infoLabel.textContent =
				"Streaming is available for Raw PCM. Note: this format produces raw bytes that will not play in the browser's audio player.";
		} else if (fmt === 'wav') {
			infoLabel.textContent =
				'Streaming is available for WAV. The server streams audio chunks for lower latency.';
		} else {
			infoLabel.textContent =
				`Streaming is available for ${fmt.toUpperCase()}. Audio is streamed and begins playing as soon as enough data arrives.`;
		}
	}

	formatSelect.addEventListener('change', updateStreamingAvailability);
	// Initialize state
	updateStreamingAvailability();

	speedSlider.addEventListener('input', () => {
		speedValue.textContent = parseFloat(speedSlider.value).toFixed(2).replace(/\.?0+$/, '');
	});

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
		const speed = parseFloat(speedSlider.value);

		generateBtn.classList.add('loading');
		generateBtn.disabled = true;
		outputSection.classList.remove('active');
		statsRow.hidden = true;
		generateStartTime = performance.now();

		function showStats(endTime) {
			const elapsed = (endTime - generateStartTime) / 1000;
			statLatency.textContent = `Latency: ${elapsed.toFixed(2)}s`;
			statDuration.textContent = '';
			statRtf.textContent = '';
			statsRow.hidden = false;
		}

		async function pollStreamStats() {
			if (!currentStatsId) return;
			try {
				const res = await fetch(`/v1/audio/speech/stats?id=${currentStatsId}`);
				if (!res.ok) return;
				const data = await res.json();
				if (data.generation_time != null && data.audio_duration != null) {
					statDuration.textContent = `Duration: ${data.audio_duration.toFixed(2)}s`;
					const rtf = data.generation_time > 0
						? data.audio_duration / data.generation_time
						: 0;
					statRtf.textContent = `RTF: ${rtf.toFixed(1)}x`;
					currentStatsId = null;
				}
			} catch {
			}
		}

		try {
			if (stream && fmt !== 'pcm') {
				currentStatsId = crypto.randomUUID();
				const params = new URLSearchParams({
					input: text,
					voice: voice,
					response_format: fmt,
					stats_id: currentStatsId,
				});
				if (speed !== 1.0) params.set('speed', String(speed));
				const streamUrl = `/v1/audio/speech?${params}`;
				outputSection.classList.add('active');
				audioPlayer.src = streamUrl;
				audioPlayer.oncanplay = () => {
					showStats(performance.now());
					audioPlayer.oncanplay = null;
				};
				audioPlayer.onended = () => {
					pollStreamStats();
					audioPlayer.onended = null;
				};
				audioPlayer.play().catch((e) => console.warn('Auto-play blocked:', e));

				downloadBtn.href = streamUrl;
				downloadBtn.download = `generated_speech.${fmt}`;
				downloadBtn.onclick = null;
			} else {
				const response = await fetch('/v1/audio/speech', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						model: 'pocket-tts',
						input: text,
						voice: voice,
						response_format: fmt,
						stream: false,
						...(speed !== 1.0 && { speed }),
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
				downloadBtn.download = `generated_speech.${fmt}`;
				downloadBtn.onclick = null;

				audioPlayer.onloadedmetadata = () => {
					const dur = audioPlayer.duration;
					const elapsed = (performance.now() - generateStartTime) / 1000;
					const rtf = dur / elapsed;
					statLatency.textContent = `Gen: ${elapsed.toFixed(2)}s`;
					statDuration.textContent = `Duration: ${dur.toFixed(2)}s`;
					statRtf.textContent = `RTF: ${rtf.toFixed(1)}x`;
					statsRow.hidden = false;
				};

				if (fmt !== 'pcm') {
					audioPlayer
						.play()
						.catch((e) => console.warn('Auto-play blocked or failed:', e));
				}
				outputSection.classList.add('active');
			}
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

// ============================================================
// Model Settings panel
// ============================================================

const modelUI = {
    activeLabel: document.getElementById('active-model-label'),
    quantizeBadge: document.getElementById('quantize-badge'),
    sessionBadge: document.getElementById('session-badge'),
    loadingIndicator: document.getElementById('loading-indicator'),
    loadingTargetLabel: document.getElementById('loading-target-label'),
    languageSelect: document.getElementById('language-select'),
    quantizeToggle: document.getElementById('quantize-toggle'),
    applyBtn: document.getElementById('apply-model-btn'),
    nonEnglishWarning: document.getElementById('non-english-warning'),
    modelPathLockedNotice: document.getElementById('model-path-locked-notice'),
    sessionOnlyNotice: document.getElementById('session-only-notice'),
    applyError: document.getElementById('apply-error'),
    generateBtn: document.getElementById('generate-btn'),
};

let currentModelState = null;
let pollTimer = null;
let pollDeadline = 0;

function populateLanguageOptions(languages) {
    if (modelUI.languageSelect.options.length > 0) return;  // already populated
    for (const lang of languages) {
        const opt = document.createElement('option');
        opt.value = lang;
        opt.textContent = lang;
        modelUI.languageSelect.appendChild(opt);
    }
}

function setHidden(el, hidden) {
    if (hidden) { el.setAttribute('hidden', ''); }
    else        { el.removeAttribute('hidden'); }
}

// Backend reports `value: null` when the server was started without a
// --language flag. Pocket-tts treats that as "english" internally, so we
// surface the same string in the UI to keep the dropdown, label, and Apply
// diff comparison consistent.
function effectiveActiveValue(state) {
    return state.active.value || 'english';
}

function updateUIForState(state) {
    currentModelState = state;
    populateLanguageOptions(state.available_languages);

    const activeLang = effectiveActiveValue(state);

    // Header labels
    modelUI.activeLabel.textContent = activeLang;
    setHidden(modelUI.quantizeBadge, !state.active.quantize);
    setHidden(modelUI.sessionBadge, !state.differs_from_boot);

    // Loading indicator
    if (state.loading && state.loading_target) {
        modelUI.loadingTargetLabel.textContent = `→ ${state.loading_target.value}`;
        setHidden(modelUI.loadingIndicator, false);
    } else {
        setHidden(modelUI.loadingIndicator, true);
    }

    // Dropdown reflects active (not the pending target).
    if (modelUI.languageSelect.value !== activeLang) {
        modelUI.languageSelect.value = activeLang;
    }
    modelUI.quantizeToggle.checked = state.active.quantize;

    // Lock state
    const locked = state.model_path_locked;
    modelUI.languageSelect.disabled = locked || state.loading;
    modelUI.quantizeToggle.disabled = locked || state.loading;
    setHidden(modelUI.modelPathLockedNotice, !locked);

    // Warnings
    const selectedLang = modelUI.languageSelect.value;
    const isEnglishVariant = selectedLang && selectedLang.startsWith('english');
    setHidden(modelUI.nonEnglishWarning, locked || isEnglishVariant);

    // Session-only notice
    setHidden(modelUI.sessionOnlyNotice, !state.differs_from_boot);

    // Apply button
    updateApplyButton();

    // Error banner from last failed reload — also clear it once the backend
    // reports no error (e.g. after a successful subsequent reload).
    if (state.last_error) {
        modelUI.applyError.textContent = state.last_error;
        setHidden(modelUI.applyError, false);
    } else {
        modelUI.applyError.textContent = '';
        setHidden(modelUI.applyError, true);
    }

    // Generate button disabled during load
    modelUI.generateBtn.disabled = state.loading;
    modelUI.generateBtn.title = state.loading ? 'Model is loading…' : '';
}

function updateApplyButton() {
    if (!currentModelState) return;
    const { active, model_path_locked, loading } = currentModelState;
    const activeLang = effectiveActiveValue(currentModelState);
    const targetLang = modelUI.languageSelect.value;
    const targetQuantize = modelUI.quantizeToggle.checked;
    const differs = targetLang !== activeLang || targetQuantize !== active.quantize;
    modelUI.applyBtn.disabled = loading || model_path_locked || !differs;
}

async function fetchModelState() {
    try {
        const resp = await fetch('/v1/model');
        if (!resp.ok) throw new Error(`GET /v1/model → ${resp.status}`);
        const state = await resp.json();
        updateUIForState(state);

        if (!state.loading && pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
    } catch (err) {
        console.warn('Failed to fetch model state:', err);
    }
}

function startPolling() {
    if (pollTimer) return;
    pollDeadline = Date.now() + 120_000;  // 2 min timeout
    pollTimer = setInterval(() => {
        if (Date.now() > pollDeadline) {
            clearInterval(pollTimer);
            pollTimer = null;
            modelUI.applyError.textContent =
                'Model load timed out after 2 minutes. Check server logs.';
            setHidden(modelUI.applyError, false);
            return;
        }
        fetchModelState();
    }, 1000);
}

async function applyModel() {
    setHidden(modelUI.applyError, true);
    modelUI.applyBtn.disabled = true;

    try {
        const resp = await fetch('/v1/model', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                language: modelUI.languageSelect.value,
                quantize: modelUI.quantizeToggle.checked,
            }),
        });
        if (resp.status === 202) {
            startPolling();
            // Immediately refresh to show loading state.
            fetchModelState();
        } else {
            const body = await resp.json();
            modelUI.applyError.textContent =
                body.error || `Server returned ${resp.status}`;
            setHidden(modelUI.applyError, false);
            // Revert dropdown to the normalized active value so we never
            // leave the select on an empty string when the backend reports
            // the default model with value=null.
            if (currentModelState) {
                modelUI.languageSelect.value =
                    effectiveActiveValue(currentModelState);
            }
            updateApplyButton();
        }
    } catch (err) {
        modelUI.applyError.textContent = `Apply failed: ${err.message}`;
        setHidden(modelUI.applyError, false);
        updateApplyButton();
    }
}

modelUI.languageSelect.addEventListener('change', updateApplyButton);
modelUI.quantizeToggle.addEventListener('change', updateApplyButton);
modelUI.applyBtn.addEventListener('click', applyModel);

// Kick off on page load.
fetchModelState().then(() => {
    if (currentModelState?.loading) startPolling();
});
