'use strict';

/**
 * V1 config. The extension never talks to Hive directly — it only ever calls
 * our own backend, which owns the Hive API key and does result caching.
 * See docs/production-roadmap.md in the repo root for why.
 */
const CONFIG = {
	backendUrl: 'http://localhost:8000/api/analyze',
	heatmapUrl: 'http://localhost:8000/api/heatmap',
	// Bands must match js/main.js:fillResults in the existing site.
	thresholds: {
		green: 0.015,	// score below this -> green
		yellow: 0.10	// score below this -> yellow, otherwise red
	}
};

// In-memory per-session cache: imageUrl -> verdict response. Cleared when the
// service worker is evicted, which is fine for v1 (real caching lives server-side).
const resultCache = new Map();
const heatmapCache = new Map();

function scoreToVerdict(score) {
	if (score < CONFIG.thresholds.green) {
		return 'green';
	}

	if (score < CONFIG.thresholds.yellow) {
		return 'yellow';
	}

	return 'red';
}

async function fetchImageAsBlob(url) {
	const response = await fetch(url, { credentials: 'omit' });

	if (!response.ok) {
		throw new Error('Failed to fetch image (' + response.status + ')');
	}

	return response.blob();
}

async function analyzeImage(imageUrl) {
	if (resultCache.has(imageUrl)) {
		return resultCache.get(imageUrl);
	}

	const blob = await fetchImageAsBlob(imageUrl);

	const form = new FormData();
	form.append('media', blob, 'image.jpg');
	form.append('source_url', imageUrl);

	const response = await fetch(CONFIG.backendUrl, {
		method: 'POST',
		body: form
	});

	if (!response.ok) {
		throw new Error('Backend returned ' + response.status);
	}

	const data = await response.json();
	const result = {
		verdict: scoreToVerdict(data.score),
		score: data.score,
		topClass: data.top_class || null
	};

	resultCache.set(imageUrl, result);
	return result;
}

async function fetchHeatmap(imageUrl) {
	if (heatmapCache.has(imageUrl)) {
		return heatmapCache.get(imageUrl);
	}

	const blob = await fetchImageAsBlob(imageUrl);

	const form = new FormData();
	form.append('media', blob, 'image.jpg');

	const response = await fetch(CONFIG.heatmapUrl, {
		method: 'POST',
		body: form
	});

	if (!response.ok) {
		throw new Error('Heatmap backend returned ' + response.status);
	}

	const data = await response.json();
	heatmapCache.set(imageUrl, data.heatmap_base64);
	return data.heatmap_base64;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
	if (!message) {
		return undefined;
	}

	if (message.type === 'VERITAS_ANALYZE_IMAGE') {
		analyzeImage(message.url)
			.then((result) => sendResponse({ ok: true, result }))
			.catch((error) => sendResponse({ ok: false, error: error.message }));

		return true; // keep the message channel open for the async response
	}

	if (message.type === 'VERITAS_GET_HEATMAP') {
		fetchHeatmap(message.url)
			.then((heatmapBase64) => sendResponse({ ok: true, heatmapBase64 }))
			.catch((error) => sendResponse({ ok: false, error: error.message }));

		return true;
	}

	return undefined;
});
