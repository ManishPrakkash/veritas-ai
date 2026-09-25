'use strict';

(() => {
	const MIN_IMAGE_SIZE = 64; // px, skip icons/avatars/spacers smaller than this
	const BADGE_INSET = 6; // px from the image's bottom-right corner

	const trackedImages = new WeakSet();
	const badgesByImage = new Map(); // img element -> badge element
	const heatmapOverlaysByImage = new Map(); // img element -> overlay <img> element

	// Reloading the extension in chrome://extensions doesn't retroactively fix
	// this content script's copy in already-open tabs — chrome.runtime becomes
	// undefined there ("extension context invalidated"). Only a fresh page load
	// gets the new script. Guard every chrome.* call so that hits a clear error
	// state instead of an uncaught crash.
	function isExtensionContextValid() {
		return typeof chrome !== 'undefined' && !!chrome.runtime && !!chrome.runtime.id;
	}

	// Full-viewport, click-through overlay that hosts every badge. A Shadow DOM
	// keeps our styles isolated from (and safe from) the host page's CSS.
	const overlayHost = document.createElement('div');
	overlayHost.id = 'veritas-ai-overlay-root';
	Object.assign(overlayHost.style, {
		position: 'fixed',
		inset: '0',
		width: '100%',
		height: '100%',
		pointerEvents: 'none',
		zIndex: '2147483647'
	});

	const shadow = overlayHost.attachShadow({ mode: 'open' });

	const style = document.createElement('style');
	style.textContent = `
		.heatmap-overlay {
			position: absolute;
			z-index: 0;
			pointer-events: none;
			object-fit: fill;
			opacity: 0;
			transition: opacity 0.25s ease;
		}
		.heatmap-overlay.visible {
			opacity: 0.92;
		}
		.badge {
			position: absolute;
			z-index: 1;
			width: 24px;
			height: 24px;
			box-sizing: border-box;
			border-radius: 50%;
			display: flex;
			align-items: center;
			justify-content: center;
			cursor: pointer;
			pointer-events: auto;
			user-select: none;
			background: rgba(255, 255, 255, 0.72);
			backdrop-filter: blur(14px) saturate(180%);
			-webkit-backdrop-filter: blur(14px) saturate(180%);
			box-shadow: 0 3px 10px rgba(0, 0, 0, 0.22), 0 0 0 0.5px rgba(0, 0, 0, 0.08);
			border: 2px solid transparent;
			color: #3a3a3c;
			transition: transform 0.1s ease, background-color 0.15s ease, box-shadow 0.15s ease;
		}
		.badge:hover {
			transform: scale(1.15);
		}
		.badge svg {
			width: 12px;
			height: 12px;
			display: block;
		}
		.badge.idle svg { opacity: 0.8; }
		.badge.red svg,
		.badge.green svg { width: 15px; height: 15px; }
		.badge.loading {
			border-color: rgba(120, 120, 128, 0.3);
			border-top-color: rgba(120, 120, 128, 0.9);
			animation: spin 0.8s linear infinite;
		}
		.badge.green {
			background: #34c759;
			color: #fff;
			box-shadow: 0 3px 10px rgba(52, 199, 89, 0.45), 0 0 0 0.5px rgba(0, 0, 0, 0.08);
		}
		.badge.yellow {
			background: #ffcc00;
			color: #1c1c1e;
			box-shadow: 0 3px 10px rgba(255, 204, 0, 0.45), 0 0 0 0.5px rgba(0, 0, 0, 0.08);
		}
		.badge.red {
			background: #ff3b30;
			color: #fff;
			box-shadow: 0 3px 10px rgba(255, 59, 48, 0.45), 0 0 0 0.5px rgba(0, 0, 0, 0.08);
		}
		.badge.error {
			background: #3a3a3c;
			color: #fff;
		}
		@keyframes spin {
			from { transform: rotate(0deg); }
			to { transform: rotate(360deg); }
		}
		@media (prefers-color-scheme: dark) {
			.badge {
				background: rgba(44, 44, 46, 0.72);
				color: #f2f2f7;
			}
		}

		.settings-panel {
			position: absolute;
			min-width: 210px;
			background: rgba(255, 255, 255, 0.82);
			backdrop-filter: blur(20px) saturate(180%);
			-webkit-backdrop-filter: blur(20px) saturate(180%);
			border-radius: 14px;
			box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28), 0 0 0 0.5px rgba(0, 0, 0, 0.08);
			padding: 10px 12px;
			font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
			color: #1c1c1e;
			pointer-events: auto;
			opacity: 0;
			transform: scale(0.94);
			transform-origin: bottom right;
			transition: opacity 0.12s ease, transform 0.12s ease;
		}
		.settings-panel.open {
			opacity: 1;
			transform: scale(1);
		}
		.settings-title {
			font-weight: 600;
			font-size: 11px;
			color: #6e6e73;
			text-transform: uppercase;
			letter-spacing: 0.03em;
			margin: 0 0 8px 2px;
		}
		.settings-row {
			display: flex;
			align-items: center;
			gap: 8px;
			padding: 2px;
		}
		.settings-label {
			flex: 1;
			display: flex;
			align-items: center;
			gap: 6px;
		}
		.settings-soon {
			font-size: 9px;
			font-weight: 700;
			color: #8e8e93;
			background: rgba(142, 142, 147, 0.18);
			padding: 2px 5px;
			border-radius: 6px;
			text-transform: uppercase;
			letter-spacing: 0.03em;
		}
		.settings-hint {
			font-size: 11px;
			color: #6e6e73;
			margin: 6px 2px 0;
			line-height: 1.35;
		}
		.toggle {
			flex-shrink: 0;
			width: 34px;
			height: 20px;
			border-radius: 999px;
			background: #d1d1d6;
			border: none;
			padding: 2px;
			cursor: pointer;
			position: relative;
			transition: background-color 0.15s ease;
		}
		.toggle[aria-checked="true"] {
			background: #34c759;
		}
		.toggle-knob {
			display: block;
			width: 16px;
			height: 16px;
			border-radius: 50%;
			background: #fff;
			box-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
			transition: transform 0.15s ease;
		}
		.toggle[aria-checked="true"] .toggle-knob {
			transform: translateX(14px);
		}
		@media (prefers-color-scheme: dark) {
			.settings-panel {
				background: rgba(32, 32, 34, 0.82);
				color: #f2f2f7;
				box-shadow: 0 10px 30px rgba(0, 0, 0, 0.55), 0 0 0 0.5px rgba(255, 255, 255, 0.08);
			}
			.settings-title { color: #98989d; }
			.settings-soon { color: #98989d; background: rgba(142, 142, 147, 0.28); }
			.settings-hint { color: #98989d; }
			.toggle { background: #48484a; }
		}
	`;
	shadow.appendChild(style);

	function mountOverlay() {
		if (document.body && !overlayHost.isConnected) {
			document.documentElement.appendChild(overlayHost);
		}
	}

	function isEligible(img) {
		const rect = img.getBoundingClientRect();
		return rect.width >= MIN_IMAGE_SIZE && rect.height >= MIN_IMAGE_SIZE && !!img.src;
	}

	function positionBadge(img, badge) {
		const rect = img.getBoundingClientRect();
		badge.style.left = (rect.right - 24 - BADGE_INSET) + 'px';
		badge.style.top = (rect.bottom - 24 - BADGE_INSET) + 'px';
	}

	function positionHeatmapOverlay(img, overlayImg) {
		const rect = img.getBoundingClientRect();
		overlayImg.style.left = rect.left + 'px';
		overlayImg.style.top = rect.top + 'px';
		overlayImg.style.width = rect.width + 'px';
		overlayImg.style.height = rect.height + 'px';
	}

	function removeHeatmapOverlay(img) {
		const overlayImg = heatmapOverlaysByImage.get(img);
		if (overlayImg) {
			overlayImg.remove();
			heatmapOverlaysByImage.delete(img);
		}
	}

	function repositionAll() {
		badgesByImage.forEach((badge, img) => {
			if (!img.isConnected) {
				badge.remove();
				badgesByImage.delete(img);
				removeHeatmapOverlay(img);
				return;
			}

			positionBadge(img, badge);
		});

		heatmapOverlaysByImage.forEach((overlayImg, img) => {
			if (!img.isConnected) {
				removeHeatmapOverlay(img);
				return;
			}

			positionHeatmapOverlay(img, overlayImg);
		});
	}

	// --- Settings panel (right-click a badge) ---------------------------------

	let heatmapEnabled = false;
	let activeSettingsPanel = null;

	if (isExtensionContextValid()) {
		chrome.storage.local.get(['veritasHeatmapEnabled'], (result) => {
			heatmapEnabled = !!result.veritasHeatmapEnabled;
		});
	}

	function closeSettingsPanel() {
		if (!activeSettingsPanel) {
			return;
		}

		activeSettingsPanel.remove();
		activeSettingsPanel = null;
	}

	function positionSettingsPanel(badge, panel) {
		const rect = badge.getBoundingClientRect();
		// Right-align the panel with the badge and sit it directly above,
		// like a popover pointing down at its trigger.
		const right = Math.max(8, window.innerWidth - rect.right);
		panel.style.left = 'auto';
		panel.style.top = 'auto';
		panel.style.right = right + 'px';
		panel.style.bottom = (window.innerHeight - rect.top + 8) + 'px';
	}

	function openSettingsPanel(badge) {
		closeSettingsPanel();

		const panel = document.createElement('div');
		panel.className = 'settings-panel';

		const title = document.createElement('div');
		title.className = 'settings-title';
		title.textContent = 'Veritas AI';
		panel.appendChild(title);

		const row = document.createElement('div');
		row.className = 'settings-row';

		const label = document.createElement('div');
		label.className = 'settings-label';
		label.innerHTML = 'Heatmap overlay <span class="settings-soon">Beta</span>';
		row.appendChild(label);

		const toggle = document.createElement('button');
		toggle.type = 'button';
		toggle.className = 'toggle';
		toggle.setAttribute('role', 'switch');
		toggle.setAttribute('aria-checked', String(heatmapEnabled));
		toggle.title = 'Highlight which part of the image looks AI-generated';

		const knob = document.createElement('span');
		knob.className = 'toggle-knob';
		toggle.appendChild(knob);

		toggle.addEventListener('click', (event) => {
			event.preventDefault();
			event.stopPropagation();
			heatmapEnabled = !heatmapEnabled;
			toggle.setAttribute('aria-checked', String(heatmapEnabled));

			if (isExtensionContextValid()) {
				chrome.storage.local.set({ veritasHeatmapEnabled: heatmapEnabled });
			}
		});

		row.appendChild(toggle);
		panel.appendChild(row);

		const hint = document.createElement('div');
		hint.className = 'settings-hint';
		hint.textContent = 'Experimental — only reliably detects SDXL-style images so far.';
		panel.appendChild(hint);

		shadow.appendChild(panel);
		positionSettingsPanel(badge, panel);
		activeSettingsPanel = panel;

		requestAnimationFrame(() => panel.classList.add('open'));
	}

	document.addEventListener('click', (event) => {
		if (activeSettingsPanel && !event.composedPath().includes(activeSettingsPanel)) {
			closeSettingsPanel();
		}
	});

	document.addEventListener('keydown', (event) => {
		if (event.key === 'Escape') {
			closeSettingsPanel();
		}
	});

	// Thin, SF-Symbols-style line/fill icons, one per badge state. Loading has
	// none — the spinning ring border (see CSS) is the whole affordance.
	const ICONS = {
		idle: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L14 10L22 12L14 14L12 22L10 14L2 12L10 10Z"/></svg>',
		green: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 1L14.5 9.5L23 12L14.5 14.5L12 23L9.5 14.5L1 12L9.5 9.5Z"/></svg>',
		yellow: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3.5L21 19.5H3Z"/><line x1="12" y1="9" x2="12" y2="13.5"/><circle cx="12" cy="16.5" r="0.9" fill="currentColor" stroke="none"/></svg>',
		red: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 1L14.5 9.5L23 12L14.5 14.5L12 23L9.5 14.5L1 12L9.5 9.5Z"/></svg>',
		error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><line x1="12" y1="7.5" x2="12" y2="13"/><circle cx="12" cy="16" r="0.9" fill="currentColor" stroke="none"/></svg>'
	};

	function setBadgeState(badge, state, title) {
		badge.className = 'badge ' + state;
		badge.innerHTML = ICONS[state] || '';
		badge.title = title || '';
	}

	function fetchAndShowHeatmap(img) {
		if (!isExtensionContextValid()) {
			return;
		}

		chrome.runtime.sendMessage(
			{ type: 'VERITAS_GET_HEATMAP', url: img.currentSrc || img.src },
			(response) => {
				if (chrome.runtime.lastError || !response || !response.ok || !img.isConnected) {
					return;
				}

				let overlayImg = heatmapOverlaysByImage.get(img);
				if (!overlayImg) {
					overlayImg = document.createElement('img');
					overlayImg.className = 'heatmap-overlay';
					shadow.appendChild(overlayImg);
					heatmapOverlaysByImage.set(img, overlayImg);
				}

				overlayImg.src = 'data:image/jpeg;base64,' + response.heatmapBase64;
				positionHeatmapOverlay(img, overlayImg);
				requestAnimationFrame(() => overlayImg.classList.add('visible'));
			}
		);
	}

	function onBadgeClick(img, badge) {
		if (badge.classList.contains('loading')) {
			return;
		}

		if (!isExtensionContextValid()) {
			setBadgeState(badge, 'error', 'Veritas AI: extension was updated — reload this page and try again.');
			return;
		}

		setBadgeState(badge, 'loading', 'Analyzing…');

		chrome.runtime.sendMessage(
			{ type: 'VERITAS_ANALYZE_IMAGE', url: img.currentSrc || img.src },
			(response) => {
				if (chrome.runtime.lastError || !response) {
					setBadgeState(badge, 'error', 'Veritas AI: extension error. Is the backend running?');
					return;
				}

				if (!response.ok) {
					setBadgeState(badge, 'error', 'Veritas AI: ' + response.error);
					return;
				}

				const { verdict, score, topClass } = response.result;
				const pct = Math.round(score * 100);
				const title = 'Veritas AI: ' + pct + '% AI likelihood' + (topClass ? ' (' + topClass + ')' : '');
				setBadgeState(badge, verdict, title);

				if (heatmapEnabled) {
					fetchAndShowHeatmap(img);
				}
			}
		);
	}

	function createBadge(img) {
		const badge = document.createElement('div');
		setBadgeState(badge, 'idle', 'Veritas AI: click to check this image');
		badge.addEventListener('click', (event) => {
			event.preventDefault();
			event.stopPropagation();
			onBadgeClick(img, badge);
		});
		badge.addEventListener('contextmenu', (event) => {
			event.preventDefault();
			event.stopPropagation();
			openSettingsPanel(badge);
		});

		shadow.appendChild(badge);
		badgesByImage.set(img, badge);
		positionBadge(img, badge);
	}

	const visibilityObserver = new IntersectionObserver(
		(entries) => {
			entries.forEach((entry) => {
				const img = entry.target;

				if (entry.isIntersecting) {
					if (!badgesByImage.has(img)) {
						createBadge(img);
					}
				} else {
					const badge = badgesByImage.get(img);
					if (badge) {
						badge.remove();
						badgesByImage.delete(img);
					}
					removeHeatmapOverlay(img);
				}
			});
		},
		{ rootMargin: '150px', threshold: 0.01 }
	);

	function considerImage(img) {
		if (trackedImages.has(img) || !isEligible(img)) {
			return;
		}

		trackedImages.add(img);
		visibilityObserver.observe(img);
	}

	function scanForImages(root) {
		root.querySelectorAll('img').forEach(considerImage);
	}

	const mutationObserver = new MutationObserver((mutations) => {
		mutations.forEach((mutation) => {
			mutation.addedNodes.forEach((node) => {
				if (!(node instanceof Element)) {
					return;
				}

				if (node.tagName === 'IMG') {
					considerImage(node);
				} else {
					scanForImages(node);
				}
			});
		});
	});

	let repositionScheduled = false;
	function scheduleReposition() {
		if (repositionScheduled) {
			return;
		}

		repositionScheduled = true;
		requestAnimationFrame(() => {
			repositionScheduled = false;
			repositionAll();
			closeSettingsPanel();
		});
	}

	function init() {
		mountOverlay();
		scanForImages(document);
		mutationObserver.observe(document.documentElement, { childList: true, subtree: true });
		window.addEventListener('scroll', scheduleReposition, { passive: true, capture: true });
		window.addEventListener('resize', scheduleReposition, { passive: true });
	}

	if (document.readyState === 'loading') {
		document.addEventListener('DOMContentLoaded', init);
	} else {
		init();
	}
})();
