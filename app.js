// ── CONFIG ────────────────────────────────────────────────────────────────
const DEFAULT_BACKEND = "https://triad-ai-helper.onrender.com";
const PDF_SIZE_WARN_BYTES = 5 * 1024 * 1024;

function getBackendUrl() { return localStorage.getItem('triad_backend_url') || DEFAULT_BACKEND; }
let BACKEND = getBackendUrl();

// ── AUTH STATE ────────────────────────────────────────────────────────────
let currentUser = null;

function getToken() { return localStorage.getItem('triad_token') || null; }
function setToken(t) { localStorage.setItem('triad_token', t); }
function clearToken() { localStorage.removeItem('triad_token'); }

function authHeaders() {
    const t = getToken();
    return t ? { 'Authorization': 'Bearer ' + t, 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' };
}

function extractErrorMsg(err) {
    if (!err || !err.detail) return null;
    if (typeof err.detail === 'string') return err.detail;
    if (Array.isArray(err.detail) && err.detail.length > 0) return err.detail.map(e => e.msg).join(', ');
    return JSON.stringify(err.detail);
}

// ── AUTH UI ───────────────────────────────────────────────────────────────
function authSwitchTab(tab) {
    document.getElementById('tab-login').classList.toggle('active', tab === 'login');
    document.getElementById('tab-register').classList.toggle('active', tab === 'register');
    document.getElementById('form-login').classList.toggle('active', tab === 'login');
    document.getElementById('form-register').classList.toggle('active', tab === 'register');
    document.getElementById('login-error').style.display = 'none';
    document.getElementById('register-error').style.display = 'none';
    document.getElementById('register-success').style.display = 'none';
}

function showAuthError(id, msg) {
    const el = document.getElementById(id);
    el.textContent = msg; el.style.display = 'block';
}

async function doLogin() {
    const email = document.getElementById('login-email').value.trim();
    const password = document.getElementById('login-password').value;
    document.getElementById('login-error').style.display = 'none';
    if (!email || !password) { showAuthError('login-error', 'Please fill in all fields.'); return; }
    const btn = document.getElementById('login-btn');
    btn.disabled = true; btn.textContent = 'Signing in...';
    try {
        const res = await fetch(`${BACKEND}/auth/login`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        const data = await res.json();
        if (!res.ok) { showAuthError('login-error', extractErrorMsg(data) || 'Login failed.'); return; }
        setToken(data.token);
        currentUser = data.user;
        enterApp();
    } catch (e) {
        showAuthError('login-error', 'Cannot reach backend. Make sure python backend.py is running.');
    } finally {
        btn.disabled = false; btn.textContent = 'Sign In';
    }
}

async function doRegister() {
    const fullName = document.getElementById('reg-fullname').value.trim();
    const username = document.getElementById('reg-username').value.trim();
    const email = document.getElementById('reg-email').value.trim();
    const password = document.getElementById('reg-password').value;
    document.getElementById('register-error').style.display = 'none';
    document.getElementById('register-success').style.display = 'none';
    if (!username || !email || !password) { showAuthError('register-error', 'Username, email and password are required.'); return; }
    const btn = document.getElementById('register-btn');
    btn.disabled = true; btn.textContent = 'Creating account...';
    try {
        const res = await fetch(`${BACKEND}/auth/register`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ full_name: fullName, username, email, password })
        });
        const data = await res.json();
        if (!res.ok) { showAuthError('register-error', extractErrorMsg(data) || 'Registration failed.'); return; }
        setToken(data.token);
        currentUser = data.user;
        enterApp();
    } catch (e) {
        showAuthError('register-error', 'Cannot reach backend. Make sure python backend.py is running.');
    } finally {
        btn.disabled = false; btn.textContent = 'Create Account';
    }
}

async function doLogout() {
    try {
        await fetch(`${BACKEND}/auth/logout`, { method: 'POST', headers: authHeaders() });
    } catch (e) { /* silent */ }
    clearToken();
    currentUser = null;
    document.getElementById('main-app').style.display = 'none';
    document.getElementById('auth-screen').style.display = 'flex';
    authSwitchTab('login');
    document.getElementById('login-email').value = '';
    document.getElementById('login-password').value = '';
}

function enterApp() {
    document.getElementById('auth-screen').style.display = 'none';
    document.getElementById('main-app').style.display = 'block';
    updateHeaderUser();
    populateProfileForm();
    checkBackend();
    // Always land on Voice tab after login — never restore last session tab
    switchTab('voice');
    startWakeListener();
}

function updateHeaderUser() {
    if (!currentUser) return;
    const initials = (currentUser.full_name || currentUser.username || '?')[0].toUpperCase();
    const avatarEl = document.getElementById('header-avatar');
    const nameEl = document.getElementById('header-username');
    nameEl.textContent = currentUser.username || currentUser.full_name || 'User';
    const localPhoto = getLocalAvatar(currentUser.id);
    const src = localPhoto || currentUser.avatar_url || null;
    if (src) {
        avatarEl.innerHTML = '<img src="' + src + '" alt="avatar" style="width:100%;height:100%;object-fit:cover;border-radius:50%;" />';
    } else {
        avatarEl.textContent = initials;
    }
}

// ── AVATAR PHOTO UPLOAD ───────────────────────────────────────────────────
// Photos are stored as base64 in localStorage keyed by user id.
// This means no server changes needed — the photo stays on this device.

function getLocalAvatar(userId) {
    return localStorage.getItem(`triad_avatar_${userId}`) || null;
}

function setLocalAvatar(userId, base64) {
    try {
        localStorage.setItem(`triad_avatar_${userId}`, base64);
    } catch (e) {
        // localStorage full — image too large
        console.warn('Could not save avatar locally:', e);
        return false;
    }
    return true;
}

function handleAvatarUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
        alert('Please choose an image file (JPG, PNG, WEBP, etc.)');
        return;
    }
    // Max 4 MB before compression
    if (file.size > 4 * 1024 * 1024) {
        alert('Image is too large. Please choose one under 4 MB.');
        return;
    }

    const wrap = document.querySelector('.profile-avatar-wrap');
    wrap.classList.add('avatar-uploading');

    const reader = new FileReader();
    reader.onload = (e) => {
        const originalDataUrl = e.target.result;
        // Compress/resize to max 200×200 via canvas
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const MAX = 200;
            let w = img.width, h = img.height;
            if (w > h) { if (w > MAX) { h = Math.round(h * MAX / w); w = MAX; } }
            else { if (h > MAX) { w = Math.round(w * MAX / h); h = MAX; } }
            canvas.width = w; canvas.height = h;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, w, h);
            const compressed = canvas.toDataURL('image/jpeg', 0.82);

            // Save to localStorage
            if (currentUser) {
                const ok = setLocalAvatar(currentUser.id, compressed);
                if (!ok) {
                    wrap.classList.remove('avatar-uploading');
                    alert('Could not save photo — browser storage may be full. Try a smaller image.');
                    return;
                }
            }

            // Update UI immediately
            applyAvatarToUI(compressed);
            wrap.classList.remove('avatar-uploading');

            // Show quick feedback
            const msg = document.getElementById('profile-info-msg');
            msg.textContent = '✓ Photo updated!'; msg.className = 'profile-msg ok';
            setTimeout(() => { msg.textContent = ''; }, 3000);
        };
        img.src = originalDataUrl;
    };
    reader.readAsDataURL(file);
    // Reset input so same file can be picked again
    event.target.value = '';
}

function applyAvatarToUI(src) {
    const initials = currentUser ? (currentUser.full_name || currentUser.username || '?')[0].toUpperCase() : '?';
    // Large profile avatar
    const bigEl = document.getElementById('profile-avatar-large');
    if (bigEl) {
        if (src) {
            bigEl.innerHTML = `<img src="${src}" alt="Profile photo" />`;
        } else {
            bigEl.textContent = initials;
        }
    }
    // Header chip avatar
    const headerEl = document.getElementById('header-avatar');
    if (headerEl) {
        if (src) {
            headerEl.innerHTML = `<img src="${src}" alt="Profile photo" />`;
        } else {
            headerEl.textContent = initials;
        }
    }
}

function populateProfileForm() {
    if (!currentUser) return;
    document.getElementById('pf-fullname').value = currentUser.full_name || '';
    document.getElementById('pf-username').value = currentUser.username || '';
    document.getElementById('pf-email').value = currentUser.email || '';

    // Load avatar — prefer locally stored photo, fall back to remote url
    const localPhoto = getLocalAvatar(currentUser.id);
    const avatarSrc = localPhoto || currentUser.avatar_url || null;
    applyAvatarToUI(avatarSrc);

    document.getElementById('profile-display-name').textContent = currentUser.full_name || currentUser.username || 'User';
    document.getElementById('profile-display-email').textContent = currentUser.email || '';
    if (currentUser.created_at) {
        const d = new Date(currentUser.created_at);
        document.getElementById('profile-joined').textContent = `Member since ${d.toLocaleDateString('en-IN', { year: 'numeric', month: 'long' })}`;
    }
}

async function saveProfile() {
    const btn = document.getElementById('btn-save-profile');
    const msg = document.getElementById('profile-info-msg');
    msg.textContent = ''; msg.className = 'profile-msg';
    btn.disabled = true; btn.textContent = 'Saving...';
    const payload = {
        full_name: document.getElementById('pf-fullname').value.trim(),
        username: document.getElementById('pf-username').value.trim(),
        email: document.getElementById('pf-email').value.trim(),
        // avatar_url kept as-is from the server; local photo handled separately
    };
    try {
        const res = await fetch(`${BACKEND}/auth/profile`, {
            method: 'PUT', headers: authHeaders(), body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) {
            msg.textContent = extractErrorMsg(data) || 'Update failed.'; msg.className = 'profile-msg err'; return;
        }
        currentUser = { ...currentUser, ...data };
        updateHeaderUser();
        populateProfileForm();
        msg.textContent = '✓ Profile updated!'; msg.className = 'profile-msg ok';
        setTimeout(() => { msg.textContent = ''; }, 3000);
    } catch (e) {
        msg.textContent = 'Network error — is the backend running?'; msg.className = 'profile-msg err';
    } finally {
        btn.disabled = false; btn.textContent = '💾 Save Changes';
    }
}

async function savePassword() {
    const btn = document.getElementById('btn-save-password');
    const msg = document.getElementById('profile-pass-msg');
    msg.textContent = ''; msg.className = 'profile-msg';
    const curPass = document.getElementById('pf-cur-pass').value;
    const newPass = document.getElementById('pf-new-pass').value;
    if (!curPass || !newPass) { msg.textContent = 'Both fields are required.'; msg.className = 'profile-msg err'; return; }
    btn.disabled = true; btn.textContent = 'Updating...';
    try {
        const res = await fetch(`${BACKEND}/auth/password`, {
            method: 'PUT', headers: authHeaders(),
            body: JSON.stringify({ current_password: curPass, new_password: newPass })
        });
        const data = await res.json();
        if (!res.ok) {
            msg.textContent = extractErrorMsg(data) || 'Update failed.'; msg.className = 'profile-msg err'; return;
        }
        document.getElementById('pf-cur-pass').value = '';
        document.getElementById('pf-new-pass').value = '';
        msg.textContent = '✓ Password changed!'; msg.className = 'profile-msg ok';
        setTimeout(() => { msg.textContent = ''; }, 3000);
    } catch (e) {
        msg.textContent = 'Network error.'; msg.className = 'profile-msg err';
    } finally {
        btn.disabled = false; btn.textContent = '🔑 Update Password';
    }
}

// ── Check for existing session on load ───────────────────────────────────
// Session management handled in the boot block below.

// ── BACKEND STATUS ────────────────────────────────────────────────────────
function toggleBackendEdit() {
    const editor = document.getElementById('backend-url-editor');
    const input = document.getElementById('backend-url-input');
    if (editor.style.display === 'none') {
        input.value = BACKEND; editor.style.display = 'flex'; input.focus();
    } else {
        editor.style.display = 'none';
    }
}

function saveBackendUrl() {
    const val = document.getElementById('backend-url-input').value.trim();
    if (!val) return;
    BACKEND = val.replace(/\/$/, '');
    localStorage.setItem('triad_backend_url', BACKEND);
    document.getElementById('backend-url-editor').style.display = 'none';
    checkBackend();
}

document.addEventListener('DOMContentLoaded', () => {
    const inp = document.getElementById('backend-url-input');
    if (inp) inp.addEventListener('keydown', e => { if (e.key === 'Enter') saveBackendUrl(); });
});

async function checkBackend() {
    try {
        const r = await fetch(`${BACKEND}/`, { signal: AbortSignal.timeout(3000) });
        if (r.ok) {
            document.getElementById('backend-dot').className = 'status-dot online';
            document.getElementById('backend-label').textContent = 'Backend online';
        } else throw new Error();
    } catch {
        document.getElementById('backend-dot').className = 'status-dot offline';
        document.getElementById('backend-label').textContent = 'Backend offline';
    }
}
setInterval(checkBackend, 30000);

// ── TAB SWITCHING ─────────────────────────────────────────────────────────
const WAKE_WORDS = ["hey triad", "hey tried", "hey trade", "triad"];
let wakeRecognition = null, commandRecognition = null, appState = "wake", currentTravelText = "";
let legalMode = "pdf", legalFile = null, legalSpeechActive = null;
let medMode = "pdf", medFile = null, medSpeechActive = null;
let micDenied = false;
let wakeRetryCount = 0, wakeRestartTimer = null;
const WAKE_MAX_RETRIES = 50, WAKE_BASE_DELAY = 500, WAKE_MAX_DELAY = 5000;
let activeTab = 'voice';

function switchTab(name, btn) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    const tabBtn = btn || document.querySelector(`[data-tab="${name}"]`);
    if (tabBtn) tabBtn.classList.add('active');
    const panel = document.getElementById('panel-' + name);
    if (panel) panel.classList.add('active');
    activeTab = name;
    if (name !== 'voice' && appState === 'wake') stopWakeListener();
    else if (name === 'voice' && appState === 'wake' && !micDenied) startWakeListener();
    if (name === 'profile') populateProfileForm();
}

// Opens the profile panel from the header chip — no nav tab is highlighted
function openProfile() {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    const panel = document.getElementById('panel-profile');
    if (panel) panel.classList.add('active');
    activeTab = 'profile';
    if (appState === 'wake') stopWakeListener();
    populateProfileForm();
}

// ── SPEECH ────────────────────────────────────────────────────────────────
if (!('speechSynthesis' in window)) {
    const el = document.getElementById('speech-support-notice');
    if (el) el.style.display = 'block';
}

function speak(text, onDone) {
    if (!('speechSynthesis' in window)) { if (onDone) onDone(); return; }
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang = 'en-IN'; utt.rate = 1.0; utt.pitch = 1.0; utt.volume = 1.0;
    utt.onend = () => { if (onDone) onDone(); };
    utt.onerror = () => { if (onDone) onDone(); };
    window.speechSynthesis.speak(utt);
}

function playChime() {
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator(); const gain = ctx.createGain();
        osc.connect(gain); gain.connect(ctx.destination); osc.type = 'sine';
        osc.frequency.setValueAtTime(880, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(1320, ctx.currentTime + 0.15);
        gain.gain.setValueAtTime(0.3, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
        osc.start(); osc.stop(ctx.currentTime + 0.3);
    } catch (e) { }
}

function setStatus(msg, active = false, isSpeaking = false) {
    const el = document.getElementById('voice-status'); if (!el) return;
    el.textContent = msg;
    el.className = 'voice-status' + (active ? ' active' : '') + (isSpeaking ? ' speaking-state' : '');
}
function setWakeDot(state) { const el = document.getElementById('wake-dot'); if (el) el.className = 'wake-dot' + (state !== 'off' ? ' ' + state : ''); }
function setOrbState(state) {
    const orb = document.getElementById('orb'); const icon = document.getElementById('orb-icon');
    if (!orb || !icon) return;
    orb.className = 'orb' + (state !== 'idle' ? ' ' + state : '');
    icon.textContent = state === 'listening' ? '👂' : state === 'speaking' ? '💬' : '🎙';
}
function setResponseText(text) {
    const el = document.getElementById('response-text'); if (!el) return;
    el.style.color = 'var(--text)'; el.textContent = text;
}

function stopWakeListener() {
    if (wakeRestartTimer) { clearTimeout(wakeRestartTimer); wakeRestartTimer = null; }
    if (wakeRecognition) { try { wakeRecognition.abort(); } catch (e) { } wakeRecognition = null; }
}

function startWakeListener() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { setStatus("⚠️ Speech recognition not supported. Use text input below.", false); setWakeDot('off'); showMicDenied(); return; }
    if (activeTab !== 'voice' || document.hidden) return;
    stopWakeListener();
    wakeRetryCount = 0;
    wakeRecognition = new SR();
    wakeRecognition.lang = 'en-IN'; wakeRecognition.continuous = false;
    wakeRecognition.interimResults = false; wakeRecognition.maxAlternatives = 3;
    wakeRecognition.onstart = () => { wakeRetryCount = 0; if (appState === "wake") { setWakeDot('active'); setStatus('Listening for "Hey Triad"...', false); setOrbState('idle'); } };
    wakeRecognition.onresult = (e) => {
        wakeRetryCount = 0;
        const alts = Array.from(e.results[0]).map(r => r.transcript.toLowerCase().trim());
        const detected = alts.some(alt => WAKE_WORDS.some(w => alt.includes(w)));
        if (detected && appState === "wake") onWakeWordDetected();
        else if (appState === "wake") restartWakeListener();
    };
    wakeRecognition.onerror = (e) => {
        if (e.error === 'not-allowed' || e.error === 'service-not-allowed') { showMicDenied(); setStatus("Microphone denied — use text input below.", false); setWakeDot('off'); return; }
        if (e.error === 'no-speech' || e.error === 'aborted') { if (appState === "wake") restartWakeListener(); return; }
        setWakeDot('off'); setStatus("Mic error: " + e.error, false);
    };
    wakeRecognition.onend = () => { if (appState === "wake") restartWakeListener(); };
    try { wakeRecognition.start(); } catch (e) { console.warn('Wake start failed:', e.message); restartWakeListener(); }
}

function showMicDenied() {
    micDenied = true;
    const el = document.getElementById('mic-denied-notice');
    if (el) el.style.display = 'block';
}

function restartWakeListener() {
    if (wakeRestartTimer) { clearTimeout(wakeRestartTimer); wakeRestartTimer = null; }
    if (appState !== "wake" || activeTab !== 'voice' || document.hidden) return;
    wakeRetryCount++;
    if (wakeRetryCount > WAKE_MAX_RETRIES) { setStatus('Voice paused — click the orb or type below.', false); setWakeDot('off'); return; }
    const delay = Math.min(WAKE_BASE_DELAY * Math.pow(1.3, Math.min(wakeRetryCount, 15)), WAKE_MAX_DELAY);
    wakeRestartTimer = setTimeout(() => {
        wakeRestartTimer = null;
        if (appState !== "wake" || activeTab !== 'voice' || document.hidden) return;
        try {
            if (!wakeRecognition) startWakeListener();
            else wakeRecognition.start();
        } catch (e) { startWakeListener(); }
    }, delay);
}

function onWakeWordDetected() {
    appState = "command"; setWakeDot('triggered'); setOrbState('listening');
    setStatus("Hey! I'm listening — what do you need?", true);
    playChime(); speak("Yes? How can I help?", () => listenForCommand());
}

function listenForCommand() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    commandRecognition = new SR();
    commandRecognition.lang = 'en-IN'; commandRecognition.continuous = false;
    commandRecognition.interimResults = false; commandRecognition.maxAlternatives = 1;
    commandRecognition.onstart = () => { setOrbState('listening'); setStatus("Listening for your command...", true); };
    commandRecognition.onresult = async (e) => {
        const command = e.results[0][0].transcript;
        const tt = document.getElementById('transcript-text');
        if (tt) { tt.style.color = 'var(--text)'; tt.textContent = command; }
        appState = "processing"; await handleCommand(command);
    };
    commandRecognition.onerror = () => { setStatus('Didn\'t catch that — say "Hey Triad" to try again', false); returnToWakeMode(); };
    commandRecognition.onend = () => { if (appState === "command") returnToWakeMode(); };
    commandRecognition.start();
}

function toggleVoice() {
    if (appState === "wake") { if (wakeRecognition) { try { wakeRecognition.stop(); } catch (e) { } } onWakeWordDetected(); }
    else if (appState === "command") { if (commandRecognition) { try { commandRecognition.stop(); } catch (e) { } } returnToWakeMode(); }
    else if (appState === "speaking") { window.speechSynthesis.cancel(); returnToWakeMode(); }
}

function returnToWakeMode() {
    appState = "wake"; setOrbState('idle'); setWakeDot('active');
    wakeRetryCount = 0; setStatus('Listening for "Hey Triad"...', false); startWakeListener();
}

async function sendManualCommand() {
    const input = document.getElementById('manual-command-input');
    const command = input.value.trim();
    if (!command) return;
    input.value = '';
    const tt = document.getElementById('transcript-text');
    if (tt) { tt.style.color = 'var(--text)'; tt.textContent = command; }
    appState = "processing"; setStatus("Processing...", false); await handleCommand(command);
}

async function handleCommand(command) {
    setOrbState('idle'); setStatus("Processing...", false);
    if (["stop", "sleep", "goodbye", "bye", "exit"].some(w => command.toLowerCase().includes(w))) {
        setResponseText("Going to sleep. Say \"Hey Triad\" to wake me up.");
        speak("Going to sleep. Say Hey Triad to wake me up.", () => returnToWakeMode()); return;
    }
    try {
        const res = await fetch(`${BACKEND}/voice/command`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command })
        });
        if (!res.ok) { const err = await res.json().catch(() => ({ detail: "Unknown error" })); throw new Error(extractErrorMsg(err) || `HTTP ${res.status}`); }
        const data = await res.json(); setResponseText(data.text);
        if (data.action === 'switch_tab' && data.tab) setTimeout(() => switchTab(data.tab), 1200);
        if (data.action === 'open_url' && data.url) setTimeout(() => window.open(data.url, '_blank'), 800);
        appState = "speaking"; setOrbState('speaking'); setStatus("Speaking...", false, true);
        speak(data.text, () => returnToWakeMode());
    } catch (err) {
        const msg = err.message.includes('fetch') ? "Cannot reach backend. Make sure python backend.py is running." : "Error: " + err.message;
        setResponseText(msg); speak("Sorry, I ran into an error.", () => returnToWakeMode());
    }
}

// ── TRAVEL ────────────────────────────────────────────────────────────────
async function planTrip() {
    const origin = document.getElementById('t-origin').value.trim();
    const dest = document.getElementById('t-dest').value.trim();
    const days = parseInt(document.getElementById('t-days').value);
    const travelers = parseInt(document.getElementById('t-travelers').value);
    const budget = document.getElementById('t-budget').value.trim();
    const errorEl = document.getElementById('travel-error');
    const loaderEl = document.getElementById('travel-loader');
    const resultEl = document.getElementById('travel-result');
    const btn = document.getElementById('plan-btn');
    errorEl.style.display = 'none';
    if (!origin) { showTravelError("Please enter your origin city."); return; }
    if (!dest) { showTravelError("Please enter your destination city."); return; }
    if (!days || days < 1) { showTravelError("Please enter a valid number of days."); return; }
    if (!travelers || travelers < 1) { showTravelError("Please enter number of travellers."); return; }
    if (!budget || isNaN(Number(budget)) || Number(budget) <= 0) { showTravelError("Please enter a valid budget amount."); return; }
    btn.disabled = true; loaderEl.style.display = 'flex'; resultEl.style.display = 'none';
    try {
        const res = await fetch(`${BACKEND}/travel/plan`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ origin, destination: dest, days, travelers, budget })
        });
        if (!res.ok) { const err = await res.json().catch(() => ({ detail: "Server error" })); throw new Error(extractErrorMsg(err) || `HTTP ${res.status}`); }
        const data = await res.json(); currentTravelText = data.itinerary;
        document.getElementById('travel-result-title').textContent = `✈ ${data.origin} → ${data.destination} · ${data.days} days · ₹${data.budget}`;
        document.getElementById('travel-result-text').textContent = data.itinerary;
        setBookingLinks(data.origin, data.destination);
        loaderEl.style.display = 'none'; resultEl.style.display = 'block';
    } catch (err) {
        loaderEl.style.display = 'none';
        showTravelError(err.message.includes('fetch') ? "Cannot reach backend. Make sure python backend.py is running." : "Error: " + err.message);
    } finally { btn.disabled = false; }
}

function showTravelError(msg) { const el = document.getElementById('travel-error'); el.textContent = msg; el.style.display = 'block'; }
function speakTravelPlan() { if (!currentTravelText) return; speak(currentTravelText.substring(0, 600) + "... Check the full plan on screen."); }
function clearTravelResult() { document.getElementById('travel-result').style.display = 'none'; currentTravelText = ''; }

function setBookingLinks(origin, destination) {
    const enc = s => encodeURIComponent(s.trim());
    document.getElementById('link-flight').href = `https://www.makemytrip.com/flights/cheap-flights-from-${enc(origin.toLowerCase())}-to-${enc(destination.toLowerCase())}.html`;
    document.getElementById('link-train').href = `https://www.irctc.co.in/nget/train-search`;
    document.getElementById('link-bus').href = `https://www.redbus.in/bus-tickets/${enc(origin.toLowerCase().replace(/\s+/g, '-'))}-to-${enc(destination.toLowerCase().replace(/\s+/g, '-'))}`;
}

// ── MEDICAL ───────────────────────────────────────────────────────────────
function updateMedCharCounter() {
    const len = document.getElementById('med-text-input').value.length;
    const el = document.getElementById('med-char-counter');
    el.textContent = `${len} / 30 min`;
    el.className = 'char-counter' + (len >= 30 ? ' ok' : len > 10 ? ' warn' : '');
}
function medSwitchMode(mode) {
    medMode = mode;
    document.getElementById('med-pdf-area').style.display = mode === 'pdf' ? 'block' : 'none';
    document.getElementById('med-text-area').style.display = mode === 'text' ? 'block' : 'none';
    document.getElementById('med-btn-pdf').classList.toggle('active', mode === 'pdf');
    document.getElementById('med-btn-text').classList.toggle('active', mode === 'text');
    medHideError(); medHideResults();
}
function medDragOver(e) { e.preventDefault(); document.getElementById('med-drop-zone').classList.add('dragover'); }
function medDragLeave() { document.getElementById('med-drop-zone').classList.remove('dragover'); }
function medDrop(e) { e.preventDefault(); document.getElementById('med-drop-zone').classList.remove('dragover'); const file = e.dataTransfer.files[0]; if (file && file.name.endsWith('.pdf')) medSetFile(file); else medShowError("Please drop a valid PDF file."); }
function medFileSelect(e) { const f = e.target.files[0]; if (f) medSetFile(f); }
function medSetFile(file) { medFile = file; document.getElementById('med-file-name').textContent = file.name; document.getElementById('med-file-chip').style.display = 'flex'; document.getElementById('med-file-size-warn').style.display = file.size > PDF_SIZE_WARN_BYTES ? 'block' : 'none'; }
function medClearFile() { medFile = null; document.getElementById('med-file-chip').style.display = 'none'; document.getElementById('med-pdf-input').value = ''; document.getElementById('med-file-size-warn').style.display = 'none'; }

async function analyzeMedical() {
    medHideError(); medHideResults();
    if (medMode === 'pdf' && !medFile) { medShowError("Please upload a PDF file first."); return; }
    if (medMode === 'text' && document.getElementById('med-text-input').value.trim().length < 30) { medShowError("Please paste your medical report text (minimum 30 characters)."); return; }
    const btn = document.getElementById('med-analyze-btn'); const loader = document.getElementById('med-loader');
    btn.disabled = true; loader.style.display = 'flex';
    try {
        let data;
        if (medMode === 'pdf') {
            const formData = new FormData(); formData.append('file', medFile);
            const res = await fetch(`${BACKEND}/medical/analyze-pdf`, { method: 'POST', body: formData });
            if (!res.ok) { const e = await res.json(); throw new Error(extractErrorMsg(e) || 'PDF analysis failed.'); }
            data = await res.json();
        } else {
            const text = document.getElementById('med-text-input').value;
            const res = await fetch(`${BACKEND}/medical/analyze-text`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
            if (!res.ok) { const e = await res.json(); throw new Error(extractErrorMsg(e) || 'Analysis failed.'); }
            data = await res.json();
        }
        document.getElementById('med-summary-text').textContent = data.summary || '—';
        document.getElementById('med-values-text').textContent = data.value_analysis || '—';
        document.getElementById('med-advice-text').textContent = data.health_advice || '—';
        document.getElementById('med-doc-badge').textContent = data.filename ? `🩺 ${data.filename}` : '✏️ Text report analyzed';
        loader.style.display = 'none'; document.getElementById('med-results').style.display = 'block';
        medShowTabByName('summary'); document.getElementById('med-results').scrollIntoView({ behavior: 'smooth' });
    } catch (err) { loader.style.display = 'none'; medShowError('Error: ' + err.message); }
    finally { btn.disabled = false; }
}

function medShowTab(name, btn) { document.querySelectorAll('#panel-medical .doc-result-panel').forEach(p => p.classList.remove('active')); document.querySelectorAll('#panel-medical .doc-result-tab').forEach(t => t.classList.remove('active')); document.getElementById('med-panel-' + name).classList.add('active'); btn.classList.add('active'); }
function medShowTabByName(name) { document.querySelectorAll('#panel-medical .doc-result-panel').forEach(p => p.classList.remove('active')); document.querySelectorAll('#panel-medical .doc-result-tab').forEach(t => t.classList.remove('active')); document.getElementById('med-panel-' + name).classList.add('active'); document.querySelectorAll('#panel-medical .doc-result-tab')[0].classList.add('active'); }
function medCopy(id, btn) { navigator.clipboard.writeText(document.getElementById(id).textContent).then(() => { btn.textContent = 'Copied!'; setTimeout(() => btn.textContent = 'Copy', 2000); }); }
function medSpeak(textId, btnId) {
    const btn = document.getElementById(btnId);
    if (medSpeechActive === btnId) { window.speechSynthesis.cancel(); medSpeechActive = null; document.querySelectorAll('#panel-medical .speak-btn').forEach(b => { b.textContent = '🔊 Read Aloud'; b.classList.remove('active-speak'); }); return; }
    window.speechSynthesis.cancel(); document.querySelectorAll('#panel-medical .speak-btn').forEach(b => { b.textContent = '🔊 Read Aloud'; b.classList.remove('active-speak'); });
    const utt = new SpeechSynthesisUtterance(document.getElementById(textId).textContent);
    utt.lang = 'en-IN'; utt.rate = 0.95;
    utt.onend = () => { btn.textContent = '🔊 Read Aloud'; btn.classList.remove('active-speak'); medSpeechActive = null; };
    btn.textContent = '⏹ Stop Speaking'; btn.classList.add('active-speak'); medSpeechActive = btnId;
    window.speechSynthesis.speak(utt);
}
function medShowError(msg) { const el = document.getElementById('med-error'); el.textContent = msg; el.style.display = 'block'; }
function medHideError() { document.getElementById('med-error').style.display = 'none'; }
function medHideResults() { document.getElementById('med-results').style.display = 'none'; }

// ── LEGAL ─────────────────────────────────────────────────────────────────
function updateLegalCharCounter() { const len = document.getElementById('legal-text-input').value.length; const el = document.getElementById('legal-char-counter'); el.textContent = `${len} / 50 min`; el.className = 'char-counter' + (len >= 50 ? ' ok' : len > 20 ? ' warn' : ''); }
function legalSwitchMode(mode) { legalMode = mode; document.getElementById('legal-pdf-area').style.display = mode === 'pdf' ? 'block' : 'none'; document.getElementById('legal-text-area').style.display = mode === 'text' ? 'block' : 'none'; document.getElementById('legal-btn-pdf').classList.toggle('active', mode === 'pdf'); document.getElementById('legal-btn-text').classList.toggle('active', mode === 'text'); legalHideError(); legalHideResults(); }
function legalDragOver(e) { e.preventDefault(); document.getElementById('legal-drop-zone').classList.add('dragover'); }
function legalDragLeave() { document.getElementById('legal-drop-zone').classList.remove('dragover'); }
function legalDrop(e) { e.preventDefault(); document.getElementById('legal-drop-zone').classList.remove('dragover'); const file = e.dataTransfer.files[0]; if (file && file.name.endsWith('.pdf')) legalSetFile(file); else legalShowError("Please drop a valid PDF file."); }
function legalFileSelect(e) { const file = e.target.files[0]; if (file) legalSetFile(file); }
function legalSetFile(file) { legalFile = file; document.getElementById('legal-file-name').textContent = file.name; document.getElementById('legal-file-chip').style.display = 'flex'; document.getElementById('legal-file-size-warn').style.display = file.size > PDF_SIZE_WARN_BYTES ? 'block' : 'none'; }
function legalClearFile() { legalFile = null; document.getElementById('legal-file-chip').style.display = 'none'; document.getElementById('legal-pdf-input').value = ''; document.getElementById('legal-file-size-warn').style.display = 'none'; }

async function analyzeLegal() {
    legalHideError(); legalHideResults();
    if (legalMode === 'pdf' && !legalFile) { legalShowError("Please upload a PDF file first."); return; }
    if (legalMode === 'text') { const txt = document.getElementById('legal-text-input').value.trim(); if (txt.length < 50) { legalShowError("Please paste a legal document (minimum 50 characters)."); return; } }
    const btn = document.getElementById('legal-analyze-btn'); const loader = document.getElementById('legal-loader');
    btn.disabled = true; loader.style.display = 'flex';
    try {
        let data;
        if (legalMode === 'pdf') {
            const formData = new FormData(); formData.append('file', legalFile);
            const res = await fetch(`${BACKEND}/legal/explain-pdf`, { method: 'POST', body: formData });
            if (!res.ok) { const err = await res.json(); throw new Error(extractErrorMsg(err) || 'PDF analysis failed.'); }
            data = await res.json();
        } else {
            const text = document.getElementById('legal-text-input').value;
            const res = await fetch(`${BACKEND}/legal/explain-text`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
            if (!res.ok) { const err = await res.json(); throw new Error(extractErrorMsg(err) || 'Text analysis failed.'); }
            data = await res.json();
        }
        document.getElementById('legal-plain-text').textContent = data.plain_english || '—';
        document.getElementById('legal-clauses-text').textContent = data.clauses || '—';
        document.getElementById('legal-keys-text').textContent = data.key_points || '—';
        document.getElementById('legal-doc-badge').textContent = data.filename ? `📄 ${data.filename}` : '✏️ Text document analyzed';
        loader.style.display = 'none'; document.getElementById('legal-results').style.display = 'block';
        legalShowTabByName('plain'); document.getElementById('legal-results').scrollIntoView({ behavior: 'smooth' });
    } catch (err) { loader.style.display = 'none'; legalShowError('Error: ' + err.message); }
    finally { btn.disabled = false; }
}

function legalShowTab(name, btn) { document.querySelectorAll('#panel-legal .doc-result-panel').forEach(p => p.classList.remove('active')); document.querySelectorAll('#panel-legal .doc-result-tab').forEach(t => t.classList.remove('active')); document.getElementById('legal-panel-' + name).classList.add('active'); btn.classList.add('active'); }
function legalShowTabByName(name) { document.querySelectorAll('#panel-legal .doc-result-panel').forEach(p => p.classList.remove('active')); document.querySelectorAll('#panel-legal .doc-result-tab').forEach(t => t.classList.remove('active')); document.getElementById('legal-panel-' + name).classList.add('active'); document.querySelectorAll('#panel-legal .doc-result-tab')[0].classList.add('active'); }
function legalCopy(id, btn) { navigator.clipboard.writeText(document.getElementById(id).textContent).then(() => { btn.textContent = 'Copied!'; setTimeout(() => btn.textContent = 'Copy', 2000); }); }
function legalSpeak(textId, btnId) {
    const btn = document.getElementById(btnId);
    if (legalSpeechActive === btnId) { window.speechSynthesis.cancel(); legalSpeechActive = null; document.querySelectorAll('#panel-legal .speak-btn').forEach(b => { b.textContent = '🔊 Read Aloud'; b.classList.remove('active-speak'); }); return; }
    window.speechSynthesis.cancel(); document.querySelectorAll('#panel-legal .speak-btn').forEach(b => { b.textContent = '🔊 Read Aloud'; b.classList.remove('active-speak'); });
    const utt = new SpeechSynthesisUtterance(document.getElementById(textId).textContent);
    utt.lang = 'en-IN'; utt.rate = 0.95;
    utt.onend = () => { btn.textContent = '🔊 Read Aloud'; btn.classList.remove('active-speak'); legalSpeechActive = null; };
    btn.textContent = '⏹ Stop Speaking'; btn.classList.add('active-speak'); legalSpeechActive = btnId;
    window.speechSynthesis.speak(utt);
}
function legalShowError(msg) { const el = document.getElementById('legal-error'); el.textContent = msg; el.style.display = 'block'; }
function legalHideError() { document.getElementById('legal-error').style.display = 'none'; }
function legalHideResults() { document.getElementById('legal-results').style.display = 'none'; }

// ── PAGE VISIBILITY ───────────────────────────────────────────────────────
document.addEventListener('visibilitychange', () => {
    if (document.hidden) { if (appState === 'wake') stopWakeListener(); }
    else { if (appState === 'wake' && activeTab === 'voice' && !micDenied) startWakeListener(); }
});

// ── BOOT ──────────────────────────────────────────────────────────────────
// Logic:
//   • No saved token  → show login screen
//   • Token exists    → verify with backend silently
//       ✓ valid       → enter app (user stays logged in)
//       ✗ expired     → clear token, show login screen
window.addEventListener('load', async () => {
    const token = getToken();
    if (!token) {
        // No token at all — first visit or after logout
        showLoginScreen();
        return;
    }
    // Token exists — check if it's still valid
    try {
        const res = await fetch(`${BACKEND}/auth/me`, {
            headers: { 'Authorization': 'Bearer ' + token },
            signal: AbortSignal.timeout(4000)
        });
        if (res.ok) {
            const data = await res.json();
            currentUser = data;
            enterApp();          // valid token — go straight in
        } else {
            clearToken();        // token expired/revoked
            showLoginScreen();
        }
    } catch (e) {
        // Backend unreachable — clear token and show login
        clearToken();
        showLoginScreen();
    }
});

function showLoginScreen() {
    document.getElementById('auth-screen').style.display = 'flex';
    document.getElementById('main-app').style.display = 'none';
}
