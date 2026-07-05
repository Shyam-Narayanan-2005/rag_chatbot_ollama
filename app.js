let indexed = false, busy = false, autoPlay = false;
let mediaRecorder = null, audioChunks = [], isRecording = false;
let currentAudio = null;

const chatToggle        = document.getElementById('chat-toggle');
const chatWindow        = document.getElementById('chat-window');
const notifDot          = document.getElementById('notif-dot');
const messages          = document.getElementById('chat-messages');
const input              = document.getElementById('chat-input');
const sendBtn            = document.getElementById('send-btn');
const micBtn             = document.getElementById('mic-btn');
const statusDot          = document.getElementById('status-dot');
const botSub             = document.getElementById('bot-sub');
const serverVoiceToggle  = document.getElementById('server-voice-toggle');
const recStatus          = document.getElementById('rec-status');
const fileInput          = document.getElementById('file-input');
const dropZone           = document.getElementById('drop-zone');
const uploadBtn          = document.getElementById('upload-btn');
const uploadStat         = document.getElementById('upload-status');
const fileChosen         = document.getElementById('file-chosen');

const apiBase = () => window.location.origin;
const now = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

/* ── server voice (pyttsx3) auto-play toggle ── */
serverVoiceToggle.addEventListener('click', () => {
  autoPlay = !autoPlay;
  serverVoiceToggle.classList.toggle('on', autoPlay);
  serverVoiceToggle.innerHTML = autoPlay
    ? `<svg viewBox="0 0 24 24"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM16.5 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/></svg> 🔊 Auto-play`
    : `<svg viewBox="0 0 24 24"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/></svg> 🔇 Auto-play`;
});

/* play audio from /speak endpoint */
async function playServerAudio(question, btnEl) {
  try {
    if (currentAudio) { currentAudio.pause(); currentAudio = null; }
    if (btnEl) { btnEl.classList.add('playing'); btnEl.textContent = '⏳'; }

    const res = await fetch(`${apiBase()}/speak`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    if (!res.ok) throw new Error('speak failed');
    await res.json();

    const audio = new Audio(`${apiBase()}/audio?t=${Date.now()}`);
    currentAudio = audio;
    audio.play();
    audio.onended = () => {
      currentAudio = null;
      if (btnEl) {
        btnEl.classList.remove('playing');
        btnEl.innerHTML = `<svg viewBox="0 0 24 24"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/></svg> ▶ Play`;
      }
    };
  } catch (e) {
    if (btnEl) {
      btnEl.classList.remove('playing');
      btnEl.innerHTML = `<svg viewBox="0 0 24 24"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/></svg> ▶ Play`;
    }
  }
}

/* ── indexed state ── */
function setIndexed(val) {
  indexed = val;
  sendBtn.disabled = !val || busy;
  micBtn.disabled  = !val || busy;
  if (val) {
    statusDot.classList.remove('offline');
    botSub.textContent = 'Dataset loaded · Ask away';
    notifDot.classList.add('show');
  }
}
micBtn.disabled = true;

/* ── check for an existing knowledge base on page load ── */
async function checkExistingIndex() {
  try {
    const res = await fetch(`${apiBase()}/status`);
    if (!res.ok) return;
    const data = await res.json();
    if (data.ready) {
      setIndexed(true);
      const chunkCount = data.chunks ?? data.total_chunks ?? '(unknown)';
      uploadStat.className = 'upload-status ok';
      uploadStat.textContent = `✅ Found existing dataset (${chunkCount} chunks) — ready to chat`;
      botSub.textContent = 'Dataset loaded · Ask away';
    }
  } catch (err) {
    // Backend not reachable yet or /status missing — just leave the
    // upload screen as the default, no need to surface an error here.
  }
}
checkExistingIndex();

/* ── chat toggle ── */
let chatOpen = false;
chatToggle.addEventListener('click', () => {
  chatOpen = !chatOpen;
  chatWindow.classList.toggle('open', chatOpen);
  if (chatOpen) { notifDot.classList.remove('show'); input.focus(); }
});

/* ── upload ── */
let selectedFile = null;
fileInput.addEventListener('change', () => {
  selectedFile = fileInput.files[0] || null;
  fileChosen.textContent = selectedFile ? selectedFile.name : '';
  uploadBtn.disabled = !selectedFile;
});
['dragover', 'dragleave', 'drop'].forEach(evt => {
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.classList.toggle('drag', evt === 'dragover');
    if (evt === 'drop') {
      selectedFile = e.dataTransfer.files[0] || null;
      fileChosen.textContent = selectedFile ? selectedFile.name : '';
      uploadBtn.disabled = !selectedFile;
    }
  });
});
uploadBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  uploadBtn.disabled = true;
  uploadStat.className = 'upload-status';
  uploadStat.textContent = '⏳ Uploading & indexing…';

  const form = new FormData();
  form.append('file', selectedFile);

  try {
    const res = await fetch(`${apiBase()}/upload`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    console.log('Upload response from server:', data);
    const added = data.chunks_added ?? data.total_chunks ?? '(unknown)';
    const skipped = data.chunks_skipped_as_duplicate ?? 0;
    uploadStat.className = 'upload-status ok';
    uploadStat.textContent = skipped > 0
      ? `✅ Added ${added} new chunks (${skipped} duplicates skipped) from "${data.file}"`
      : `✅ Indexed ${added} new chunks from "${data.file}"`;
    setIndexed(true);
  } catch (err) {
    uploadStat.className = 'upload-status err';
    uploadStat.textContent = `❌ Upload failed (${err.message})`;
    uploadBtn.disabled = false;
  }
});

/* ── messages ── */
function appendMsg(role, text, questionForSpeak) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;

  const bub = document.createElement('div');
  bub.className = 'bubble';
  bub.textContent = text;

  const time = document.createElement('div');
  time.className = 'msg-time';
  time.textContent = now();

  div.appendChild(bub);

  if (role === 'bot' && questionForSpeak) {
    const sp = document.createElement('button');
    sp.className = 'speak-btn';
    sp.innerHTML = `<svg viewBox="0 0 24 24"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/></svg> ▶ Play`;
    sp.onclick = () => playServerAudio(questionForSpeak, sp);
    time.appendChild(sp);
  }

  div.appendChild(time);
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

function showTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'msg bot';
  wrap.id = 'typing';
  const bub = document.createElement('div');
  bub.className = 'typing-bubble';
  bub.innerHTML = '<span></span><span></span><span></span>';
  wrap.appendChild(bub);
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
}
function hideTyping() {
  const t = document.getElementById('typing');
  if (t) t.remove();
}

/* ── send text question ── */
async function sendMessage(question) {
  question = question || input.value.trim();
  if (!question || busy || !indexed) return;

  busy = true;
  sendBtn.disabled = true;
  micBtn.disabled  = true;
  input.value = '';
  input.style.height = '';

  appendMsg('user', question);
  showTyping();
  botSub.textContent = 'Thinking…';

  try {
    const res = await fetch(`${apiBase()}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    hideTyping();

    const answer = data.answer || 'No answer returned.';
    appendMsg('bot', answer, question);

    if (autoPlay) playServerAudio(question, null);
    botSub.textContent = 'Dataset loaded · Ask away';
  } catch (err) {
    hideTyping();
    appendMsg('bot', `⚠️ Could not reach the API. (${err.message})`);
    botSub.textContent = 'Connection error';
  } finally {
    busy = false;
    sendBtn.disabled = !indexed;
    micBtn.disabled  = !indexed;
    input.focus();
  }
}

sendBtn.addEventListener('click', () => sendMessage());
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
input.addEventListener('input', () => {
  input.style.height = '';
  input.style.height = Math.min(input.scrollHeight, 100) + 'px';
});

/* ── microphone recording ── */
micBtn.addEventListener('click', async () => {
  if (!indexed) return;

  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        recStatus.classList.remove('show');
        micBtn.classList.remove('recording');
        isRecording = false;

        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        const form = new FormData();
        form.append('file', blob, 'recording.webm');

        botSub.textContent = 'Transcribing…';
        appendMsg('user', '🎤 (voice question)');
        showTyping();
        busy = true;
        sendBtn.disabled = true;
        micBtn.disabled  = true;

        try {
          const res = await fetch(`${apiBase()}/voice-browser`, { method: 'POST', body: form });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          hideTyping();

          const userMsgs = messages.querySelectorAll('.msg.user');
          const last = userMsgs[userMsgs.length - 1];
          if (last) last.querySelector('.bubble').textContent = '🎤 ' + (data.question || 'voice question');

          const answer = data.answer || 'No answer returned.';
          appendMsg('bot', answer, data.question);

          if (autoPlay && data.question) playServerAudio(data.question, null);
          botSub.textContent = 'Dataset loaded · Ask away';
        } catch (err) {
          hideTyping();
          appendMsg('bot', `⚠️ Voice error: ${err.message}`);
          botSub.textContent = 'Voice error';
        } finally {
          busy = false;
          sendBtn.disabled = !indexed;
          micBtn.disabled  = !indexed;
        }
      };

      mediaRecorder.start();
      isRecording = true;
      micBtn.classList.add('recording');
      recStatus.classList.add('show');
    } catch (err) {
      appendMsg('bot', '⚠️ Microphone access denied. Please allow microphone in your browser settings.');
    }
  } else {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  }
});
