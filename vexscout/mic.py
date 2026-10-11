"""Microphone button: speech-to-text in the browser (Web Speech API), sent back to
Python as a trigger value. Works in Chrome, Edge and Safari on localhost or
https; other browsers get a disabled button with an explanation."""
import os

import streamlit as st

HTTPS_PORT = int(os.getenv("HTTPS_PORT", "8443"))  # where scripts/serve_https.sh serves the app

HTML = """
<div class="mic-wrap">
  <button type="button" class="mic" aria-label="Tap to talk" title="Tap to talk">
    <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
      <rect x="9" y="3" width="6" height="11" rx="3" fill="currentColor"/>
      <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    </svg>
  </button>
  <div class="status"></div>
</div>
"""

CSS = """
.mic-wrap {display: flex; flex-direction: row; align-items: center; gap: 12px; min-height: 52px;}
.mic {width: 48px; height: 48px; border-radius: 50%; border: 1px solid #3A3F4B; background: #1A1C22;
      color: #E8EAED; cursor: pointer; display: grid; place-items: center; transition: background .15s;}
.mic:hover {background: #262930;}
.mic.on {background: #E5484D; border-color: #E5484D; color: #fff; animation: pulse 1.2s infinite;}
.mic:disabled {opacity: .4; cursor: not-allowed;}
.mic.off {opacity: .6; border-style: dashed;}
.status {font: 14px/1.35 Inter, sans-serif; color: #9AA0AB; text-align: left; flex: 1; min-width: 0;}
.status.live {color: #F2F3F5;} .status.ok {color: #6BD58A;} .status.err {color: #F5B95A;}
@keyframes pulse {0% {box-shadow: 0 0 0 0 #E5484D88;} 70% {box-shadow: 0 0 0 12px #E5484D00;} 100% {box-shadow: 0 0 0 0 #E5484D00;}}
"""

JS = """
export default function(component) {
  const { parentElement, setTriggerValue, data } = component;
  const btn = parentElement.querySelector('.mic');
  const status = parentElement.querySelector('.status');
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    btn.disabled = true;
    status.textContent = 'Use Chrome, Edge or Safari';
    return;
  }
  if (!window.isSecureContext) {
    // Browsers only allow the mic on https (or localhost): offer a one-tap switch to the https server.
    const url = new URL(window.location.href);
    url.protocol = 'https:';
    url.port = String((data && data.https_port) || 8443);
    btn.classList.add('off');
    btn.title = 'Voice needs a secure connection - tap to reopen this page over https';
    status.textContent = 'Tap to enable voice';
    btn.onclick = () => { window.location.href = url.toString(); };
    return;
  }
  let rec = null, finalText = '', heard = false;
  const say = (text, cls) => { status.textContent = text; status.className = 'status' + (cls ? ' ' + cls : ''); };
  say('Tap the mic and talk - it stops when you pause');
  btn.onclick = () => {
    if (rec) { rec.stop(); return; }
    finalText = ''; heard = false;
    rec = new SR();
    // single phrase: recognition ends by itself when the speaker pauses
    rec.lang = 'en-US'; rec.interimResults = true; rec.continuous = false;
    rec.onresult = (e) => {
      heard = true;
      let interim = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) finalText += r[0].transcript + ' '; else interim += r[0].transcript;
      }
      say('Listening… “' + (finalText + interim).trim() + '”', 'live');
    };
    rec.onerror = (e) => {
      const msg = {'not-allowed': 'Microphone blocked - allow it in your browser settings, then try again',
                   'service-not-allowed': 'Voice input is turned off on this device (check Siri & Dictation)',
                   'no-speech': "Didn't hear anything - tap and try again",
                   'audio-capture': 'No microphone found',
                   'network': 'Voice service unreachable - check your connection'}[e.error] || ('Voice error: ' + e.error);
      say(msg, 'err');
    };
    rec.onend = () => {
      btn.classList.remove('on'); rec = null;
      const text = finalText.trim();
      if (text) { say('Added to the text box ✓  “' + text + '”', 'ok'); setTriggerValue('transcript', text); }
      else if (!status.classList.contains('err')) say(heard ? "Didn't catch that - tap and try again" : "Didn't hear anything - tap and try again", 'err');
    };
    rec.start();
    btn.classList.add('on');
    say('Listening… start talking', 'live');
  };
  return () => { if (rec) rec.abort(); };
}
"""

def _register():
    return st.components.v2.component("vexscout_mic", html=HTML, css=CSS, js=JS)


_mic = _register()


def mic(key):
    """Render the button; returns the new transcript string once, after the user stops talking."""
    global _mic
    try:
        return _mic(key=key, data={"https_port": HTTPS_PORT}, on_transcript_change=lambda: None).transcript
    except st.errors.StreamlitAPIException:  # registry was reset (new runtime): register again
        _mic = _register()
        return _mic(key=key, data={"https_port": HTTPS_PORT}, on_transcript_change=lambda: None).transcript
