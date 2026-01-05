// Real-time ASL fingerspelling helper using MediaPipe Hands + simple feature KNN
// - Mirrors video visually
// - Extracts normalized distances & angles from 21 landmarks
// - Uses a small KNN-style template store (localStorage) for A-Z

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const translated = document.getElementById('translated');
const handState = document.getElementById('handState');
const sampleCountEl = document.getElementById('sampleCount');
const body = document.body;

// user state
let currentUser = null; // null = guest

// templates will be assigned after seededTemplates is defined
let templates = {};

// ML model support
let tfModel = null;
const MODEL_URL = '/model/model.json'; // put a TF.js model here or use IndexedDB saved model name 'localstorage://asl-model'

async function tryLoadModel(){
  // try localstorage first
  try{
    tfModel = await tf.loadLayersModel('localstorage://asl-model');
    console.log('Loaded model from IndexedDB');
    return true;
  }catch(e){ /* ignore */ }
  try{
    tfModel = await tf.loadLayersModel(MODEL_URL);
    console.log('Loaded model from', MODEL_URL);
    return true;
  }catch(e){ console.warn('No model at', MODEL_URL); tfModel = null; return false; }
}
// Listen for external changes to templates (other tabs) and reload for current user
window.addEventListener('storage', (ev)=>{
  try{
    if(!currentUser) return;
    const key = getUserKey(currentUser);
    if(ev.key === key){
      console.log('Storage event: reloading templates for', currentUser);
      templates = loadTemplates(currentUser);
      updateSampleCount();
      if(document.getElementById('recorder') && !document.getElementById('recorder').classList.contains('hidden')) buildLetterGrid();
    }
  }catch(e){ console.warn('storage event handler error', e); }
});

async function trainModelFromTemplates(epochs=50){
  // build dataset from `templates` object
  const X=[]; const Y=[]; const labels = Object.keys(templates).filter(k=>templates[k] && templates[k].length);
  if(labels.length===0){ alert('No templates to train on'); return null; }
  const labelIndex = {}; labels.forEach((l,i)=>labelIndex[l]=i);
  labels.forEach(l=>{
    templates[l].forEach(sample=>{ if(Array.isArray(sample)) { X.push(sample); Y.push(labelIndex[l]); } });
  });
  const xs = tf.tensor2d(X);
  const ys = tf.oneHot(tf.tensor1d(Y,'int32'), labels.length);
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape:[X[0].length], units:64, activation:'relu'}));
  model.add(tf.layers.dropout({rate:0.2}));
  model.add(tf.layers.dense({units:labels.length, activation:'softmax'}));
  model.compile({optimizer:'adam', loss:'categoricalCrossentropy', metrics:['accuracy']});
  await model.fit(xs, ys, {epochs, batchSize:16});
  // save to IndexedDB
  await model.save('localstorage://asl-model');
  tfModel = model;
  alert('Model trained and saved to IndexedDB (localstorage://asl-model)');
  xs.dispose(); ys.dispose();
  return model;
}

// If no stored templates, seed with heuristic templates for immediate demo
const seededTemplates = (function(){
  // helper to build vector: tips(5), tipToMid(5), pairs(5), angles(3) => 18 dims
  const E = (v)=>v; // identity for clarity
  const ext = 0.9, cur=0.32, midExt=0.6, midCur=0.12;
  const near = 0.08, midNear=0.14, far=0.45;
  const a = {};
  // A: fist (all curled)
  a['A'] = [[cur,cur,cur,cur,cur, midCur,midCur,midCur,midCur,midCur, near, near, near, near, near, 2.6,2.6,2.6]];
  // B: flat hand (all extended except thumb)
  a['B'] = [[cur,ext,ext,ext,ext, midCur,midExt,midExt,midExt,midExt, far, far, far, far, far, 0.4,0.45,0.5]];
  // C: curved hand
  a['C'] = [[0.7,0.75,0.78,0.72,0.7, 0.35,0.38,0.4,0.36,0.34, 0.25,0.22,0.26,0.28,0.2, 1.2,1.1,1.15]];
  // D: index extended, others curled
  a['D'] = [[cur,ext,cur,cur,cur, midCur,midExt,midCur,midCur,midCur, 0.25,0.18,0.2,0.2,0.18, 0.6,1.8,1.9]];
  // E: all curled with thumb over
  a['E'] = [[cur,cur,cur,cur,cur, midCur,midCur,midCur,midCur,midCur, near, near, near, near, near, 2.4,2.4,2.4]];
  // F: thumb+index touch
  a['F'] = [[0.45,0.7,0.7,0.65,0.6, midCur,midExt,midExt,midExt,midExt, 0.06,0.22,0.25,0.28,0.22, 0.6,0.7,0.8]];
  // G: index pointing sideways
  a['G'] = [[0.5,0.82,0.5,0.48,0.46, midCur,midExt,midCur,midCur,midCur, 0.28,0.18,0.22,0.2,0.18, 0.5,1.4,1.5]];
  // H: index+middle extended together
  a['H'] = [[0.4,0.86,0.86,0.48,0.44, midCur,midExt,midExt,midCur,midCur, 0.12,0.22,0.22,0.2,0.2, 0.4,0.45,1.6]];
  // I: pinky extended
  a['I'] = [[0.4,0.45,0.42,0.4,0.88, midCur,midCur,midCur,midCur,midExt, 0.28,0.18,0.22,0.26,0.14, 2.0,2.0,2.0]];
  // J: pinky swipe (use I-like)
  a['J'] = a['I'];
  // K: index+middle extended, thumb extended
  a['K'] = [[0.6,0.9,0.9,0.5,0.45, midCur,midExt,midExt,midCur,midCur, 0.22,0.2,0.22,0.2,0.18, 0.5,0.45,1.4]];
  // L: index+thumb forming L
  a['L'] = [[0.9,0.85,0.4,0.38,0.36, midExt,midExt,midCur,midCur,midCur, 0.18,0.26,0.28,0.24,0.2, 0.5,1.7,1.8]];
  // M: three fingers over thumb (curled)
  a['M'] = [[cur,cur,cur,cur,cur, midCur,midCur,midCur,midCur,midCur, near, near, near, near, near, 2.5,2.5,2.5]];
  // N: two fingers over thumb
  a['N'] = a['M'];
  // O: all tips touching to make circle
  a['O'] = [[0.44,0.44,0.44,0.44,0.44, midCur,midCur,midCur,midCur,midCur, 0.08,0.08,0.08,0.08,0.08, 1.9,1.9,1.9]];
  // P: like K but rotated
  a['P'] = a['K'];
  // Q: like G but thumb down
  a['Q'] = a['G'];
  // R: crossed index+middle
  a['R'] = [[cur,0.8,0.8,cur,cur, midCur,midExt,midExt,midCur,midCur, 0.1,0.08,0.14,0.18,0.16, 0.5,0.5,1.8]];
  // S: fist
  a['S'] = a['A'];
  // T: thumb between fingers
  a['T'] = [[cur,cur,cur,cur,cur, midCur,midCur,midCur,midCur,midCur, near, near, near, near, near, 2.5,2.5,2.5]];
  // U: index+middle together upright
  a['U'] = [[cur,0.9,0.9,cur,cur, midCur,midExt,midExt,midCur,midCur, 0.12,0.12,0.18,0.18,0.16, 0.4,0.45,1.6]];
  // V: index+middle spread
  a['V'] = [[cur,0.9,0.9,cur,cur, midCur,midExt,midExt,midCur,midCur, 0.3,0.25,0.3,0.22,0.18, 0.4,0.6,1.6]];
  // W: index+middle+ring extended
  a['W'] = [[cur,0.9,0.9,0.9,cur, midCur,midExt,midExt,midExt,midCur, 0.3,0.3,0.3,0.26,0.22, 0.4,0.45,0.5]];
  // X: index bent
  a['X'] = [[cur,0.45,cur,cur,cur, midCur,midCur,midCur,midCur,midCur, 0.18,0.16,0.18,0.16,0.14, 1.9,2.0,2.0]];
  // Y: thumb+pinky
  a['Y'] = [[0.9,0.36,0.36,0.36,0.9, midExt,midCur,midCur,midCur,midExt, 0.4,0.22,0.2,0.18,0.4, 0.5,1.9,2.0]];
  // Z: index traces (approx as extended)
  a['Z'] = [[cur,0.9,cur,cur,cur, midCur,midExt,midCur,midCur,midCur, 0.22,0.18,0.2,0.18,0.16, 0.5,1.5,1.6]];
  return a;
})();

// now that seededTemplates is declared, initialize templates for guest view
templates = JSON.parse(JSON.stringify(seededTemplates));
updateSampleCount();

function getUserKey(user){ return `asl_templates_${user}`; }
function loadTemplates(user){
  try{
    if(!user) return JSON.parse(JSON.stringify(seededTemplates));
    const v = localStorage.getItem(getUserKey(user));
    if(!v){
      const copy = JSON.parse(JSON.stringify(seededTemplates));
      localStorage.setItem(getUserKey(user), JSON.stringify(copy));
      return copy;
    }
    return JSON.parse(v);
  }catch(e){ return JSON.parse(JSON.stringify(seededTemplates)); }
}
function saveTemplates(){
  if(currentUser){
    try{
      localStorage.setItem(getUserKey(currentUser), JSON.stringify(templates));
      // write a timestamp key for debug/persistence checks
      localStorage.setItem(getUserKey(currentUser)+'__ts', String(Date.now()));
      // verify readback
      const back = localStorage.getItem(getUserKey(currentUser));
      if(!back){ console.warn('saveTemplates: write failed (no backread)'); }
      else {
        try{ const parsed = JSON.parse(back); const cnt = Object.values(parsed).reduce((s,a)=>s+(Array.isArray(a)?a.length:0),0); console.log('saveTemplates: saved', cnt, 'samples for', currentUser); }
        catch(e){ console.warn('saveTemplates: readback parse failed', e); }
      }
    }catch(e){ console.error('saveTemplates error', e); }
  }
  updateSampleCount();
}
function updateSampleCount(){ let c=0; Object.values(templates).forEach(arr=>c+=arr.length); sampleCountEl.textContent=c }

function downloadTemplates(){ const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([JSON.stringify(templates)],{type:'application/json'})); a.download=`asl_templates_${currentUser||'guest'}.json`; a.click(); }
function importTemplates(obj){ templates = obj; saveTemplates(); }

document.getElementById('clearStorage').onclick = ()=>{
  if(!currentUser){ alert('Please login to clear your templates.'); return; }
  templates={}; saveTemplates(); translated.textContent='—';
}
document.getElementById('downloadTemplates').onclick = downloadTemplates;
document.getElementById('importBtn').onclick = ()=>{
  if(!currentUser){ alert('Please login to import templates to your account.'); return; }
  document.getElementById('importFile').click();
};
document.getElementById('importFile').onchange = async (e)=>{
  const f = e.target.files[0]; if(!f) return; if(!currentUser){ alert('Login first to import into your account.'); return; }
  const txt = await f.text(); importTemplates(JSON.parse(txt));
}

// Utils: compute feature vector from landmarks (normalized distances + a few angles)
function computeFeatures(landmarks){
  // landmarks: array of {x,y,z}
  // normalize by distance wrist(0) to middle_finger_mcp(9)
  const p = l => ({x:landmarks[l].x, y:landmarks[l].y, z:landmarks[l].z});
  const dist = (a,b)=>Math.hypot(a.x-b.x,a.y-b.y,a.z-b.z);
  const wrist = p(0), ref = p(9);
  const norm = dist(wrist, ref) || 1e-6;

  const fingers = [4,8,12,16,20]; // tips
  const mids = [3,7,11,15,19];
  const features = [];

  // normalized tip-to-wrist distances
  for(let i=0;i<fingers.length;i++) features.push(dist(p(fingers[i]), wrist)/norm);

  // tip-to-mid distances (finger curl)
  for(let i=0;i<fingers.length;i++) features.push(dist(p(fingers[i]), p(mids[i]))/norm);

  // selected inter-finger distances
  const pairs = [[8,12],[8,4],[12,16],[16,20],[4,8]];
  pairs.forEach(([a,b])=>features.push(dist(p(a),p(b))/norm));

  // angles at MCPs (using vectors) for index/middle/ring
  function angle(a,b,c){
    const u = {x:a.x-b.x,y:a.y-b.y,z:a.z-b.z};
    const v = {x:c.x-b.x,y:c.y-b.y,z:c.z-b.z};
    const du = Math.hypot(u.x,u.y,u.z), dv = Math.hypot(v.x,v.y,v.z);
    if(!du||!dv) return 0;
    const dot = (u.x*v.x+u.y*v.y+u.z*v.z)/(du*dv);
    return Math.acos(Math.max(-1,Math.min(1,dot)));
  }
  features.push(angle(p(5),p(6),p(8))); // index
  features.push(angle(p(9),p(10),p(12))); // middle
  features.push(angle(p(13),p(14),p(16))); // ring

  return features;
}

// Rule-based classifier for immediate recognition, fallback to nearest-neighbor templates
// ruleBasedClassify now uses both features and landmarks for finer checks
function ruleBasedClassify(features, landmarks){
  // features layout:
  // 0-4: tip-to-wrist distances (thumb,index,middle,ring,pinky)
  // 5-9: tip-to-mid distances (curl)
  // 10-14: inter-finger pair distances ([8,12],[8,4],[12,16],[16,20],[4,8])
  // 15-17: angles for index/middle/ring
  if(!features || features.length<18 || !landmarks) return null;
  const tip = features.slice(0,5);
  const tipToMid = features.slice(5,10);
  const pairs = features.slice(10,15);

  const thumbDist = tip[0], indexDist = tip[1], middleDist = tip[2], ringDist = tip[3], pinkyDist = tip[4];

  // normalized distance reference
  const p = i => ({x:landmarks[i].x, y:landmarks[i].y, z:landmarks[i].z});
  const dist = (a,b)=>Math.hypot(a.x-b.x,a.y-b.y,a.z-b.z);
  const norm = dist(p(0), p(9)) || 1e-6;

  // helpers
  const isExtended = (tipIdx, mcpIdx)=> ((p(mcpIdx).y - p(tipIdx).y) / norm) > 0.03 || (dist(p(tipIdx), p(mcpIdx))/norm)>0.2;
  const isCurled = (tipIdx, pipIdx)=> (dist(p(tipIdx), p(pipIdx))/norm) < 0.12;

  const extendedIndex = isExtended(8,5);
  const extendedMiddle = isExtended(12,9);
  const extendedRing = isExtended(16,13);
  const extendedPinky = isExtended(20,17);
  const extendedThumb = (thumbDist > 0.55);

  const thumbTip = p(4), idxMCP = p(5), midMCP = p(9), ringMCP = p(13);
  const d_thumb_index_mcp = dist(thumbTip, idxMCP)/norm;
  const d_thumb_middle_mcp = dist(thumbTip, midMCP)/norm;
  const d_thumb_ring_mcp = dist(thumbTip, ringMCP)/norm;

  // A/E/M/N/S/T (fist variants) -> all fingers curled
  if(!extendedIndex && !extendedMiddle && !extendedRing && !extendedPinky){
    // thumb near palm center -> A
    if(d_thumb_index_mcp < 0.08 && d_thumb_middle_mcp < 0.08) return 'A';
    // thumb over fingers (on top) -> S
    if(d_thumb_index_mcp < 0.12 && (thumbTip.y < p(8).y)) return 'S';
    // thumb between fingers -> T (thumb tucked)
    if(d_thumb_index_mcp < 0.06 && thumbTip.y > p(8).y) return 'T';
    // thumb near middle MCP -> N or M depending on how many fingers slightly above thumb
    if(d_thumb_middle_mcp < d_thumb_ring_mcp && d_thumb_middle_mcp < d_thumb_index_mcp) {
      // decide M vs N by approximate overlap count
      const smallCount = [8,12,16].reduce((c,i)=> c + (dist(p(i), thumbTip)/norm < 0.09 ? 1:0), 0);
      if(smallCount>=3) return 'M';
      if(smallCount===2) return 'N';
    }
    // E: fingers curled with thumb over tips (close to tips)
    if(d_thumb_index_mcp < 0.06) return 'E';
  }

  // D: index extended alone
  if(extendedIndex && !extendedMiddle && !extendedRing && !extendedPinky) return 'D';

  // I, L, U, V, W, Y: based on extended count and thumb
  const extCount = [extendedIndex, extendedMiddle, extendedRing, extendedPinky].filter(Boolean).length + (extendedThumb?1:0);
  if(extendedPinky && !extendedIndex && !extendedMiddle && !extendedRing) return 'I';
  if(extendedIndex && !extendedMiddle && !extendedRing && !extendedPinky && extendedThumb) return 'L';
  if(extendedIndex && extendedMiddle && !extendedRing && !extendedPinky){
    const idxMidTipDist = dist(p(8), p(12))/norm;
    if(idxMidTipDist < 0.18) return 'U';
    return 'V';
  }
  if(extendedIndex && extendedMiddle && extendedRing && !extendedPinky) return 'W';
  if(extendedThumb && extendedPinky && !extendedIndex && !extendedMiddle && !extendedRing) return 'Y';

  // C and O: curved shapes: distance thumb to finger tips
  const tips = [p(8), p(12), p(16), p(20)];
  const dthumbTips = tips.map(t=>dist(thumbTip, t)/norm);
  const avgThumbTip = dthumbTips.reduce((s,d)=>s+d,0)/dthumbTips.length;
  if(avgThumbTip < 0.12) return 'O';
  if(avgThumbTip < 0.22) return 'C';

  // F: thumb-index touching
  if(dist(p(4), p(8))/norm < 0.08) return 'F';

  // R: crossed index+middle (small tip distance while both extended)
  if(extendedIndex && extendedMiddle && (dist(p(8), p(12))/norm) < 0.08) return 'R';

  // X: index bent (curled) while others not extended
  if(isCurled(8,7) && !extendedMiddle && !extendedRing && !extendedPinky) return 'X';

  // K/P/Q: best-effort heuristics
  if(extendedIndex && extendedMiddle && !extendedRing && !extendedPinky){
    // hand rotated downwards -> P
    if(p(8).y > p(12).y) return 'P';
    // thumb between -> K
    if(dist(p(4), p(6))/norm < 0.12) return 'K';
    return 'K';
  }
  // Q: like G but with thumb down and index pointing down
  if(extendedIndex && !extendedMiddle && !extendedRing && !extendedPinky && (p(8).y > p(4).y)) return 'Q';

  // fallback
  return null;
}

function classify(features, landmarks){
  // Prefer model if available
  if(tfModel){
    try{
      const t = tf.tensor2d([features]);
      const out = tfModel.predict(t);
      const arr = out.arraySync()[0];
      const idx = arr.indexOf(Math.max(...arr));
      // map idx back to label: use templates keys order
      const keys = Object.keys(templates).filter(k=>templates[k] && templates[k].length);
      if(keys[idx]) return {letter: keys[idx], score: 1-arr[idx]};
    }catch(e){ console.warn('Model predict failed', e); }
  }

  const rule = ruleBasedClassify(features, landmarks);
  if(rule) return {letter:rule, score:0};

  // nearest-neighbor against stored templates (average per letter)
  let best = {letter:'—',score:Infinity};
  for(const [letter, samples] of Object.entries(templates)){
    if(!samples || !samples.length) continue;
    const numericSamples = samples.filter(s => Array.isArray(s));
    if(numericSamples.length===0) continue;
    const mean = numericSamples[0].slice();
    for(let i=1;i<numericSamples.length;i++) for(let j=0;j<mean.length;j++) mean[j]+=numericSamples[i][j];
    for(let j=0;j<mean.length;j++) mean[j]/=numericSamples.length;
    let s=0; for(let k=0;k<mean.length;k++){ const d=(features[k]-mean[k]); s+=d*d; }
    s=Math.sqrt(s);
    if(s<best.score){ best={letter,score:s}; }
  }
  if(best.score>0.45) return {letter:'?',score:best.score};
  return {letter:best.letter, score:best.score};
}

// Keyboard recording
window.addEventListener('keydown', e=>{
  const k = e.key.toUpperCase();
  if(k.length===1 && k>='A' && k<='Z'){
    if(!lastFeatures){ alert('No hand/features detected to record. Show the letter to the camera first.'); return; }
    if(!currentUser){ alert('Please login as a user to save templates (admin / 1234).'); return; }
    templates[k] = templates[k] || [];
    templates[k].push(lastFeatures.slice());
    saveTemplates();
  }
});

// MediaPipe Hands setup
const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});
hands.setOptions({
  selfieMode: true,
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.65,
  minTrackingConfidence: 0.5
});

let lastFeatures = null;
let lastPrediction = null;

// stability/debounce: require N consecutive identical frames
let stableLetter = '';
let stableCount = 0;
const STABLE_REQUIRED = 5;
// recording mode flag
let recordingMode = false;
let currentRecordingLetter = null;

hands.onResults(results=>{
  // ensure canvas matches video
  overlay.width = video.videoWidth || overlay.clientWidth;
  overlay.height = video.videoHeight || overlay.clientHeight;
  ctx.clearRect(0,0,overlay.width,overlay.height);

  if(results.multiHandLandmarks && results.multiHandLandmarks.length){
    body.classList.add('hand-on');
    handState.textContent='Yes';
    const lm = results.multiHandLandmarks[0];
    // draw
    drawConnectors(ctx, lm, HAND_CONNECTIONS, {color:'#00f0ff', lineWidth:2});
    drawLandmarks(ctx, lm, {color:'#7c3aed', lineWidth:1});
    // compute features and classify
    lastFeatures = computeFeatures(lm);
    if(!recordingMode){
      const res = classify(lastFeatures, lm);
      const letter = res ? res.letter : '';
      if(letter === stableLetter){
        stableCount++;
      } else {
        stableLetter = letter; stableCount = 1;
      }
      if(stableLetter && stableLetter !== '' && stableCount >= STABLE_REQUIRED){
        translated.textContent = stableLetter;
      }
    } else {
      // recordingMode: show live landmarks only, update overlay and lastFeatures for capture
      // indicate current letter being recorded
      if(currentRecordingLetter){
        // draw a label
        ctx.font = '22px Inter'; ctx.fillStyle = 'rgba(124,58,237,0.95)'; ctx.fillText('Recording: '+currentRecordingLetter, 12, 28);
      }
    }
  } else {
    body.classList.remove('hand-on');
    handState.textContent='No';
    lastFeatures = null;
    translated.textContent = '—';
    // reset stability state
    stableLetter = ''; stableCount = 0;
  }
});

// Use MediaPipe Camera helper for reliable frame delivery
function initCamera(){
  if(typeof Camera === 'undefined'){
    console.warn('MediaPipe Camera not found — falling back to getUserMedia');
    return fallbackCamera();
  }
  const cam = new Camera(video, {
    onFrame: async () => { await hands.send({image: video}); },
    width: 1280,
    height: 720,
    fps: 30
  });
  cam.start();
}

async function fallbackCamera(){
  try{
    const stream = await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720, frameRate:30}, audio:false});
    video.srcObject = stream;
    await video.play();
    const interval = 1000/30;
    setInterval(async ()=>{
      if(video.readyState < 2) return;
      await hands.send({image: video});
    }, interval);
  }catch(err){ console.error(err); alert('Camera error: '+err.message); }
}

initCamera();

// expose small helper to update templates from import
window.__asl = { templates, saveTemplates, importTemplates };

// wire model buttons
document.getElementById('loadModelBtn').onclick = async ()=>{
  const ok = await tryLoadModel();
  alert(ok ? 'Model loaded' : `No model found at ${MODEL_URL} or IndexedDB`);
};
document.getElementById('trainModelBtn').onclick = async ()=>{
  if(!currentUser){ alert('Login first to train a model with your templates'); return; }
  await trainModelFromTemplates(40);
};

// --- Login modal behavior ---
const loginModal = document.getElementById('loginModal');
const loginBtn = document.getElementById('loginBtn');
const logoutBtn = document.getElementById('logoutBtn');
const loginConfirm = document.getElementById('loginConfirm');
const loginCancel = document.getElementById('loginCancel');
const usernameInput = document.getElementById('usernameInput');
const passwordInput = document.getElementById('passwordInput');
const currentUserEl = document.getElementById('currentUser');

loginBtn.onclick = ()=>{ loginModal.classList.remove('hidden'); };
loginCancel.onclick = ()=>{ loginModal.classList.add('hidden'); };

loginConfirm.onclick = ()=>{
  const user = (usernameInput.value||'').trim();
  const pass = passwordInput.value||'';
    if(user === 'admin' && pass === '1234'){
    currentUser = user;
    templates = loadTemplates(currentUser);
    updateSampleCount();
    currentUserEl.textContent = currentUser;
    loginModal.classList.add('hidden');
    loginBtn.classList.add('hidden');
    logoutBtn.classList.remove('hidden');
    // show record button when logged in
    const recBtn = document.getElementById('recordModeBtn'); if(recBtn) recBtn.classList.remove('hidden');
    alert('Logged in as '+currentUser);
  } else {
    alert('Invalid credentials');
  }
};

logoutBtn.onclick = ()=>{
  currentUser = null;
  templates = JSON.parse(JSON.stringify(seededTemplates));
  updateSampleCount();
  currentUserEl.textContent = 'Guest';
  loginBtn.classList.remove('hidden');
  logoutBtn.classList.add('hidden');
  const recBtn = document.getElementById('recordModeBtn'); if(recBtn) recBtn.classList.add('hidden');
  alert('Logged out');
};

// Recorder UI wiring
const recordModeBtn = document.getElementById('recordModeBtn');
const recorder = document.getElementById('recorder');
const letterGrid = document.getElementById('letterGrid');
const recDoneBtn = document.getElementById('recDoneBtn');
const recClearLetter = document.getElementById('recClearLetter');

// build letter grid A-Z
function buildLetterGrid(){
  letterGrid.innerHTML = '';
  for(let i=0;i<26;i++){
    const L = String.fromCharCode(65+i);
    const btn = document.createElement('button');
    btn.textContent = L;
    btn.dataset.letter = L;
    const ct = document.createElement('span'); ct.className='letter-count';
    ct.textContent = (templates[L]||[]).length + ' samples';
    const wrap = document.createElement('div');
    wrap.appendChild(btn); wrap.appendChild(ct);
    btn.onclick = ()=>{ captureSample(L, btn, ct); };
    letterGrid.appendChild(wrap);
  }
}

function captureSample(letter, btn, countEl){
  if(!lastFeatures){ alert('No hand detected. Position your hand in view.'); return; }
  templates[letter] = templates[letter] || [];
  templates[letter].push(lastFeatures.slice());
  saveTemplates();
  if(countEl) countEl.textContent = templates[letter].length + ' samples';
  // flash button
  btn.classList.add('active'); setTimeout(()=>btn.classList.remove('active'),200);
}

recordModeBtn && (recordModeBtn.onclick = ()=>{
  if(!currentUser){ alert('Login first'); return; }
  recordingMode = true; currentRecordingLetter = null;
  recorder.classList.remove('hidden');
  buildLetterGrid();
});

recDoneBtn && (recDoneBtn.onclick = ()=>{
  recordingMode = false; currentRecordingLetter = null; recorder.classList.add('hidden');
});

recClearLetter && (recClearLetter.onclick = ()=>{
  if(!currentUser) { alert('Login first'); return; }
  const letter = prompt('Clear which letter? (A-Z)');
  if(!letter) return; const L = letter.trim().toUpperCase(); if(!L.match(/^[A-Z]$/)){ alert('Invalid letter'); return; }
  templates[L] = []; saveTemplates(); buildLetterGrid();
});
