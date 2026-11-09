<?php
declare(strict_types=1);

require dirname(__DIR__) . '/bootstrap.php';

use Mcvt\App;

$user = App::requireUser();
?><!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MCVT Vision · Live tracking</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/app.css">
</head>
<body class="app">
<div class="switch-flash" id="switchFlash"></div>

<header class="app-header">
  <div class="header-left">
    <div class="logo">
      <div class="logo-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
      </div>
      <span class="logo-text">MCVT Vision</span>
      <span class="logo-badge">FYP AI ReID</span>
    </div>
  </div>

  <div class="header-center">
    <div class="segmented" role="group" aria-label="View mode">
      <button id="btnGrid" class="btn btn-sm btn-primary"><span aria-hidden="true">▦</span> Grid View (<span id="camCount">0</span> Cams)</button>
      <button id="btnFocus" class="btn btn-sm"><span aria-hidden="true">▣</span> Focus Stream</button>
    </div>
    <label class="toggle-wrap" title="When on, the view jumps to whichever camera sees the tracked vehicle best">
      <span class="toggle-switch"><input type="checkbox" id="autoSwitch" checked><span class="toggle-slider"></span></span>
      <span>Cross-Camera Auto-Switch</span>
    </label>
  </div>

  <div class="header-right">
    <div class="status-indicator" id="engineStatus"><span class="status-dot offline"></span><span class="txt">AI Engine Offline</span></div>
    <div class="status-indicator" id="hubStatus"><span class="status-dot offline"></span><span class="txt">Connecting…</span></div>
    <div class="user-box">
      <span class="mono muted" id="clock"></span>
      <span><?= App::e($user['username']) ?></span>
      <a class="btn btn-sm" href="logout.php">Sign out</a>
    </div>
  </div>
</header>

<main class="app-main">
  <section id="cameraPanel" class="camera-panel grid-view">
    <div class="main-camera" id="mainCamera">
      <div class="main-camera-header">
        <div class="main-camera-title">
          <span>SURVEILLANCE FEED:</span>
          <span id="mainLabel" class="cam-id">–</span>
          <span class="muted small" id="modeChip">[Auto]</span>
        </div>
        <button id="btnBackGrid" class="btn btn-sm"><span aria-hidden="true">◂</span> Return to Camera Grid</button>
      </div>
      <div class="main-camera-stream" id="viewer">
        <canvas id="mainCanvas" aria-label="Live camera view"></canvas>
        <!-- Journey replay: the recorded video of each camera the vehicle passed, in order -->
        <div class="replay hidden" id="replay" aria-label="Journey replay">
          <video id="rVideo" muted playsinline preload="auto"></video>
          <canvas id="rCanvas"></canvas>
          <div class="replay-top">
            <span class="replay-title">JOURNEY REPLAY · <strong id="rGid"></strong></span>
            <span class="mono" id="rClock"></span>
          </div>
          <div class="replay-bar">
            <div class="replay-steps" id="rSteps"></div>
            <div class="replay-controls">
              <button class="btn btn-sm" id="rPrev" title="Previous camera">⏮</button>
              <button class="btn btn-sm" id="rPlay" title="Pause / play">⏸</button>
              <button class="btn btn-sm" id="rNext" title="Next camera">⏭</button>
              <button class="btn btn-sm" id="rSpeed" title="Playback speed">1×</button>
              <button class="btn btn-sm btn-primary" id="rClose">Back to live</button>
            </div>
          </div>
        </div>
        <div class="banner hidden" id="switchBanner" role="status"></div>
        <div class="follow-chip hidden" id="followChip">
          <span class="status-dot tracking"></span>
          <span>Tracking <strong id="followGid"></strong></span>
          <span class="muted" id="followReason"></span>
          <button class="chip-x" id="unfollow" title="Stop tracking">×</button>
        </div>
      </div>
      <div class="main-camera-footer">
        <span id="perf" class="mono">Waiting for the AI engine…</span>
        <span class="mono">AUTOSWITCH: <span id="autoState" class="ok">ENGAGED</span></span>
      </div>
    </div>

    <div id="camGrid" class="camera-grid"></div>
    <div class="grid-empty" id="gridEmpty">
      <div class="no-tracking-icon" aria-hidden="true">📡</div>
      <p><strong>Waiting for camera streams…</strong></p>
      <p class="muted">Start the WebSocket hub and the AI engine (see README). Cameras appear here as soon as the engine connects.</p>
    </div>
  </section>

  <aside class="app-sidebar">
    <!-- Active vehicle target -->
    <div class="sidebar-section target-section">
      <div class="sidebar-section-header">
        <span class="sidebar-section-title">Active Vehicle Target</span>
        <span class="accent small">Cross-Cam Lock</span>
      </div>
      <div class="sidebar-section-content" id="targetBody">
        <div class="no-tracking" id="noTarget">
          <div class="no-tracking-icon" aria-hidden="true">🎯</div>
          <p>Click a vehicle (in the gallery or on the video) to track it across cameras.<br>
            The system will auto-switch to follow it.</p>
        </div>
        <div class="tracking-vehicle-info hidden" id="target">
          <div class="target-top">
            <div>
              <div class="tracking-label" id="tLabel">● Tracking</div>
              <div class="tracking-gid" id="tGid"></div>
            </div>
            <img id="tThumb" class="tthumb" alt="">
          </div>
          <div class="tracking-details">
            <div class="tracking-detail-item"><div class="label">Type</div><div class="value" id="tClass">–</div></div>
            <div class="tracking-detail-item"><div class="label">Colour</div><div class="value" id="tColor">–</div></div>
            <div class="tracking-detail-item"><div class="label">State</div><div class="value" id="tMotion">–</div></div>
            <div class="tracking-detail-item"><div class="label">Now on</div><div class="value" id="tNow">–</div></div>
            <div class="tracking-detail-item"><div class="label">First seen</div><div class="value" id="tFirst">–</div></div>
            <div class="tracking-detail-item"><div class="label">Last seen</div><div class="value" id="tLast">–</div></div>
          </div>
          <div class="camera-path" id="tPath"></div>
          <div class="journey">
            <div class="journey-head">
              <span class="sub-title">Complete journey</span>
              <button class="btn btn-sm btn-primary" id="btnReplay" disabled>▶ Replay journey</button>
            </div>
            <ol class="journey-list" id="tJourney"></ol>
          </div>
          <div class="features" id="tFeatures">
            <div class="journey-head">
              <span class="sub-title">Extracted features (ReID)</span>
              <span class="muted small" id="fModel"></span>
            </div>
            <div class="feat-attrs" id="fAttrs"></div>
            <div class="feat-label">Appearance fingerprint <span class="muted" id="fEmbInfo"></span></div>
            <canvas class="feat-emb" id="fEmb" width="64" height="32" title="Mean embedding"></canvas>
            <div class="feat-scale"><span>−</span><span>0</span><span>+</span></div>
            <div class="feat-label">Stored samples <span class="muted" id="fSampInfo"></span></div>
            <div class="feat-samples">
              <canvas class="feat-pair" id="fPair" width="1" height="1" title="Similarity between stored samples"></canvas>
              <ul class="feat-samp-list" id="fSamples"></ul>
            </div>
            <div class="feat-label">Most similar other vehicles <span class="muted" id="fNearInfo"></span></div>
            <ul class="feat-near" id="fNear"></ul>
          </div>
          <div class="target-actions">
            <button class="btn btn-sm" id="btnStop">Stop tracking</button>
            <button class="btn btn-sm" id="btnJourney">Request full video</button>
          </div>
          <details class="more">
            <summary>Timeline, switches &amp; requests</summary>
            <div class="sub-title">Camera timeline</div>
            <div class="timeline" id="tTimeline"></div>
            <div class="sub-title">Camera switches</div>
            <ul class="mini" id="tSwitches"></ul>
            <div class="sub-title">Video requests</div>
            <ul class="mini" id="tJobs"></ul>
          </details>
        </div>
      </div>
    </div>

    <!-- Vehicle gallery -->
    <div class="sidebar-section gallery-section">
      <div class="sidebar-section-header">
        <span class="sidebar-section-title">Vehicle Gallery (Global ID)</span>
        <div class="tabs" role="tablist">
          <button class="tab active" data-tab="live" role="tab">Live <span id="liveCount">0</span></button>
          <button class="tab" data-tab="history" role="tab">All</button>
          <button class="tab" data-tab="requests" role="tab">Requests</button>
        </div>
      </div>
      <div class="sidebar-section-content gallery-body">
        <input class="search" id="search" placeholder="Filter by id, type, colour, camera">
        <div class="vehicle-list" id="vehicleList"></div>
        <ul class="mini hidden" id="jobList"></ul>
      </div>
    </div>

    <!-- Analytics -->
    <div class="sidebar-section">
      <div class="sidebar-section-header"><span class="sidebar-section-title">ReID Performance Analytics</span></div>
      <div class="sidebar-section-content">
        <div class="stats-grid">
          <div class="stat-card"><div class="stat-value" id="statKnown">0</div><div class="stat-label">Unique Vehicles</div></div>
          <div class="stat-card"><div class="stat-value" id="statLive">0</div><div class="stat-label">In Live View</div></div>
          <div class="stat-card"><div class="stat-value" id="statCross">0</div><div class="stat-label">Cross-Cam ReIDs</div></div>
          <div class="stat-card"><div class="stat-value" id="statNew">0</div><div class="stat-label">New Signatures</div></div>
        </div>
      </div>
    </div>

    <!-- Event log -->
    <div class="sidebar-section log-section">
      <div class="sidebar-section-header">
        <span class="sidebar-section-title">ReID Event Log</span>
        <span class="live-tag">● LIVE</span>
      </div>
      <div class="sidebar-section-content log-body">
        <div class="events-log" id="log"></div>
      </div>
    </div>
  </aside>
</main>

<dialog id="journeyDialog">
  <form method="dialog" id="journeyForm">
    <h2>Full journey video</h2>
    <p class="muted">All clips of <strong id="jGid" class="mono"></strong> from every camera are joined in time order, with overlaps removed, and emailed when ready.</p>
    <label>Email to<input type="email" id="jEmail" placeholder="name@example.com" autocomplete="email"></label>
    <p class="muted small">Leave empty to only build the video for download.</p>
    <p class="form-error hidden" id="jError"></p>
    <div class="dialog-actions">
      <button class="btn" value="cancel" formnovalidate>Cancel</button>
      <button class="btn btn-primary" id="jSend" value="send">Send</button>
    </div>
  </form>
</dialog>

<div class="toast-container" id="toasts"></div>
<script src="assets/app.js"></script>
</body>
</html>
