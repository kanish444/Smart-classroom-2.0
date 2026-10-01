/**
 * SmartClass Vision AI - Smart Board Dashboard Interactive Engine
 * Handles SSE live telemetry, polling fallback, dynamic attendance rendering,
 * search/filtering, session controls, and export downloads.
 */

(function () {
  'use strict';

  // State
  let currentSession = null;
  let allRosterRecords = [];
  let currentFilter = 'ALL';
  let searchQuery = '';
  let viewMode = 'normal';
  let sseSource = null;
  let pollTimer = null;
  let isOperator = false;

  // Cached DOM elements
  const el = {
    clockTime: document.getElementById('clock-time'),
    clockDate: document.getElementById('clock-date'),
    sessClass: document.getElementById('sess-class'),
    sessSubject: document.getElementById('sess-subject'),
    sessStatus: document.getElementById('sess-status'),
    sessStart: document.getElementById('sess-start'),
    sessCurrent: document.getElementById('sess-current'),
    sessEnd: document.getElementById('sess-end'),
    btnStart: document.getElementById('btn-start-session'),
    btnPause: document.getElementById('btn-pause-session'),
    btnResume: document.getElementById('btn-resume-session'),
    btnEnd: document.getElementById('btn-end-session'),
    kpiTotal: document.getElementById('kpi-total'),
    kpiPresent: document.getElementById('kpi-present'),
    kpiLate: document.getElementById('kpi-late'),
    kpiNotseen: document.getElementById('kpi-notseen'),
    kpiUnknown: document.getElementById('kpi-unknown'),
    kpiTracks: document.getElementById('kpi-tracks'),
    kpiFpsText: document.getElementById('kpi-fps-text'),
    cameraStream: document.getElementById('camera-stream'),
    videoOfflineOverlay: document.getElementById('video-offline-overlay'),
    unknownAlert: document.getElementById('unknown-alert'),
    viewModeIndicator: document.getElementById('view-mode-indicator'),
    btnModeNormal: document.getElementById('btn-mode-normal'),
    btnModeDebug: document.getElementById('btn-mode-debug'),
    btnModeRaw: document.getElementById('btn-mode-raw'),
    linkDroidcamRemote: document.getElementById('link-droidcam-remote'),
    btnFullscreen: document.getElementById('btn-fullscreen'),

    videoWrapper: document.getElementById('video-wrapper'),
    btnExportCsv: document.getElementById('btn-export-csv'),
    btnExportJson: document.getElementById('btn-export-json'),
    studentSearchInput: document.getElementById('student-search-input'),
    filterBtns: document.querySelectorAll('.filter-btn'),
    rosterTbody: document.getElementById('roster-tbody'),
    rosterCountBadge: document.getElementById('roster-count-badge'),
    countAll: document.getElementById('count-all'),
    countPresent: document.getElementById('count-present'),
    countLate: document.getElementById('count-late'),
    countNotseen: document.getElementById('count-notseen'),
    endSessionModal: document.getElementById('end-session-modal'),
    btnModalCancel: document.getElementById('btn-modal-cancel'),
    btnModalConfirmEnd: document.getElementById('btn-modal-confirm-end'),
    btnOperatorToggle: document.getElementById('btn-operator-toggle'),
    operatorLabel: document.getElementById('operator-label'),
    operatorModal: document.getElementById('operator-modal'),
    inputOperatorToken: document.getElementById('input-operator-token'),
    btnOperatorCancel: document.getElementById('btn-operator-cancel'),
    btnOperatorSave: document.getElementById('btn-operator-save'),
    healthCamera: document.getElementById('health-camera'),
    healthRec: document.getElementById('health-recognition'),
    healthTrack: document.getElementById('health-tracking'),
    healthDb: document.getElementById('health-db'),

    // Camera Source UI Elements
    cardCamPc: document.getElementById('card-cam-pc'),
    cardCamDroidcam: document.getElementById('card-cam-droidcam'),
    cardCamExtension: document.getElementById('card-cam-extension'),
    cardCamEsp32: document.getElementById('card-cam-esp32'),

    panelCamPc: document.getElementById('panel-cam-pc'),
    panelCamDroidcam: document.getElementById('panel-cam-droidcam'),
    panelCamExtension: document.getElementById('panel-cam-extension'),
    panelCamEsp32: document.getElementById('panel-cam-esp32'),

    btnConnectPc: document.getElementById('btn-connect-pc'),
    btnTestDroidcam: document.getElementById('btn-test-droidcam'),
    btnConnectDroidcam: document.getElementById('btn-connect-droidcam'),
    btnTestExtension: document.getElementById('btn-test-extension'),
    btnConnectExtension: document.getElementById('btn-connect-extension'),
    btnTestEsp32: document.getElementById('btn-test-esp32'),
    btnConnectEsp32: document.getElementById('btn-connect-esp32'),
    btnStopCamera: document.getElementById('btn-stop-camera'),
    btnStartCamera: document.getElementById('btn-start-camera'),

    inputDroidcamHost: document.getElementById('input-droidcam-host'),
    inputDroidcamPort: document.getElementById('input-droidcam-port'),
    inputDroidcamVideourl: document.getElementById('input-droidcam-videourl'),
    linkDroidcamRemote: document.getElementById('link-droidcam-remote'),
    selectExtensionDevice: document.getElementById('select-extension-device'),
    inputEsp32Url: document.getElementById('input-esp32-url'),

    activeCamName: document.getElementById('active-cam-name'),
    activeCamStatusPill: document.getElementById('active-cam-status-pill'),
    activeCamRes: document.getElementById('active-cam-res'),
    activeCamFps: document.getElementById('active-cam-fps'),

    camTroubleshootBox: document.getElementById('cam-troubleshoot-box'),
    troubleshootTitle: document.getElementById('troubleshoot-title'),
    troubleshootList: document.getElementById('troubleshoot-list'),

    // Backward compatibility refs
    cameraSourceSelect: document.getElementById('camera-source-select'),
    droidcamIpGroup: document.getElementById('droidcam-ip-group'),
    btnTestCamera: document.getElementById('btn-test-camera'),
    btnApplyCamera: document.getElementById('btn-apply-camera'),
    camActiveBadge: document.getElementById('cam-active-badge'),
    camResBadge: document.getElementById('cam-res-badge'),
    camFpsBadge: document.getElementById('cam-fps-badge'),
    camStatusPill: document.getElementById('cam-status-pill')
  };


  // =========================================================================
  // 1. Clock Engine
  // =========================================================================
  function updateClock() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
    const dateStr = now.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
    if (el.clockTime) el.clockTime.textContent = timeStr;
    if (el.clockDate) el.clockDate.textContent = dateStr;
    if (el.sessCurrent) el.sessCurrent.textContent = timeStr.substring(0, 5);
  }
  setInterval(updateClock, 1000);
  updateClock();

  // =========================================================================
  // 2. Authentication Context
  // =========================================================================
  function getOperatorToken() {
    return localStorage.getItem('smartclass_operator_token') || '';
  }

  function setOperatorToken(token) {
    if (token && token.trim()) {
      localStorage.setItem('smartclass_operator_token', token.trim());
      isOperator = true;
      if (el.operatorLabel) el.operatorLabel.textContent = 'Operator';
    } else {
      localStorage.removeItem('smartclass_operator_token');
      isOperator = false;
      if (el.operatorLabel) el.operatorLabel.textContent = 'Viewer';
    }
  }

  function getAuthHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    const token = getOperatorToken();
    if (token) {
      headers['X-Operator-Token'] = token;
    }
    return headers;
  }

  // Initialize operator status
  setOperatorToken(getOperatorToken());

  // =========================================================================
  // 3. Real-Time Telemetry: SSE with Polling Fallback
  // =========================================================================
  function initSSE() {
    if (sseSource) {
      sseSource.close();
    }

    try {
      sseSource = new EventSource('/api/events/sse');

      sseSource.onmessage = function (event) {
        try {
          const data = JSON.parse(event.data);
          applyTelemetry(data);
        } catch (err) {
          console.warn('Error parsing SSE event:', err);
        }
      };

      sseSource.onerror = function () {
        console.warn('SSE disconnected. Falling back to efficient polling.');
        if (sseSource) {
          sseSource.close();
          sseSource = null;
        }
        startPolling();
      };
    } catch (e) {
      console.warn('SSE not supported or failed. Starting polling.');
      startPolling();
    }
  }

  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(pollSummary, 1500);
    pollSummary();
  }

  async function pollSummary() {
    try {
      const res = await fetch('/api/dashboard/summary');
      if (res.ok) {
        const json = await res.json();
        if (json.success && json.data) {
          applyTelemetry(json.data);
        }
      }
    } catch (err) {
      console.debug('Polling error:', err);
    }
  }

  function applyTelemetry(data) {
    // 1. Update KPIs
    if (el.kpiTotal && data.total_enrolled !== undefined) el.kpiTotal.textContent = data.total_enrolled;
    if (el.kpiPresent && data.present_count !== undefined) el.kpiPresent.textContent = data.present_count;
    if (el.kpiLate && data.late_count !== undefined) el.kpiLate.textContent = data.late_count;
    if (el.kpiNotseen && data.not_seen_count !== undefined) el.kpiNotseen.textContent = data.not_seen_count;
    if (el.kpiUnknown && data.unknown_count !== undefined) el.kpiUnknown.textContent = data.unknown_count;
    if (el.kpiTracks && (data.active_track_count !== undefined || data.active_tracks)) {
      const trackCount = data.active_track_count !== undefined ? data.active_track_count : (data.active_tracks ? data.active_tracks.length : 0);
      el.kpiTracks.textContent = trackCount;
    }
    if (el.kpiFpsText) {
      const camFps = data.camera_fps !== undefined ? data.camera_fps : (data.fps !== undefined ? data.fps : 30.0);
      const aiFps = data.ai_fps !== undefined ? data.ai_fps : (data.ai_inference_fps || 0.0);
      const aiLat = data.ai_latency_ms !== undefined ? data.ai_latency_ms : 0.0;
      el.kpiFpsText.textContent = `Cam: ${Number(camFps).toFixed(1)} FPS | AI: ${Number(aiFps).toFixed(1)} FPS (${Number(aiLat).toFixed(0)}ms)`;
    }

    // 2. Camera status overlay
    const isCamOnline = data.camera_online !== undefined ? data.camera_online : (data.system_status !== 'OFFLINE');
    if (el.videoOfflineOverlay) {
      el.videoOfflineOverlay.style.display = isCamOnline ? 'none' : 'flex';
    }

    // 3. Unknown face alert banner
    if (el.unknownAlert) {
      el.unknownAlert.style.display = (data.unknown_count && data.unknown_count > 0) ? 'flex' : 'none';
    }

    // 3b. Update Institutional Overview Strip
    const mTotStudents = document.getElementById('metric-total-students');
    const mPresStudents = document.getElementById('metric-students-present');
    const mAbsStudents = document.getElementById('metric-students-absent');
    const mAttRate = document.getElementById('metric-attendance-rate');
    const mAttSub = document.getElementById('metric-attendance-sub');
    const mActCam = document.getElementById('metric-active-cameras');
    const mCamStatusSub = document.getElementById('metric-camera-status-sub');
    const mAiStatus = document.getElementById('dept-ai-status');

    if (mTotStudents && data.total_enrolled !== undefined) {
      mTotStudents.textContent = data.total_enrolled;
      const pres = data.present_count || 0;
      if (mPresStudents) mPresStudents.textContent = pres;
      if (mAbsStudents) mAbsStudents.textContent = Math.max(0, data.total_enrolled - pres);
      if (mAttRate) {
        const rate = data.total_enrolled > 0 ? Math.round((pres / data.total_enrolled) * 100) : 0;
        mAttRate.textContent = `${rate}%`;
      }
      if (mAttSub) {
        if (data.session) {
          mAttSub.textContent = `Session: ${data.session.subject || 'Active'}`;
        } else {
          mAttSub.textContent = 'Standby (No Active Session)';
        }
      }
    }
    if (mActCam) {
      mActCam.textContent = isCamOnline ? '1 Active' : '0 Active';
    }
    if (mCamStatusSub) {
      mCamStatusSub.textContent = isCamOnline ? 'PC Camera (Online)' : 'Camera not connected';
    }
    if (mAiStatus && data.system_status) {
      mAiStatus.innerHTML = `<span class="pulse-dot"></span> AI Pipeline: ${data.system_status}`;
    }

    // 4. Session status & details
    if (data.session) {
      currentSession = data.session;
      if (el.sessClass) el.sessClass.textContent = data.session.class_section || '--';
      if (el.sessSubject) el.sessSubject.textContent = data.session.subject || '--';
      if (el.sessStart) el.sessStart.textContent = formatTimeOnly(data.session.planned_start_time);
      if (el.sessEnd) el.sessEnd.textContent = formatTimeOnly(data.session.planned_end_time);

      updateSessionStatusBadge(data.session.status);
      updateSessionButtons(data.session.status);
    } else {
      currentSession = null;
      if (el.sessClass) el.sessClass.textContent = 'None';
      if (el.sessSubject) el.sessSubject.textContent = 'Standby';
      if (el.sessStart) el.sessStart.textContent = '--:--';
      if (el.sessEnd) el.sessEnd.textContent = '--:--';
      updateSessionStatusBadge('NO_ACTIVE_SESSION');
      updateSessionButtons('NO_ACTIVE_SESSION');
    }
  }

  function formatTimeOnly(timeStr) {
    if (!timeStr) return '--:--';
    if (timeStr.includes('T')) {
      const parts = timeStr.split('T')[1].split(':');
      return `${parts[0]}:${parts[1]}`;
    }
    return timeStr.substring(0, 5);
  }

  function updateSessionStatusBadge(status) {
    if (!el.sessStatus) return;
    el.sessStatus.textContent = status || 'STANDBY';
    el.sessStatus.className = 'status-badge';
    switch (status) {
      case 'ACTIVE':
        el.sessStatus.classList.add('badge-active');
        break;
      case 'SCHEDULED':
        el.sessStatus.classList.add('badge-scheduled');
        break;
      case 'PAUSED':
        el.sessStatus.classList.add('badge-paused');
        break;
      case 'ENDED':
        el.sessStatus.classList.add('badge-ended');
        break;
      default:
        el.sessStatus.classList.add('badge-neutral');
        break;
    }
  }

  function updateSessionButtons(status) {
    if (!el.btnStart) return;
    el.btnStart.style.display = 'none';
    el.btnPause.style.display = 'none';
    el.btnResume.style.display = 'none';
    el.btnEnd.style.display = 'none';

    if (status === 'SCHEDULED') {
      el.btnStart.style.display = 'inline-block';
    } else if (status === 'ACTIVE') {
      el.btnPause.style.display = 'inline-block';
      el.btnEnd.style.display = 'inline-block';
    } else if (status === 'PAUSED' || status === 'RECOVERING') {
      el.btnResume.style.display = 'inline-block';
      el.btnEnd.style.display = 'inline-block';
    } else if (status === 'NO_ACTIVE_SESSION' || status === 'ENDED') {
      el.btnStart.style.display = 'inline-block';
    }
  }

  // =========================================================================
  // 4. Attendance Roster Fetching & Rendering
  // =========================================================================
  async function fetchRosterData() {
    try {
      if (currentSession && currentSession.session_id) {
        const res = await fetch(`/api/attendance/session/${encodeURIComponent(currentSession.session_id)}`);
        if (res.ok) {
          const json = await res.json();
          if (json.success && json.data) {
            processReport(json.data);
            return;
          }
        }
      }

      // Fallback: Fetch all students directory
      const stuRes = await fetch('/api/students?page=1&page_size=100');
      if (stuRes.ok) {
        const json = await stuRes.json();
        if (json.success && json.data && json.data.items) {
          allRosterRecords = json.data.items.map(s => ({
            student_id: s.student_id,
            student_name: s.student_name,
            status: 'NOT_SEEN',
            first_seen: '--',
            last_seen: '--'
          }));
          renderRoster();
        }
      }
    } catch (err) {
      console.warn('Error fetching roster data:', err);
    }
  }

  function processReport(report) {
    const list = [];
    const seenMap = {};

    // 1. Seen records (PRESENT or LATE)
    if (report.records) {
      for (const r of report.records) {
        seenMap[r.student_id] = true;
        const details = (report.student_details && report.student_details[r.student_id]) || null;
        let displayName = r.student_name || (details && details.student_name) || r.student_id;
        if (displayName && typeof displayName === 'string' && displayName.startsWith("STU_")) {
          displayName = displayName.substring(4);
        }
        let regNo = (details && details.register_no) || r.student_id;
        if (regNo && typeof regNo === 'string' && regNo.startsWith("STU_")) {
          regNo = regNo.substring(4);
        }
        list.push({
          student_id: regNo,
          student_name: displayName,
          status: r.status,
          first_seen: formatTimeOnly(r.first_seen),
          last_seen: formatTimeOnly(r.last_seen)
        });
      }
    }

    // 2. Not seen records
    if (report.not_seen_students) {
      for (const id of report.not_seen_students) {
        const details = (report.student_details && report.student_details[id]) || null;
        let displayName = (details && details.student_name) || id;
        if (displayName && typeof displayName === 'string' && displayName.startsWith("STU_")) {
          displayName = displayName.substring(4);
        }
        let regNo = (details && details.register_no) || id;
        if (regNo && typeof regNo === 'string' && regNo.startsWith("STU_")) {
          regNo = regNo.substring(4);
        }
        list.push({
          student_id: regNo,
          student_name: displayName,
          status: 'NOT_SEEN',
          first_seen: '--',
          last_seen: '--'
        });
      }
    }

    allRosterRecords = list;
    renderRoster();
  }

  function renderRoster() {
    if (!el.rosterTbody) return;

    let filtered = allRosterRecords.slice();

    // 1. Status Filter
    if (currentFilter !== 'ALL') {
      filtered = filtered.filter(item => item.status === currentFilter);
    }

    // 2. Text Search
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter(item =>
        item.student_id.toLowerCase().includes(q) ||
        (item.student_name && item.student_name.toLowerCase().includes(q))
      );
    }

    // Update Counter Badges
    const totalCount = allRosterRecords.length;
    const presentCount = allRosterRecords.filter(r => r.status === 'PRESENT').length;
    const lateCount = allRosterRecords.filter(r => r.status === 'LATE').length;
    const notseenCount = allRosterRecords.filter(r => r.status === 'NOT_SEEN').length;

    if (el.countAll) el.countAll.textContent = totalCount;
    if (el.countPresent) el.countPresent.textContent = presentCount;
    if (el.countLate) el.countLate.textContent = lateCount;
    if (el.countNotseen) el.countNotseen.textContent = notseenCount;
    if (el.rosterCountBadge) el.rosterCountBadge.textContent = `${filtered.length} of ${totalCount}`;

    // Render Table Body
    if (filtered.length === 0) {
      el.rosterTbody.innerHTML = '<tr><td colspan="5" class="empty-state">No student records match the filter criteria.</td></tr>';
      return;
    }

    let html = '';
    for (const item of filtered) {
      let badgeClass = 'status-notseen';
      if (item.status === 'PRESENT') badgeClass = 'status-present';
      else if (item.status === 'LATE') badgeClass = 'status-late';

      html += `
        <tr>
          <td class="font-mono"><strong>${escapeHtml(item.student_id)}</strong></td>
          <td>${escapeHtml(item.student_name)}</td>
          <td><span class="status-tag ${badgeClass}">${escapeHtml(item.status)}</span></td>
          <td class="font-mono">${escapeHtml(item.first_seen)}</td>
          <td class="font-mono">${escapeHtml(item.last_seen)}</td>
        </tr>
      `;
    }
    el.rosterTbody.innerHTML = html;
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // Refresh roster every 3 seconds
  setInterval(fetchRosterData, 3000);
  fetchRosterData();

  // =========================================================================
  // 5. System Health Check
  // =========================================================================
  async function updateHealthIndicators() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const json = await res.json();
        if (json.success && json.data && json.data.components) {
          const comps = json.data.components;
          setHealthPill(el.healthCamera, comps.camera);
          setHealthPill(el.healthRec, comps.recognition);
          setHealthPill(el.healthTrack, comps.tracking);
          setHealthPill(el.healthDb, comps.database);
        }
      }
    } catch (e) {
      console.debug('Health check error:', e);
    }
  }

  function setHealthPill(pillEl, compData) {
    if (!pillEl || !compData) return;
    const dot = pillEl.querySelector('.health-dot');
    if (!dot) return;
    dot.className = 'health-dot';
    if (compData.status === 'ONLINE') {
      dot.classList.add('online');
    } else if (compData.status === 'DEGRADED') {
      dot.classList.add('degraded');
    } else {
      dot.classList.add('offline');
    }
  }

  setInterval(updateHealthIndicators, 5000);
  updateHealthIndicators();

  // =========================================================================
  // 6. View Mode Toggling & Fullscreen
  // =========================================================================
  function setViewMode(mode) {
    viewMode = mode;
    if (el.cameraStream) {
      el.cameraStream.src = `/api/video/feed?view_mode=${mode}&t=${Date.now()}`;
    }
    if (el.btnModeNormal) el.btnModeNormal.classList.toggle('active', mode === 'normal');
    if (el.btnModeDebug) el.btnModeDebug.classList.toggle('active', mode === 'debug');
    if (el.btnModeRaw) el.btnModeRaw.classList.toggle('active', mode === 'raw');

    if (el.viewModeIndicator) {
      if (mode === 'debug') {
        el.viewModeIndicator.textContent = 'DEBUG HUD VIEW';
      } else if (mode === 'raw') {
        el.viewModeIndicator.textContent = 'RAW DETECTOR VIEW';
      } else {
        el.viewModeIndicator.textContent = 'NORMAL VIEW';
      }
    }
  }

  if (el.btnModeNormal) el.btnModeNormal.addEventListener('click', () => setViewMode('normal'));
  if (el.btnModeDebug) el.btnModeDebug.addEventListener('click', () => setViewMode('debug'));
  if (el.btnModeRaw) el.btnModeRaw.addEventListener('click', () => setViewMode('raw'));


  if (el.btnFullscreen && el.videoWrapper) {
    el.btnFullscreen.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        el.videoWrapper.requestFullscreen().catch(err => alert(`Fullscreen error: ${err.message}`));
      } else {
        document.exitFullscreen();
      }
    });
  }

  // =========================================================================
  // 7. Search & Filter Handlers
  // =========================================================================
  if (el.studentSearchInput) {
    el.studentSearchInput.addEventListener('input', (e) => {
      searchQuery = e.target.value.trim();
      renderRoster();
    });
  }

  if (el.filterBtns) {
    el.filterBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        el.filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.getAttribute('data-filter') || 'ALL';
        renderRoster();
      });
    });
  }

  // =========================================================================
  // 8. Session Action Handlers (Operator Protected)
  // =========================================================================
  async function performSessionAction(action, extraPayload = {}) {
    if (!currentSession || !currentSession.session_id) {
      alert('No session selected.');
      return;
    }

    const sid = encodeURIComponent(currentSession.session_id);
    const url = `/api/sessions/${sid}/${action}`;

    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify(extraPayload)
      });
      const json = await res.json();
      if (!res.ok || !json.success) {
        const errMsg = json.error ? json.error.message : 'Operation failed';
        alert(`Session action error: ${errMsg}`);
      } else {
        currentSession = json.data;
        updateSessionStatusBadge(currentSession.status);
        updateSessionButtons(currentSession.status);
        fetchRosterData();
      }
    } catch (err) {
      alert(`Network error performing ${action}: ${err.message}`);
    }
  }

  if (el.btnStart) el.btnStart.addEventListener('click', () => performSessionAction('start'));
  if (el.btnPause) el.btnPause.addEventListener('click', () => performSessionAction('pause'));
  if (el.btnResume) el.btnResume.addEventListener('click', () => performSessionAction('resume'));

  // End Session Confirmation Modal
  if (el.btnEnd && el.endSessionModal) {
    el.btnEnd.addEventListener('click', () => {
      el.endSessionModal.style.display = 'flex';
    });
  }

  if (el.btnModalCancel && el.endSessionModal) {
    el.btnModalCancel.addEventListener('click', () => {
      el.endSessionModal.style.display = 'none';
    });
  }

  if (el.btnModalConfirmEnd && el.endSessionModal) {
    el.btnModalConfirmEnd.addEventListener('click', async () => {
      el.endSessionModal.style.display = 'none';
      await performSessionAction('end', { confirm: true });
    });
  }

  // =========================================================================
  // 9. Operator Authentication Modal
  // =========================================================================
  if (el.btnOperatorToggle && el.operatorModal) {
    el.btnOperatorToggle.addEventListener('click', () => {
      if (el.inputOperatorToken) el.inputOperatorToken.value = getOperatorToken();
      el.operatorModal.style.display = 'flex';
    });
  }

  if (el.btnOperatorCancel && el.operatorModal) {
    el.btnOperatorCancel.addEventListener('click', () => {
      el.operatorModal.style.display = 'none';
    });
  }

  if (el.btnOperatorSave && el.operatorModal) {
    el.btnOperatorSave.addEventListener('click', () => {
      const val = el.inputOperatorToken ? el.inputOperatorToken.value : '';
      setOperatorToken(val);
      el.operatorModal.style.display = 'none';
    });
  }

  // =========================================================================
  // 10. Export Actions (CSV & JSON)
  // =========================================================================
  if (el.btnExportCsv) {
    el.btnExportCsv.addEventListener('click', () => {
      if (!currentSession || !currentSession.session_id) {
        alert('Please start or select a session to export attendance data.');
        return;
      }
      const sid = encodeURIComponent(currentSession.session_id);
      window.location.href = `/api/export/attendance/${sid}/csv`;
    });
  }

  if (el.btnExportJson) {
    el.btnExportJson.addEventListener('click', () => {
      if (!currentSession || !currentSession.session_id) {
        alert('Please start or select a session to export attendance data.');
        return;
      }
      const sid = encodeURIComponent(currentSession.session_id);
      window.open(`/api/export/attendance/${sid}/json`, '_blank');
    });
  }

  // =========================================================================
  // 11. Camera Source Management (PC, DROIDCAM, EXTENSION, ESP32)
  // =========================================================================
  let selectedSourceType = 'pc';

  function updateDroidCamUrlPreview() {
    const host = el.inputDroidcamHost ? (el.inputDroidcamHost.value.trim() || '10.140.159.218') : '10.140.159.218';
    const port = el.inputDroidcamPort ? (el.inputDroidcamPort.value.trim() || '4747') : '4747';
    if (el.inputDroidcamVideourl) {
      el.inputDroidcamVideourl.value = `http://${host}:${port}/video`;
    }
    if (el.linkDroidcamRemote) {
      el.linkDroidcamRemote.href = `http://${host}:${port}`;
    }
  }

  function showCameraSourcePanel(source) {
    selectedSourceType = source;
    // 1. Update cards active state
    const cards = [
      { id: 'pc', card: el.cardCamPc, panel: el.panelCamPc },
      { id: 'droidcam', card: el.cardCamDroidcam, panel: el.panelCamDroidcam },
      { id: 'extension', card: el.cardCamExtension, panel: el.panelCamExtension },
      { id: 'esp32', card: el.cardCamEsp32, panel: el.panelCamEsp32 }
    ];

    cards.forEach(item => {
      const match = (item.id === source) || (item.id === 'pc' && source === 'laptop') || (item.id === 'extension' && source === 'external');
      if (item.card) {
        if (match) item.card.classList.add('active');
        else item.card.classList.remove('active');
      }
      if (item.panel) {
        item.panel.style.display = match ? 'flex' : 'none';
      }
    });

    if (source === 'droidcam') {
      updateDroidCamUrlPreview();
    }
  }

  // Card click listeners
  if (el.cardCamPc) {
    el.cardCamPc.addEventListener('click', () => showCameraSourcePanel('pc'));
  }
  if (el.cardCamDroidcam) {
    el.cardCamDroidcam.addEventListener('click', () => showCameraSourcePanel('droidcam'));
  }
  if (el.cardCamExtension) {
    el.cardCamExtension.addEventListener('click', () => showCameraSourcePanel('extension'));
  }
  if (el.cardCamEsp32) {
    el.cardCamEsp32.addEventListener('click', () => showCameraSourcePanel('esp32'));
  }

  // Input listeners for DroidCam IP & Port
  if (el.inputDroidcamHost) {
    el.inputDroidcamHost.addEventListener('input', updateDroidCamUrlPreview);
  }
  if (el.inputDroidcamPort) {
    el.inputDroidcamPort.addEventListener('input', updateDroidCamUrlPreview);
  }

  async function loadCameraSources() {
    try {
      const res = await fetch('/api/camera/sources');
      if (res.ok) {
        const json = await res.json();
        if (json.success && json.data) {
          const d = json.data;
          const activeSrc = (d.active_source || 'pc').toLowerCase();
          showCameraSourcePanel(activeSrc);

          if (d.droidcam_config) {
            if (el.inputDroidcamHost && d.droidcam_config.host) {
              el.inputDroidcamHost.value = d.droidcam_config.host;
            }
            if (el.inputDroidcamPort && d.droidcam_config.port) {
              el.inputDroidcamPort.value = d.droidcam_config.port;
            }
            updateDroidCamUrlPreview();
          }

          if (d.esp32_config && el.inputEsp32Url && d.esp32_config.stream_url) {
            el.inputEsp32Url.value = d.esp32_config.stream_url;
          }

          if (d.available_devices && el.selectExtensionDevice) {
            el.selectExtensionDevice.innerHTML = '';
            d.available_devices.forEach(dev => {
              const opt = document.createElement('option');
              opt.value = dev.index;
              opt.textContent = dev.name;
              el.selectExtensionDevice.appendChild(opt);
            });
          }
        }
      }
    } catch (e) {
      console.warn('Error loading camera sources:', e);
    }
  }

  async function switchCamera(sourceType, payload, triggerBtn) {
    if (triggerBtn) {
      triggerBtn.disabled = true;
      triggerBtn.textContent = 'Connecting...';
    }

    try {
      const res = await fetch('/api/camera/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(Object.assign({ source: sourceType }, payload))
      });
      const json = await res.json();
      if (res.ok && json.success) {
        updateCameraStatus();
        if (el.cameraStream) {
          el.cameraStream.src = `/api/video/feed?view_mode=${viewMode}&t=${Date.now()}`;
        }
      } else {
        const errMsg = json.error ? json.error.message : (json.data ? json.data.message : 'Failed to switch camera source');
        alert(errMsg);
        updateCameraStatus();
      }
    } catch (err) {
      alert(`Network error switching camera: ${err.message}`);
    } finally {
      if (triggerBtn) {
        triggerBtn.disabled = false;
        triggerBtn.textContent = 'Connect Camera';
      }
    }
  }

  async function testCamera(sourceType, payload, triggerBtn) {
    if (triggerBtn) {
      triggerBtn.disabled = true;
      triggerBtn.textContent = 'Testing...';
    }

    try {
      const res = await fetch('/api/camera/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(Object.assign({ source: sourceType }, payload))
      });
      const json = await res.json();
      if (res.ok && json.success) {
        const d = json.data;
        if (d.connected) {
          alert(`● CONNECTED: ${d.message}`);
          if (el.camTroubleshootBox) el.camTroubleshootBox.style.display = 'none';
        } else {
          const reasons = d.reasons ? d.reasons.join('\n- ') : '';
          alert(`● NOT CONNECTED: ${d.message}\n\nTroubleshooting:\n- ${reasons}`);
          showTroubleshoot(sourceType, d.reasons);
        }
      } else {
        alert('Camera test failed or timed out.');
      }
    } catch (err) {
      alert(`Test error: ${err.message}`);
    } finally {
      if (triggerBtn) {
        triggerBtn.disabled = false;
        triggerBtn.textContent = 'Test Connection';
      }
    }
  }

  function showTroubleshoot(source, customReasons) {
    if (!el.camTroubleshootBox) return;
    el.camTroubleshootBox.style.display = 'flex';

    if (source === 'droidcam') {
      if (el.troubleshootTitle) el.troubleshootTitle.textContent = 'Unable to connect to DroidCam.';
      if (el.troubleshootList) {
        el.troubleshootList.innerHTML = `
          <li>Check phone DroidCam app</li>
          <li>Check same Wi-Fi</li>
          <li>Check IP address</li>
          <li>Check port</li>
          <li>Check firewall</li>
        `;
      }
    } else if (source === 'esp32') {
      if (el.troubleshootTitle) el.troubleshootTitle.textContent = 'Unable to connect to ESP32 Wi-Fi Camera.';
      if (el.troubleshootList) {
        el.troubleshootList.innerHTML = `
          <li>Check ESP32 is powered on and streaming</li>
          <li>Check same Wi-Fi network</li>
          <li>Check stream URL</li>
          <li>Check firewall</li>
        `;
      }
    } else {
      if (el.troubleshootTitle) el.troubleshootTitle.textContent = `Unable to connect to ${source.toUpperCase()} camera.`;
      if (el.troubleshootList && customReasons) {
        el.troubleshootList.innerHTML = customReasons.map(r => `<li>${r}</li>`).join('');
      }
    }
  }

  // 1. PC Camera Connect
  if (el.btnConnectPc) {
    el.btnConnectPc.addEventListener('click', () => {
      switchCamera('pc', { index: 0 }, el.btnConnectPc);
    });
  }

  // 2. DroidCam Test & Connect
  if (el.btnTestDroidcam) {
    el.btnTestDroidcam.addEventListener('click', () => {
      const host = el.inputDroidcamHost ? el.inputDroidcamHost.value.trim() : '10.140.159.218';
      const port = el.inputDroidcamPort ? parseInt(el.inputDroidcamPort.value.trim()) || 4747 : 4747;
      testCamera('droidcam', { host: host, port: port, video_path: '/video' }, el.btnTestDroidcam);
    });
  }
  if (el.btnConnectDroidcam) {
    el.btnConnectDroidcam.addEventListener('click', () => {
      const host = el.inputDroidcamHost ? el.inputDroidcamHost.value.trim() : '10.140.159.218';
      const port = el.inputDroidcamPort ? parseInt(el.inputDroidcamPort.value.trim()) || 4747 : 4747;
      switchCamera('droidcam', { host: host, port: port, video_path: '/video' }, el.btnConnectDroidcam);
    });
  }

  // 3. Extension Camera Test & Connect
  if (el.btnTestExtension) {
    el.btnTestExtension.addEventListener('click', () => {
      const devIdx = el.selectExtensionDevice ? parseInt(el.selectExtensionDevice.value) || 0 : 0;
      testCamera('extension', { index: devIdx }, el.btnTestExtension);
    });
  }
  if (el.btnConnectExtension) {
    el.btnConnectExtension.addEventListener('click', () => {
      const devIdx = el.selectExtensionDevice ? parseInt(el.selectExtensionDevice.value) || 0 : 0;
      switchCamera('extension', { index: devIdx }, el.btnConnectExtension);
    });
  }

  // 4. ESP32 Camera Test & Connect
  if (el.btnTestEsp32) {
    el.btnTestEsp32.addEventListener('click', () => {
      const streamUrl = el.inputEsp32Url ? el.inputEsp32Url.value.trim() : 'http://192.168.1.100:81/stream';
      testCamera('esp32', { stream_url: streamUrl }, el.btnTestEsp32);
    });
  }
  if (el.btnConnectEsp32) {
    el.btnConnectEsp32.addEventListener('click', () => {
      const streamUrl = el.inputEsp32Url ? el.inputEsp32Url.value.trim() : 'http://192.168.1.100:81/stream';
      switchCamera('esp32', { stream_url: streamUrl }, el.btnConnectEsp32);
    });
  }

  // STOP CAMERA Action
  if (el.btnStopCamera) {
    el.btnStopCamera.addEventListener('click', async () => {
      el.btnStopCamera.disabled = true;
      el.btnStopCamera.innerHTML = '<span class="ctrl-icon">⏹</span> Stopping...';
      try {
        const res = await fetch('/api/camera/stop', { method: 'POST' });
        const json = await res.json();
        if (res.ok && json.success) {
          if (el.activeCamStatusPill) {
            el.activeCamStatusPill.className = 'status-pill status-pill-stopped';
            el.activeCamStatusPill.textContent = '⏹ Stopped';
          }
          if (el.btnStopCamera) el.btnStopCamera.style.display = 'none';
          if (el.btnStartCamera) el.btnStartCamera.style.display = 'inline-flex';
          if (el.camTroubleshootBox) el.camTroubleshootBox.style.display = 'none';
          updateCameraStatus();
        } else {
          alert('Failed to stop camera feed: ' + (json.error ? json.error.message : 'Unknown error'));
        }
      } catch (err) {
        alert('Network error stopping camera: ' + err.message);
      } finally {
        if (el.btnStopCamera) {
          el.btnStopCamera.disabled = false;
          el.btnStopCamera.innerHTML = '<span class="ctrl-icon">⏹</span> STOP CAMERA';
        }
      }
    });
  }

  // START CAMERA Action
  if (el.btnStartCamera) {
    el.btnStartCamera.addEventListener('click', async () => {
      el.btnStartCamera.disabled = true;
      el.btnStartCamera.innerHTML = '<span class="ctrl-icon">▶</span> Starting...';
      if (el.activeCamStatusPill) {
        el.activeCamStatusPill.className = 'status-pill status-pill-connecting';
        el.activeCamStatusPill.textContent = '◌ Connecting...';
      }
      try {
        const res = await fetch('/api/camera/start', { method: 'POST' });
        const json = await res.json();
        if (res.ok && json.success) {
          if (el.btnStartCamera) el.btnStartCamera.style.display = 'none';
          if (el.btnStopCamera) el.btnStopCamera.style.display = 'inline-flex';
          if (el.cameraStream) {
            el.cameraStream.src = `/api/video/feed?view_mode=${viewMode}&t=${Date.now()}`;
          }
          updateCameraStatus();
        } else {
          alert('Failed to start camera: ' + (json.error ? json.error.message : 'Unknown error'));
        }
      } catch (err) {
        alert('Network error starting camera: ' + err.message);
      } finally {
        if (el.btnStartCamera) {
          el.btnStartCamera.disabled = false;
          el.btnStartCamera.innerHTML = '<span class="ctrl-icon">▶</span> START CAMERA';
        }
      }
    });
  }

  // Status updates
  async function updateCameraStatus() {
    try {
      const res = await fetch('/api/camera/status');
      if (res.ok) {
        const json = await res.json();
        if (json.success && json.data) {
          const d = json.data;
          const isOnline = Boolean(d.camera_online);
          const isWorkerRunning = Boolean(d.worker_running);
          const details = d.details || {};
          const source = (d.active_source || 'pc').toLowerCase();

          const labelMap = {
            'pc': 'PC Camera',
            'laptop': 'PC Camera',
            'droidcam': 'DroidCam',
            'extension': 'Extension Camera',
            'external': 'Extension Camera',
            'esp32': 'ESP32 Camera',
            'smart_board': 'Smart Board Camera'
          };
          const camDisplayName = labelMap[source] || source.toUpperCase();

          if (el.activeCamName) {
            el.activeCamName.textContent = camDisplayName;
          }
          if (el.activeCamRes) {
            el.activeCamRes.textContent = details.resolution || '--';
          }
          if (el.activeCamFps) {
            el.activeCamFps.textContent = isOnline ? `${Number(d.fps || details.measured_fps || 0).toFixed(1)} FPS` : '-- FPS';
          }

          // Toggle Stop / Start Camera buttons based on actual worker status
          if (el.btnStopCamera && el.btnStartCamera) {
            if (isWorkerRunning || isOnline) {
              el.btnStopCamera.style.display = 'inline-flex';
              el.btnStartCamera.style.display = 'none';
            } else {
              el.btnStopCamera.style.display = 'none';
              el.btnStartCamera.style.display = 'inline-flex';
            }
          }

          // Update Status Pill with distinct states: connected, connecting, stopped, disconnected
          if (el.activeCamStatusPill) {
            if (isOnline) {
              el.activeCamStatusPill.className = 'status-pill status-pill-connected';
              el.activeCamStatusPill.textContent = '● Connected';
              if (el.camTroubleshootBox) el.camTroubleshootBox.style.display = 'none';
            } else if (!isWorkerRunning) {
              el.activeCamStatusPill.className = 'status-pill status-pill-stopped';
              el.activeCamStatusPill.textContent = '⏹ Stopped';
              if (el.camTroubleshootBox) el.camTroubleshootBox.style.display = 'none';
            } else {
              el.activeCamStatusPill.className = 'status-pill status-pill-disconnected';
              el.activeCamStatusPill.textContent = '● Disconnected';
              if (source === 'droidcam' || source === 'esp32') {
                showTroubleshoot(source);
              }
            }
          }
        }
      }
    } catch (e) {
      console.debug('Camera status update failed:', e);
    }
  }

  // =========================================================================
  // Phase 10: New One-By-One Student Face Enrollment Controller
  // =========================================================================
  const enrollModal = document.getElementById('enrollment-modal');
  const btnOpenEnroll = document.getElementById('btn-open-enrollment');
  const btnEnrollClose = document.getElementById('btn-enroll-close');
  const btnEnrollCancel = document.getElementById('btn-enroll-cancel');

  const enrollStepForm = document.getElementById('enroll-step-form');
  const enrollStepSource = document.getElementById('enroll-step-source');
  const enrollStepConn = document.getElementById('enroll-step-conn');
  const enrollStepCamera = document.getElementById('enroll-step-camera');

  const enrollInputReg = document.getElementById('enroll-input-reg');
  const enrollInputName = document.getElementById('enroll-input-name');
  const enrollInputClass = document.getElementById('enroll-input-class');
  const enrollInputDept = document.getElementById('enroll-input-dept');
  const enrollInputSec = document.getElementById('enroll-input-sec');

  const btnEnrollToStep2 = document.getElementById('btn-enroll-to-step2');
  const btnEnrollCapture = document.getElementById('btn-enroll-capture');
  const btnEnrollRetake = document.getElementById('btn-enroll-retake');
  const btnEnrollSave = document.getElementById('btn-enroll-save');

  const enrollCurrentName = document.getElementById('enroll-current-name');
  const enrollCurrentReg = document.getElementById('enroll-current-reg');
  const enrollCurrentDept = document.getElementById('enroll-current-dept');
  const enrollCurrentSource = document.getElementById('enroll-current-source');

  const enrollQualityHud = document.getElementById('enroll-quality-hud');
  const enrollHudText = document.getElementById('enroll-hud-text');
  const enrollSamplesCount = document.getElementById('enroll-samples-count');

  const duplicateRegModal = document.getElementById('duplicate-reg-modal');
  const dupRegNumber = document.getElementById('dup-reg-number');
  const dupStudentDetails = document.getElementById('dup-student-details');
  const dupStudentName = document.getElementById('dup-student-name');
  const dupStudentClass = document.getElementById('dup-student-class');
  const dupStudentDept = document.getElementById('dup-student-dept');
  const dupStudentSec = document.getElementById('dup-student-sec');
  const dupStudentDate = document.getElementById('dup-student-date');
  const btnDupCancel = document.getElementById('btn-dup-cancel');
  const btnDupView = document.getElementById('btn-dup-view');

  let currentEnrollStep = 1;
  let currentEnrollStudent = null;
  let selectedEnrollSource = 'pc';
  let capturedSamples = []; // Array of base64 frame strings
  let capturedThumbnails = []; // Array of crop base64 strings
  let hudPollingInterval = null;
  let latestValidationResult = null;

  function openEnrollmentModal() {
    if (enrollModal) {
      resetEnrollmentForm();
      enrollModal.style.display = 'flex';
      setIdxEnrollStep(1);
      if (enrollInputReg) enrollInputReg.focus();
    }
  }

  function closeEnrollmentModal() {
    stopQualityHudPolling();
    if (enrollModal) enrollModal.style.display = 'none';
    if (duplicateRegModal) duplicateRegModal.style.display = 'none';
  }

  function setIdxEnrollStep(step) {
    currentEnrollStep = step;
    const steps = [
      { num: 1, pane: enrollStepForm, nav: document.getElementById('idx-wiz-step-1') },
      { num: 2, pane: enrollStepSource, nav: document.getElementById('idx-wiz-step-2') },
      { num: 3, pane: enrollStepConn, nav: document.getElementById('idx-wiz-step-3') },
      { num: 4, pane: enrollStepCamera, nav: document.getElementById('idx-wiz-step-4') }
    ];

    steps.forEach(s => {
      if (s.pane) s.pane.style.display = (s.num === step) ? 'block' : 'none';
      if (s.nav) {
        if (s.num === step) {
          s.nav.style.color = '#2563eb';
          s.nav.style.fontWeight = '700';
        } else if (s.num < step) {
          s.nav.style.color = '#16a34a';
          s.nav.style.fontWeight = '600';
        } else {
          s.nav.style.color = '#64748b';
          s.nav.style.fontWeight = '500';
        }
      }
    });

    if (step === 4) {
      startQualityHudPolling();
      renderSampleSlots();
      const feed = document.getElementById('enroll-camera-feed');
      if (feed) feed.src = `/api/video/feed?view_mode=normal&t=${Date.now()}`;
    } else {
      stopQualityHudPolling();
    }
  }

  function selectEnrollCamera(src) {
    selectedEnrollSource = src;
    ['pc', 'droidcam', 'extension', 'esp32'].forEach(s => {
      const card = document.getElementById(`idx-cam-${s}`);
      const box = document.getElementById(`idx-conn-box-${s}`);
      if (card) {
        if (s === src) card.classList.add('active');
        else card.classList.remove('active');
      }
      if (box) box.style.display = (s === src) ? 'block' : 'none';
    });

    const labelMap = { pc: 'PC Camera', droidcam: 'DroidCam', extension: 'Extension Camera', esp32: 'ESP32 Wi-Fi Camera' };
    if (enrollCurrentSource) enrollCurrentSource.textContent = labelMap[src] || src.toUpperCase();
  }

  function resetEnrollmentForm() {
    stopQualityHudPolling();
    currentEnrollStudent = null;
    capturedSamples = [];
    capturedThumbnails = [];
    latestValidationResult = null;
    selectedEnrollSource = 'pc';

    if (enrollInputReg) enrollInputReg.value = '';
    if (enrollInputName) enrollInputName.value = '';
    if (enrollInputClass) enrollInputClass.value = '3rd Year';
    if (enrollInputDept) enrollInputDept.value = 'AI&DS';
    if (enrollInputSec) enrollInputSec.value = 'B';

    selectEnrollCamera('pc');
    renderSampleSlots();
  }

  function renderSampleSlots() {
    const grid = document.getElementById('idx-samples-grid');
    const count = capturedSamples.length;
    if (enrollSamplesCount) {
      enrollSamplesCount.textContent = `${count} of 10 captured (min 5 required)`;
    }

    if (grid) {
      grid.innerHTML = '';
      for (let slot = 1; slot <= 10; slot++) {
        const slotEl = document.createElement('div');
        slotEl.style.cssText = 'aspect-ratio: 1; background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; overflow: hidden; font-size: 0.75rem; color: #94a3b8; font-weight: 600;';

        if (slot <= count) {
          slotEl.style.border = '2px solid #16a34a';
          slotEl.style.borderStyle = 'solid';
          const src = capturedThumbnails[slot - 1] || capturedSamples[slot - 1];
          const fullSrc = src.startsWith('data:') ? src : `data:image/jpeg;base64,${src}`;
          slotEl.innerHTML = `<img src="${fullSrc}" alt="Sample ${slot}" style="width:100%; height:100%; object-fit:cover;"><span style="position:absolute; bottom:2px; right:2px; background:rgba(0,0,0,0.6); color:#fff; font-size:0.65rem; padding:1px 4px; border-radius:3px;">#${slot}</span>`;
        } else {
          slotEl.textContent = `#${slot}`;
        }
        grid.appendChild(slotEl);
      }
    }

    // Update Progress checklist
    const chkSamples = document.getElementById('chk-samples-status');
    const chkSave = document.getElementById('chk-save-status');
    const hasEnough = (count >= 5);

    if (chkSamples) {
      chkSamples.textContent = hasEnough ? `✓ Face Samples (${count} / 10 captured)` : `○ Face Samples (${count} / 10, min 5 required)`;
      chkSamples.style.color = hasEnough ? '#16a34a' : '#64748b';
    }
    if (chkSave) {
      chkSave.textContent = hasEnough ? '✓ Ready to Save Enrollment' : '○ Save Enrollment';
      chkSave.style.color = hasEnough ? '#16a34a' : '#64748b';
    }

    if (btnEnrollRetake) btnEnrollRetake.disabled = (count === 0);
    if (btnEnrollSave) btnEnrollSave.disabled = !hasEnough;
  }

  async function handleProceedToSource() {
    const regNo = (enrollInputReg?.value || '').trim();
    const name = (enrollInputName?.value || '').trim();
    const cls = (enrollInputClass?.value || '').trim();
    const dept = (enrollInputDept?.value || '').trim();
    const sec = (enrollInputSec?.value || '').trim();

    if (!regNo || !name || !cls || !dept || !sec) {
      alert('Please fill in all 5 student details:\n- Register Number\n- Student Name\n- Class\n- Department\n- Section');
      return;
    }

    // Check duplicate register number in new enrollment database
    try {
      const res = await fetch(`/api/enrollment/one-by-one/check/${encodeURIComponent(regNo)}`);
      const json = await res.json();
      if (json.success && json.data.exists) {
        showDuplicateModal(regNo, json.data.student);
        return;
      }
    } catch (e) {
      console.warn('Duplicate check check failed:', e);
    }

    // Set current student info
    currentEnrollStudent = {
      register_number: regNo,
      name: name,
      class_name: cls,
      department: dept,
      section: sec
    };

    if (enrollCurrentName) enrollCurrentName.textContent = name;
    if (enrollCurrentReg) enrollCurrentReg.textContent = regNo;
    if (enrollCurrentDept) enrollCurrentDept.textContent = `${cls} · ${dept} · ${sec}`;

    // Switch to Step 2
    setIdxEnrollStep(2);
  }

  function showDuplicateModal(regNo, studentData) {
    if (dupRegNumber) dupRegNumber.textContent = regNo;
    if (dupStudentDetails) {
      if (studentData) {
        dupStudentDetails.style.display = 'flex';
        if (dupStudentName) dupStudentName.textContent = studentData.name || '--';
        if (dupStudentClass) dupStudentClass.textContent = studentData.class || '--';
        if (dupStudentDept) dupStudentDept.textContent = studentData.department || '--';
        if (dupStudentSec) dupStudentSec.textContent = studentData.section || '--';
        if (dupStudentDate) dupStudentDate.textContent = studentData.created_at || '--';
      } else {
        dupStudentDetails.style.display = 'none';
      }
    }

    if (duplicateRegModal) duplicateRegModal.style.display = 'flex';
  }

  function startQualityHudPolling() {
    stopQualityHudPolling();
    pollFaceQuality();
    hudPollingInterval = setInterval(pollFaceQuality, 600);
  }

  function stopQualityHudPolling() {
    if (hudPollingInterval) {
      clearInterval(hudPollingInterval);
      hudPollingInterval = null;
    }
  }

  async function pollFaceQuality() {
    try {
      const res = await fetch('/api/enrollment/one-by-one/validate-sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const json = await res.json();
      if (json.success && json.data) {
        latestValidationResult = json.data;
        updateQualityHud(json.data);
      }
    } catch (e) {}
  }

  function updateQualityHud(data) {
    if (!enrollQualityHud || !enrollHudText) return;

    enrollQualityHud.className = 'enroll-quality-hud';

    const numFaces = data.num_faces || 0;
    const canCapture = Boolean(data.can_capture);
    const status = data.status || '';

    if (numFaces === 0) {
      enrollQualityHud.classList.add('hud-noface');
      enrollHudText.textContent = 'No face detected. Please position your face clearly.';
    } else if (numFaces > 1) {
      enrollQualityHud.classList.add('hud-multiface');
      enrollHudText.textContent = 'Multiple faces detected. Only one person can be enrolled at a time.';
    } else {
      if (status === 'PASS') {
        enrollQualityHud.classList.add('hud-good');
        enrollHudText.textContent = 'GOOD QUALITY ✓';
      } else {
        enrollQualityHud.classList.add('hud-low');
        enrollHudText.textContent = data.message || 'LOW QUALITY — RETAKE';
      }
    }

    if (btnEnrollCapture) {
      btnEnrollCapture.disabled = !canCapture || capturedSamples.length >= 10;
    }
  }

  async function handleCaptureSample() {
    if (capturedSamples.length >= 10) {
      alert('Maximum 10 face samples already captured.');
      return;
    }

    if (btnEnrollCapture) btnEnrollCapture.disabled = true;

    try {
      const res = await fetch('/api/enrollment/one-by-one/capture-current', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const json = await res.json();

      if (!json.success || !json.data) {
        alert('Failed to capture frame from camera.');
        return;
      }

      const d = json.data;
      if (!d.can_capture) {
        alert(`Cannot capture frame:\n${d.message || d.rejection_reason || 'Face quality condition not met.'}`);
        return;
      }

      // Flash animation
      const flash = document.getElementById('enroll-capture-flash');
      if (flash) {
        flash.style.display = 'block';
        setTimeout(() => { flash.style.display = 'none'; }, 150);
      }

      const sampleFrame = d.frame_base64;
      const sampleThumb = d.crop_base64 || d.frame_base64;

      if (!sampleFrame) {
        alert('Unable to extract frame data from capture.');
        return;
      }

      capturedSamples.push(sampleFrame);
      capturedThumbnails.push(sampleThumb);
      renderSampleSlots();

    } catch (err) {
      alert(`Capture error: ${err.message}`);
    } finally {
      if (btnEnrollCapture) btnEnrollCapture.disabled = false;
    }
  }

  function handleRetakeSample() {
    if (capturedSamples.length === 0) return;
    capturedSamples.pop();
    capturedThumbnails.pop();
    renderSampleSlots();
  }

  async function handleSaveEnrollment() {
    if (!currentEnrollStudent) {
      alert('No student details available to save.');
      return;
    }

    if (capturedSamples.length < 5) {
      alert(`Please capture at least 5 face samples before saving (currently: ${capturedSamples.length}).`);
      return;
    }

    if (btnEnrollSave) {
      btnEnrollSave.disabled = true;
      btnEnrollSave.textContent = 'Saving...';
    }

    const payload = {
      register_number: currentEnrollStudent.register_number,
      name: currentEnrollStudent.name,
      class_name: currentEnrollStudent.class_name,
      department: currentEnrollStudent.department,
      section: currentEnrollStudent.section,
      samples: capturedSamples
    };

    try {
      const res = await fetch('/api/enrollment/one-by-one/enroll', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const json = await res.json();

      if (!res.ok || !json.success) {
        const msg = json.error ? json.error.message : (json.detail || 'Enrollment save failed');
        alert(`Enrollment Error:\n${msg}`);
        return;
      }

      const studentName = currentEnrollStudent.name;
      alert(`Student "${studentName}" enrolled successfully!\n\nMoving to next student.`);

      // Reset for next student
      resetEnrollmentForm();
      closeEnrollmentModal();

      // Refresh attendance roster if available
      if (typeof fetchRosterData === 'function') {
        fetchRosterData();
      }

    } catch (err) {
      alert(`Network error saving enrollment: ${err.message}`);
    } finally {
      if (btnEnrollSave) {
        btnEnrollSave.disabled = capturedSamples.length < 5;
        btnEnrollSave.innerHTML = '<i class="ti ti-device-floppy"></i> SAVE ENROLLMENT';
      }
    }
  }

  // Event Listeners for Phase 10 / Multi-Source
  if (btnOpenEnroll) btnOpenEnroll.addEventListener('click', openEnrollmentModal);
  if (btnEnrollClose) btnEnrollClose.addEventListener('click', closeEnrollmentModal);
  if (btnEnrollCancel) btnEnrollCancel.addEventListener('click', closeEnrollmentModal);
  if (btnEnrollToStep2) btnEnrollToStep2.addEventListener('click', handleProceedToSource);

  // Step 2 & 3 navigation
  const btnIdxBack1 = document.getElementById('btn-idx-back-1');
  const btnIdxToStep3 = document.getElementById('btn-idx-to-step3');
  const btnIdxBack2 = document.getElementById('btn-idx-back-2');
  const btnIdxToStep4 = document.getElementById('btn-idx-to-step4');

  if (btnIdxBack1) btnIdxBack1.addEventListener('click', () => setIdxEnrollStep(1));
  if (btnIdxToStep3) btnIdxToStep3.addEventListener('click', () => setIdxEnrollStep(3));
  if (btnIdxBack2) btnIdxBack2.addEventListener('click', () => setIdxEnrollStep(2));
  if (btnIdxToStep4) btnIdxToStep4.addEventListener('click', () => setIdxEnrollStep(4));

  // Source selection cards
  ['pc', 'droidcam', 'extension', 'esp32'].forEach(src => {
    const el = document.getElementById(`idx-cam-${src}`);
    if (el) el.addEventListener('click', () => selectEnrollCamera(src));
  });

  // Step 3 Test, Connect, Stop buttons
  const btnIdxTest = document.getElementById('btn-idx-test-cam');
  const btnIdxConnect = document.getElementById('btn-idx-connect-cam');
  const btnIdxStop = document.getElementById('btn-idx-stop-cam');
  const idxPill = document.getElementById('idx-conn-pill');

  if (btnIdxTest) {
    btnIdxTest.addEventListener('click', async () => {
      btnIdxTest.disabled = true;
      btnIdxTest.textContent = 'TESTING...';
      let payload = { source: selectedEnrollSource };
      if (selectedEnrollSource === 'droidcam') {
        payload.host = document.getElementById('idx-dc-host').value.trim();
        payload.port = parseInt(document.getElementById('idx-dc-port').value.trim()) || 4747;
        payload.video_path = '/video';
      } else if (selectedEnrollSource === 'extension') {
        payload.index = parseInt(document.getElementById('idx-ext-index').value) || 0;
      } else if (selectedEnrollSource === 'esp32') {
        payload.stream_url = document.getElementById('idx-esp-url').value.trim();
      }

      try {
        const res = await fetch('/api/camera/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const json = await res.json();
        if (json.success && json.data && json.data.connected) {
          alert(`● CONNECTED: ${json.data.message}`);
          if (idxPill) { idxPill.className = 'status-pill status-pill-connected'; idxPill.textContent = '● CONNECTED'; }
        } else {
          alert(`● NOT CONNECTED: ${json.data?.message || 'Camera unreachable'}`);
          if (idxPill) { idxPill.className = 'status-pill status-pill-disconnected'; idxPill.textContent = '● DISCONNECTED'; }
        }
      } catch (e) {
        alert(`Test error: ${e.message}`);
      } finally {
        btnIdxTest.disabled = false;
        btnIdxTest.textContent = 'TEST CONNECTION';
      }
    });
  }

  if (btnIdxConnect) {
    btnIdxConnect.addEventListener('click', async () => {
      btnIdxConnect.disabled = true;
      btnIdxConnect.textContent = 'CONNECTING...';
      let payload = { source: selectedEnrollSource };
      if (selectedEnrollSource === 'droidcam') {
        payload.host = document.getElementById('idx-dc-host').value.trim();
        payload.port = parseInt(document.getElementById('idx-dc-port').value.trim()) || 4747;
        payload.video_path = '/video';
      } else if (selectedEnrollSource === 'extension') {
        payload.index = parseInt(document.getElementById('idx-ext-index').value) || 0;
      } else if (selectedEnrollSource === 'esp32') {
        payload.stream_url = document.getElementById('idx-esp-url').value.trim();
      }

      try {
        const res = await fetch('/api/camera/select', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const json = await res.json();
        if (res.ok && json.success) {
          if (idxPill) { idxPill.className = 'status-pill status-pill-connected'; idxPill.textContent = '● CONNECTED'; }
          alert(`Camera connected: ${selectedEnrollSource.toUpperCase()}`);
        } else {
          alert('Failed to connect: ' + (json.error ? json.error.message : 'Unknown error'));
        }
      } catch (e) {
        alert(`Connect error: ${e.message}`);
      } finally {
        btnIdxConnect.disabled = false;
        btnIdxConnect.textContent = 'CONNECT CAMERA';
      }
    });
  }

  if (btnIdxStop) {
    btnIdxStop.addEventListener('click', async () => {
      try {
        await fetch('/api/camera/stop', { method: 'POST' });
        if (idxPill) { idxPill.className = 'status-pill status-pill-stopped'; idxPill.textContent = '⏹ STOPPED'; }
      } catch (e) {
        console.warn('Error stopping camera:', e);
      }
    });
  }

  // Wizard nav step clicks
  [1, 2, 3, 4].forEach(s => {
    const el = document.getElementById(`idx-wiz-step-${s}`);
    if (el) el.addEventListener('click', () => {
      if (s === 1 || currentEnrollStudent) setIdxEnrollStep(s);
    });
  });

  if (btnEnrollCapture) btnEnrollCapture.addEventListener('click', handleCaptureSample);
  if (btnEnrollRetake) btnEnrollRetake.addEventListener('click', handleRetakeSample);
  if (btnEnrollSave) btnEnrollSave.addEventListener('click', handleSaveEnrollment);

  if (btnDupCancel) {
    btnDupCancel.addEventListener('click', () => {
      if (duplicateRegModal) duplicateRegModal.style.display = 'none';
    });
  }

  if (btnDupView) {
    btnDupView.addEventListener('click', () => {
      if (dupStudentDetails) {
        dupStudentDetails.style.display = dupStudentDetails.style.display === 'none' ? 'flex' : 'none';
      }
    });
  }

  // Camera initialization and periodic status polling
  loadCameraSources();
  updateCameraStatus();
  setInterval(updateCameraStatus, 3000);

  // Start Real-Time Connection
  initSSE();

})();

