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
    cameraSourceSelect: document.getElementById('camera-source-select'),
    droidcamIpGroup: document.getElementById('droidcam-ip-group'),
    inputDroidcamHost: document.getElementById('input-droidcam-host'),
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
    if (el.kpiFpsText && data.fps !== undefined) {
      el.kpiFpsText.textContent = `Camera: ${Number(data.fps).toFixed(1)} FPS`;
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
        list.push({
          student_id: r.student_id,
          student_name: r.student_name || r.student_id,
          status: r.status,
          first_seen: formatTimeOnly(r.first_seen),
          last_seen: formatTimeOnly(r.last_seen)
        });
      }
    }

    // 2. Not seen records
    if (report.not_seen_students) {
      for (const id of report.not_seen_students) {
        list.push({
          student_id: id,
          student_name: id,
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
  // 11. Camera Source & DroidCam Management
  // =========================================================================
  async function loadCameraSources() {
    try {
      const res = await fetch('/api/camera/sources');
      if (res.ok) {
        const json = await res.json();
        if (json.success && json.data) {
          const d = json.data;
          if (el.cameraSourceSelect) {
            el.cameraSourceSelect.value = d.active_source;
            toggleDroidCamInputs(d.active_source);
          }
          if (el.inputDroidcamHost && d.droidcam_config && d.droidcam_config.host) {
            el.inputDroidcamHost.value = d.droidcam_config.host;
            if (el.linkDroidcamRemote) {
              el.linkDroidcamRemote.href = `http://${d.droidcam_config.host}:4747`;
            }
          }

        }
      }
    } catch (e) {
      console.warn('Error loading camera sources:', e);
    }
  }

  function toggleDroidCamInputs(source) {
    if (el.droidcamIpGroup) {
      el.droidcamIpGroup.style.display = (source === 'droidcam') ? 'flex' : 'none';
    }
    const labelMap = {
      'droidcam': 'DroidCam Wi-Fi',
      'laptop': 'Laptop Camera',
      'smart_board': 'Smart Board Camera',
      'external': 'External Camera'
    };
    if (el.camActiveBadge) {
      el.camActiveBadge.textContent = `Camera: ${labelMap[source] || source}`;
    }
  }

  async function updateCameraStatus() {
    try {
      const res = await fetch('/api/camera/status');
      if (res.ok) {
        const json = await res.json();
        if (json.success && json.data) {
          const d = json.data;
          const isOnline = d.camera_online;
          const details = d.details || {};
          const source = d.active_source || 'unknown';

          const labelMap = {
            'droidcam': 'DroidCam Wi-Fi',
            'laptop': 'Laptop Camera',
            'smart_board': 'Smart Board Camera',
            'external': 'External Camera'
          };
          if (el.camActiveBadge) {
            el.camActiveBadge.textContent = `Camera: ${labelMap[source] || source}`;
          }
          if (el.camResBadge) {
            el.camResBadge.textContent = `Res: ${details.resolution || '--'}`;
          }
          if (el.camFpsBadge) {
            el.camFpsBadge.textContent = `FPS: ${d.fps || details.measured_fps || 0}`;
          }
          if (el.camStatusPill) {
            if (isOnline) {
              el.camStatusPill.className = 'cam-status-pill status-connected';
              el.camStatusPill.textContent = 'CONNECTED';
            } else {
              el.camStatusPill.className = 'cam-status-pill status-disconnected';
              el.camStatusPill.textContent = source === 'droidcam' ? 'DROIDCAM UNAVAILABLE' : 'DISCONNECTED';
            }
          }
        }
      }
    } catch (e) {
      console.debug('Camera status update failed:', e);
    }
  }

  async function applyCameraSource() {
    if (!el.cameraSourceSelect) return;
    const source = el.cameraSourceSelect.value;
    const host = el.inputDroidcamHost ? el.inputDroidcamHost.value.trim() : '10.140.159.218';

    if (el.btnApplyCamera) {
      el.btnApplyCamera.disabled = true;
      el.btnApplyCamera.textContent = 'Connecting...';
    }

    try {
      const res = await fetch('/api/camera/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: source, host: host, port: 4747 })
      });
      const json = await res.json();
      if (res.ok && json.success) {
        updateCameraStatus();
        if (el.cameraStream) {
          el.cameraStream.src = `/api/video/feed?view_mode=${viewMode}&t=${Date.now()}`;
        }
      } else {
        alert(json.error ? json.error.message : 'Failed to switch camera source');
      }
    } catch (err) {
      alert(`Network error switching camera: ${err.message}`);
    } finally {
      if (el.btnApplyCamera) {
        el.btnApplyCamera.disabled = false;
        el.btnApplyCamera.textContent = 'Connect';
      }
    }
  }

  async function testCameraConnection() {
    const host = el.inputDroidcamHost ? el.inputDroidcamHost.value.trim() : '10.140.159.218';
    if (el.btnTestCamera) {
      el.btnTestCamera.disabled = true;
      el.btnTestCamera.textContent = 'Testing...';
    }

    try {
      const res = await fetch('/api/camera/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: 'droidcam', host: host, port: 4747 })
      });
      const json = await res.json();
      if (res.ok && json.success) {
        const d = json.data;
        if (d.connected) {
          alert(`CONNECTED: ${d.message}`);
        } else {
          alert(`NOT CONNECTED: ${d.message}\n\nTroubleshooting:\n` + (d.reasons || []).join('\n'));
        }
      } else {
        alert('Test failed.');
      }
    } catch (err) {
      alert(`Test error: ${err.message}`);
    } finally {
      if (el.btnTestCamera) {
        el.btnTestCamera.disabled = false;
        el.btnTestCamera.textContent = 'Test Camera';
      }
    }
  }

  if (el.cameraSourceSelect) {
    el.cameraSourceSelect.addEventListener('change', (e) => {
      toggleDroidCamInputs(e.target.value);
    });
  }

  if (el.inputDroidcamHost) {
    el.inputDroidcamHost.addEventListener('input', (e) => {
      const host = e.target.value.trim() || '10.140.159.218';
      if (el.linkDroidcamRemote) {
        el.linkDroidcamRemote.href = `http://${host}:4747`;
      }
    });
  }

  if (el.btnApplyCamera) {
    el.btnApplyCamera.addEventListener('click', applyCameraSource);
  }

  if (el.btnTestCamera) {
    el.btnTestCamera.addEventListener('click', testCameraConnection);
  }


  // Camera initialization and periodic status polling
  loadCameraSources();
  updateCameraStatus();
  setInterval(updateCameraStatus, 3000);

  // Start Real-Time Connection
  initSSE();

})();
