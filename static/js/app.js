    const fileInput = document.getElementById('file');
    const dropzone = document.getElementById('dropzone');
    const dzText = document.getElementById('dz-text');
    const extractionPanel = document.getElementById('extraction-panel');
    const extractionBody = document.getElementById('extraction-body');
    const reportPanel = document.getElementById('report-panel');
    const reportBody = document.getElementById('report-body');
    const continueExtractBtn = document.getElementById('continue-extract-btn');
    const verifyBtn = document.getElementById('verify-btn');
    const cancelVerifyBtn = document.getElementById('cancel-verify-btn');
    // These are defaults; /config endpoint may override them based on ENABLE_CLIENT_MODELS flag.
    // All Anthropic usage runs on the server-side ANTHROPIC_API_KEY env var.
    let EXTRACTOR_MODEL = 'claude-sonnet-4-6';
    let ANALYZER_MODEL  = 'claude-sonnet-4-6';

    let currentContextId = null;
    let currentFailedPages = [];  // 1-indexed slides the local vision model couldn't process
    let savedCompanyHtml = '';   // snapshot of extractionBody after basic extraction (no side-effects re-render)
    let currentFilename = '';
    let lastReport = null;  // accumulated report for download

    // ── Session helpers ─────────────────────────────────────────────────────

    /** Return (or create) a stable UUID for this browser session. */
    function getSessionId() {
      let sid = localStorage.getItem('compliance_session_id');
      if (!sid) {
        sid = crypto.randomUUID();
        localStorage.setItem('compliance_session_id', sid);
      }
      return sid;
    }

    /** Headers to attach to every fetch() call. The Anthropic API key is
     *  server-side only — never passed through the browser. */
    function apiHeaders() {
      return { 'X-Session-ID': getSessionId() };
    }

    // ── Toast / snackbar helper ─────────────────────────────────────────────

    /**
     * Show a dismissable error snackbar at the bottom of the page.
     * @param {string} message - text to show
     * @param {Object} [opts] - {duration?: number, variant?: string}
     */
    function showToast(message, { duration = 6000, variant = 'error' } = {}) {
      const container = document.getElementById('toast-container');
      if (!container) return;
      const toast = document.createElement('div');
      toast.className = `toast toast-${variant}`;
      toast.innerHTML = `
        <span class="material-symbols-rounded">error</span>
        <span class="toast-msg"></span>
        <button class="close-btn" aria-label="Dismiss">
          <span class="material-symbols-rounded">close</span>
        </button>
      `;
      toast.querySelector('.toast-msg').textContent = message;
      const remove = () => toast.remove();
      toast.querySelector('.close-btn').onclick = remove;
      container.appendChild(toast);
      if (duration > 0) setTimeout(remove, duration);
    }

    // ── Google Slides OAuth2 integration ──────────────────────────────────

    /**
     * Start Google Slides OAuth2 flow.
     * User is redirected to Google login, then back to /auth/google/slides-callback
     */
    function startGoogleSlidesAuth() {
      window.location.href = '/auth/google/slides-auth';
    }

    /**
     * Handle OAuth2 callback: check for token in URL and save to localStorage
     */
    function handleGoogleSlidesCallback() {
      const params = new URLSearchParams(window.location.search);
      const token = params.get('google_slides_token');
      if (token) {
        localStorage.setItem('google_slides_token', token);
        // Clean URL
        window.history.replaceState({}, document.title, window.location.pathname);
        // Show modal to paste presentation ID
        showGoogleSlidesModal();
      }
    }

    function showGoogleSlidesModal() {
      // Create and show a simple modal for presentation ID input
      const modal = document.createElement('div');
      modal.innerHTML = `
        <div style="position: fixed; inset: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 9999;">
          <div style="background: white; padding: 24px; border-radius: 8px; max-width: 500px; box-shadow: 0 8px 32px rgba(0,0,0,0.15);">
            <h3 style="margin: 0 0 12px; font-size: 18px;">Google Slides Authentication Successful</h3>
            <p style="margin: 0 0 16px; color: var(--md-on-surface-variant); font-size: 14px;">
              Now, paste your Google Slides presentation ID from the URL:
              <br/><code style="background: var(--md-surface-container-lowest); padding: 4px 8px; border-radius: 4px;">docs.google.com/presentation/d/<strong>{ID}</strong>/edit</code>
            </p>
            <input type="text" id="presentation-id" placeholder="Paste ID here..."
              style="width: 100%; padding: 12px; border: 1px solid var(--md-outline-variant); border-radius: 4px; margin-bottom: 16px; box-sizing: border-box;" />
            <div style="display: flex; gap: 12px; justify-content: flex-end;">
              <button onclick="this.closest('div').parentElement.parentElement.remove()" style="padding: 8px 16px; border: 1px solid var(--md-outline-variant); background: transparent; border-radius: 4px; cursor: pointer;">Cancel</button>
              <button onclick="uploadGoogleSlides()" style="padding: 8px 16px; background: var(--md-primary); color: white; border: none; border-radius: 4px; cursor: pointer;">Analyze</button>
            </div>
          </div>
        </div>
      `;
      document.body.appendChild(modal);
      document.getElementById('presentation-id').focus();
    }

    async function uploadGoogleSlides() {
      const presentationId = document.getElementById('presentation-id')?.value;
      const token = localStorage.getItem('google_slides_token');

      if (!presentationId || !token) {
        showToast('Missing presentation ID or authentication token');
        return;
      }

      dzText.innerHTML = progressHTML('Downloading Google Slides presentation…');

      try {
        const fd = new FormData();
        fd.append('presentation_id', presentationId);
        fd.append('access_token', token);
        fd.append('extractor_model', EXTRACTOR_MODEL);

        const r = await fetch('/extract-from-google-slides', {
          method: 'POST',
          body: fd,
          headers: apiHeaders()
        });

        if (!r.ok) {
          const error = await r.json();
          throw new Error(error.detail || 'Extraction failed');
        }

        const data = await r.json();
        currentContextId = data.context_id;
        renderBasicExtraction(data.extraction);
        restoreDropzone(null);
        document.querySelector('[style*="position: fixed"]')?.parentElement?.remove();
      } catch (e) {
        showToast(e.message || 'Failed to extract Google Slides');
        restoreDropzone(null);
      }
    }

    /** Apply UI changes based on client_models feature flag. */
    function applyClientModelsUX(enableClientModels) {
      const cardIds = [
        'model-selection-card',    // Hide model dropdown (if it exists)
        'api-key-card',            // Hide API key input (if it exists)
      ];

      if (enableClientModels) {
        // Hide model/API key cards
        cardIds.forEach(id => {
          const card = document.getElementById(id);
          if (card) card.classList.add('hidden');
        });
      } else {
        // Keep cards visible (current behavior)
        cardIds.forEach(id => {
          const card = document.getElementById(id);
          if (card) card.classList.remove('hidden');
        });
      }
    }

    /** Load and display regulatory update notifications. */
    async function loadNotifications() {
      try {
        const resp = await fetch('/notifications', { credentials: 'same-origin' });
        if (!resp.ok) return;
        const notes = await resp.json();
        if (!notes || notes.length === 0) return;

        const banner = document.getElementById('notification-banner');
        const chips = notes.map(n => `
          <div class="notification-chip" data-id="${n.id}" style="
            display: flex; align-items: center; gap: 12px;
            background: var(--md-secondary-container);
            border: 1px solid var(--md-secondary);
            border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
            font-size: 0.875rem; color: var(--md-on-secondary-container);
          ">
            <span class="material-symbols-rounded" style="font-size: 20px;">update</span>
            <div style="flex: 1;">
              <div style="font-weight: 500;">${escapeHtml(n.title)}</div>
              <div style="font-size: 0.8rem; color: var(--md-on-secondary-container); opacity: 0.8;">${escapeHtml(n.body)}</div>
            </div>
            <button class="btn btn-text" onclick="dismissNote(${n.id})" style="white-space: nowrap;">
              Dismiss
            </button>
          </div>
        `).join('');
        banner.innerHTML = chips;
        banner.style.display = 'block';
      } catch (e) {
        console.warn('[notifications] Error loading:', e);
      }
    }

    /** Dismiss a notification by ID. */
    async function dismissNote(id) {
      try {
        const resp = await fetch(`/notifications/${id}/dismiss`, {
          method: 'POST',
          credentials: 'same-origin',
        });
        if (!resp.ok) {
          console.error('[notifications] Dismiss failed:', resp.statusText);
          return;
        }
        const chip = document.querySelector(`.notification-chip[data-id="${id}"]`);
        if (chip) chip.remove();
      } catch (e) {
        console.error('[notifications] Error dismissing:', e);
      }
    }

    /** Escape HTML to prevent XSS. */
    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    // Fetch current user on page load
    document.addEventListener('DOMContentLoaded', () => {
      fetchCurrentUser();
      loadNotifications();
      handleGoogleSlidesCallback();

      // Fetch feature flags and config from server
      fetch('/config')
        .then(r => r.json())
        .then(config => {
          // Store config globally for use throughout app
          window.appConfig = config;

          // Update hardcoded models if server specifies them
          if (config.extractor_model) {
            EXTRACTOR_MODEL = config.extractor_model;
          }
          if (config.analyzer_model) {
            ANALYZER_MODEL = config.analyzer_model;
          }

          // Toggle UI visibility based on feature flag
          applyClientModelsUX(config.enable_client_models);
        })
        .catch(e => console.warn('Failed to load config:', e));
    });

    /** Populate the app-bar user widget if the user is logged in via Google SSO. */
    async function fetchCurrentUser() {
      try {
        const resp = await fetch('/auth/me', { headers: apiHeaders() });
        if (!resp.ok) return;
        const user = await resp.json();
        if (!user || !user.email) return;

        const widget = document.getElementById('user-widget');
        document.getElementById('user-name').textContent  = user.name  || user.email;
        document.getElementById('user-email').textContent = user.email || '';

        const avatar = document.getElementById('user-avatar');
        if (user.picture) {
          avatar.src = user.picture;
          avatar.style.display = 'block';
        } else {
          avatar.style.display = 'none';
        }
        widget.style.display = 'flex';
      } catch (_) { /* SSO not enabled or network error — silently skip */ }
    }

    // ── Navigation ─────────────────────────────────────────────────────────
    function showDashboard() {
      document.getElementById('dashboard-view').classList.remove('hidden');
      document.getElementById('analysis-view').classList.add('hidden');
      document.getElementById('drilldown-view').classList.add('hidden');
      loadSavedReports();
      loadSavedExtractionsList();
    }
    function showAnalysisView(reset) {
      document.getElementById('dashboard-view').classList.add('hidden');
      document.getElementById('analysis-view').classList.remove('hidden');
      document.getElementById('drilldown-view').classList.add('hidden');
      if (reset) resetAnalysisState();
    }
    function resetAnalysisState() {
      // Clear extraction
      currentContextId = null;
      const ep = document.getElementById('extraction-panel');
      if (ep) { ep.classList.add('hidden'); document.getElementById('extraction-body').innerHTML = ''; }
      // Clear report
      const rp = document.getElementById('report-panel');
      if (rp) { rp.classList.add('hidden'); rp.innerHTML = ''; }
      // Reset upload dropzone
      const fileInput = document.getElementById('file');
      if (fileInput) fileInput.value = '';
      const dzText = document.getElementById('dz-text');
      if (dzText) dzText.innerHTML = `<span class="material-symbols-rounded icon">upload_file</span><div><strong>Click or drop a file here</strong></div><div class="hint">PDF, PowerPoint, Word, or Excel • Null fields indicate the deck did not state that value.</div>`;
      const dropzone = document.getElementById('dropzone');
      if (dropzone) dropzone.classList.remove('uploading');
    }
    function showDrilldownView() {
      document.getElementById('dashboard-view').classList.add('hidden');
      document.getElementById('analysis-view').classList.add('hidden');
      document.getElementById('drilldown-view').classList.remove('hidden');
      document.getElementById('drilldown-view').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ── Saved Reports Dashboard ───────────────────────────────────────────
    function getMetricBadgeClass(pctStr) {
      if (pctStr === "N/A") return "";
      const pct = parseInt(pctStr);
      if (pct >= 80) return "high";
      if (pct >= 50) return "med";
      return "low";
    }

    function shortModel(m) {
      if (!m) return '—';
      return m.replace('claude-', '').replace('-latest', '').replace(/-(\d)/g, ' $1');
    }

    async function loadSavedReports() {
      const tbody = document.getElementById('reports-table-body');
      try {
        const r = await fetch('/reports', { headers: apiHeaders() });
        const items = await r.json();
        if (!items.length) {
          tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;padding:24px;color:var(--md-on-surface-variant);">No compliance reports yet. Click <strong>New Analysis</strong> to start.</td></tr>';
          return;
        }
        tbody.innerHTML = items.map(it => {
          const dt = it.generated_at ? it.generated_at.replace('T',' ').slice(0,16) : '—';
          const rid = escape(it.report_id);

          // Verdict bar
          const total = it.claims_analyzed || 0;
          const vbar = `<div class="verdict-bar">
            ${it.consistent   ? `<span class="vb-item vb-ok"><span class="material-symbols-rounded">check_circle</span>${it.consistent}</span>` : ''}
            ${it.contradicts  ? `<span class="vb-item vb-flag"><span class="material-symbols-rounded">error</span>${it.contradicts}</span>` : ''}
            ${it.unsupported  ? `<span class="vb-item vb-warn"><span class="material-symbols-rounded">help</span>${it.unsupported}</span>` : ''}
            ${it.insufficient ? `<span class="vb-item vb-grey"><span class="material-symbols-rounded">help_outline</span>${it.insufficient}</span>` : ''}
            <span style="color:var(--md-on-surface-variant);font-size:0.75rem;margin-left:2px;">${total} claims</span>
          </div>`;

          // Model line
          const modelLine = `<div style="font-size:0.75rem;color:var(--md-on-surface-variant);margin-top:4px;">
            <span title="Extractor">${escape(shortModel(it.extractor_model))}${it.extractor_version ? ' v'+escape(it.extractor_version) : ''}</span>
            <span style="margin:0 4px;">→</span>
            <span title="Analyzer">${escape(shortModel(it.analyzer_model))}${it.analyzer_version ? ' v'+escape(it.analyzer_version) : ''}</span>
            ${it.assumed_industry ? `<span style="margin-left:6px;opacity:0.7;">· ${escape(it.assumed_industry)}</span>` : ''}
          </div>`;

          // Flag preview rows
          const flagRows = (it.top_flags || []).map(f =>
            `<div class="flag-preview"><span class="material-symbols-rounded">error</span>${escape(f.claim)}</div>`
          ).join('');
          const detailId = `rpd-${rid}`;
          const hasDetail = (it.top_flags || []).length > 0;

          return `<tr style="cursor:${hasDetail?'pointer':'default'}" ${hasDetail ? `onclick="toggleRowDetail('${detailId}')"` : ''}>
            <td>
              <div style="font-weight:500;color:var(--md-on-surface);">${escape(it.company_name)}</div>
              ${modelLine}
            </td>
            <td>${vbar}</td>
            <td style="white-space:nowrap;color:var(--md-on-surface-variant);font-size:0.8125rem;">${escape(dt)}</td>
            <td>
              <div style="display:flex;gap:4px;">
                <button class="btn btn-outlined" style="height:30px;padding:0 10px;font-size:0.8rem" onclick="event.stopPropagation();loadReportDrilldown('${rid}')">
                  <span class="material-symbols-rounded" style="font-size:15px">open_in_new</span>
                </button>
                <button class="btn btn-text" style="height:30px;padding:0 6px;color:var(--md-primary)" title="Copy share link" onclick="event.stopPropagation();shareReport('${rid}', this)">
                  <span class="material-symbols-rounded" style="font-size:15px">${it.is_public ? 'link' : 'share'}</span>
                </button>
                <button class="btn btn-text" style="height:30px;padding:0 6px;color:var(--md-error)" onclick="event.stopPropagation();deleteReport('${rid}')">
                  <span class="material-symbols-rounded" style="font-size:15px">delete</span>
                </button>
              </div>
            </td>
          </tr>
          ${hasDetail ? `<tr><td colspan="4" style="padding:0"><div id="${detailId}" class="report-row-detail">${flagRows}</div></td></tr>` : ''}`;
        }).join('');
      } catch (e) {
        tbody.innerHTML = `<tr><td colspan="4" style="color:var(--md-error);">Could not load reports: ${escape(e.message)}</td></tr>`;
      }
    }

    function toggleRowDetail(id) {
      const el = document.getElementById(id);
      if (el) el.classList.toggle('open');
    }

    async function loadReportDrilldown(reportId) {
      try {
        const r = await fetch(`/reports/${reportId}`, { headers: apiHeaders() });
        if (!r.ok) throw new Error("Report not found");
        const report = await r.json();
        
        document.getElementById('drilldown-title').textContent = `${report.company_name || 'Unknown'} - Compliance Report`;
        
        let html = buildReportSummary(report);
        for (const res of report.results) {
          html += renderResultEntry(res);
        }
        
        document.getElementById('drilldown-body').innerHTML = html;
        showDrilldownView();
      } catch (e) {
        showToast("Failed to load report: " + e.message);
      }
    }

    async function deleteReport(reportId) {
      if (!confirm('Are you sure you want to delete this report?')) return;
      try {
        await fetch(`/reports/${reportId}`, { method: 'DELETE', headers: apiHeaders() });
        loadSavedReports();
      } catch (e) {
        showToast("Failed to delete report.");
      }
    }

    // ── Report download as Markdown ───────────────────────────────────────
    document.getElementById('download-report-btn').addEventListener('click', () => {
      if (!lastReport) return;
      const md = reportToMarkdown(lastReport);
      const blob = new Blob([md], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const company = (lastReport.company_name || 'report').replace(/[^a-z0-9]+/gi, '-').toLowerCase();
      const shortModel = (m) => m ? m.replace(/^claude-/, '').replace(/^qwen/, 'qwen').replace(/[.:]/g, '').toLowerCase() : 'unknown';
      const extractorShort = shortModel(lastReport.extractor_model);
      const analyzerShort = shortModel(lastReport.analyzer_model);
      const date = lastReport.generated_at.slice(0, 10);
      a.href = url;
      a.download = `${company}-${date}-${extractorShort}-${analyzerShort}.md`;
      a.click();
      URL.revokeObjectURL(url);
    });

    function reportToMarkdown(rep) {
      const lines = [];
      lines.push(`# Compliance Report: ${rep.company_name || 'Unknown Company'}`);
      lines.push(`**Generated:** ${rep.generated_at}  `);
      lines.push(`**SEC CIK:** ${rep.cik || 'unresolved'}  `);
      if (rep.assumed_industry) lines.push(`**Industry:** ${rep.assumed_industry}  `);
      lines.push(`**Extractor:** ${rep.extractor_model || 'unknown'}${rep.extractor_version ? ' (v' + rep.extractor_version + ')' : ''}  `);
      lines.push(`**Analyzer:** ${rep.analyzer_model || 'unknown'}${rep.analyzer_version ? ' (v' + rep.analyzer_version + ')' : ''}  `);
      lines.push(`**Claims analyzed:** ${rep.claims_analyzed}  `);
      lines.push(`**Flagged FLS contradictions:** ${rep.flagged_forward_looking_contradictions}`);
      lines.push('');
      if ((rep.warnings || []).length) {
        lines.push('## Warnings');
        for (const w of rep.warnings) lines.push(`> ⚠️ ${w}`);
        lines.push('');
      }
      lines.push('## Findings');
      for (let i = 0; i < (rep.results || []).length; i++) {
        const r = rep.results[i];
        const flag = r.verdict === 'CONTRADICTS' && r.forward_looking ? ' 🚨' : '';
        lines.push(`### ${i + 1}. ${r.claim}${flag}`);
        lines.push('');
        lines.push(`**Verdict:** \`${r.verdict}\`  `);
        lines.push(`**Severity:** ${r.severity}  `);
        lines.push(`**Forward-looking:** ${r.forward_looking ? 'Yes' : 'No'}  `);
        if (r.analysis_method) lines.push(`**Method:** ${r.analysis_method}  `);
        lines.push('');
        lines.push(r.explanation);
        lines.push('');
        if (r.missing_information) {
          lines.push(`> **Missing information:** ${r.missing_information}`);
          lines.push('');
        }
        for (const c of (r.cited_passages || [])) {
          lines.push(`**SEC P${c.passage_num}** · ${c.form} filed ${c.filing_date} · [${c.accession}](${c.url})`);
          lines.push('');
          lines.push(`> ${c.excerpt.replace(/\n/g, '  \n> ')}…`);
          lines.push('');
        }
        if ((r.web_sources || []).length) {
          lines.push('**Web sources consulted:**');
          for (const s of r.web_sources) {
            lines.push(`- [${s.title || s.url}](${s.url})${s.page_age ? ' (' + s.page_age + ')' : ''}`);
          }
          lines.push('');
        }
        lines.push('---');
        lines.push('');
      }
      if (rep.log_path) lines.push(`_Log file: ${rep.log_path}_`);
      return lines.join('\n');
    }

    // Initialize Dashboard
    loadSavedReports();
    loadSavedExtractionsList();

    fileInput.addEventListener('change', () => uploadFile(fileInput.files[0]));
    dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
    dropzone.addEventListener('drop', e => {
      e.preventDefault(); dropzone.classList.remove('drag');
      if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
    });

    function escape(s) {
      return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    function progressHTML(label) {
      return `<div class="progress"><div class="progress-bar"></div><span>${escape(label)}</span></div>`;
    }

    async function uploadFile(file) {
      if (!file) return;
      currentFilename = file.name;
      dzText.innerHTML = progressHTML(`Analyzing stage and basic info with ${EXTRACTOR_MODEL}...`);
      const fd = new FormData();
      fd.append('file', file);
      fd.append('extractor_model', EXTRACTOR_MODEL);
      try {
        const r = await fetch('/extract', { method: 'POST', body: fd, headers: apiHeaders() });
        if (!r.ok) throw new Error((await r.json()).detail || 'Basic extraction failed');
        const data = await r.json();
        currentContextId = data.context_id;
        renderBasicExtraction(data.extraction);
        restoreDropzone(file.name);
      } catch (e) {
        // Surface errors as a dismissable toast and reset the dropzone so the
        // user can immediately retry without a page refresh.
        showToast(e.message || 'Extraction failed');
        restoreDropzone(null);
      }
    }

    function restoreDropzone(filename) {
      if (filename) {
        dropzone.classList.add('compact');
        dzText.innerHTML = `
          <span class="material-symbols-rounded icon" style="color:var(--md-primary)">picture_as_pdf</span>
          <div><strong>${escape(filename)}</strong></div>
          <div class="hint">Drop a different PDF to replace</div>`;
      } else {
        dropzone.classList.remove('compact');
        dzText.innerHTML = `
          <span class="material-symbols-rounded icon">upload_file</span>
          <div><strong>Click or drop a PDF here</strong></div>
          <div class="hint">Null fields indicate the deck did not state that value.</div>`;
      }
    }

    function toggleAccordion(header) {
      header.classList.toggle('open');
      const body = header.nextElementSibling;
      if (body) body.classList.toggle('open');
    }

    function emptyOr(v) {
      return v ? `<dd>${escape(v)}</dd>` : '<dd class="empty">not stated in deck</dd>';
    }

    function renderBasicExtraction(ex) {
      const c = ex.company;
      const stageLabel = ex.stage_assessment?.stage
        ? ex.stage_assessment.stage.replace(/_/g,' ').replace(/\b\w/g, l => l.toUpperCase())
        : null;

      let html = `<div class="acc-header open" onclick="toggleAccordion(this)">
        <span class="material-symbols-rounded acc-chevron">expand_more</span>
        Company identity
        ${stageLabel ? `<span class="chip chip-primary" style="margin-left:8px;text-transform:none;letter-spacing:0;">${escape(stageLabel)}</span>` : ''}
      </div>
      <div class="acc-body open"><dl class="kv">
        <dt>Name</dt><dd>${escape(c.name)}</dd>
        <dt>Ticker</dt>${emptyOr(c.ticker)}
        <dt>CIK</dt>${emptyOr(c.cik)}
        <dt>Industry</dt>${emptyOr(c.industry)}
        <dt>Website</dt>${emptyOr(c.website)}
      </dl></div>`;

      extractionBody.innerHTML = html;
      savedCompanyHtml = html;  // snapshot for clean re-render in completeWithClaude()

      if (ex.stage_assessment && ex.stage_assessment.stage) {
        document.getElementById('startup-stage').value = ex.stage_assessment.stage;
        updateModulesFromStage(ex.stage_assessment.stage);
      }

      continueExtractBtn.classList.remove('hidden');
      verifyBtn.classList.add('hidden');
      
      extractionPanel.classList.remove('hidden');
      reportPanel.classList.add('hidden');
      reportBody.innerHTML = '';
      extractionPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function buildClaimsAccordion(claims, notes) {
      // Category colour map for chips
      const catColor = { financial:'#0B57D0', market:'#6B5778', traction:'#146C2E',
                         projection:'#7A5900', regulatory:'#BA1A1A' };
      let claimsHtml = '';
      for (const cl of claims) {
        const cc = catColor[cl.category] || 'var(--md-on-surface-variant)';
        claimsHtml += `<div class="claim">
          <div class="text">${escape(cl.text)}</div>
          <div class="verbatim">"${escape(cl.verbatim)}"</div>
          <div class="meta">
            <span class="chip"><span class="material-symbols-rounded">photo_library</span>slide ${cl.slide}</span>
            <span class="chip" style="border-color:${cc};color:${cc};">${escape(cl.category)}</span>
            ${cl.likely_forward_looking ? '<span class="chip chip-primary" title="Forward-Looking Statement — this claim is a projection, target, or expectation rather than a stated historical fact."><span class="material-symbols-rounded">trending_up</span>FLS</span>' : ''}
          </div>
        </div>`;
      }

      // Count by category for the header badge
      const cats = {};
      for (const cl of claims) cats[cl.category] = (cats[cl.category] || 0) + 1;
      const catSummary = Object.entries(cats).map(([k,v]) => `${v} ${k}`).join(', ');

      let html = `
        <div class="acc-header open" onclick="toggleAccordion(this)">
          <span class="material-symbols-rounded acc-chevron">expand_more</span>
          Claims (${claims.length})
          <span style="margin-left:6px;font-weight:400;opacity:0.7;font-size:0.75rem;text-transform:none;letter-spacing:0;">${catSummary}</span>
        </div>
        <div class="acc-body open">${claimsHtml}</div>`;

      if (notes) {
        html += `
        <div class="acc-header" onclick="toggleAccordion(this)">
          <span class="material-symbols-rounded acc-chevron">expand_more</span>
          Extractor notes
        </div>
        <div class="acc-body"><pre class="notes">${escape(notes)}</pre></div>`;
      }
      return html;
    }

    function renderDeepExtraction(ex, failedPages) {
      failedPages = failedPages || [];
      currentFailedPages = failedPages;

      let html = extractionBody.innerHTML;
      html += buildClaimsAccordion(ex.claims, ex.extraction_notes);

      if (failedPages.length > 0) {
        const pagesStr = failedPages.join(', ');
        html += `
        <div class="failed-pages-panel" id="failed-pages-panel">
          <div class="fp-header">
            <span class="material-symbols-rounded">warning</span>
            ${failedPages.length} slide${failedPages.length > 1 ? 's' : ''} skipped by local model (OOM)
          </div>
          <div class="fp-body">
            Slide${failedPages.length > 1 ? 's' : ''} <strong>${pagesStr}</strong> exceeded the local
            vision model's memory capacity and could not be processed. Select a cloud model below to
            extract claims from those pages and merge the results.
          </div>
          <div class="fp-actions">
            <div class="select-field">
              <label>Complete with cloud model</label>
              <select id="cloud-complete-model">
                <option value="claude-haiku-4-5" selected>Claude Haiku 4.5 — fast &amp; cheap</option>
                <option value="claude-sonnet-4-6">Claude Sonnet 4.6 — balanced</option>
                <option value="claude-opus-4-6">Claude Opus 4.6 — highest quality</option>
              </select>
              <span class="material-symbols-rounded arrow">arrow_drop_down</span>
            </div>
            <button class="btn btn-filled" id="complete-with-claude-btn" onclick="completeWithClaude()">
              <span class="material-symbols-rounded">cloud_sync</span>
              Complete with Claude
            </button>
          </div>
        </div>`;
      }

      extractionBody.innerHTML = html;

      document.getElementById('analysis-config').classList.add('hidden');
      continueExtractBtn.classList.add('hidden');

      if (failedPages.length > 0) {
        // Don't show compliance/save until the partial extraction is completed
        verifyBtn.classList.add('hidden');
        document.getElementById('save-btn').classList.add('hidden');
      } else {
        verifyBtn.classList.remove('hidden');
        document.getElementById('save-btn').classList.remove('hidden');
      }
    }

    async function completeWithClaude() {
      if (!currentContextId || currentFailedPages.length === 0) return;

      const btn = document.getElementById('complete-with-claude-btn');
      const modelSel = document.getElementById('cloud-complete-model');
      const cloudModel = modelSel ? modelSel.value : 'claude-haiku-4-5';

      btn.disabled = true;
      btn.innerHTML = '<span class="material-symbols-rounded">hourglass_empty</span> Completing…';

      const fd = new FormData();
      fd.append('context_id', currentContextId);
      fd.append('cloud_model', cloudModel);
      const modules = Array.from(document.querySelectorAll('.metric-cb:checked')).map(cb => cb.value);
      fd.append('modules', modules.join(','));

      try {
        const r = await fetch('/extract/complete', { method: 'POST', body: fd, headers: apiHeaders() });
        if (!r.ok) throw new Error((await r.json()).detail || 'Cloud completion failed');
        const data = await r.json();

        // Restore the company block without triggering scroll, button flashes, or
        // report panel reset — then append the merged claims accordion on top.
        extractionBody.innerHTML = savedCompanyHtml;
        renderDeepExtraction(data.extraction, []);

        // Prepend a success banner above the company block
        const successBanner = document.createElement('div');
        successBanner.className = 'banner banner-info';
        successBanner.innerHTML = `<span class="material-symbols-rounded">check_circle</span>
          <div>Cloud completion done — ${data.completed_pages.length} slide(s) merged using ${escape(cloudModel)}.</div>`;
        extractionBody.prepend(successBanner);

        currentFailedPages = [];
      } catch (e) {
        // Inline error — keep the failed-pages panel visible so the user can retry
        btn.disabled = false;
        btn.innerHTML = '<span class="material-symbols-rounded">cloud_sync</span> Complete with Claude';
        const errBanner = document.createElement('div');
        errBanner.className = 'banner banner-error';
        errBanner.style.marginTop = '8px';
        errBanner.innerHTML = `<span class="material-symbols-rounded">error</span>
          <div>Cloud completion failed: ${escape(e.message)}</div>`;
        document.getElementById('failed-pages-panel')?.append(errBanner);
      }
    }

    function updateModulesFromStage(stage) {
      const isSeed = (stage === 'pre_seed' || stage === 'seed');
      const isSeriesAB = (stage === 'series_a' || stage === 'series_b');
      const isLate = (stage === 'series_c_plus');

      document.querySelectorAll('.metric-cb.seed').forEach(cb => cb.checked = isSeed);
      document.querySelectorAll('.metric-cb.series-ab').forEach(cb => cb.checked = isSeriesAB);
      document.querySelectorAll('.metric-cb.late').forEach(cb => cb.checked = isLate);
      document.querySelectorAll('.metric-cb.universal').forEach(cb => cb.checked = true);
    }

    document.getElementById('startup-stage').addEventListener('change', (e) => {
      updateModulesFromStage(e.target.value);
    });

    let activeAbortController = null;

    cancelVerifyBtn.addEventListener('click', () => {
      if (activeAbortController) {
        activeAbortController.abort();
      }
    });

    continueExtractBtn.addEventListener('click', async () => {
      if (!currentContextId) return;
      
      continueExtractBtn.disabled = true;
      continueExtractBtn.innerHTML = '<span class="material-symbols-rounded">hourglass_empty</span> Extracting...';
      
      const fd = new FormData();
      fd.append('context_id', currentContextId);
      fd.append('extractor_model', EXTRACTOR_MODEL);
      fd.append('startup_stage', document.getElementById('startup-stage').value);
      
      const modules = Array.from(document.querySelectorAll('.metric-cb:checked')).map(cb => cb.value);
      fd.append('modules', modules.join(','));

      try {
        const r = await fetch('/extract/deep', { method: 'POST', body: fd, headers: apiHeaders() });
        if (!r.ok) throw new Error((await r.json()).detail || 'Deep extraction failed');
        const data = await r.json();
        renderDeepExtraction(data.extraction, data.failed_pages || []);
      } catch (e) {
        showToast("Deep Extraction Failed: " + e.message);
      } finally {
        continueExtractBtn.disabled = false;
        continueExtractBtn.innerHTML = '<span class="material-symbols-rounded">manage_search</span> Continue Extraction';
      }
    });

    // ── Streaming compliance check ────────────────────────────────────────
    verifyBtn.addEventListener('click', async () => {
      if (!currentContextId) return;
      verifyBtn.disabled = true;
      cancelVerifyBtn.classList.remove('hidden');
      reportPanel.classList.remove('hidden');
      reportBody.innerHTML = progressHTML(`Connecting to compliance engine · ${ANALYZER_MODEL}…`);
      reportPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

      // State accumulated while streaming
      let claimsCount = 0;
      let flaggedCount = 0;

      activeAbortController = new AbortController();

      const fd = new FormData();
      fd.append('context_id', currentContextId);
      fd.append('analyzer_model', ANALYZER_MODEL);
      fd.append('extractor_model', EXTRACTOR_MODEL);
      fd.append('startup_stage', document.getElementById('startup-stage').value);
      const modules = Array.from(document.querySelectorAll('.metric-cb:checked')).map(cb => cb.value);
      fd.append('modules', modules.join(','));

      try {
        const resp = await fetch('/verify/stream', {
          method: 'POST',
          body: fd,
          signal: activeAbortController.signal,
          headers: apiHeaders(),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ detail: resp.statusText }));
          throw new Error(err.detail || 'Stream failed');
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });

          // SSE lines: "data: {...}\n\n"
          const parts = buf.split('\n\n');
          buf = parts.pop(); // keep any incomplete chunk
          for (const part of parts) {
            const line = part.trim();
            if (!line.startsWith('data:')) continue;
            let evt;
            try { evt = JSON.parse(line.slice(5).trim()); } catch { continue; }
            handleStreamEvent(evt);
          }
        }
      } catch (e) {
        if (e.name === 'AbortError') {
          reportBody.innerHTML += `<div class="banner banner-error"><span class="material-symbols-rounded">cancel</span><div>Analysis cancelled by user.</div></div>`;
        } else {
          reportBody.innerHTML = `<div class="banner banner-error"><span class="material-symbols-rounded">error</span><div>${escape(e.message)}</div></div>`;
        }
      } finally {
        verifyBtn.disabled = false;
        cancelVerifyBtn.classList.add('hidden');
        activeAbortController = null;
        // Remove any lingering status/progress indicator
        document.getElementById('stream-status')?.remove();
      }

      function handleStreamEvent(evt) {
        const { event, data } = evt;

        if (event === 'start') {
          // Replace progress bar with summary stats skeleton + status line
          const cikLabel = data.cik || 'unresolved';
          reportBody.innerHTML = `
            <div id="report-summary" class="report-summary">
              <div class="stat"><div class="stat-value" id="stat-analyzed">0</div><div class="stat-label">Analyzed</div></div>
              <div class="stat"><div class="stat-value" id="stat-flagged">0</div><div class="stat-label">Flagged FLS</div></div>
              <div class="stat"><div class="stat-value" style="font-size:1rem;padding-top:8px">${escape(String(data.total_claims))}</div><div class="stat-label">Total claims</div></div>
              <div class="stat"><div class="stat-value" style="font-size:1rem;padding-top:8px">${escape(cikLabel)}</div><div class="stat-label">SEC CIK</div></div>
              ${data.assumed_industry ? `<div class="stat"><div class="stat-value" style="font-size:1rem;padding-top:8px">${escape(data.assumed_industry)}</div><div class="stat-label">Industry</div></div>` : ''}
            </div>
            <div id="stream-status" class="progress"><div class="progress-bar"></div><span id="stream-status-text">Initialising…</span></div>
            <div id="results-list"></div>`;
        }

        if (event === 'warning') {
          const el = document.createElement('div');
          el.className = 'banner';
          el.innerHTML = `<span class="material-symbols-rounded">warning</span><div>${escape(data.message)}</div>`;
          document.getElementById('results-list')?.prepend(el);
        }

        if (event === 'status') {
          const t = document.getElementById('stream-status-text');
          if (t) t.textContent = data.message;
        }

        if (event === 'claim_result') {
          claimsCount++;
          const el = document.getElementById('stat-analyzed');
          if (el) el.textContent = claimsCount;

          const entry = data.entry;
          if (entry.verdict === 'CONTRADICTS' && entry.forward_looking) {
            flaggedCount++;
            const fel = document.getElementById('stat-flagged');
            if (fel) { fel.textContent = flaggedCount; fel.classList.add('flag'); }
          }

          const node = document.createElement('div');
          node.innerHTML = renderResultEntry(entry);
          document.getElementById('results-list')?.appendChild(node.firstElementChild);

          // Update status line
          const t = document.getElementById('stream-status-text');
          if (t) t.textContent = `[${data.index}/${data.total}] done — ${entry.verdict}`;
        }

        if (event === 'done') {
          document.getElementById('stream-status')?.remove();
          const rep = data.report;
          lastReport = rep;
          // Build final report
          let html = buildReportSummary(rep);
          for (const res of rep.results) {
            html += renderResultEntry(res);
          }
          reportBody.innerHTML = html;
          // Show download button
          document.getElementById('download-report-btn').style.display = '';
          // Force a load of the dashboard in the background so it's fresh
          loadSavedReports();
        }

        if (event === 'error') {
          reportBody.innerHTML += `<div class="banner banner-error"><span class="material-symbols-rounded">error</span><div>Stream error: ${escape(data.message)}</div></div>`;
        }
      }
    });

    // ── Shared rendering helpers ──────────────────────────────────────────
    function verdictChipClass(verdict) {
      return { 'CONTRADICTS':'chip-contradicts','CONSISTENT':'chip-consistent',
               'UNSUPPORTED':'chip-unsupported','INSUFFICIENT_EVIDENCE':'chip-insufficient' }[verdict] || 'chip-insufficient';
    }
    function verdictIcon(verdict) {
      return { 'CONTRADICTS':'error','CONSISTENT':'check_circle',
               'UNSUPPORTED':'help','INSUFFICIENT_EVIDENCE':'help_outline' }[verdict] || 'help_outline';
    }

    function buildReportSummary(rep) {
      const claimCount  = rep.claims_analyzed || 0;
      const flagCount   = rep.flagged_forward_looking_contradictions || 0;
      const results     = rep.results || [];
      const consistent  = results.filter(r => r.verdict === 'CONSISTENT').length;
      const contradicts = results.filter(r => r.verdict === 'CONTRADICTS').length;
      const unsupported = results.filter(r => r.verdict === 'UNSUPPORTED').length;
      const insufficient= results.filter(r => r.verdict === 'INSUFFICIENT_EVIDENCE').length;

      let html = '<div style="margin-bottom:16px;">';

      // ── Compact metadata bar ────────────────────────────────────────────
      html += '<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:12px;">';
      if (rep.generated_at) {
        html += `<span class="chip"><span class="material-symbols-rounded">schedule</span>${escape(rep.generated_at.replace('T',' ').slice(0,16))}</span>`;
      }
      if (rep.extractor_model) {
        html += `<span class="chip" title="Extractor"><span class="material-symbols-rounded">psychology</span>${escape(shortModel(rep.extractor_model))}${rep.extractor_version ? ' v'+escape(rep.extractor_version) : ''}</span>`;
      }
      if (rep.analyzer_model) {
        html += `<span class="chip" title="Analyzer"><span class="material-symbols-rounded">manage_search</span>${escape(shortModel(rep.analyzer_model))}${rep.analyzer_version ? ' v'+escape(rep.analyzer_version) : ''}</span>`;
      }
      if (rep.cik) html += `<span class="chip"><span class="material-symbols-rounded">receipt_long</span>CIK ${escape(rep.cik)}</span>`;
      if (rep.assumed_industry) html += `<span class="chip">${escape(rep.assumed_industry)}</span>`;
      html += '</div>';

      // ── Verdict breakdown tiles ─────────────────────────────────────────
      html += '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">';
      const tile = (val, label, color, icon) =>
        `<div style="background:var(--md-surface-container);border-radius:10px;padding:10px 14px;display:flex;align-items:center;gap:8px;min-width:100px;">
          <span class="material-symbols-rounded" style="color:${color};font-size:20px;">${icon}</span>
          <div>
            <div style="font-size:1.25rem;font-weight:600;color:${color};line-height:1;">${val}</div>
            <div style="font-size:0.7rem;color:var(--md-on-surface-variant);text-transform:uppercase;letter-spacing:0.4px;">${label}</div>
          </div>
        </div>`;
      html += tile(consistent,   'Consistent',   'var(--md-success)',              'check_circle');
      html += tile(contradicts,  'Contradicts',  contradicts ? 'var(--md-error)' : 'var(--md-on-surface-variant)', 'error');
      html += tile(unsupported,  'Unsupported',  'var(--md-warning)',              'help');
      html += tile(insufficient, 'Insufficient', 'var(--md-on-surface-variant)',   'help_outline');
      if (flagCount) {
        html += tile(flagCount,  'FLS flags',    'var(--md-error)',                'flag');
      }
      html += '</div>';

      // ── Warnings ────────────────────────────────────────────────────────
      for (const w of (rep.warnings || [])) {
        html += `<div class="banner" style="margin-bottom:8px;"><span class="material-symbols-rounded">warning</span><div>${escape(w)}</div></div>`;
      }

      html += '</div>';

      if (claimCount > 0) {
        html += `<div style="font-size:0.8rem;color:var(--md-on-surface-variant);margin-bottom:8px;">
          Click any finding to expand · <strong style="color:var(--md-error)">CONTRADICTS</strong> shown expanded by default
        </div>`;
        html += `<div class="section-heading" style="margin-top:0;">Findings (${claimCount} claims)</div>`;
      }

      return html;
    }

    // ── Save extraction ───────────────────────────────────────────────────
    const saveBtn = document.getElementById('save-btn');
    saveBtn.addEventListener('click', async () => {
      if (!currentContextId) return;
      saveBtn.disabled = true;
      saveBtn.innerHTML = '<span class="material-symbols-rounded">hourglass_empty</span> Saving…';
      const fd = new FormData();
      fd.append('context_id', currentContextId);
      fd.append('original_filename', currentFilename || '');
      fd.append('extractor_model', EXTRACTOR_MODEL);
      try {
        const r = await fetch('/saved-extractions', { method: 'POST', body: fd, headers: apiHeaders() });
        if (!r.ok) throw new Error((await r.json()).detail || 'Save failed');
        const data = await r.json();
        saveBtn.innerHTML = '<span class="material-symbols-rounded">bookmark_added</span> Saved!';
        saveBtn.disabled = true;  // prevent double-save
      } catch (e) {
        showToast('Failed to save extraction: ' + e.message);
        saveBtn.disabled = false;
        saveBtn.innerHTML = '<span class="material-symbols-rounded">bookmark_add</span> Save extraction';
      }
    });

    // ── Saved extractions library ──────────────────────────────────────────
    async function loadSavedExtractionsList() {
      const container = document.getElementById('saved-extractions-list');
      if (!container) return;
      try {
        const r = await fetch('/saved-extractions', { headers: apiHeaders() });
        const items = await r.json();
        if (!items.length) {
          container.innerHTML = '<div class="saved-empty">No saved extractions yet. After running a deep extraction, click "Save extraction" to add it to this library.</div>';
          return;
        }
        container.innerHTML = '<div class="saved-list">' + items.map(it => {
          const savedAt = it.saved_at ? it.saved_at.replace('T', ' ').slice(0, 16) : '—';
          const model = it.extractor_model || '—';
          const filename = it.original_filename ? ` · ${escape(it.original_filename)}` : '';

          // ── Info popover content ──────────────────────────────────────────
          const byCategory = it.claims_by_category || {};
          const totalClaims = it.claims_count || 0;
          const catOrder = ['financial','market','traction','projection','product','team','regulatory','other'];
          const catLabel = { financial:'Financial', market:'Market', traction:'Traction',
                             projection:'Projection', product:'Product', team:'Team',
                             regulatory:'Regulatory', other:'Other' };

          let popHtml = '';
          if (it.stage) {
            const stageLabel = it.stage.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            popHtml += `<div class="pop-stage"><strong>Stage:</strong> ${escape(stageLabel)}</div>`;
          }
          popHtml += '<div class="pop-title">Claims by category</div>';
          const activeCats = catOrder.filter(c => byCategory[c]);
          if (activeCats.length) {
            activeCats.forEach(cat => {
              popHtml += `<div class="pop-row"><span class="cat">${escape(catLabel[cat] || cat)}</span><span class="cnt">${byCategory[cat]}</span></div>`;
            });
          } else {
            popHtml += '<div class="pop-row"><span class="cat">No claims extracted</span></div>';
          }
          popHtml += `<div class="pop-total"><span>Total</span><span>${totalClaims}</span></div>`;
          if (it.key_metrics && it.key_metrics.length) {
            popHtml += `<div class="pop-metrics"><strong>Key metrics:</strong> ${it.key_metrics.map(escape).join(', ')}</div>`;
          }

          return `<div class="saved-item">
            <span class="material-symbols-rounded" style="color:var(--md-primary);flex-shrink:0;">picture_as_pdf</span>
            <div class="saved-info">
              <div class="saved-name">${escape(it.company_name)}</div>
              <div class="saved-meta">${escape(savedAt)}${filename} · model: ${escape(model)}</div>
            </div>
            <div class="saved-actions">
              <div class="info-wrap">
                <button class="info-btn" aria-label="Extraction stats">
                  <span class="material-symbols-rounded">info</span>
                </button>
                <div class="info-popup">${popHtml}</div>
              </div>
              <button class="btn btn-outlined" style="height:32px;padding:0 12px;font-size:0.8rem"
                      onclick="loadSavedExtraction('${escape(it.save_id)}')">
                <span class="material-symbols-rounded" style="font-size:16px">play_arrow</span> Load &amp; Analyze
              </button>
              <button class="btn btn-text" style="height:32px;padding:0 8px;color:var(--md-primary)" title="Copy share link"
                      onclick="shareExtraction('${escape(it.save_id)}', this)">
                <span class="material-symbols-rounded" style="font-size:16px">${it.is_public ? 'link' : 'share'}</span>
              </button>
              <button class="btn btn-text" style="height:32px;padding:0 8px;color:var(--md-error)"
                      onclick="deleteSavedExtraction('${escape(it.save_id)}')">
                <span class="material-symbols-rounded" style="font-size:16px">delete</span>
              </button>
            </div>
          </div>`;
        }).join('') + '</div>';
      } catch (e) {
        container.innerHTML = `<div class="saved-empty" style="color:var(--md-error);">Could not load saved extractions: ${escape(e.message)}</div>`;
      }
    }

    async function loadSavedExtraction(saveId) {
      const container = document.getElementById('saved-extractions-list');
      if (container) {
        container.innerHTML = '<div style="text-align:center;color:var(--md-on-surface-variant);padding:16px;">' +
          progressHTML('Loading saved extraction…') + '</div>';
      }
      try {
        const r = await fetch(`/saved-extractions/${saveId}/load`, { method: 'POST', headers: apiHeaders() });
        if (!r.ok) throw new Error((await r.json()).detail || 'Load failed');
        const data = await r.json();
        currentContextId = data.context_id;
        currentFilename = data.meta?.original_filename || '';
        showAnalysisView();
        renderLoadedExtraction(data.extraction, data.meta);
      } catch (e) {
        showToast('Failed to load extraction: ' + e.message);
        loadSavedExtractionsList(); // restore the list
      }
    }

    async function shareReport(reportId, btn) {
      try {
        const r = await fetch(`/reports/${reportId}/share`, { method: 'POST', headers: apiHeaders() });
        if (!r.ok) throw new Error((await r.json()).detail || 'Share failed');
        const { share_url } = await r.json();
        await navigator.clipboard.writeText(share_url);
        if (btn) btn.querySelector('.material-symbols-rounded').textContent = 'link';
        showToast('Share link copied to clipboard!');
      } catch (e) {
        showToast('Could not create share link: ' + e.message);
      }
    }

    async function shareExtraction(saveId, btn) {
      try {
        const r = await fetch(`/saved-extractions/${saveId}/share`, { method: 'POST', headers: apiHeaders() });
        if (!r.ok) throw new Error((await r.json()).detail || 'Share failed');
        const { share_url } = await r.json();
        await navigator.clipboard.writeText(share_url);
        if (btn) btn.querySelector('.material-symbols-rounded').textContent = 'link';
        showToast('Share link copied to clipboard!');
      } catch (e) {
        showToast('Could not create share link: ' + e.message);
      }
    }

    async function deleteSavedExtraction(saveId) {
      if (!confirm('Delete this saved extraction?')) return;
      try {
        await fetch(`/saved-extractions/${saveId}`, { method: 'DELETE', headers: apiHeaders() });
        loadSavedExtractionsList();
      } catch (e) {
        showToast('Failed to delete: ' + e.message);
      }
    }

    function renderLoadedExtraction(extraction, meta) {
      // Compact dropzone to show "loaded from saved"
      const filename = meta?.original_filename || '';
      const modelUsed = meta?.extractor_model || '';
      dropzone.classList.add('compact');
      dzText.innerHTML = `
        <span class="material-symbols-rounded icon" style="color:var(--md-primary)">bookmark</span>
        <div><strong>${filename ? escape(filename) : escape(extraction.company?.name || 'Saved extraction')}</strong></div>
        <div class="hint">Loaded from library${modelUsed ? ' · ' + escape(shortModel(modelUsed)) : ''}</div>`;

      // Company identity
      const c = extraction.company || {};
      let html = `<div class="banner" style="margin-bottom:12px;">
        <span class="material-symbols-rounded">info</span>
        <div>Loaded from saved extraction${meta?.saved_at ? ' · ' + escape(meta.saved_at.replace('T',' ').slice(0,16)) : ''}. Click <strong>Run compliance check</strong> to analyze.</div>
      </div>`;

      html += `<div class="acc-header open" onclick="toggleAccordion(this)">
        <span class="material-symbols-rounded acc-chevron">expand_more</span>
        Company identity
      </div>
      <div class="acc-body open"><dl class="kv">
        <dt>Name</dt><dd>${escape(c.name || '—')}</dd>
        <dt>Ticker</dt>${emptyOr(c.ticker)}
        <dt>CIK</dt>${emptyOr(c.cik)}
        <dt>Industry</dt>${emptyOr(c.industry)}
        <dt>Website</dt>${emptyOr(c.website)}
      </dl></div>`;

      // Set stage
      if (extraction.stage_assessment?.stage) {
        document.getElementById('startup-stage').value = extraction.stage_assessment.stage;
        updateModulesFromStage(extraction.stage_assessment.stage);
      }

      // Claims + notes via shared helper
      html += buildClaimsAccordion(extraction.claims || [], extraction.extraction_notes);
      extractionBody.innerHTML = html;

      // UI state
      document.getElementById('analysis-config').classList.add('hidden');
      continueExtractBtn.classList.add('hidden');
      verifyBtn.classList.remove('hidden');
      document.getElementById('save-btn').classList.add('hidden');
      document.getElementById('cancel-verify-btn').classList.add('hidden');

      extractionPanel.classList.remove('hidden');
      reportPanel.classList.add('hidden');
      reportBody.innerHTML = '';
      extractionPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function toggleResult(el) {
      // Don't toggle when the user clicks a link inside the detail panel
      el.classList.toggle('open');
    }

    function renderResultEntry(r) {
      const verdict = r.verdict || 'INSUFFICIENT_EVIDENCE';
      const isContradicts = verdict === 'CONTRADICTS';
      const isFlag = isContradicts && r.forward_looking;

      // Base classes — CONTRADICTS starts open, others collapsed
      let cls = 'result';
      if (isFlag) cls += ' flag';
      else if (verdict === 'CONSISTENT') cls += ' consistent';
      else if (verdict === 'UNSUPPORTED' || verdict === 'INSUFFICIENT_EVIDENCE') cls += ' unsupported';
      if (isContradicts) cls += ' open'; // expand flagged findings by default

      // ── Always-visible summary row ─────────────────────────────────────
      const VERDICT_TIPS = {
        CONTRADICTS: 'The deck claim conflicts with evidence found in SEC filings or web sources.',
        CONSISTENT: 'The deck claim is supported by or consistent with available evidence.',
        UNSUPPORTED: 'No corroborating evidence found, but no direct contradiction either.',
        INSUFFICIENT_EVIDENCE: 'Not enough public information available to verify this claim.',
      };
      const SEVERITY_TIPS = {
        HIGH: 'High severity — material discrepancy that may warrant further investigation.',
        MEDIUM: 'Medium severity — notable inconsistency worth flagging to the deal team.',
        LOW: 'Low severity — minor or likely immaterial difference.',
      };
      let chips = `<span class="chip ${verdictChipClass(verdict)}" title="${VERDICT_TIPS[verdict] || verdict}"><span class="material-symbols-rounded">${verdictIcon(verdict)}</span>${escape(verdict)}</span>`;
      if (r.forward_looking) chips += `<span class="chip chip-primary" style="height:20px;font-size:0.7rem;" title="Forward-Looking Statement — this claim is a projection, target, or expectation rather than a stated historical fact."><span class="material-symbols-rounded">trending_up</span>FLS</span>`;
      if (r.severity && r.severity !== 'NONE') chips += `<span class="chip" style="height:20px;font-size:0.7rem;" title="${SEVERITY_TIPS[r.severity] || r.severity}">${escape(r.severity)}</span>`;
      if (r.analysis_method === 'web_search') chips += `<span class="chip" style="height:20px;font-size:0.7rem;" title="Verified via web search — no SEC filing found for this company, so the claim was cross-referenced against public web sources."><span class="material-symbols-rounded">travel_explore</span>web</span>`;

      let html = `<div class="${cls}" onclick="if(!event.target.closest('a'))toggleResult(this)">`;
      html += `<div style="display:flex;align-items:flex-start;gap:8px;">`;
      html += `<div style="flex:1;min-width:0;">`;
      html += `<div class="claim-text" style="margin-bottom:6px;">${escape(r.claim)}</div>`;
      html += `<div style="display:flex;gap:4px;flex-wrap:wrap;">${chips}</div>`;
      html += `</div>`;
      html += `<span class="material-symbols-rounded result-chevron">expand_more</span>`;
      html += `</div>`;

      // ── Collapsible detail panel ────────────────────────────────────────
      let detail = '';
      if (r.explanation) detail += `<div class="explanation">${escape(r.explanation)}</div>`;
      if (r.missing_information) {
        detail += `<div class="missing"><span class="material-symbols-rounded">info</span><div>Missing: ${escape(r.missing_information)}</div></div>`;
      }
      for (const c of (r.cited_passages || [])) {
        detail += `<div class="cite" onclick="event.stopPropagation()">`;
        detail += `<div><strong>P${c.passage_num}</strong> · ${escape(c.form)} filed ${escape(c.filing_date)} · <a href="${escape(c.url)}" target="_blank" rel="noopener">${escape(c.accession)}</a></div>`;
        detail += `<div class="excerpt">${escape(c.excerpt)}…</div>`;
        detail += `</div>`;
      }
      if ((r.web_sources || []).length > 0) {
        detail += `<div class="section-heading" style="margin-top:10px;margin-bottom:4px;">Web sources</div>`;
        for (const s of r.web_sources) {
          detail += `<div class="cite" onclick="event.stopPropagation()">`;
          detail += `<div><span class="material-symbols-rounded" style="font-size:13px;vertical-align:-2px;color:var(--md-primary)">open_in_new</span> `;
          detail += `<a href="${escape(s.url)}" target="_blank" rel="noopener">${escape(s.title || s.url)}</a>`;
          if (s.page_age) detail += ` <span style="color:var(--md-on-surface-variant);font-size:0.75rem">(${escape(s.page_age)})</span>`;
          detail += `</div></div>`;
        }
      }

      if (detail) html += `<div class="result-detail">${detail}</div>`;
      html += `</div>`;
      return html;
    }

    // ── Inactivity auto-logout (30 min total; warn at 25 min) ─────────────
    const INACTIVITY_WARN_MS = 25 * 60 * 1000;   // 25 min → show warning
    const INACTIVITY_LOGOUT_MS = 30 * 60 * 1000;  // 30 min → force logout

    let _inactivityWarnTimer   = null;
    let _inactivityLogoutTimer = null;
    let _inactivityCountdown   = null;

    function _clearInactivityTimers() {
      clearTimeout(_inactivityWarnTimer);
      clearTimeout(_inactivityLogoutTimer);
      clearInterval(_inactivityCountdown);
    }

    function _closeInactivityModal() {
      const m = document.getElementById('inactivity-modal');
      if (m) m.style.display = 'none';
      clearInterval(_inactivityCountdown);
    }

    function _showInactivityWarning() {
      const modal = document.getElementById('inactivity-modal');
      if (!modal) return;
      modal.style.display = 'flex';

      let secsLeft = 5 * 60;
      const countEl = document.getElementById('inactivity-countdown');
      function tick() {
        const m = String(Math.floor(secsLeft / 60)).padStart(1, '0');
        const s = String(secsLeft % 60).padStart(2, '0');
        if (countEl) countEl.textContent = `${m}:${s}`;
        if (secsLeft <= 0) { clearInterval(_inactivityCountdown); doLogout(); }
        secsLeft--;
      }
      tick();
      _inactivityCountdown = setInterval(tick, 1000);
    }

    function resetInactivityTimer() {
      _clearInactivityTimers();
      _closeInactivityModal();
      _inactivityWarnTimer   = setTimeout(_showInactivityWarning, INACTIVITY_WARN_MS);
      _inactivityLogoutTimer = setTimeout(doLogout, INACTIVITY_LOGOUT_MS);
    }

    function stayLoggedIn() {
      resetInactivityTimer();
    }

    function doLogout() {
      _clearInactivityTimers();
      window.location.href = '/auth/logout';
    }

    // Only activate inactivity timer when a user is authenticated
    (async () => {
      try {
        const r = await fetch('/auth/me', { credentials: 'same-origin' });
        const user = await r.json();
        if (user && user.email) {
          ['mousemove', 'keydown', 'click', 'scroll', 'touchstart'].forEach(evt =>
            document.addEventListener(evt, resetInactivityTimer, { passive: true })
          );
          resetInactivityTimer();
        }
      } catch (_) { /* unauthenticated or fetch failed — no timer needed */ }
    })();
