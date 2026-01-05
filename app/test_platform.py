"""
Test Platform Routes - Apple-Style UI

Clean, minimal design inspired by Apple's Human Interface Guidelines:
- Light mode with subtle grays
- SF Pro-like typography (Inter)
- Generous whitespace
- Subtle shadows and rounded corners
"""

import json
import logging
from typing import Optional

import redis
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/test", tags=["test"])

# =============================================================================
# Demo Patient Data
# =============================================================================

DEMO_PATIENTS = [
    {
        "id": "P001",
        "name": "张三",
        "surgery_date": "2024-06-15",
        "product": "DBS_PD",
        "phone": "138****1001",
        "age": 58,
        "gender": "男",
        "days_post_op": 179,
    },
    {
        "id": "P002",
        "name": "李四",
        "surgery_date": "2024-08-20",
        "product": "VNS",
        "phone": "138****1002",
        "age": 45,
        "gender": "女",
        "days_post_op": 113,
    },
    {
        "id": "P003",
        "name": "王五",
        "surgery_date": "2024-09-10",
        "product": "DBS_PD",
        "phone": "138****1003",
        "age": 62,
        "gender": "男",
        "days_post_op": 92,
    },
    {
        "id": "P004",
        "name": "赵六",
        "surgery_date": "2024-10-05",
        "product": "SCS",
        "phone": "138****1004",
        "age": 51,
        "gender": "女",
        "days_post_op": 67,
    },
    {
        "id": "P005",
        "name": "钱七",
        "surgery_date": "2024-11-01",
        "product": "SNM",
        "phone": "138****1005",
        "age": 67,
        "gender": "男",
        "days_post_op": 40,
    },
]


def get_redis_client() -> Optional[redis.Redis]:
    try:
        import os

        return redis.Redis(
            host=os.environ.get("REDIS_HOST", "redis"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            db=int(os.environ.get("REDIS_DB", 0)),
            decode_responses=True,
        )
    except Exception:
        return None


def get_all_results() -> list[dict]:
    r = get_redis_client()
    if not r:
        return []
    try:
        results = []
        keys = r.keys("sop:call:*")
        for key in keys:
            data = r.get(key)
            if data:
                results.append(json.loads(data))
        results.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        return results[:10]
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        return []


# =============================================================================
# Apple-Style Dashboard
# =============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>品驰关爱中心</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            /* Apple-inspired light color system */
            --bg-primary: #ffffff;
            --bg-secondary: #f5f5f7;
            --bg-tertiary: #fbfbfd;

            --text-primary: #1d1d1f;
            --text-secondary: #86868b;
            --text-tertiary: #aeaeb2;

            --border-color: rgba(0, 0, 0, 0.08);
            --divider: rgba(0, 0, 0, 0.06);

            /* Apple blue accent */
            --accent: #007aff;
            --accent-hover: #0066d6;
            --accent-light: rgba(0, 122, 255, 0.1);

            /* Semantic colors */
            --green: #34c759;
            --orange: #ff9500;
            --red: #ff3b30;
            --purple: #af52de;
            --pink: #ff2d55;
            --teal: #5ac8fa;

            /* Shadows */
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
            --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
            --shadow-lg: 0 12px 40px rgba(0, 0, 0, 0.12);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
            background: var(--bg-secondary);
            min-height: 100vh;
            color: var(--text-primary);
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 48px 24px;
        }

        /* Header */
        .header {
            text-align: center;
            margin-bottom: 48px;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: var(--text-primary);
            margin-bottom: 8px;
        }

        .header p {
            font-size: 1.125rem;
            color: var(--text-secondary);
            font-weight: 400;
        }

        /* Stats Row */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 32px;
        }

        .stat-card {
            background: var(--bg-primary);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
        }

        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }

        .stat-label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        /* Main Card */
        .main-card {
            background: var(--bg-primary);
            border-radius: 20px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color);
            overflow: hidden;
        }

        .card-header {
            padding: 24px 28px;
            border-bottom: 1px solid var(--divider);
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        /* Patient List */
        .patient-list {
            padding: 8px 0;
        }

        .patient-row {
            display: flex;
            align-items: center;
            padding: 16px 28px;
            gap: 16px;
            cursor: pointer;
            transition: background 0.15s ease;
            border-bottom: 1px solid var(--divider);
        }

        .patient-row:last-child {
            border-bottom: none;
        }

        .patient-row:hover {
            background: var(--bg-tertiary);
        }

        .patient-row:active {
            background: var(--bg-secondary);
        }

        .avatar {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 1.125rem;
            color: white;
            flex-shrink: 0;
        }

        .avatar.blue { background: linear-gradient(180deg, #5ac8fa 0%, #007aff 100%); }
        .avatar.pink { background: linear-gradient(180deg, #ff6b9d 0%, #ff2d55 100%); }

        .patient-info {
            flex: 1;
            min-width: 0;
        }

        .patient-name {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 2px;
        }

        .patient-meta {
            font-size: 0.875rem;
            color: var(--text-secondary);
        }

        .product-badge {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 5px 12px;
            border-radius: 100px;
            background: var(--bg-secondary);
            color: var(--text-secondary);
        }

        .product-badge.dbs { background: rgba(0, 122, 255, 0.1); color: #007aff; }
        .product-badge.vns { background: rgba(52, 199, 89, 0.1); color: #34c759; }
        .product-badge.scs { background: rgba(255, 149, 0, 0.1); color: #ff9500; }
        .product-badge.snm { background: rgba(175, 82, 222, 0.1); color: #af52de; }

        .days-text {
            font-size: 0.875rem;
            color: var(--text-tertiary);
            margin-right: 16px;
        }

        .call-btn {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            border: none;
            background: var(--green);
            color: white;
            font-size: 1.25rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .call-btn:hover {
            transform: scale(1.08);
            box-shadow: 0 4px 16px rgba(52, 199, 89, 0.4);
        }

        .call-btn:active {
            transform: scale(0.96);
        }

        .chevron {
            color: var(--text-tertiary);
            font-size: 1.25rem;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .stats-row { grid-template-columns: repeat(2, 1fr); }
            .days-text { display: none; }
            .header h1 { font-size: 2rem; }
        }

        @media (max-width: 480px) {
            .stats-row { grid-template-columns: 1fr; }
            .product-badge { display: none; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>品驰关爱中心</h1>
            <p>术后随访管理平台</p>
        </header>

        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">5</div>
                <div class="stat-label">待随访</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="completedCount">0</div>
                <div class="stat-label">已完成</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">98%</div>
                <div class="stat-label">满意度</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">1.2s</div>
                <div class="stat-label">响应时间</div>
            </div>
        </div>

        <div class="main-card">
            <div class="card-header">
                <div class="card-title">患者列表</div>
            </div>
            <div class="patient-list" id="patientList"></div>
        </div>
    </div>

    <script>
        const patients = PATIENTS_DATA;
        const results = RESULTS_DATA;

        const patientList = document.getElementById('patientList');
        patients.forEach(p => {
            const avatarClass = p.gender === '女' ? 'pink' : 'blue';
            const productClass = p.product.toLowerCase().replace('_pd', '');
            patientList.innerHTML += `
                <div class="patient-row" onclick="location.href='/test/call/${p.id}'">
                    <div class="avatar ${avatarClass}">${p.name[0]}</div>
                    <div class="patient-info">
                        <div class="patient-name">${p.name}</div>
                        <div class="patient-meta">${p.id} · ${p.age}岁 · ${p.phone}</div>
                    </div>
                    <span class="product-badge ${productClass}">${p.product}</span>
                    <span class="days-text">术后 ${p.days_post_op} 天</span>
                    <button class="call-btn" onclick="event.stopPropagation(); location.href='/test/call/${p.id}'">📞</button>
                    <span class="chevron">›</span>
                </div>
            `;
        });

        if (results.length > 0) {
            document.getElementById('completedCount').textContent = results.filter(r => r.completed).length;
        }
    </script>
</body>
</html>
"""


# =============================================================================
# Apple-Style Call Interface
# =============================================================================

CALL_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>通话 - PATIENT_NAME</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f5f5f7;
            --text-primary: #1d1d1f;
            --text-secondary: #86868b;
            --border-color: rgba(0, 0, 0, 0.08);
            --accent: #007aff;
            --green: #34c759;
            --red: #ff3b30;
            --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-secondary);
            min-height: 100vh;
            color: var(--text-primary);
            -webkit-font-smoothing: antialiased;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 32px 24px;
        }

        .back-link {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            color: var(--accent);
            text-decoration: none;
            font-size: 1rem;
            font-weight: 500;
            margin-bottom: 24px;
        }

        .back-link:hover {
            text-decoration: underline;
        }

        .patient-card {
            background: var(--bg-primary);
            border-radius: 16px;
            padding: 24px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 24px;
        }

        .avatar-lg {
            width: 64px;
            height: 64px;
            border-radius: 50%;
            background: linear-gradient(180deg, #5ac8fa 0%, #007aff 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            font-weight: 600;
            color: white;
        }

        .patient-info h1 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .patient-meta {
            color: var(--text-secondary);
            font-size: 0.9375rem;
        }

        .call-section {
            background: var(--bg-primary);
            border-radius: 16px;
            padding: 48px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color);
            text-align: center;
            margin-bottom: 24px;
        }

        #status {
            font-size: 1.25rem;
            font-weight: 500;
            margin-bottom: 32px;
            color: var(--text-secondary);
        }

        #status.active { color: var(--green); }

        .call-buttons {
            display: flex;
            justify-content: center;
            gap: 32px;
        }

        .call-btn {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            font-size: 2rem;
            transition: all 0.2s ease;
        }

        .call-btn.answer {
            background: var(--green);
            color: white;
        }

        .call-btn.answer:hover:not(:disabled) {
            transform: scale(1.08);
            box-shadow: 0 8px 24px rgba(52, 199, 89, 0.4);
        }

        .call-btn.hangup {
            background: var(--red);
            color: white;
        }

        .call-btn.hangup:hover:not(:disabled) {
            transform: scale(1.08);
            box-shadow: 0 8px 24px rgba(255, 59, 48, 0.4);
        }

        .call-btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none !important;
            box-shadow: none !important;
        }

        .panels {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .panel {
            background: var(--bg-primary);
            border-radius: 16px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color);
            max-height: 360px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .panel-header {
            padding: 16px 20px;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }

        .panel-body {
            padding: 16px 20px;
            overflow-y: auto;
            flex: 1;
        }

        .message {
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 85%;
            margin-bottom: 10px;
            font-size: 0.9375rem;
            line-height: 1.4;
        }

        .message.user {
            background: var(--accent);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 6px;
        }

        .message.ai {
            background: var(--bg-secondary);
            color: var(--text-primary);
            border-bottom-left-radius: 6px;
        }

        .message-time {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 4px;
            opacity: 0.7;
        }

        .message.user .message-time { color: rgba(255,255,255,0.7); }

        .slot-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid var(--border-color);
        }

        .slot-item:last-child { border-bottom: none; }

        .slot-name { color: var(--text-secondary); }

        .slot-value {
            font-weight: 600;
            color: var(--accent);
        }

        /* === Formal Report Styles === */
        .report-section {
            margin-top: 24px;
        }

        .report-btn {
            width: 100%;
            padding: 14px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .report-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
        }

        .report-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.6);
            z-index: 1000;
            padding: 24px;
            overflow-y: auto;
        }

        .report-modal.active { display: flex; justify-content: center; align-items: flex-start; }

        .report-content {
            background: white;
            border-radius: 16px;
            max-width: 700px;
            width: 100%;
            box-shadow: 0 24px 48px rgba(0, 0, 0, 0.2);
            margin: 24px 0;
        }

        .report-header {
            background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
            color: white;
            padding: 24px;
            border-radius: 16px 16px 0 0;
            text-align: center;
        }

        .report-header h2 {
            font-size: 1.25rem;
            margin-bottom: 8px;
        }

        .report-header .subtitle {
            font-size: 0.875rem;
            opacity: 0.8;
        }

        .report-patient {
            background: #f5f7fa;
            padding: 16px 24px;
            border-bottom: 1px solid #e0e0e0;
        }

        .report-patient-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }

        .report-patient-item {
            display: flex;
            gap: 8px;
        }

        .report-patient-item .label {
            color: #666;
            font-size: 0.875rem;
        }

        .report-patient-item .value {
            font-weight: 600;
            font-size: 0.875rem;
        }

        .report-body {
            padding: 24px;
        }

        .report-table {
            width: 100%;
            border-collapse: collapse;
        }

        .report-table th,
        .report-table td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }

        .report-table th {
            background: #f5f7fa;
            font-weight: 600;
            color: #333;
            font-size: 0.875rem;
        }

        .report-table tr:last-child td {
            border-bottom: none;
        }

        .report-table .category {
            background: #e3f2fd;
            font-weight: 600;
            color: #1565c0;
        }

        .report-table .field-name {
            color: #555;
            font-size: 0.875rem;
            width: 40%;
        }

        .report-table .field-value {
            font-weight: 500;
            color: #1a1a1a;
        }

        .report-table .field-value.empty {
            color: #999;
            font-style: italic;
        }

        .report-footer {
            padding: 16px 24px;
            background: #f5f7fa;
            border-radius: 0 0 16px 16px;
            display: flex;
            justify-content: space-between;
            gap: 12px;
        }

        .report-footer button {
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }

        .report-close {
            background: #e0e0e0;
            color: #333;
            border: none;
        }

        .report-print {
            background: #1565c0;
            color: white;
            border: none;
        }

        .report-print:hover { background: #0d47a1; }

        @media print {
            body * { visibility: hidden; }
            .report-modal, .report-modal * { visibility: visible; }
            .report-modal { position: absolute; left: 0; top: 0; background: white; }
            .report-footer { display: none; }
        }

        @media (max-width: 640px) {
            .panels { grid-template-columns: 1fr; }
            .report-patient-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/test/" class="back-link">‹ 返回</a>

        <div class="patient-card">
            <div class="avatar-lg">PATIENT_INITIAL</div>
            <div class="patient-info">
                <h1>PATIENT_NAME</h1>
                <div class="patient-meta">PATIENT_PRODUCT · 手术日期 PATIENT_SURGERY_DATE</div>
            </div>
        </div>

        <div class="call-section">
            <div id="status">点击接听开始通话</div>
            <div class="call-buttons">
                <button id="answerBtn" class="call-btn answer" onclick="startCall()">📞</button>
                <button id="hangupBtn" class="call-btn hangup" onclick="endCall()" disabled>📵</button>
            </div>
        </div>

        <div class="panels">
            <div class="panel">
                <div class="panel-header">对话记录</div>
                <div class="panel-body" id="transcript"></div>
            </div>
            <div class="panel">
                <div class="panel-header">收集的信息</div>
                <div class="panel-body" id="slots">
                    <div class="slot-item" style="color: var(--text-secondary); opacity: 0.6;">等待通话开始...</div>
                </div>
            </div>
        </div>

        <!-- Report Section -->
        <div class="report-section">
            <button class="report-btn" onclick="generateReport()">
                📋 生成随访报告
            </button>
        </div>
    </div>

    <!-- Report Modal -->
    <div class="report-modal" id="reportModal">
        <div class="report-content">
            <div class="report-header">
                <h2>品驰医疗 · 术后随访报告</h2>
                <div class="subtitle">Post-Operative Follow-up Report</div>
            </div>
            <div class="report-patient">
                <div class="report-patient-grid">
                    <div class="report-patient-item">
                        <span class="label">患者姓名：</span>
                        <span class="value" id="reportPatientName">PATIENT_NAME</span>
                    </div>
                    <div class="report-patient-item">
                        <span class="label">产品线：</span>
                        <span class="value" id="reportProductLine">PATIENT_PRODUCT</span>
                    </div>
                    <div class="report-patient-item">
                        <span class="label">随访日期：</span>
                        <span class="value" id="reportDate"></span>
                    </div>
                    <div class="report-patient-item">
                        <span class="label">随访方式：</span>
                        <span class="value">电话随访 (AI)</span>
                    </div>
                </div>
            </div>
            <div class="report-body">
                <table class="report-table">
                    <thead>
                        <tr>
                            <th colspan="2">随访信息汇总</th>
                        </tr>
                    </thead>
                    <tbody id="reportTableBody">
                        <!-- Filled by JS -->
                    </tbody>
                </table>
            </div>
            <div class="report-footer">
                <button class="report-close" onclick="closeReport()">关闭</button>
                <button class="report-print" onclick="window.print()">🖨️ 打印报告</button>
            </div>
        </div>
    </div>


    <script>
        const patientId = 'PATIENT_ID';
        const patientName = 'PATIENT_NAME';
        const productLine = 'PATIENT_PRODUCT';

        let ws = null;
        let audioContext = null;
        let playbackCtx = null;
        let processor = null;
        let micStream = null;

        // === P0 Fix: Echo prevention ===
        let isAISpeaking = false;  // Mute mic when AI speaks

        // === P0 Fix: Audio queue for sequential playback ===
        const audioQueue = [];
        let isPlaying = false;

        function startCall() {
            document.getElementById('status').textContent = '连接中...';
            document.getElementById('answerBtn').disabled = true;

            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/streaming/ws?patient_id=${patientId}&patient_name=${encodeURIComponent(patientName)}&product_line=${productLine}`);

            ws.onopen = () => {
                document.getElementById('status').textContent = '通话中';
                document.getElementById('status').className = 'active';
                document.getElementById('hangupBtn').disabled = false;
                startRecording();
            };

            ws.onmessage = async (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'user_text') {
                    addMessage('user', data.text);
                }
                else if (data.type === 'ai_text') {
                    addMessage('ai', data.text);
                }
                else if (data.type === 'audio') {
                    // Queue audio instead of playing immediately
                    queueAudio(data.data);
                }
                else if (data.type === 'speaking') {
                    // AI started/stopped speaking
                    isAISpeaking = data.value;
                    console.log('[Audio] AI speaking:', isAISpeaking);
                    if (!isAISpeaking) {
                        // AI stopped - clear any remaining queue
                        // audioQueue.length = 0;
                    }
                }
                else if (data.type === 'audio_end') {
                    // AI finished this response
                    console.log('[Audio] Response complete');
                }
                else if (data.type === 'slot_update') {
                    updateSlot(data.name, data.value);
                }
            };

            ws.onclose = () => {
                document.getElementById('status').textContent = '通话已结束';
                document.getElementById('status').className = '';
                stopRecording();
                isAISpeaking = false;
            };
        }

        function endCall() {
            if (ws) ws.close();
            stopRecording();
            document.getElementById('status').textContent = '通话已结束';
            document.getElementById('status').className = '';
            document.getElementById('hangupBtn').disabled = true;
            isAISpeaking = false;
            audioQueue.length = 0;
        }

        async function startRecording() {
            try {
                micStream = await navigator.mediaDevices.getUserMedia({ audio: true });

                // Create AudioContext WITHOUT specifying sampleRate - let browser use native rate
                audioContext = new AudioContext();
                const deviceRate = audioContext.sampleRate;
                const targetRate = 16000;
                console.log('[Mic] Device sample rate:', deviceRate, '-> resampling to', targetRate);

                const source = audioContext.createMediaStreamSource(micStream);

                // Use larger buffer for better resampling (8192 samples)
                processor = audioContext.createScriptProcessor(8192, 1, 1);

                // Resampling function - linear interpolation
                function resample(inputBuffer, fromRate, toRate) {
                    const ratio = fromRate / toRate;
                    const outputLength = Math.floor(inputBuffer.length / ratio);
                    const output = new Float32Array(outputLength);

                    for (let i = 0; i < outputLength; i++) {
                        const srcIndex = i * ratio;
                        const srcIndexFloor = Math.floor(srcIndex);
                        const srcIndexCeil = Math.min(srcIndexFloor + 1, inputBuffer.length - 1);
                        const t = srcIndex - srcIndexFloor;

                        // Linear interpolation
                        output[i] = inputBuffer[srcIndexFloor] * (1 - t) + inputBuffer[srcIndexCeil] * t;
                    }
                    return output;
                }

                processor.onaudioprocess = (e) => {
                    // Don't send audio while AI is speaking (prevents echo)
                    if (isAISpeaking) {
                        return;
                    }

                    if (ws && ws.readyState === WebSocket.OPEN) {
                        const inputFloat32 = e.inputBuffer.getChannelData(0);

                        // Resample from device rate to 16kHz
                        let float32;
                        if (deviceRate !== targetRate) {
                            float32 = resample(inputFloat32, deviceRate, targetRate);
                        } else {
                            float32 = inputFloat32;
                        }

                        // Convert to 16-bit PCM
                        const int16 = new Int16Array(float32.length);
                        for (let i = 0; i < float32.length; i++) {
                            int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
                        }

                        const base64 = btoa(String.fromCharCode(...new Uint8Array(int16.buffer)));
                        ws.send(JSON.stringify({
                            type: 'user_audio',
                            data: base64,
                            sample_rate: targetRate  // Always 16kHz after resampling
                        }));
                    }
                };

                source.connect(processor);
                processor.connect(audioContext.destination);
            } catch (err) {
                console.error('[Mic] Error:', err);
                document.getElementById('status').textContent = '麦克风权限被拒绝';
            }
        }

        function stopRecording() {
            if (processor) {
                processor.disconnect();
                processor = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            if (micStream) {
                micStream.getTracks().forEach(track => track.stop());
                micStream = null;
            }
        }

        // === P0 Fix: Audio queue for sequential playback ===
        function queueAudio(base64) {
            audioQueue.push(base64);
            if (!isPlaying) {
                playNextInQueue();
            }
        }

        async function playNextInQueue() {
            if (audioQueue.length === 0) {
                isPlaying = false;
                return;
            }

            isPlaying = true;
            const base64 = audioQueue.shift();

            try {
                if (!playbackCtx) {
                    playbackCtx = new AudioContext({ sampleRate: 24000 });
                }
                if (playbackCtx.state === 'suspended') {
                    await playbackCtx.resume();
                }

                const binary = atob(base64);
                const bytes = new Uint8Array(binary.length);
                for (let i = 0; i < binary.length; i++) {
                    bytes[i] = binary.charCodeAt(i);
                }

                const int16 = new Int16Array(bytes.buffer);
                const float32 = new Float32Array(int16.length);
                for (let i = 0; i < int16.length; i++) {
                    float32[i] = int16[i] / 32768;
                }

                const buffer = playbackCtx.createBuffer(1, float32.length, 24000);
                buffer.copyToChannel(float32, 0);

                const source = playbackCtx.createBufferSource();
                source.buffer = buffer;
                source.connect(playbackCtx.destination);

                // Play next chunk when this one ends
                source.onended = () => {
                    playNextInQueue();
                };

                source.start();
            } catch (err) {
                console.error('[Audio] Playback error:', err);
                // Continue with next chunk even on error
                playNextInQueue();
            }
        }

        function addMessage(role, text) {
            const div = document.getElementById('transcript');
            div.innerHTML += `<div class="message ${role}">${text}<div class="message-time">${new Date().toLocaleTimeString()}</div></div>`;
            div.scrollTop = div.scrollHeight;
        }

        function updateSlot(name, value) {
            const div = document.getElementById('slots');
            div.querySelector('[style]')?.remove();
            const existing = div.querySelector(`[data-slot="${name}"]`);
            if (existing) {
                existing.querySelector('.slot-value').textContent = value;
            } else {
                div.innerHTML += `<div class="slot-item" data-slot="${name}"><span class="slot-name">${name}</span><span class="slot-value">${value}</span></div>`;
            }
        }

        // === Report Generation ===
        const slotLabels = {
            'is_patient': '接听人',
            'symptom_improvement': '症状改善情况',
            'control_score': '症状控制评分',
            'life_quality_score': '生活质量评分',
            'programming_count': '程控次数',
            'programming_satisfaction': '程控满意度',
            'side_effects': '不良反应/副作用',
            'mental_issues': '情绪/心理问题',
            'medication_issues': '用药情况',
            'pre_op_informed': '术前告知情况',
            'device_explained': '设备讲解情况',
            'id_card_explained': '识别卡讲解情况',
            'insurance_type': '医保类型',
            'total_cost': '手术总费用',
            'self_pay': '自费金额',
            'huimin_bao': '惠民保购买',
            'other_concerns': '其他问题/诉求'
        };

        const slotCategories = {
            '术后效果': ['symptom_improvement', 'control_score', 'life_quality_score', 'side_effects', 'mental_issues'],
            '程控与用药': ['programming_count', 'programming_satisfaction', 'medication_issues'],
            '术前与出院指导': ['pre_op_informed', 'device_explained', 'id_card_explained'],
            '费用与医保': ['insurance_type', 'total_cost', 'self_pay', 'huimin_bao'],
            '其他': ['is_patient', 'other_concerns']
        };

        function generateReport() {
            // Set date
            document.getElementById('reportDate').textContent = new Date().toLocaleDateString('zh-CN');

            // Collect all slots
            const slots = {};
            document.querySelectorAll('#slots .slot-item').forEach(item => {
                const name = item.dataset.slot;
                const value = item.querySelector('.slot-value')?.textContent;
                if (name && value) slots[name] = value;
            });

            // Build table HTML
            let tableHtml = '';
            for (const [category, fields] of Object.entries(slotCategories)) {
                tableHtml += `<tr><td colspan="2" class="category">${category}</td></tr>`;
                for (const field of fields) {
                    const label = slotLabels[field] || field;
                    const value = slots[field];
                    const valueClass = value ? 'field-value' : 'field-value empty';
                    const displayValue = value || '未收集';
                    tableHtml += `<tr><td class="field-name">${label}</td><td class="${valueClass}">${displayValue}</td></tr>`;
                }
            }

            document.getElementById('reportTableBody').innerHTML = tableHtml;
            document.getElementById('reportModal').classList.add('active');
        }

        function closeReport() {
            document.getElementById('reportModal').classList.remove('active');
        }

        // Close modal on backdrop click
        document.getElementById('reportModal').addEventListener('click', (e) => {
            if (e.target.id === 'reportModal') closeReport();
        });
    </script>
</body>
</html>
"""


# =============================================================================
# Results Page
# =============================================================================

RESULTS_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>通话详情</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f5f5f7;
            --text-primary: #1d1d1f;
            --text-secondary: #86868b;
            --border-color: rgba(0, 0, 0, 0.08);
            --accent: #007aff;
            --green: #34c759;
            --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-secondary);
            min-height: 100vh;
            color: var(--text-primary);
            padding: 32px 24px;
        }
        .container { max-width: 640px; margin: 0 auto; }
        .back-link {
            display: inline-flex;
            align-items: center;
            color: var(--accent);
            text-decoration: none;
            font-weight: 500;
            margin-bottom: 24px;
        }
        .back-link:hover { text-decoration: underline; }
        .card {
            background: var(--bg-primary);
            border-radius: 16px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color);
            margin-bottom: 20px;
            overflow: hidden;
        }
        .card-header {
            padding: 16px 20px;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
        }
        .card-body { padding: 20px; }
        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .info-item {}
        .info-label { font-size: 0.8125rem; color: var(--text-secondary); margin-bottom: 4px; }
        .info-value { font-weight: 600; }
        .slot-row {
            display: flex;
            justify-content: space-between;
            padding: 14px 0;
            border-bottom: 1px solid var(--border-color);
        }
        .slot-row:last-child { border-bottom: none; }
        .slot-name { color: var(--text-secondary); }
        .slot-value { font-weight: 600; color: var(--accent); }
    </style>
</head>
<body>
    <div class="container">
        <a href="/test/" class="back-link">‹ 返回</a>

        <div class="card">
            <div class="card-header">通话信息</div>
            <div class="card-body">
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">患者ID</div>
                        <div class="info-value">PATIENT_ID</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">通话ID</div>
                        <div class="info-value" style="font-size: 0.875rem;">CALL_ID</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">开始时间</div>
                        <div class="info-value">STARTED_AT</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">状态</div>
                        <div class="info-value" style="color: var(--green);">FINAL_STATUS</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">收集的信息</div>
            <div class="card-body">SLOTS_ROWS</div>
        </div>
    </div>
</body>
</html>
"""


# =============================================================================
# Routes
# =============================================================================


@router.get("/", response_class=HTMLResponse)
async def dashboard():
    results = get_all_results()
    html = DASHBOARD_HTML.replace(
        "PATIENTS_DATA", json.dumps(DEMO_PATIENTS, ensure_ascii=False)
    ).replace("RESULTS_DATA", json.dumps(results, ensure_ascii=False, default=str))
    return HTMLResponse(content=html)


@router.get("/patients")
async def get_patients():
    return DEMO_PATIENTS


@router.get("/call/{patient_id}", response_class=HTMLResponse)
async def call_page(patient_id: str):
    patient = next((p for p in DEMO_PATIENTS if p["id"] == patient_id), None)
    if not patient:
        return HTMLResponse(content="Patient not found", status_code=404)

    html = CALL_HTML
    html = html.replace("PATIENT_ID", patient["id"])
    html = html.replace("PATIENT_NAME", patient["name"])
    html = html.replace("PATIENT_INITIAL", patient["name"][0])
    html = html.replace("PATIENT_PRODUCT", patient["product"])
    html = html.replace("PATIENT_SURGERY_DATE", patient["surgery_date"])
    html = html.replace("PATIENT_PHONE", patient["phone"])
    return HTMLResponse(content=html)


@router.get("/results/{call_id}", response_class=HTMLResponse)
async def results_page(call_id: str):
    r = get_redis_client()
    if not r:
        return HTMLResponse(content="Redis not available", status_code=500)

    data = r.get(f"sop:call:{call_id}")
    if not data:
        return HTMLResponse(content="Call not found", status_code=404)

    call_data = json.loads(data)
    slots = call_data.get("slots_collected", {})

    slots_html = ""
    for name, value in slots.items():
        slots_html += f'<div class="slot-row"><span class="slot-name">{name}</span><span class="slot-value">{value}</span></div>'
    if not slots_html:
        slots_html = '<div style="color: var(--text-secondary);">暂无数据</div>'

    html = RESULTS_HTML
    html = html.replace("PATIENT_ID", call_data.get("patient_id", "-"))
    html = html.replace("CALL_ID", call_data.get("call_id", "-"))
    html = html.replace(
        "STARTED_AT", str(call_data.get("started_at", "-"))[:19].replace("T", " ")
    )
    html = html.replace("FINAL_STATUS", call_data.get("final_status", "进行中"))
    html = html.replace("SLOTS_ROWS", slots_html)
    return HTMLResponse(content=html)
