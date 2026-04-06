#!/usr/bin/env python3
"""
Identity Support Bot — Comprehensive Analysis v4
New: Full engineer resolution analysis per cluster (all messages, pattern detection, automation rec),
     Week-by-week deep dive tabs (click week → see all intents with clusters for that week).
"""

import pandas as pd
import numpy as np
import json, re, html as html_lib
from collections import Counter, defaultdict
from datetime import datetime

CSV_PATH = "/Users/sshivarudra/Documents/support_insights/bot_insights_guide/data/identity_summary_data_march.csv"
OUTPUT_PATH = "/Users/sshivarudra/Documents/support_insights/bot_insights_guide/data/support_bot_report_v4.html"

# ── Loading ───────────────────────────────────────────────────────────────────
def load_and_prep(csv_path=None):
    path = csv_path or CSV_PATH
    df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Rename ai_orchestrator_trace → ai_orchestration_trace
    if 'ai_orchestrator_trace' in df.columns and 'ai_orchestration_trace' not in df.columns:
        df = df.rename(columns={'ai_orchestrator_trace': 'ai_orchestration_trace'})

    # Booleans
    for col in ['escalated', 'faq_rendered', 'prefiltered', 'rca_human_available', 'is_ambiguous', 'rca_llm_available']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper().map({
                'TRUE': True, 'FALSE': False, 'YES': True, 'NO': False,
                '1': True, '0': False, 'NAN': False, 'NONE': False, '': False
            }).fillna(False)

    # Weeks
    df['_wdate'] = pd.to_datetime(df['week_start'], errors='coerce', dayfirst=False)
    unique_weeks = sorted(df['_wdate'].dropna().unique())
    wmap = {w: i+1 for i, w in enumerate(unique_weeks)}
    df['week_seq'] = df['_wdate'].map(wmap).fillna(1).astype(int)
    df['week_label'] = df['_wdate'].apply(
        lambda d: f"Wk {wmap.get(d,'?')}: {d.strftime('%b %d')}" if pd.notna(d) else 'Unknown'
    )
    df['week_date_str'] = df['_wdate'].apply(
        lambda d: d.strftime('%b %d') if pd.notna(d) else '?'
    )

    # Intent normalization
    IMAP = {'howTo': 'how-to', 'troubleshooting': 'troubleshooting',
            'accessRequest': 'access-request', 'notification': 'notification'}
    df['intent_norm'] = df['intent_category'].apply(
        lambda x: IMAP.get(str(x).strip(), 'enquiry/other')
    )

    df['prefilter_action'] = df['prefilter_action'].fillna('') if 'prefilter_action' in df.columns else ''

    # Parse JSON columns
    for col in ['conversation_history', 'ai_orchestration_trace']:
        if col in df.columns:
            df[f'{col}_parsed'] = df[col].apply(safe_json)

    return df, wmap

def safe_json(val):
    if pd.isna(val) or str(val).strip() in ('', 'nan', 'None', '[]', '{}'):
        return None
    if isinstance(val, (dict, list)):
        return val
    s = str(val).strip()
    try:
        return json.loads(s)
    except:
        try:
            if s.startswith('"') and s.endswith('"'):
                return json.loads(s[1:-1].replace('""', '"'))
        except:
            pass
        return None

def esc(text):
    """HTML-escape."""
    return html_lib.escape(str(text or ''), quote=True)

def strip_pii(text):
    if not text:
        return ''
    text = re.sub(r'<@[A-Z0-9]+>', '[USER]', text)
    text = re.sub(r'<!subteam\^[^>]+>', '[TEAM]', text)
    text = re.sub(r'<![^>]+>', '[MENTION]', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
    text = re.sub(r'https?://\S+', '[URL]', text)
    return text.strip()

def clean_question(q):
    """Remove Slack noise from question."""
    if not q or str(q) == 'nan':
        return ''
    q = re.sub(r'<!subteam\^[^>]+>', '', q)
    q = re.sub(r'<@[A-Z0-9]+>', '', q)
    q = re.sub(r'<https?://[^|>]+\|([^>]+)>', r'\1', q)
    q = re.sub(r'<https?://\S+>', '[URL]', q)
    return q.strip()[:400]

# ── Conversation parsing ──────────────────────────────────────────────────────
def get_first_user_and_question(conv_parsed):
    if not conv_parsed or not isinstance(conv_parsed, list):
        return None, ''
    for msg in conv_parsed:
        if isinstance(msg, dict) and msg.get('senderType') == 'user':
            return msg.get('senderName', ''), strip_pii(str(msg.get('content', '')))
    return None, ''

def get_engineer_messages(conv_parsed, original_user):
    """Messages from support engineers (different user from original requester)."""
    msgs = []
    if not conv_parsed or not isinstance(conv_parsed, list):
        return msgs
    for msg in conv_parsed:
        if not isinstance(msg, dict):
            continue
        stype = msg.get('senderType', '')
        sname = msg.get('senderName', '')
        mt = msg.get('messageType', '')
        content = str(msg.get('content', ''))
        if (stype == 'user' and sname and sname != original_user
                and sname not in ('', 'assistant')
                and mt not in ('IA_FEEDBACK_REQUEST',)):
            if content.strip():
                msgs.append({'name': sname, 'content': strip_pii(content), 'messageType': mt})
    return msgs

def parse_trace_steps(trace_parsed):
    """Parse orchestration trace into structured steps — full reasoning, no truncation."""
    steps = []
    if not trace_parsed or not isinstance(trace_parsed, list):
        return steps
    for i, step in enumerate(trace_parsed):
        if not isinstance(step, dict):
            continue
        s = {
            'index': i,
            'event_type': step.get('event_type', ''),
            'llm_status': step.get('llm_status', ''),
            'llm_reasoning': str(step.get('llm_reasoning', '') or ''),   # full, no truncation
            'turn_user_input': str(step.get('turn_user_input', '') or ''),  # full
            'final_disposition': step.get('final_disposition', ''),
            'pf_verdict': step.get('pf_verdict', ''),
            'pf_rule_hit': str(step.get('pf_rule_hit', '') or ''),
        }
        steps.append(s)
    return steps


def classify_escalation_root_cause(row, conv, trace_steps):
    """
    Classify the TRUE root cause of escalation using trace + conversation signals.

    Returns: (tag: str, explanation: str)

    Tags (in priority order):
      WRONG_CHANNEL        — Engineer redirected to another team/channel; not identity's domain
      PREFILTER_DIRECT_ESC — Prefilter directly escalated before bot attempted
      NO_KB_COVERAGE       — Bot searched KB, found nothing (NO_CONTEXT from first turn)
      BOT_OVER_CLARIFIED   — Bot found docs (READY) but entered clarification loop; frustrated user
      ANSWER_INSUFFICIENT  — Bot found docs, gave answer, user explicitly said insufficient
      KB_GAP_PARTIAL       — Bot found some docs, follow-up query hit NO_CONTEXT
      BOT_CHOSE_ESCALATE   — Bot reasoned to escalate despite having partial context (ESCALATE status)
      USER_DIRECT_ESC      — User explicitly clicked escalate without bot failure
    """
    events = [s['event_type'] for s in trace_steps]
    statuses = [s['llm_status'] for s in trace_steps]
    pf_verdicts = [s['pf_verdict'] for s in trace_steps]

    # ── Signal 1: Wrong channel / not identity's domain ──────────────────────
    # Engineer message redirects user to another channel or explicitly says not our domain
    if conv and isinstance(conv, list):
        # Find original requester
        orig_user = None
        for msg in conv:
            if isinstance(msg, dict) and msg.get('senderType') == 'user':
                mt = msg.get('messageType', '')
                if mt == 'ASKING_QUESTION':
                    orig_user = msg.get('senderName', '')
                    break
        for msg in conv:
            if not isinstance(msg, dict):
                continue
            stype = msg.get('senderType', '')
            sname = msg.get('senderName', '')
            mt = msg.get('messageType', '')
            if mt == 'IA_FEEDBACK_REQUEST':
                continue
            content = str(msg.get('content', '') or '').lower()
            # Engineer (different user) redirecting to another channel/team
            is_eng = stype == 'user' and sname and sname != orig_user
            if is_eng and any(kw in content for kw in [
                'reach out to #', 'directed to #', 'please check with',
                'not identity', 'not our domain', '#cmty-', 'for queries related to',
                'queries related to', 'questions on idps', 'please contact', 'owned by',
                'please raise with', 'wrong channel', 'not the right channel',
                'not in identity', 'belong to ', 'not belong here',
                'out of scope for identity', 'have queries related to',
                'if you have queries', 'please reach out to <#', 'reach out to <#',
                'contact the ', 'check with the ', 'owned by the ',
            ]):
                # Extract channel/team name if possible
                ch_match = re.search(r'<#\w+\|?([^>]*)>', content)
                ch_name = ch_match.group(1) if ch_match else ''
                detail = f'Engineer redirected to {ch_name or "another team/channel"}. Topic is outside Identity team scope.'
                return 'WRONG_CHANNEL', detail

    # ── Signal 2: Prefilter direct escalation ────────────────────────────────
    if 'SMART_PREFILTER_ESCALATE' in events and 'RESPOND' not in pf_verdicts:
        pf_rule = next((s['pf_rule_hit'] for s in trace_steps if s['pf_rule_hit']), '')
        detail = f'Prefilter rule triggered direct escalation without bot attempting answer. Rule: {pf_rule or "unknown"}'
        return 'PREFILTER_DIRECT_ESC', detail

    has_ready = 'READY' in statuses
    has_no_ctx = 'NO_CONTEXT' in statuses
    has_clarify = 'CLARIFY' in statuses
    has_escalate_status = 'ESCALATE' in statuses
    has_feedback_esc = 'FEEDBACK_ESCALATED' in events
    clarify_count = statuses.count('CLARIFY')

    # ── Signal 3: Bot found docs but over-clarified (bot platform issue) ─────
    # READY appeared, then CLARIFY loop, user frustrated → escalated
    if has_ready and has_clarify and clarify_count >= 2:
        reasoning = ' '.join(s['llm_reasoning'] for s in trace_steps if s['llm_status'] == 'CLARIFY')[:300]
        detail = f'Bot found relevant documents (READY) but entered a clarification loop ({clarify_count} CLARIFY turns) instead of answering. Bot asked for assetId, env, etc. when docs contained enough to answer. Bot platform issue: over-cautious clarification strategy.'
        return 'BOT_OVER_CLARIFIED', detail

    # ── Signal 4: User explicitly said answer was insufficient ───────────────
    if has_feedback_esc or has_ready:
        for step in trace_steps:
            tui = step['turn_user_input'].lower()
            if 'insufficient_answer' in tui or 'need_further_explanation' in tui:
                extra = re.search(r'additional context:\s*(.+)', step['turn_user_input'], re.IGNORECASE)
                extra_text = extra.group(1).strip()[:200] if extra else ''
                detail = f'Bot found docs and answered (READY), but user marked answer as insufficient. {("User added context: " + extra_text) if extra_text else ""} KB content may be incomplete or bot response too generic.'
                return 'ANSWER_INSUFFICIENT', detail

    # ── Signal 5: KB partial — READY then NO_CONTEXT on follow-up ───────────
    if has_ready and has_no_ctx:
        detail = 'Bot found documents and answered initial question (READY), but follow-up question hit NO_CONTEXT. KB covers the topic partially but lacks depth for multi-turn conversations.'
        return 'KB_GAP_PARTIAL', detail

    # ── Signal 6: No KB coverage — distinguish "docs found but insufficient" vs "no docs" ─────
    if has_no_ctx and not has_ready:
        reasoning = next((s['llm_reasoning'] for s in trace_steps if s['llm_status'] == 'NO_CONTEXT'), '')
        # Check if bot actually found and reviewed documents (shows in reasoning) but couldn't answer
        docs_reviewed = bool(reasoning) and any(kw in reasoning.lower() for kw in [
            'document', 'documents describe', 'doc 1', 'doc 2', 'reviewed provided',
            'none of the documents', 'the document', 'based on the document',
        ])
        if docs_reviewed:
            detail = f'Bot found and reviewed relevant documents but could not extract a specific answer (topic too narrow or question too specific for existing docs). Bot reasoning: {reasoning[:350]}'
            return 'KB_GAP_PARTIAL', detail
        else:
            detail = f'Bot searched KB but found no relevant documents (NO_CONTEXT). Topic is absent from the knowledge base. {("Bot reasoning: " + reasoning[:200]) if reasoning else ""}'
            return 'NO_KB_COVERAGE', detail

    # ── Signal 7: Bot chose to escalate with reasoning (ESCALATE status) ────
    if has_escalate_status:
        reasoning = next((s['llm_reasoning'][:300] for s in trace_steps if s['llm_status'] == 'ESCALATE'), '')
        detail = f'Bot decided to escalate based on its reasoning despite having some context available. {("Reasoning: " + reasoning) if reasoning else ""}'
        return 'BOT_CHOSE_ESCALATE', detail

    # ── Signal 8: User directly requested escalation ─────────────────────────
    for step in trace_steps:
        tui = step['turn_user_input'].lower()
        if 'escalate' in tui and len(tui.strip()) < 60:
            return 'USER_DIRECT_ESC', 'User directly clicked escalate or typed "escalate" without receiving a bot answer.'

    return 'OTHER', 'Could not determine specific root cause from trace and conversation signals.'

def classify_user_feedback(steps):
    """Classify why user escalated based on turn_user_input patterns."""
    for step in steps:
        tui = step['turn_user_input'].lower()
        if 'insufficient_answer' in tui:
            extra = re.search(r'additional context:\s*(.+)', tui, re.IGNORECASE)
            return 'insufficient_answer', extra.group(1).strip()[:150] if extra else ''
        if 'need_further_explanation' in tui:
            extra = re.search(r'additional context:\s*(.+)', tui, re.IGNORECASE)
            return 'need_further_explanation', extra.group(1).strip()[:150] if extra else ''
        if 'escalate' in tui and len(tui) < 50:
            return 'direct_escalation', step['turn_user_input']
    return '', ''

# ── How-To Classification ─────────────────────────────────────────────────────
def classify_howto(row):
    """
    Classify how-to escalation source. Categories:
    - faq_and_rag_then_no_context: faq_rendered=True, READY in trace, then NO_CONTEXT on follow-up
    - faq_only: faq_rendered=True, only NO_CONTEXT (no READY)
    - rag_only: faq=False, READY in trace, no NO_CONTEXT
    - rag_insufficient: READY in trace, then NO_CONTEXT (user found answer insufficient)
    - no_context: NO_CONTEXT from the start, no READY at all
    """
    faq = bool(row.get('faq_rendered', False))
    steps = parse_trace_steps(row.get('ai_orchestration_trace_parsed'))

    has_ready = any(s['llm_status'] == 'READY' for s in steps)
    has_no_ctx = any(s['llm_status'] == 'NO_CONTEXT' for s in steps)

    # Detect user dissatisfaction pattern
    feedback_type, feedback_detail = classify_user_feedback(steps)
    rag_then_nc = has_ready and has_no_ctx  # Bot answered but follow-up hit no context

    if faq and rag_then_nc:
        return 'faq_and_rag_then_no_context', feedback_type, feedback_detail
    elif faq and has_ready and not has_no_ctx:
        return 'faq_and_rag', feedback_type, feedback_detail
    elif faq and not has_ready:
        return 'faq_only', feedback_type, feedback_detail
    elif rag_then_nc:
        return 'rag_insufficient', feedback_type, feedback_detail
    elif has_ready and not has_no_ctx:
        return 'rag_only', feedback_type, feedback_detail
    else:
        return 'no_context', feedback_type, feedback_detail

# ── Weekly Summary ────────────────────────────────────────────────────────────
def compute_weekly_summary(df):
    rows = []
    for week, wdf in df.groupby('week_seq'):
        total = len(wdf)
        total_esc = int((wdf['escalated'] == True).sum())
        prefiltered = int(wdf['prefiltered'].sum())
        pf_rows = wdf[wdf['prefiltered'] == True]
        pf_esc = int((pf_rows['prefilter_action'].str.upper() == 'ESCALATED').sum())
        pf_skip = int((pf_rows['prefilter_action'].str.upper() == 'SKIPPED').sum())

        non_pf = wdf[wdf['prefiltered'] == False]
        pf_skip_rows = wdf[(wdf['prefiltered'] == True) & (wdf['prefilter_action'].str.upper() == 'SKIPPED')]
        bot_df = pd.concat([non_pf, pf_skip_rows])
        bot_att = len(bot_df)
        bot_res = int((bot_df['resolution_type'] == 'BOT').sum())
        res_rate = round(bot_res / bot_att * 100, 1) if bot_att > 0 else 0

        surv = int(wdf['rca_human_available'].sum()) if 'rca_human_available' in wdf.columns else 0
        surv_rate = round(surv / total_esc * 100, 1) if total_esc > 0 else 0
        faq_cnt = int(wdf['faq_rendered'].sum())

        rows.append({
            'week': int(week),
            'week_label': wdf['week_label'].iloc[0],
            'total': total, 'escalated': total_esc,
            'prefiltered': prefiltered, 'pf_escalated': pf_esc, 'pf_skipped': pf_skip,
            'bot_attempted': bot_att, 'bot_resolved': bot_res,
            'resolution_rate': res_rate,
            'survey_filled': surv, 'survey_rate': surv_rate,
            'faq_rendered': faq_cnt,
        })
    return rows

def compute_intent_weekly(df):
    """Per-intent, per-week breakdown."""
    result = {}
    for intent, idf in df.groupby('intent_norm'):
        weeks = []
        for week, widf in idf.groupby('week_seq'):
            total = len(widf)
            esc = int((widf['escalated'] == True).sum())
            bot = int((widf['resolution_type'] == 'BOT').sum())
            faq = int(widf['faq_rendered'].sum())
            wlabel = widf['week_label'].iloc[0]
            weeks.append({'week': int(week), 'week_label': wlabel,
                         'total': total, 'bot_resolved': bot, 'escalated': esc, 'faq_rendered': faq})
        result[intent] = weeks
    return result

# ── How-To Analysis ───────────────────────────────────────────────────────────
def analyze_howto(df):
    """Full how-to analysis with 5 sub-categories, per-week, and thread details."""
    howto_df = df[(df['intent_norm'] == 'how-to') & (df['escalated'] == True)].copy()
    if len(howto_df) == 0:
        return {}, {}, []

    # Classify each row
    sources, fb_types, fb_details = [], [], []
    for _, row in howto_df.iterrows():
        src, fbt, fbd = classify_howto(row)
        sources.append(src)
        fb_types.append(fbt)
        fb_details.append(fbd)

    howto_df['howto_source'] = sources
    howto_df['feedback_type'] = fb_types
    howto_df['feedback_detail'] = fb_details

    breakdown = howto_df['howto_source'].value_counts().to_dict()

    # Per-week breakdown
    weekly_breakdown = {}
    for week, wdf in howto_df.groupby('week_seq'):
        wlabel = wdf['week_label'].iloc[0]
        weekly_breakdown[int(week)] = {
            'week_label': wlabel,
            'by_source': wdf['howto_source'].value_counts().to_dict(),
            'total': len(wdf),
        }

    # Per-category thread details + engineer analysis
    category_details = []
    for source, grp in howto_df.groupby('howto_source'):
        threads = []
        all_eng_flat = []
        for _, row in grp.iterrows():
            conv = row.get('conversation_history_parsed')
            trace_steps = parse_trace_steps(row.get('ai_orchestration_trace_parsed'))
            orig_user, question = get_first_user_and_question(conv)
            eng_msgs = get_engineer_messages(conv, orig_user)
            fbt = row.get('feedback_type', '')
            fbd = row.get('feedback_detail', '')
            tid = str(row.get('threadid', ''))

            for m in eng_msgs:
                all_eng_flat.append({'thread_id': tid, 'name': m['name'], 'content': m['content']})

            # New heuristic root cause + all bot answers
            rc_tag, rc_detail = classify_escalation_root_cause(row, conv, trace_steps)
            bot_answers_full = get_all_bot_answers(conv)

            threads.append({
                'thread_id': tid,
                'slack_url': str(row.get('slack_thread_url', '')),
                'week_label': row.get('week_label', ''),
                'question': clean_question(question),
                'faq_rendered': bool(row.get('faq_rendered', False)),
                'faq_ids': str(row.get('faq_ids', '')),
                'feedback_type': fbt,
                'feedback_detail': fbd,
                'trace_steps': trace_steps,
                'engineer_msgs': [{'name': m['name'], 'content': m['content']} for m in eng_msgs],
                'bot_answers': bot_answers_full,
                'bot_answer': bot_answers_full[0] if bot_answers_full else '',
                'rca_summary': strip_pii(str(row.get('rca_llm_analysis_summary', '') or ''))[:400],
                'rca_correct': strip_pii(str(row.get('rca_correct_answer', '') or ''))[:300],
                'root_cause': rc_tag,
                'root_cause_detail': rc_detail,
            })

        eng_analysis = analyze_engineer_resolutions(all_eng_flat)
        category_details.append({
            'source': source,
            'count': len(grp),
            'threads': threads,
            'eng_analysis': eng_analysis,
        })

    return breakdown, weekly_breakdown, category_details

# ── Cluster Analysis ──────────────────────────────────────────────────────────
def extract_cluster_text(row):
    parts = []
    q = str(row.get('question', '') or '')
    q = re.sub(r'<[^>]+>', ' ', q)
    q = re.sub(r'https?://\S+', ' ', q)
    q = re.sub(r'```[\s\S]*?```', ' code ', q)
    if q.strip():
        parts.append(q[:500].lower())
    rca = str(row.get('rca_llm_analysis_summary', '') or '')
    if rca and rca != 'nan':
        parts.append(rca[:200].lower())
    rt = str(row.get('rca_llm_root_cause_tag', '') or '')
    if rt and rt != 'nan':
        parts.append(rt.replace('_', ' ').lower())
    return ' '.join(parts) if parts else 'no data'

def cluster_df(intent_df, min_k=2, max_k=8):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    texts = [extract_cluster_text(row) for _, row in intent_df.iterrows()]
    n = len(texts)
    if n < min_k:
        return [(0, 'All Issues', list(range(n)), -1)]

    vec = TfidfVectorizer(max_features=400, stop_words='english', ngram_range=(1, 2),
                          min_df=max(1, min(2, n // 15)), max_df=0.9)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return [(0, 'All Issues', list(range(n)), -1)]

    fn = vec.get_feature_names_out()
    best_k, best_sc = min_k, -1
    max_k_ = min(max_k, max(2, n - 1), n // 2 + 1)
    for k in range(min_k, max_k_ + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbs = km.fit_predict(X)
            if len(set(lbs)) > 1:
                sc = silhouette_score(X, lbs, sample_size=min(500, n))
                if sc > best_sc:
                    best_sc, best_k = sc, k
        except:
            pass

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    clusters = []
    for i in range(best_k):
        idxs = [j for j, l in enumerate(labels) if l == i]
        if idxs:
            mv = np.asarray(X[idxs].mean(axis=0)).flatten()
            top = [fn[t] for t in mv.argsort()[-5:][::-1]]
            lbl = ' / '.join(top[:3])
        else:
            lbl = f'Cluster {i+1}'
        clusters.append((i, lbl, idxs, round(best_sc, 3)))
    return sorted(clusters, key=lambda c: -len(c[2]))

def analyze_engineer_resolutions(all_eng_msgs):
    """
    Analyze ALL engineer messages across a cluster to extract resolution patterns.
    Returns a structured summary dict.
    """
    if not all_eng_msgs:
        return {
            'total_eng_threads': 0,
            'total_eng_messages': 0,
            'resolution_patterns': {},
            'sample_resolutions': [],
            'automation_signals': [],
        }

    all_text = ' '.join(m.get('content', '').lower() for m in all_eng_msgs)
    threads_with_eng = len({m.get('thread_id', i) for i, m in enumerate(all_eng_msgs)})

    # Pattern detection across ALL messages
    patterns = Counter()
    for m in all_eng_msgs:
        c = m.get('content', '').lower()
        if any(kw in c for kw in ['doc', 'wiki', 'confluence', 'link', 'guide', 'runbook', 'readme', 'article']):
            patterns['Shared documentation/links'] += 1
        if any(kw in c for kw in ['channel', 'reach out', 'contact', 'team', 'slack', 'ping', 'message']):
            patterns['Routed to another team/channel'] += 1
        if any(kw in c for kw in ['ticket', 'jira', 'servicenow', 'request', 'raise', 'file']):
            patterns['Created/advised ticket'] += 1
        if any(kw in c for kw in ['access', 'permission', 'grant', 'role', 'entitlement', 'provisioning', 'okta', 'sso']):
            patterns['Resolved access/permissions'] += 1
        if any(kw in c for kw in ['config', 'setting', 'setup', 'configure', 'install', 'deploy']):
            patterns['Provided config/setup steps'] += 1
        if any(kw in c for kw in ['command', 'run', 'execute', 'script', 'cli', 'api call']):
            patterns['Provided CLI/API commands'] += 1
        if any(kw in c for kw in ['onboard', 'new hire', 'orientation', 'first day', 'getting started']):
            patterns['Handled onboarding task'] += 1
        if any(kw in c for kw in ['escalat', 'forward', 'loop in', 'cc', 'involve']):
            patterns['Escalated to specialist'] += 1
        if any(kw in c for kw in ['wait', 'pending', 'investigating', 'checking', 'looking into']):
            patterns['Actively investigated/researched'] += 1

    # Automation signals (what the bot could do instead)
    automation = []
    if patterns.get('Shared documentation/links', 0) > 0:
        automation.append(f"📄 Add linked docs/wikis to RAG knowledge base ({patterns['Shared documentation/links']} messages referenced docs) — bot can serve these directly.")
    if patterns.get('Routed to another team/channel', 0) > 0:
        automation.append(f"🔀 Build smart routing: bot can detect these issue types and route to correct team without engineer involvement ({patterns['Routed to another team/channel']} routing resolutions).")
    if patterns.get('Resolved access/permissions', 0) > 0:
        automation.append(f"🔐 Automate access provisioning flow: {patterns['Resolved access/permissions']} resolutions involved access/permissions. Bot can collect details and trigger automated provisioning or submit to access management queue.")
    if patterns.get('Created/advised ticket', 0) > 0:
        automation.append(f"🎫 Auto-create tickets: {patterns['Created/advised ticket']} resolutions involved ticketing. Bot can intake details and auto-file tickets without engineer hand-off.")
    if patterns.get('Provided config/setup steps', 0) > 0:
        automation.append(f"⚙️ Add step-by-step setup guides to KB ({patterns['Provided config/setup steps']} config resolutions) — bot can walk users through setup via structured conversation.")
    if patterns.get('Provided CLI/API commands', 0) > 0:
        automation.append(f"💻 Document CLI/API commands in KB ({patterns['Provided CLI/API commands']} command-based resolutions) — bot can provide exact commands in response.")
    if patterns.get('Handled onboarding task', 0) > 0:
        automation.append(f"👋 Create an onboarding FAQ/workflow ({patterns['Handled onboarding task']} onboarding resolutions) — high-value automation for predictable repeat questions.")
    if not automation:
        automation.append("🔍 Review individual thread resolutions to identify patterns for KB content or workflow automation.")

    # Sample resolutions (up to 6 most informative messages)
    samples = sorted(all_eng_msgs, key=lambda m: -len(m.get('content', '')))[:6]

    return {
        'total_eng_threads': threads_with_eng,
        'total_eng_messages': len(all_eng_msgs),
        'resolution_patterns': dict(patterns),
        'sample_resolutions': samples,
        'automation_signals': automation,
    }


def get_all_bot_answers(conv):
    """Extract ALL bot PRESENTING_INFORMATION messages (full content) from conversation."""
    answers = []
    if not conv or not isinstance(conv, list):
        return answers
    for msg in conv:
        if isinstance(msg, dict) and msg.get('senderType') == 'assistant' and msg.get('messageType') == 'PRESENTING_INFORMATION':
            content = strip_pii(str(msg.get('content', '') or ''))
            if content.strip():
                answers.append(content)
    return answers


def build_cluster_data(intent_label, intent_df):
    """Build detailed cluster data with trace, engineer resolution, slack URLs."""
    if len(intent_df) == 0:
        return []

    clusters_raw = cluster_df(intent_df)
    rows_list = list(intent_df.iterrows())
    results = []

    for ci, cl_label, idxs, sil in clusters_raw:
        cl_rows = [rows_list[j][1] for j in idxs]

        threads = []
        gap_types = Counter()
        root_causes_new = Counter()   # our new heuristic tags
        bot_answers_agg = []
        all_eng_msgs_flat = []  # ALL engineer messages across ALL threads in cluster

        for row in cl_rows:
            conv = row.get('conversation_history_parsed')
            trace_steps = parse_trace_steps(row.get('ai_orchestration_trace_parsed'))
            orig_user, question = get_first_user_and_question(conv)
            eng_msgs = get_engineer_messages(conv, orig_user)
            thread_id = str(row.get('threadid', ''))

            # ALL bot answers from conversation history (full content)
            bot_answers_full = get_all_bot_answers(conv)
            if bot_answers_full and len(bot_answers_agg) < 3:
                bot_answers_agg.append(bot_answers_full[0])  # first answer for cluster sample

            # Collect ALL engineer messages (tag with thread_id for counting)
            for m in eng_msgs:
                all_eng_msgs_flat.append({
                    'thread_id': thread_id,
                    'name': m['name'],
                    'content': m['content'],
                })

            # Gaps from trace
            for step in trace_steps:
                ls = step['llm_status'].upper()
                tui = step['turn_user_input'].lower()
                if ls == 'NO_CONTEXT':
                    gap_types['Knowledge Gap (NO_CONTEXT)'] += 1
                elif ls == 'CLARIFY':
                    gap_types['Needed Clarification'] += 1
                elif 'turn limit' in step['llm_reasoning'].lower() or 'turn limit' in tui:
                    gap_types['Turn Limit Exceeded'] += 1
                elif ls == 'ESCALATE' and step['llm_reasoning']:
                    gap_types['Bot Chose to Escalate'] += 1

            # New heuristic root cause classification
            rc_tag, rc_detail = classify_escalation_root_cause(row, conv, trace_steps)
            root_causes_new[rc_tag] += 1

            # Thread record — include full bot answers and new root cause
            rca_s = strip_pii(str(row.get('rca_llm_analysis_summary', '') or ''))
            rca_c = strip_pii(str(row.get('rca_correct_answer', '') or ''))
            threads.append({
                'thread_id': thread_id,
                'slack_url': str(row.get('slack_thread_url', '')),
                'week_label': str(row.get('week_label', '')),
                'question': clean_question(question),
                'bot_answers': bot_answers_full,           # ALL bot answers, full content
                'bot_answer': bot_answers_full[0] if bot_answers_full else '',  # compat
                'trace_steps': trace_steps,
                'engineer_msgs': [{'name': m['name'], 'content': m['content']} for m in eng_msgs],
                'rca_summary': rca_s[:400] if rca_s and rca_s != 'nan' else '',
                'rca_correct': rca_c[:300] if rca_c and rca_c != 'nan' else '',
                'root_cause': rc_tag,
                'root_cause_detail': rc_detail,
                'faq_rendered': bool(row.get('faq_rendered', False)),
            })

        # Analyze ALL engineer messages across the entire cluster
        eng_analysis = analyze_engineer_resolutions(all_eng_msgs_flat)
        recommendation = make_recommendation_v4(
            intent_label, cl_label, gap_types, root_causes_new,
            eng_analysis, len(idxs)
        )

        results.append({
            'label': cl_label,
            'count': len(idxs),
            'silhouette': sil,
            'gap_types': dict(gap_types),
            'root_causes': dict(root_causes_new),   # now uses our heuristic tags
            'bot_answers': bot_answers_agg,
            'engineer_resolutions': eng_analysis.get('sample_resolutions', []),
            'eng_analysis': eng_analysis,
            'threads': threads,
            'recommendation': recommendation,
        })

    return results


def make_recommendation_v4(intent, label, gaps, root_causes, eng_analysis, count):
    """Generate rich recommendation using heuristic root causes and engineer analysis."""
    lines = []
    no_ctx = gaps.get('Knowledge Gap (NO_CONTEXT)', 0)
    turn_lim = gaps.get('Turn Limit Exceeded', 0)
    bot_esc = gaps.get('Bot Chose to Escalate', 0)
    clarify = gaps.get('Needed Clarification', 0)
    eng_threads = eng_analysis.get('total_eng_threads', 0)
    total_eng_msgs = eng_analysis.get('total_eng_messages', 0)

    # ── New heuristic root cause recommendations ──────────────────────────────
    wrong_ch = root_causes.get('WRONG_CHANNEL', 0)
    no_kb = root_causes.get('NO_KB_COVERAGE', 0)
    kb_partial = root_causes.get('KB_GAP_PARTIAL', 0)
    over_clarify = root_causes.get('BOT_OVER_CLARIFIED', 0)
    insuff = root_causes.get('ANSWER_INSUFFICIENT', 0)
    prefilter = root_causes.get('PREFILTER_DIRECT_ESC', 0)
    bot_chose = root_causes.get('BOT_CHOSE_ESCALATE', 0)

    if wrong_ch > 0:
        lines.append(f"🚫 Wrong Channel ({wrong_ch}/{count} threads): Questions redirected by engineers to another team/channel — these are out-of-scope for Identity. [Capability Team] Add channel routing rules so bot proactively redirects users, reducing engineer load.")
    if no_kb > count * 0.2:
        lines.append(f"📭 No KB Coverage ({no_kb}/{count} threads): Bot found zero relevant docs. [Capability Team] Identify the wikis/runbooks engineers used in these threads and index them. [Bot Platform] Run KB coverage audit against this cluster's top questions.")
    if kb_partial > 0:
        lines.append(f"📄 Partial KB Coverage ({kb_partial}/{count} threads): Bot answered initial question but follow-up hit NO_CONTEXT. [Capability Team] Expand existing articles with follow-up Q&A sections. [Bot Platform] Enable per-turn re-retrieval for multi-turn conversations.")
    if over_clarify > 0:
        lines.append(f"❓ Bot Over-Clarified ({over_clarify}/{count} threads): Bot found docs (READY) but looped on clarification questions (assetId, env, etc.) instead of answering. [Bot Platform] Tune clarification strategy — bot should give best-effort answer from docs + optionally ask for specifics, not block answer on clarification.")
    if insuff > 0:
        lines.append(f"⚠ Answer Insufficient ({insuff}/{count} threads): Bot found docs and answered, but user explicitly marked answer as insufficient or needed further explanation. [Capability Team] Review what engineers added beyond the bot's answer and enrich KB articles with that specificity.")
    if prefilter > 0:
        lines.append(f"⏭ Prefilter Direct Escalation ({prefilter}/{count} threads): Prefilter rule bypassed bot entirely. [Bot Platform] Review prefilter rules for this cluster — ensure real support queries aren't caught by announcement/notification rules.")
    if bot_chose > 0:
        lines.append(f"⚡ Bot Over-Escalated ({bot_chose}/{count} threads): Bot reasoned to escalate despite having some context. [Bot Platform] Implement 'best-effort answer + offer human' fallback instead of direct escalation when docs are available.")

    # ── Trace-based gap signals ───────────────────────────────────────────────
    if no_ctx > count * 0.3 and no_kb == 0:  # fallback if heuristic didn't fire
        lines.append(f"🔍 KB Gap: {no_ctx}/{count} trace steps had NO_CONTEXT status. Index the exact resources engineers shared into the RAG knowledge base.")
    if turn_lim > 0:
        lines.append(f"⏱ Turn Limit: Exceeded in {turn_lim} step(s). Increase turn budget or build a structured sub-flow for complex multi-step questions.")

    # ── Engineer resolution automation signals ────────────────────────────────
    for signal in eng_analysis.get('automation_signals', []):
        lines.append(signal)

    # Coverage stat
    if eng_threads > 0:
        lines.append(f"ℹ️ Based on {total_eng_msgs} engineer messages across {eng_threads}/{count} threads with engineer involvement.")

    if not lines:
        lines.append(f"Review the {count} threads in this cluster to identify bot KB and platform gaps for '{label}' topics.")

    return lines  # Return list so HTML can render as bullet points

# ── Themed Recommendations ────────────────────────────────────────────────────
def extract_themed_recommendations(howto_bd, ts_clusters, ar_clusters, notif_clusters, weekly):
    """
    Extract two recommendation lists by owner:
    - Bot Platform: RAG, turn budget, escalation threshold, intent classifier, KB indexing
    - Capability Team: KB content, runbooks, access automation, routing, documentation
    Returns {'bot_platform': [...], 'capability_team': [...]}
    Each item: {'theme': str, 'evidence': str, 'action': str, 'priority': 'HIGH'|'MEDIUM'|'LOW'}
    """
    bot_platform = []
    capability_team = []

    # ── Theme 1: RAG/KB coverage gap (NO_CONTEXT dominant) ──────────────────
    no_ctx = howto_bd.get('no_context', 0) + howto_bd.get('rag_insufficient', 0) + howto_bd.get('faq_and_rag_then_no_context', 0)
    howto_total = sum(howto_bd.values()) or 1
    if no_ctx > 0:
        bot_platform.append({
            'theme': 'RAG Knowledge Base Coverage',
            'evidence': f'{no_ctx}/{howto_total} how-to escalations hit NO_CONTEXT (bot found no relevant docs). This is the single biggest escalation driver.',
            'action': 'Audit the current KB index against top how-to question topics (account selectors, offline tickets, GraphQL endpoints, SSO config, onboarding). Re-index or add missing articles. Enable semantic similarity matching to improve recall.',
            'priority': 'HIGH',
        })
        capability_team.append({
            'theme': 'KB Article Creation — How-To Gaps',
            'evidence': f'{no_ctx} escalated how-to threads got no RAG docs. Engineers resolved by sharing wikis/runbooks that are not indexed.',
            'action': 'Identify the wikis, Confluence pages, and runbooks engineers linked in escalated threads (visible in engineer resolution patterns). Create or index these as KB articles so RAG can serve them.',
            'priority': 'HIGH',
        })

    # ── Theme 2: Multi-turn RAG (RAG insufficient) ───────────────────────────
    rag_insuff = howto_bd.get('rag_insufficient', 0)
    if rag_insuff > 0:
        bot_platform.append({
            'theme': 'Multi-Turn RAG Retrieval',
            'evidence': f'{rag_insuff} how-to threads: bot found docs on first turn, user said answer was insufficient, follow-up hit NO_CONTEXT. Bot does not re-retrieve on follow-up turns.',
            'action': 'Enable per-turn RAG retrieval so every user follow-up triggers a fresh document search. Add conversation context to follow-up queries so retrievals are relevant to the dialogue.',
            'priority': 'HIGH',
        })

    # ── Theme 3: Intent classification errors ────────────────────────────────
    all_clusters = ts_clusters + ar_clusters + notif_clusters
    total_intent_err = sum(cl.get('root_causes', {}).get('Intent Error', 0) for cl in all_clusters)
    if total_intent_err > 5:
        bot_platform.append({
            'theme': 'Intent Classifier Accuracy',
            'evidence': f'INTENT_ERROR root cause detected in {total_intent_err} threads across clusters. How-to and troubleshooting intents overlap (both use "how do I fix/use X" phrasing).',
            'action': 'Retrain intent classifier using this dataset\'s actual question distribution. Add how-to vs troubleshooting boundary examples. Review notification/enquiry classification — many "how do I" questions land in wrong intent.',
            'priority': 'HIGH',
        })

    # ── Theme 4: Escalation threshold tuning ────────────────────────────────
    total_bot_esc = sum(cl.get('gap_types', {}).get('Bot Chose to Escalate', 0) for cl in all_clusters)
    if total_bot_esc > 5:
        bot_platform.append({
            'theme': 'Escalation Threshold Tuning',
            'evidence': f'Bot chose to escalate in {total_bot_esc} threads even when partial context was available (READY status in trace). Bot confidence threshold may be too conservative.',
            'action': 'Implement "best-effort answer + offer human" fallback before auto-escalating. Lower escalation threshold for topics where engineer resolution shows simple routing or doc sharing.',
            'priority': 'MEDIUM',
        })

    # ── Theme 5: Engineer routing patterns → smart routing ──────────────────
    # Aggregate all routing-related resolutions across all clusters
    total_routed = sum(
        cl.get('eng_analysis', {}).get('resolution_patterns', {}).get('Routed to another team/channel', 0)
        for cl in all_clusters
    )
    if total_routed > 5:
        bot_platform.append({
            'theme': 'Smart Routing (Team/Channel Detection)',
            'evidence': f'{total_routed} engineer resolutions involved routing the user to a specific team or Slack channel. Engineers detected the correct owner without bot involvement.',
            'action': 'Build team ownership rules: detect issue patterns (offline jobs→data lake, SSO→IT/Okta, onboarding→HR, GraphQL→platform team) and route proactively without requiring escalation.',
            'priority': 'MEDIUM',
        })
        capability_team.append({
            'theme': 'Team Ownership Documentation',
            'evidence': f'{total_routed} threads were resolved by routing to another team. No documented routing map exists for the bot.',
            'action': 'Create and maintain a team ownership matrix: which topics belong to which team/channel. Contribute routing rules to the bot configuration so it can route without engineer involvement.',
            'priority': 'MEDIUM',
        })

    # ── Theme 6: Access provisioning automation ──────────────────────────────
    total_access = sum(
        cl.get('eng_analysis', {}).get('resolution_patterns', {}).get('Resolved access/permissions', 0)
        for cl in ar_clusters
    )
    if total_access > 3:
        bot_platform.append({
            'theme': 'Access Request Automation',
            'evidence': f'{total_access} access-request resolutions involved engineers granting/modifying access, permissions, roles, or SSO entitlements.',
            'action': 'Integrate with OKTA / access management API. Bot can collect access details (system, role, justification) and auto-submit provisioning requests or route to access management queue without engineer hand-off.',
            'priority': 'MEDIUM',
        })
        capability_team.append({
            'theme': 'Access Workflow Standardization',
            'evidence': f'{total_access} access-request threads needed engineer intervention for provisioning. Each engineer resolves differently (ticket vs direct grant vs routing).',
            'action': 'Standardize access request workflows: define which access types are self-service vs ticket-based vs engineer-required. Document SLAs and provisioning steps so the bot can set expectations.',
            'priority': 'MEDIUM',
        })

    # ── Theme 7: Doc/wiki sharing → KB content ───────────────────────────────
    total_docs = sum(
        cl.get('eng_analysis', {}).get('resolution_patterns', {}).get('Shared documentation/links', 0)
        for cl in all_clusters
    )
    if total_docs > 5:
        capability_team.append({
            'theme': 'Existing Docs/Wikis Not in KB',
            'evidence': f'{total_docs} engineer resolutions involved sharing documentation links, wikis, or Confluence pages that the bot could not find.',
            'action': 'Audit all wikis/Confluence pages shared by engineers in escalated threads. Index them into the RAG knowledge base. Ensure internal links are crawlable and up-to-date.',
            'priority': 'HIGH',
        })

    # ── Theme 8: Prefilter tuning ────────────────────────────────────────────
    total_pf = sum(w['prefiltered'] for w in weekly)
    total_pf_esc = sum(w['pf_escalated'] for w in weekly)
    if total_pf > 0 and total_pf_esc / total_pf > 0.7:
        bot_platform.append({
            'theme': 'Prefilter Rule Review',
            'evidence': f'{total_pf_esc}/{total_pf} prefiltered threads ({round(total_pf_esc/total_pf*100,0):.0f}%) were directly escalated by the prefilter. Some real support queries may be incorrectly caught by notification/announcement rules.',
            'action': 'Review prefilter rules with sample of prefiltered threads. Adjust rules to distinguish bot-answerable support queries from true notifications/announcements.',
            'priority': 'MEDIUM',
        })

    # ── Theme 9: Survey completion ───────────────────────────────────────────
    total_esc = sum(w['escalated'] for w in weekly)
    total_surv = sum(w['survey_filled'] for w in weekly)
    if total_esc > 0 and total_surv / total_esc < 0.1:
        capability_team.append({
            'theme': 'Post-Escalation RCA Surveys',
            'evidence': f'Only {total_surv}/{total_esc} escalations ({round(total_surv/total_esc*100,1)}%) have human-completed RCA surveys. Analysis relies almost entirely on LLM auto-analysis.',
            'action': 'Implement mandatory post-escalation survey for support engineers: root cause (1 of 5 tags), resolution method (1 of 6 types), and KB gap identified. Target 80%+ completion. This data will directly improve bot training.',
            'priority': 'HIGH',
        })

    # Sort by priority
    pri_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    bot_platform.sort(key=lambda x: pri_order[x['priority']])
    capability_team.sort(key=lambda x: pri_order[x['priority']])

    return {'bot_platform': bot_platform, 'capability_team': capability_team}


def render_themed_recommendations(themed):
    """Render themed recommendations HTML split by Bot Platform vs Capability Team."""
    def render_items(items):
        html = ''
        for item in items:
            badge_cls = f'badge-{item["priority"].lower()}'
            html += f'''
            <div class="themed-rec-item">
                <div class="rec-meta">
                    <span class="badge {badge_cls}">{esc_val(item["priority"])}</span>
                    <span class="rec-category">{esc_val(item["theme"])}</span>
                </div>
                <div class="themed-rec-body">
                    <p class="themed-rec-evidence"><strong>Evidence:</strong> {esc_val(item["evidence"])}</p>
                    <p class="themed-rec-action"><strong>Action:</strong> {esc_val(item["action"])}</p>
                </div>
            </div>'''
        return html or '<p class="text-muted text-sm">No recommendations for this group.</p>'

    bp_html = render_items(themed.get('bot_platform', []))
    ct_html = render_items(themed.get('capability_team', []))

    return f'''
    <div class="two-col" style="gap:24px">
        <div>
            <h3 class="themed-rec-group-title" style="color:var(--accent-light)">🤖 Bot Platform Team</h3>
            <p class="section-intro" style="margin-bottom:12px">Fixes to the bot engine, RAG pipeline, intent classifier, escalation threshold, and routing logic.</p>
            {bp_html}
        </div>
        <div>
            <h3 class="themed-rec-group-title" style="color:var(--success)">👥 Capability / Content Team</h3>
            <p class="section-intro" style="margin-bottom:12px">KB content to create, documentation to write, access workflows to standardize, and survey discipline.</p>
            {ct_html}
        </div>
    </div>'''


# ── HTML Generation ───────────────────────────────────────────────────────────
CHART_COLORS = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#f97316', '#84cc16']

def pct_color(rate):
    if rate >= 60: return 'var(--success)'
    if rate >= 30: return 'var(--warning)'
    return 'var(--danger)'

ROOT_CAUSE_META = {
    'WRONG_CHANNEL':        ('🚫', 'var(--warning)',  'Wrong Channel — Not Identity Scope'),
    'PREFILTER_DIRECT_ESC': ('⏭',  'var(--text-sec)', 'Prefilter Direct Escalation'),
    'NO_KB_COVERAGE':       ('📭', 'var(--danger)',   'No KB Coverage'),
    'BOT_OVER_CLARIFIED':   ('❓', 'var(--warning)',  'Bot Over-Clarified (Platform Issue)'),
    'ANSWER_INSUFFICIENT':  ('⚠',  'var(--warning)',  'Answer Insufficient'),
    'KB_GAP_PARTIAL':       ('📄', 'var(--info)',     'Partial KB Coverage'),
    'BOT_CHOSE_ESCALATE':   ('⚡', 'var(--danger)',   'Bot Chose to Escalate'),
    'USER_DIRECT_ESC':      ('👤', 'var(--text-sec)', 'User Directly Escalated'),
    'OTHER':                ('•',  'var(--text-muted)','Unclassified'),
}

# Groups for prioritized root cause view
ROOT_CAUSE_GROUPS = [
    {
        'id': 'kb_content',
        'label': '📚 KB / Content Gaps',
        'color': 'var(--danger)',
        'priority': 1,
        'description': 'The bot could not resolve because the knowledge base is missing, incomplete, or too shallow. Fix: Capability Team adds/expands KB articles.',
        'tags': ['NO_KB_COVERAGE', 'KB_GAP_PARTIAL'],
        'tagging_logic': {
            'NO_KB_COVERAGE': 'Detected when: trace status=NO_CONTEXT on first turn AND bot reasoning shows no relevant documents found (no "Document 1/2/3..." in reasoning). Means the topic is entirely absent from the KB.',
            'KB_GAP_PARTIAL': 'Detected when: trace has READY→NO_CONTEXT sequence (multi-turn), OR trace shows NO_CONTEXT but bot reasoning mentions reviewing specific documents that didn\'t contain the answer. Means the KB covers the topic broadly but lacks depth.',
        },
    },
    {
        'id': 'bot_platform',
        'label': '🤖 Bot Platform Issues',
        'color': 'var(--accent-light)',
        'priority': 2,
        'description': 'The bot had relevant information available but failed to use it correctly — over-clarified, over-escalated, or gave a too-generic answer. Fix: Bot Platform Team tunes strategy.',
        'tags': ['BOT_OVER_CLARIFIED', 'BOT_CHOSE_ESCALATE', 'ANSWER_INSUFFICIENT'],
        'tagging_logic': {
            'BOT_OVER_CLARIFIED': 'Detected when: trace has READY status (docs found) AND 2+ CLARIFY turns. Bot found docs but kept asking the user for assetId/env/other specifics instead of answering with what was available.',
            'BOT_CHOSE_ESCALATE': 'Detected when: trace has ESCALATE status in llm_status. Bot explicitly reasoned to escalate — typically triggered when confidence is below threshold despite having partial context.',
            'ANSWER_INSUFFICIENT': 'Detected when: user feedback in turn_user_input contains "insufficient_answer" or "need_further_explanation" AND trace shows READY or FEEDBACK_ESCALATED. Bot answered but user explicitly marked it as not good enough.',
        },
    },
    {
        'id': 'routing',
        'label': '🚦 Routing / Scope Issues',
        'color': 'var(--warning)',
        'priority': 3,
        'description': 'Question is out of scope for Identity support. Engineer redirected user to the correct team or channel. Fix: Build bot routing rules; or add channel scope guidance.',
        'tags': ['WRONG_CHANNEL'],
        'tagging_logic': {
            'WRONG_CHANNEL': 'Detected when: an engineer message (senderType=user, senderName≠original requester) contains routing keywords like "reach out to #...", "queries related to", "have queries related to", "if you have queries" pointing to another channel/team. Engineer is explicitly saying "this is not identity\'s domain."',
        },
    },
    {
        'id': 'bypass',
        'label': '⏩ Bypassed Bot (Prefilter / Direct)',
        'color': 'var(--text-sec)',
        'priority': 4,
        'description': 'Bot was never given a chance — prefilter or user directly escalated. These are not bot failures but reduce resolution rate metrics.',
        'tags': ['PREFILTER_DIRECT_ESC', 'USER_DIRECT_ESC'],
        'tagging_logic': {
            'PREFILTER_DIRECT_ESC': 'Detected when: trace event=SMART_PREFILTER_ESCALATE AND pf_verdict is not RESPOND. The prefilter rule matched before the bot could process the question.',
            'USER_DIRECT_ESC': 'Detected when: turn_user_input contains a short "escalate" command (<60 chars). User explicitly requested human escalation without waiting for a bot answer.',
        },
    },
    {
        'id': 'other',
        'label': '❓ Other / Unclassified',
        'color': 'var(--text-muted)',
        'priority': 5,
        'description': 'Could not classify from available trace and conversation signals.',
        'tags': ['OTHER'],
        'tagging_logic': {
            'OTHER': 'None of the above signals matched the trace or conversation history.',
        },
    },
]


def render_root_cause_summary(rc_counts_all, total_esc):
    """Render grouped + prioritized root cause summary with explanations and legend."""
    # Compute group totals
    tag_to_group = {}
    for grp in ROOT_CAUSE_GROUPS:
        for tag in grp['tags']:
            tag_to_group[tag] = grp

    group_totals = {grp['id']: 0 for grp in ROOT_CAUSE_GROUPS}
    for tag, cnt in rc_counts_all.items():
        grp = tag_to_group.get(tag)
        if grp:
            group_totals[grp['id']] += cnt

    html = '<div class="rc-summary">'

    # ── Priority bar — visual overview ───────────────────────────────────────
    html += '<div class="rc-priority-bar">'
    for grp in sorted(ROOT_CAUSE_GROUPS, key=lambda g: g['priority']):
        gcnt = group_totals.get(grp['id'], 0)
        if gcnt == 0:
            continue
        pct = round(gcnt / max(total_esc, 1) * 100)
        html += f'''<div class="rc-bar-seg" style="width:{pct}%;background:{grp["color"]}" title="{esc_val(grp["label"])}: {gcnt} ({pct}%)">
            <span class="rc-bar-label">{pct}%</span>
        </div>'''
    html += '</div>'
    html += f'<p class="text-muted text-sm" style="text-align:center;margin-top:4px">Proportion of {total_esc} escalated threads by root cause group</p>'

    # ── Group cards ──────────────────────────────────────────────────────────
    html += '<div class="rc-groups">'
    for grp in sorted(ROOT_CAUSE_GROUPS, key=lambda g: g['priority']):
        gcnt = group_totals.get(grp['id'], 0)
        pct = round(gcnt / max(total_esc, 1) * 100)
        color = grp['color']
        priority_badge = ['', '🔴 P1 — Critical', '🟠 P2 — High', '🟡 P3 — Medium', '⚪ P4 — Low', '⚫ P5 — Informational'][grp['priority']]

        # Tag breakdown within group
        tag_rows = ''
        for tag in grp['tags']:
            tag_cnt = rc_counts_all.get(tag, 0)
            if tag_cnt == 0:
                continue
            tag_pct = round(tag_cnt / max(gcnt, 1) * 100)
            t_icon, t_color, t_label = ROOT_CAUSE_META.get(tag, ('•', color, tag))
            logic = grp['tagging_logic'].get(tag, '')
            tag_rows += f'''
            <div class="rc-tag-row">
                <div class="rc-tag-header">
                    <span style="color:{t_color};font-size:.9rem">{t_icon}</span>
                    <span class="rc-tag-name" style="color:{t_color}">{esc_val(t_label)}</span>
                    <span class="rc-tag-count">{tag_cnt} threads ({tag_pct}% of group)</span>
                </div>
                {f'<p class="rc-tag-logic">{esc_val(logic)}</p>' if logic else ''}
            </div>'''

        if not tag_rows:
            continue

        html += f'''
        <div class="rc-group-card" style="border-left:4px solid {color}">
            <div class="rc-group-header">
                <div>
                    <span class="rc-group-label" style="color:{color}">{grp["label"]}</span>
                    <span class="rc-priority-label">{priority_badge}</span>
                </div>
                <div class="rc-group-count">
                    <span class="rc-group-num" style="color:{color}">{gcnt}</span>
                    <span class="rc-group-pct text-muted">{pct}% of escalations</span>
                </div>
            </div>
            <p class="rc-group-desc">{esc_val(grp["description"])}</p>
            <div class="rc-tag-list">{tag_rows}</div>
        </div>'''

    html += '</div>'  # rc-groups
    html += '</div>'  # rc-summary
    return html


def render_trace_html(trace_steps):
    """Render full orchestration trace as timeline HTML — full reasoning, no truncation."""
    if not trace_steps:
        return '<p class="text-muted text-sm">No trace data</p>'

    STATUS_COLOR = {
        'READY': 'var(--success)', 'NO_CONTEXT': 'var(--danger)',
        'CLARIFY': 'var(--warning)', 'ESCALATE': 'var(--danger)', '': 'var(--text-muted)',
    }
    EVENT_ICON = {
        'TURN_EXECUTE_TOOL': '🔍', 'TURN_ESCALATE': '🚨', 'TURN_CLARIFY': '❓',
        'FEEDBACK_ESCALATED': '📤', 'SMART_PREFILTER_SKIP': '⏭️',
        'SMART_PREFILTER_ESCALATE': '🚨',
    }
    html = '<div class="trace-timeline">'
    for step in trace_steps:
        et = step['event_type']
        ls = step['llm_status']
        color = STATUS_COLOR.get(ls, 'var(--text-muted)')
        icon = EVENT_ICON.get(et, '•')
        reasoning = esc(step['llm_reasoning'])   # full, no truncation
        tui = esc(step['turn_user_input'])         # full
        html += f'''
        <div class="trace-step">
            <div class="trace-step-header">
                <span class="trace-icon">{icon}</span>
                <span class="trace-event">{esc(et)}</span>
                {f'<span class="trace-status" style="color:{color}">{esc(ls)}</span>' if ls else ''}
            </div>
            {f'<div class="trace-user-input"><span class="trace-field-label">User Input:</span> {tui}</div>' if tui else ''}
            {f'<div class="trace-reasoning"><span class="trace-field-label">Bot Reasoning:</span> {reasoning}</div>' if reasoning else ''}
        </div>'''
    html += '</div>'
    return html


def render_thread_card(thread, collapsed=True):
    """Render a single thread card with full trace, all bot answers, root cause badge, engineer resolution."""
    slack_url = thread.get('slack_url', '')
    thread_id = thread.get('thread_id', '')
    question = esc(thread.get('question', ''))
    week = esc(thread.get('week_label', ''))
    trace_html = render_trace_html(thread.get('trace_steps', []))
    rca_s = esc(thread.get('rca_summary', ''))
    rca_c = esc(thread.get('rca_correct', ''))
    faq = thread.get('faq_rendered', False)

    # Root cause — new heuristic tag
    root_cause = thread.get('root_cause', '') or ''
    root_cause_detail = thread.get('root_cause_detail', '') or ''
    rc_icon, rc_color, rc_label = ROOT_CAUSE_META.get(root_cause, ('•', 'var(--text-muted)', root_cause or 'Unknown'))

    # ALL bot answers from conversation history (full content)
    bot_answers = thread.get('bot_answers', [])
    if not bot_answers and thread.get('bot_answer'):
        bot_answers = [thread['bot_answer']]
    bot_answers_html = ''
    if bot_answers:
        for i, ba in enumerate(bot_answers):
            turn_label = f'Bot Answer — Turn {i+1}' if len(bot_answers) > 1 else 'Bot Answer'
            bot_answers_html += f'<div style="margin-top:8px"><span class="field-label">{esc(turn_label)}</span><div class="bot-quote bot-answer-full">{esc(ba)}</div></div>'
    else:
        bot_answers_html = '<p class="text-muted text-sm" style="margin-top:8px">No bot answer in conversation history</p>'

    # Engineer messages — all, full content
    eng_html = ''
    for em in thread.get('engineer_msgs', []):
        eng_html += f'''
        <div class="eng-block">
            <span class="eng-name">{esc(em["name"])}</span>
            <p class="text-sm">{esc(em["content"])}</p>
        </div>'''
    if not eng_html:
        if rca_c and rca_c != 'nan':
            eng_html = f'<div class="eng-block"><span class="eng-name">[RCA — Correct Answer]</span><p class="text-sm">{rca_c}</p></div>'
        else:
            eng_html = '<p class="text-muted text-sm">No engineer messages captured</p>'

    slack_badge = ''
    if slack_url and slack_url != 'nan':
        slack_badge = f'<a href="{esc(slack_url)}" target="_blank" class="slack-link">↗ Slack Thread</a>'

    feedback_html = ''
    fb_type = thread.get('feedback_type', '')
    fb_detail = thread.get('feedback_detail', '')
    if fb_type:
        fb_label = {'insufficient_answer': '⚠ Insufficient Answer', 'need_further_explanation': '⚠ Needs Further Explanation', 'direct_escalation': '⚠ User Requested Escalation'}.get(fb_type, fb_type)
        feedback_html = f'<div class="feedback-badge">{esc(fb_label)}{(": " + esc(fb_detail)) if fb_detail else ""}</div>'

    content_state = 'active' if not collapsed else ''

    rc_badge_html = f'''
    <div class="rc-badge" style="border-left:3px solid {rc_color}">
        <span class="rc-icon">{rc_icon}</span>
        <div>
            <span class="rc-label" style="color:{rc_color}">{esc(rc_label)}</span>
            {f'<p class="rc-detail">{esc(root_cause_detail)}</p>' if root_cause_detail else ''}
        </div>
    </div>''' if root_cause else ''

    return f'''
    <div class="thread-card">
        <div class="collapsible-header" onclick="toggleCollapsible(this)">
            <div class="thread-card-header">
                <span class="week-pill">{week}</span>
                {f'<span class="faq-pill">FAQ</span>' if faq else ''}
                {f'<span class="rc-mini" style="color:{rc_color}" title="{esc(rc_label)}">{rc_icon}</span>' if root_cause else ''}
                <p class="thread-question">{question[:120]}{'...' if len(question)>120 else ''}</p>
            </div>
            <div style="display:flex;align-items:center;gap:8px;flex-shrink:0">
                {slack_badge}
                <span class="arrow">▶</span>
            </div>
        </div>
        <div class="collapsible-content {content_state}">
            {f'<div class="thread-full-question"><span class="field-label">Full Question</span><p class="sample-q">{question}</p></div>' if question else ''}
            {rc_badge_html}
            {feedback_html}
            {bot_answers_html}
            <div class="two-col" style="margin-top:14px">
                <div>
                    <span class="field-label">Orchestration Trace</span>
                    {trace_html}
                </div>
                <div>
                    <span class="field-label">Engineer Resolution</span>
                    {eng_html}
                    {f'<div style="margin-top:8px"><span class="field-label">LLM Analysis</span><p class="text-sm insight-summary">{rca_s}</p></div>' if rca_s and rca_s != 'nan' else ''}
                </div>
            </div>
        </div>
    </div>'''

def render_cluster_section_full(clusters, intent_label):
    if not clusters:
        return f'<div class="card"><p class="text-muted">No escalated {esc(intent_label)} threads.</p></div>'

    html = ''
    for ci, cl in enumerate(clusters):
        color = CHART_COLORS[ci % len(CHART_COLORS)]
        label = esc(cl['label'].title())
        count = cl['count']
        sil = cl.get('silhouette', -1)

        # Gap analysis
        gaps = cl.get('gap_types', {})
        total_gaps = sum(gaps.values()) or 1
        gap_html = ''
        for gap, cnt in sorted(gaps.items(), key=lambda x: -x[1]):
            pct = cnt / total_gaps * 100
            gap_html += f'''
            <div class="bar-row">
                <span class="bar-label">{esc(gap)}</span>
                <div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div>
                <span class="bar-val">{cnt}</span>
            </div>'''

        # Root causes — use new heuristic tags with icons and colors
        rc_html = ''
        for rc, cnt in sorted(cl.get('root_causes', {}).items(), key=lambda x: -x[1]):
            rc_icon, rc_color, rc_label = ROOT_CAUSE_META.get(rc, ('•', 'var(--text-muted)', rc))
            pct_rc = round(cnt / count * 100)
            rc_html += f'<span class="rc-tag" style="border-color:{rc_color};color:{rc_color}" title="{esc(rc_label)}">{rc_icon} {esc(rc_label)} <strong>{cnt}</strong> ({pct_rc}%)</span> '

        # Gap explanations
        gap_explain = ''
        if 'Knowledge Gap (NO_CONTEXT)' in gaps:
            gap_explain += f'<li><strong>Knowledge Gap (NO_CONTEXT) — {gaps["Knowledge Gap (NO_CONTEXT)"]} threads</strong>: The RAG system searched the knowledge base but found no relevant documents for the user\'s query. Bot had nothing to answer with.</li>'
        if 'Needed Clarification' in gaps:
            gap_explain += f'<li><strong>Needed Clarification — {gaps["Needed Clarification"]} threads</strong>: Bot sent CLARIFY status — it could not determine intent precisely and asked the user follow-up questions before proceeding.</li>'
        if 'Turn Limit Exceeded' in gaps:
            gap_explain += f'<li><strong>Turn Limit Exceeded — {gaps["Turn Limit Exceeded"]} threads</strong>: The conversation went through the maximum allowed back-and-forth turns without resolving, so the bot automatically escalated. This indicates the topic is too complex for the current turn budget.</li>'
        if 'Bot Chose to Escalate' in gaps:
            gap_explain += f'<li><strong>Bot Chose to Escalate — {gaps["Bot Chose to Escalate"]} threads</strong>: Bot decided to escalate to a human based on its reasoning, even when some docs were available. The bot\'s confidence threshold may be too conservative.</li>'

        # Aggregated bot answers and eng resolutions
        bot_html = ''
        for ba in cl.get('bot_answers', [])[:2]:
            bot_html += f'<blockquote class="bot-quote">{esc(ba)}</blockquote>'

        # Engineer resolution analysis (full cluster)
        eng_analysis = cl.get('eng_analysis', {})
        eng_threads = eng_analysis.get('total_eng_threads', 0)
        eng_total_msgs = eng_analysis.get('total_eng_messages', 0)
        resolution_patterns = eng_analysis.get('resolution_patterns', {})
        sample_resolutions = eng_analysis.get('sample_resolutions', [])

        # Resolution patterns bar chart
        eng_patterns_html = ''
        if resolution_patterns:
            total_pat = max(sum(resolution_patterns.values()), 1)
            for pat, cnt in sorted(resolution_patterns.items(), key=lambda x: -x[1]):
                pct = cnt / total_pat * 100
                eng_patterns_html += f'''
                <div class="bar-row">
                    <span class="bar-label">{esc(pat)}</span>
                    <div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:var(--success)"></div></div>
                    <span class="bar-val">{cnt}</span>
                </div>'''

        # Sample engineer messages
        eng_samples_html = ''
        for er in sample_resolutions[:5]:
            eng_samples_html += f'<div class="eng-block"><span class="eng-name">{esc(er["name"])}</span><p class="text-sm">{esc(er["content"][:350])}</p></div>'
        if not eng_samples_html:
            eng_samples_html = '<p class="text-muted text-sm">No engineer messages captured in conversation history for this cluster.</p>'

        # Recommendation as bullet list
        rec_items = cl.get('recommendation', [])
        if isinstance(rec_items, str):
            rec_items = [rec_items]
        rec_html = ''.join(f'<li class="rec-bullet">{esc(r)}</li>' for r in rec_items)

        # Individual threads
        threads_html = ''
        for thread in cl.get('threads', []):
            threads_html += render_thread_card(thread, collapsed=True)

        html += f'''
        <div class="cluster-card card" style="border-left-color:{color}">
            <div class="collapsible-header" onclick="toggleCollapsible(this)">
                <div class="cluster-header-row">
                    <div class="color-dot" style="background:{color}"></div>
                    <h4 class="cluster-label">{label}</h4>
                    <span class="badge badge-neutral">{count} threads</span>
                    {f'<span class="text-muted text-sm" style="margin-left:4px">silhouette={sil}</span>' if sil >= 0 else ''}
                </div>
                <span class="arrow">▶</span>
            </div>
            <div class="collapsible-content">

                <div class="two-col">
                    <div>
                        <h5 class="subsection-title">Bot Gap Analysis</h5>
                        <div class="bar-chart">{gap_html if gap_html else '<p class="text-muted text-sm">No trace data</p>'}</div>
                        {f'<div style="margin-top:8px">{rc_html}</div>' if rc_html else ''}
                        {f'<ul class="gap-explain-list">{gap_explain}</ul>' if gap_explain else ''}

                        <h5 class="subsection-title" style="margin-top:16px">What Bot Answered (samples)</h5>
                        {bot_html if bot_html else '<p class="text-muted text-sm">No bot responses extracted</p>'}
                    </div>
                    <div>
                        <h5 class="subsection-title">How Engineers Resolved — All {count} Threads</h5>
                        {f'<p class="text-muted text-sm mb-md">{eng_total_msgs} engineer messages across {eng_threads} of {count} threads analyzed</p>' if eng_threads > 0 else '<p class="text-muted text-sm mb-md">No engineer messages found in conversation history for this cluster.</p>'}
                        {f'<div class="bar-chart">{eng_patterns_html}</div>' if eng_patterns_html else ''}
                        <h5 class="subsection-title" style="margin-top:12px">Sample Engineer Messages</h5>
                        {eng_samples_html}
                    </div>
                </div>

                <div class="recommendation-block">
                    <h5 class="subsection-title">💡 Automation Recommendations — based on engineer resolution patterns</h5>
                    <ul class="rec-bullet-list">{rec_html}</ul>
                </div>

                <div style="margin-top:16px">
                    <h5 class="subsection-title">Individual Threads ({count}) — with full trace &amp; engineer resolution</h5>
                    {threads_html}
                </div>
            </div>
        </div>'''

    return html

def render_howto_section(breakdown, weekly_bd, category_details):
    LABELS = {
        'faq_and_rag_then_no_context': ('FAQ + RAG → No Context Follow-up', '#8b5cf6',
            'FAQ was rendered and bot found RAG docs for the initial answer, but when the user asked a follow-up, the RAG hit NO_CONTEXT. These 3 threads are the user-reported FAQ-rendered escalations.'),
        'faq_and_rag': ('FAQ + RAG Served', '#6366f1',
            'Both FAQ and RAG were used and answered the query, but user still escalated.'),
        'faq_only': ('FAQ Only', '#3b82f6',
            'Only FAQ was rendered (no RAG docs found). User escalated.'),
        'rag_insufficient': ('RAG Served → Insufficient Answer', '#f59e0b',
            'Bot found RAG docs and answered. User explicitly said answer was insufficient or needed further explanation, then follow-up hit NO_CONTEXT. This is the most common pattern.'),
        'rag_only': ('RAG Only (Answered but Escalated)', '#22c55e',
            'Bot found docs and answered. User requested escalation without giving insufficient_answer feedback. Possible: answer was correct but user wanted human confirmation.'),
        'no_context': ('No Context from Start', '#ef4444',
            'RAG found no relevant docs at all from the first turn. Bot had nothing to offer.'),
    }
    total = sum(breakdown.values())

    stat_html = ''
    for key, (label, color, desc) in LABELS.items():
        cnt = breakdown.get(key, 0)
        if cnt == 0:
            continue
        pct = round(cnt / total * 100, 1) if total else 0
        stat_html += f'''
        <div class="stat-card">
            <div class="stat-num" style="color:{color}">{cnt}</div>
            <div class="stat-label" style="color:{color}">{label}</div>
            <div class="stat-sub">{pct}% of {total} escalated how-to threads</div>
            <div class="stat-desc">{desc}</div>
        </div>'''

    # Per-category thread detail with engineer resolution analysis
    detail_html = ''
    for cat in category_details:
        src = cat['source']
        if src not in LABELS:
            continue
        label, color, desc = LABELS[src]
        threads_html = ''.join(render_thread_card(t, collapsed=True) for t in cat['threads'])
        count = cat['count']

        # Engineer resolution analysis for this how-to sub-category
        eng_analysis = cat.get('eng_analysis', {})
        eng_threads = eng_analysis.get('total_eng_threads', 0)
        eng_total_msgs = eng_analysis.get('total_eng_messages', 0)
        resolution_patterns = eng_analysis.get('resolution_patterns', {})
        sample_resolutions = eng_analysis.get('sample_resolutions', [])
        automation_signals = eng_analysis.get('automation_signals', [])

        # Resolution patterns bar chart
        eng_patterns_html = ''
        if resolution_patterns:
            total_pats = max(sum(resolution_patterns.values()), 1)
            for pat, cnt in sorted(resolution_patterns.items(), key=lambda x: -x[1]):
                pp = cnt / total_pats * 100
                eng_patterns_html += f'<div class="bar-row"><span class="bar-label">{esc(pat)}</span><div class="bar-track"><div class="bar-fill" style="width:{pp:.0f}%;background:var(--success)"></div></div><span class="bar-val">{cnt}</span></div>'

        eng_samples_html = ''.join(
            f'<div class="eng-block"><span class="eng-name">{esc(em["name"])}</span><p class="text-sm">{esc(em["content"][:300])}</p></div>'
            for em in sample_resolutions[:4]
        ) or '<p class="text-muted text-sm">No engineer messages captured in this sub-category.</p>'

        rec_html = ''.join(f'<li class="rec-bullet">{esc(sig)}</li>' for sig in automation_signals)

        eng_block = f'''
        <div class="two-col" style="margin-bottom:16px">
            <div>
                <h5 class="subsection-title">How Engineers Resolved — All {count} Threads</h5>
                <p class="text-muted text-sm mb-md">{eng_total_msgs} engineer messages from {eng_threads}/{count} threads</p>
                {f'<div class="bar-chart">{eng_patterns_html}</div>' if eng_patterns_html else '<p class="text-muted text-sm">No patterns detected.</p>'}
            </div>
            <div>
                <h5 class="subsection-title">Sample Engineer Resolutions</h5>
                {eng_samples_html}
                {f'<div class="recommendation-block" style="margin-top:10px"><h5 class="subsection-title">💡 Automation Opportunities</h5><ul class="rec-bullet-list">{rec_html}</ul></div>' if rec_html else ''}
            </div>
        </div>''' if eng_total_msgs > 0 else ''

        detail_html += f'''
        <div class="card collapsible-wrap" style="border-left:3px solid {color}">
            <div class="collapsible-header" onclick="toggleCollapsible(this)">
                <div style="display:flex;align-items:center;gap:10px">
                    <div class="color-dot" style="background:{color}"></div>
                    <h4>{esc(label)} — <span style="color:{color}">{count} threads</span></h4>
                </div>
                <span class="arrow">▶</span>
            </div>
            <div class="collapsible-content">
                <p class="text-secondary text-sm mb-md">{esc(desc)}</p>
                {eng_block}
                <h5 class="subsection-title">Individual Threads — with full trace &amp; engineer resolution</h5>
                {threads_html}
            </div>
        </div>'''

    return stat_html, detail_html

def render_intent_weekly_table(intent_weekly):
    """Render per-intent weekly drill-down tables."""
    INTENT_ORDER = ['troubleshooting', 'how-to', 'access-request', 'notification', 'enquiry/other']
    INTENT_LABELS = {
        'troubleshooting': 'Troubleshooting',
        'how-to': 'How-To',
        'access-request': 'Access Requests',
        'notification': 'Notification',
        'enquiry/other': 'Enquiry / Other',
    }
    html = '<div class="intent-weekly-tabs">'
    # Tab buttons
    html += '<div class="tab-buttons">'
    for i, intent in enumerate(INTENT_ORDER):
        if intent not in intent_weekly:
            continue
        active = 'active' if i == 0 else ''
        html += f'<button class="tab-btn {active}" onclick="switchTab(this, \'iw-{intent.replace("/","-")}\')">{esc(INTENT_LABELS.get(intent, intent))}</button>'
    html += '</div>'

    # Tab panels
    for i, intent in enumerate(INTENT_ORDER):
        if intent not in intent_weekly:
            continue
        weeks = intent_weekly[intent]
        active = 'active' if i == 0 else ''
        total_all = sum(w['total'] for w in weeks)
        esc_all = sum(w['escalated'] for w in weeks)
        bot_all = sum(w['bot_resolved'] for w in weeks)
        faq_all = sum(w['faq_rendered'] for w in weeks)

        rows_html = ''
        for w in weeks:
            esc_pct = round(w['escalated'] / w['total'] * 100, 1) if w['total'] else 0
            bot_pct = round(w['bot_resolved'] / w['total'] * 100, 1) if w['total'] else 0
            rc = pct_color(bot_pct)
            rows_html += f'''
            <tr>
                <td><strong>{esc(w["week_label"])}</strong></td>
                <td>{w["total"]}</td>
                <td style="color:var(--success)">{w["bot_resolved"]} <span class="text-muted text-sm">({bot_pct}%)</span></td>
                <td style="color:var(--danger)">{w["escalated"]} <span class="text-muted text-sm">({esc_pct}%)</span></td>
                <td>{w["faq_rendered"]}</td>
            </tr>'''

        rows_html += f'''
        <tr style="background:rgba(99,102,241,.08);font-weight:600">
            <td>TOTAL</td><td>{total_all}</td>
            <td style="color:var(--success)">{bot_all} ({round(bot_all/total_all*100,1) if total_all else 0}%)</td>
            <td style="color:var(--danger)">{esc_all} ({round(esc_all/total_all*100,1) if total_all else 0}%)</td>
            <td>{faq_all}</td>
        </tr>'''

        html += f'''
        <div class="tab-panel {active}" id="iw-{intent.replace("/","-")}">
            <div class="table-wrap" style="margin-top:12px">
                <table>
                    <thead><tr><th>Week</th><th>Total Requests</th><th>Bot Resolved</th><th>Escalated</th><th>FAQ Rendered</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
        </div>'''

    html += '</div>'
    return html

def compute_week_deepdive(df, week_seq):
    """
    For a given week: run the FULL cluster/howto analysis on that week's threads only.
    Pre-renders HTML for each intent so clicking week→intent shows identical content
    to the dedicated sections (howto, troubleshooting, access, notification) but week-filtered.
    """
    wdf = df[df['week_seq'] == week_seq].copy()
    if len(wdf) == 0:
        return None

    week_label = wdf['week_label'].iloc[0]
    INTENT_LABELS = {
        'troubleshooting': 'Troubleshooting', 'how-to': 'How-To',
        'access-request': 'Access Requests', 'notification': 'Notification', 'enquiry/other': 'Enquiry/Other',
    }

    intent_html_map = {}  # intent → pre-rendered HTML

    # ── How-To ──────────────────────────────────────────────────────────────────
    ht_esc = wdf[(wdf['intent_norm'] == 'how-to') & (wdf['escalated'] == True)]
    if len(ht_esc) > 0:
        bd, wbd, cats = analyze_howto_subset(wdf)
        stat_html, detail_html = render_howto_section(bd, wbd, cats)
        total_ht = len(wdf[wdf['intent_norm'] == 'how-to'])
        esc_ht = len(ht_esc)
        bot_ht = int((wdf[(wdf['intent_norm'] == 'how-to')]['resolution_type'] == 'BOT').sum())
        intent_html_map['how-to'] = {
            'total': total_ht, 'escalated': esc_ht, 'bot_resolved': bot_ht,
            'html': f'<div class="howto-stats">{stat_html}</div>{detail_html}',
        }

    # ── Troubleshooting ──────────────────────────────────────────────────────────
    ts_esc = wdf[(wdf['intent_norm'] == 'troubleshooting') & (wdf['escalated'] == True)]
    if len(ts_esc) > 0:
        ts_cls = build_cluster_data('troubleshooting', ts_esc)
        total_ts = len(wdf[wdf['intent_norm'] == 'troubleshooting'])
        bot_ts = int((wdf[(wdf['intent_norm'] == 'troubleshooting')]['resolution_type'] == 'BOT').sum())
        intent_html_map['troubleshooting'] = {
            'total': total_ts, 'escalated': len(ts_esc), 'bot_resolved': bot_ts,
            'html': render_cluster_section_full(ts_cls, 'troubleshooting'),
        }

    # ── Access Request ───────────────────────────────────────────────────────────
    ar_esc = wdf[(wdf['intent_norm'] == 'access-request') & (wdf['escalated'] == True)]
    if len(ar_esc) > 0:
        ar_cls = build_cluster_data('access-request', ar_esc)
        total_ar = len(wdf[wdf['intent_norm'] == 'access-request'])
        bot_ar = int((wdf[(wdf['intent_norm'] == 'access-request')]['resolution_type'] == 'BOT').sum())
        intent_html_map['access-request'] = {
            'total': total_ar, 'escalated': len(ar_esc), 'bot_resolved': bot_ar,
            'html': render_cluster_section_full(ar_cls, 'access-request'),
        }

    # ── Notification / Enquiry ───────────────────────────────────────────────────
    notif_esc = wdf[wdf['intent_norm'].isin(['notification', 'enquiry/other']) & (wdf['escalated'] == True)]
    if len(notif_esc) > 0:
        notif_cls = build_cluster_data('notification/other', notif_esc)
        total_notif = len(wdf[wdf['intent_norm'].isin(['notification', 'enquiry/other'])])
        bot_notif = int((wdf[wdf['intent_norm'].isin(['notification', 'enquiry/other'])]['resolution_type'] == 'BOT').sum())
        intent_html_map['notification'] = {
            'total': total_notif, 'escalated': len(notif_esc), 'bot_resolved': bot_notif,
            'html': render_cluster_section_full(notif_cls, 'notification/other'),
        }

    return {
        'week_seq': week_seq,
        'week_label': week_label,
        'intent_html_map': intent_html_map,
        'INTENT_LABELS': INTENT_LABELS,
    }


def analyze_howto_subset(wdf):
    """Run howto analysis on a week-subset dataframe (mirrors analyze_howto but takes any df)."""
    howto_df = wdf[(wdf['intent_norm'] == 'how-to') & (wdf['escalated'] == True)].copy()
    if len(howto_df) == 0:
        return {}, {}, []

    sources, fb_types, fb_details = [], [], []
    for _, row in howto_df.iterrows():
        src, fbt, fbd = classify_howto(row)
        sources.append(src)
        fb_types.append(fbt)
        fb_details.append(fbd)

    howto_df['howto_source'] = sources
    howto_df['feedback_type'] = fb_types
    howto_df['feedback_detail'] = fb_details

    breakdown = howto_df['howto_source'].value_counts().to_dict()

    weekly_breakdown = {}
    for week, grp in howto_df.groupby('week_seq'):
        wlabel = grp['week_label'].iloc[0]
        weekly_breakdown[int(week)] = {
            'week_label': wlabel,
            'by_source': grp['howto_source'].value_counts().to_dict(),
            'total': len(grp),
        }

    category_details = []
    for source, grp in howto_df.groupby('howto_source'):
        threads = []
        all_eng_flat = []
        for _, row in grp.iterrows():
            conv = row.get('conversation_history_parsed')
            trace_steps = parse_trace_steps(row.get('ai_orchestration_trace_parsed'))
            orig_user, question = get_first_user_and_question(conv)
            eng_msgs = get_engineer_messages(conv, orig_user)
            tid = str(row.get('threadid', ''))
            for m in eng_msgs:
                all_eng_flat.append({'thread_id': tid, 'name': m['name'], 'content': m['content']})
            rc_tag, rc_detail = classify_escalation_root_cause(row, conv, trace_steps)
            bot_answers_full = get_all_bot_answers(conv)
            threads.append({
                'thread_id': tid,
                'slack_url': str(row.get('slack_thread_url', '')),
                'week_label': row.get('week_label', ''),
                'question': clean_question(question),
                'faq_rendered': bool(row.get('faq_rendered', False)),
                'faq_ids': str(row.get('faq_ids', '')),
                'feedback_type': row.get('feedback_type', ''),
                'feedback_detail': row.get('feedback_detail', ''),
                'trace_steps': trace_steps,
                'engineer_msgs': [{'name': m['name'], 'content': m['content']} for m in eng_msgs],
                'bot_answers': bot_answers_full,
                'bot_answer': bot_answers_full[0] if bot_answers_full else '',
                'rca_summary': strip_pii(str(row.get('rca_llm_analysis_summary', '') or ''))[:400],
                'rca_correct': strip_pii(str(row.get('rca_correct_answer', '') or ''))[:300],
                'root_cause': rc_tag,
                'root_cause_detail': rc_detail,
            })
        eng_analysis = analyze_engineer_resolutions(all_eng_flat)
        category_details.append({
            'source': source, 'count': len(grp),
            'threads': threads, 'eng_analysis': eng_analysis,
        })

    return breakdown, weekly_breakdown, category_details


def render_week_deepdive_panel(week_data):
    """Render the full deep dive HTML for one week — uses pre-rendered intent HTML."""
    if not week_data:
        return '<p class="text-muted">No data for this week.</p>'

    intent_html_map = week_data.get('intent_html_map', {})
    INTENT_LABELS = week_data.get('INTENT_LABELS', {})
    INTENT_ORDER = ['how-to', 'troubleshooting', 'access-request', 'notification']
    INTENT_SECTION_TITLES = {
        'how-to': 'How-To Escalation Deep Dive',
        'troubleshooting': 'Troubleshooting Cluster Analysis',
        'access-request': 'Access Request Cluster Analysis',
        'notification': 'Notification / Enquiry Cluster Analysis',
    }

    if not intent_html_map:
        return '<p class="text-muted text-sm">No escalated threads this week.</p>'

    html = ''
    for intent in INTENT_ORDER:
        if intent not in intent_html_map:
            continue
        d = intent_html_map[intent]
        total = d['total']
        esc = d['escalated']
        bot = d['bot_resolved']
        bot_pct = round(bot / total * 100, 1) if total else 0
        esc_pct = round(esc / total * 100, 1) if total else 0
        color_bot = pct_color(bot_pct)
        section_title = INTENT_SECTION_TITLES.get(intent, INTENT_LABELS.get(intent, intent))
        ilabel = INTENT_LABELS.get(intent, intent)

        html += f'''
        <div class="week-intent-block">
            <div class="week-intent-header">
                <h4 class="week-intent-title">{esc_val(ilabel)}</h4>
                <div class="week-intent-metrics">
                    <span class="wim-stat"><span class="wim-val">{total}</span><span class="wim-lbl">Total</span></span>
                    <span class="wim-stat"><span class="wim-val" style="color:{color_bot}">{bot}</span><span class="wim-lbl">Bot Resolved ({bot_pct}%)</span></span>
                    <span class="wim-stat"><span class="wim-val" style="color:var(--danger)">{esc}</span><span class="wim-lbl">Escalated ({esc_pct}%)</span></span>
                </div>
            </div>
            {d['html']}
        </div>'''

    return html

# ── Old panel approach (unused) ────────────────────────────────────────────────
def _render_week_deepdive_panel_old(week_data):
    """Old render for reference — superseded by pre-rendered intent_html_map."""
    if not week_data:
        return '<p class="text-muted">No data for this week.</p>'

    html = ''
    for intent_stat in week_data.get('intents', []):
        intent = intent_stat['intent']
        label = intent_stat['label']
        total = intent_stat['total']
        bot = intent_stat['bot_resolved']
        esc = intent_stat['escalated']
        faq = intent_stat['faq_rendered']
        bot_pct = round(bot / total * 100, 1) if total else 0
        esc_pct = round(esc / total * 100, 1) if total else 0
        color_bot = pct_color(bot_pct)
        eng_analysis = intent_stat.get('eng_analysis', {})
        eng_threads = eng_analysis.get('total_eng_threads', 0)
        eng_total_msgs = eng_analysis.get('total_eng_messages', 0)
        resolution_patterns = eng_analysis.get('resolution_patterns', {})
        sample_resolutions = eng_analysis.get('sample_resolutions', [])
        automation_signals = eng_analysis.get('automation_signals', [])

        # Intent header
        html += f'''
        <div class="week-intent-block">
            <div class="week-intent-header">
                <h4 class="week-intent-title">{esc_val(label)}</h4>
                <div class="week-intent-metrics">
                    <span class="wim-stat"><span class="wim-val">{total}</span><span class="wim-lbl">Total</span></span>
                    <span class="wim-stat"><span class="wim-val" style="color:{color_bot}">{bot}</span><span class="wim-lbl">Bot Resolved ({bot_pct}%)</span></span>
                    <span class="wim-stat"><span class="wim-val" style="color:var(--danger)">{esc}</span><span class="wim-lbl">Escalated ({esc_pct}%)</span></span>
                    {f'<span class="wim-stat"><span class="wim-val" style="color:var(--info)">{faq}</span><span class="wim-lbl">FAQ</span></span>' if faq > 0 else ''}
                </div>
            </div>'''

        if esc > 0:
            html += '<div class="two-col" style="margin-top:12px">'
            # Left col: how-to sub-breakdown OR engineer resolution patterns
            html += '<div>'
            if intent == 'how-to' and intent_stat.get('howto_sub'):
                ht_sub = intent_stat['howto_sub']
                ht_total = max(sum(ht_sub.values()), 1)
                html += '<h5 class="subsection-title">How-To Sub-Categories</h5><div class="bar-chart">'
                for key, cnt in sorted(ht_sub.items(), key=lambda x: -x[1]):
                    lbl = HOWTO_DISPLAY.get(key, key)
                    clr = HOWTO_COLORS.get(key, '#6366f1')
                    pct = cnt / ht_total * 100
                    html += f'<div class="bar-row"><span class="bar-label" style="color:{clr}">{esc_val(lbl)}</span><div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:{clr}"></div></div><span class="bar-val">{cnt}</span></div>'
                html += '</div>'

            if resolution_patterns:
                total_pats = max(sum(resolution_patterns.values()), 1)
                html += '<h5 class="subsection-title" style="margin-top:14px">Engineer Resolution Patterns</h5>'
                html += f'<p class="text-muted text-sm mb-md">{eng_total_msgs} messages from {eng_threads}/{esc} escalated threads</p>'
                html += '<div class="bar-chart">'
                for pat, cnt in sorted(resolution_patterns.items(), key=lambda x: -x[1]):
                    pp = cnt / total_pats * 100
                    html += f'<div class="bar-row"><span class="bar-label">{esc_val(pat)}</span><div class="bar-track"><div class="bar-fill" style="width:{pp:.0f}%;background:var(--success)"></div></div><span class="bar-val">{cnt}</span></div>'
                html += '</div>'
            elif esc > 0:
                html += f'<p class="text-muted text-sm" style="margin-top:14px">No engineer messages captured for {esc} escalated threads.</p>'
            html += '</div>'

            # Right col: sample engineer messages + automation recommendations
            html += '<div>'
            if sample_resolutions:
                html += '<h5 class="subsection-title">Sample Engineer Resolutions</h5>'
                for em in sample_resolutions[:4]:
                    html += f'<div class="eng-block"><span class="eng-name">{esc_val(em["name"])}</span><p class="text-sm">{esc_val(em["content"][:300])}</p></div>'
            if automation_signals:
                html += '<div class="recommendation-block" style="margin-top:10px"><h5 class="subsection-title">💡 Automation Opportunities</h5><ul class="rec-bullet-list">'
                for sig in automation_signals:
                    html += f'<li class="rec-bullet">{esc_val(sig)}</li>'
                html += '</ul></div>'
            html += '</div>'
            html += '</div>'  # end two-col

            # Escalated threads (collapsible)
            esc_threads = intent_stat.get('escalated_threads', [])
            if esc_threads:
                uid = f'wk{week_data["week_seq"]}-{intent.replace("/","-").replace("-","")}'
                html += f'''
                <div class="collapsible-wrap" style="margin-top:12px">
                    <div class="collapsible-header" onclick="toggleCollapsible(this)">
                        <span style="font-size:.82rem;font-weight:600">All {len(esc_threads)} Escalated Threads — {esc_val(label)} (with engineer resolutions)</span>
                        <span class="arrow">▶</span>
                    </div>
                    <div class="collapsible-content">'''
                for t in esc_threads:
                    # Build a thread dict compatible with render_thread_card
                    thread_card_data = {
                        'thread_id': t['thread_id'],
                        'slack_url': t['slack_url'],
                        'week_label': week_data['week_label'],
                        'question': t['question'],
                        'bot_answer': t['bot_answer'],
                        'trace_steps': [],  # not re-parsed here for speed
                        'engineer_msgs': t['engineer_msgs'],
                        'rca_summary': '',
                        'rca_correct': '',
                        'root_cause': t['root_cause'],
                        'faq_rendered': t['faq_rendered'],
                        'feedback_type': '',
                        'feedback_detail': '',
                    }
                    html += render_thread_card(thread_card_data, collapsed=True)
                html += '</div></div>'
        else:
            html += '<p class="text-muted text-sm" style="margin-top:8px">No escalations this week.</p>'

        html += '</div>'  # week-intent-block
    return html


def render_week_deepdive_tabs(weekly_deepdives):
    """Render week tabs — clicking a week reveals its full intent deep dive panel."""
    html = '<div class="week-deepdive-tabs">'
    html += '<div class="tab-buttons">'
    for i, wd in enumerate(weekly_deepdives):
        if wd is None:
            continue
        active = 'active' if i == 0 else ''
        html += f'<button class="tab-btn {active}" onclick="switchTab(this, \'wd-week-{wd["week_seq"]}\'">{esc_val(wd["week_label"])}</button>'
    html += '</div>'
    for i, wd in enumerate(weekly_deepdives):
        if wd is None:
            continue
        active = 'active' if i == 0 else ''
        panel_html = render_week_deepdive_panel(wd)
        html += f'''
        <div class="tab-panel {active}" id="wd-week-{wd["week_seq"]}">
            <div class="week-deepdive-panel">{panel_html}</div>
        </div>'''
    html += '</div>'
    return html


# Helper alias to avoid name collision with html.escape used in f-strings
def esc_val(text):
    return html_lib.escape(str(text or ''), quote=True)


def render_thread_explorer(df):
    """Build the thread explorer with all threads filterable by intent and sub-category."""
    threads_data = []
    howto_classify_cache = {}

    # Pre-classify how-to threads
    howto_esc = df[(df['intent_norm'] == 'how-to') & (df['escalated'] == True)]
    for _, row in howto_esc.iterrows():
        src, fbt, fbd = classify_howto(row)
        howto_classify_cache[str(row.get('threadid', ''))] = src

    for _, row in df.iterrows():
        thread_id = str(row.get('threadid', ''))
        intent = str(row.get('intent_norm', ''))
        is_esc = bool(row.get('escalated', False))
        is_bot = str(row.get('resolution_type', '')) == 'BOT'
        slack_url = str(row.get('slack_thread_url', '') or '')
        question = clean_question(str(row.get('question', '') or ''))[:200]
        week = str(row.get('week_label', ''))
        faq = bool(row.get('faq_rendered', False))
        root_cause = str(row.get('rca_llm_root_cause_tag', '') or '')
        created = str(row.get('created_date', '') or '')[:10]

        sub_category = ''
        if intent == 'how-to' and is_esc:
            sub_category = howto_classify_cache.get(thread_id, 'no_context')

        threads_data.append({
            'id': thread_id,
            'date': created,
            'week': week,
            'intent': intent,
            'sub': sub_category,
            'esc': is_esc,
            'bot': is_bot,
            'faq': faq,
            'url': slack_url,
            'q': question,
            'rca': root_cause,
        })

    return json.dumps(threads_data, ensure_ascii=False)

def generate_html(analysis, df):
    weekly = analysis['weekly_summaries']
    week_labels_js = [w['week_label'] for w in weekly]
    week_totals_js = [w['total'] for w in weekly]
    week_esc_js = [w['escalated'] for w in weekly]
    week_bot_js = [w['bot_resolved'] for w in weekly]
    week_res_js = [w['resolution_rate'] for w in weekly]
    week_pf_js = [w['prefiltered'] for w in weekly]
    week_att_js = [w['bot_attempted'] for w in weekly]

    total_requests = sum(w['total'] for w in weekly)
    total_esc = sum(w['escalated'] for w in weekly)
    total_pf = sum(w['prefiltered'] for w in weekly)
    total_pf_esc = sum(w['pf_escalated'] for w in weekly)
    total_att = sum(w['bot_attempted'] for w in weekly)
    total_bot = sum(w['bot_resolved'] for w in weekly)
    overall_res = round(total_bot / total_att * 100, 1) if total_att else 0
    total_surv = sum(w['survey_filled'] for w in weekly)
    overall_surv = round(total_surv / total_esc * 100, 1) if total_esc else 0

    mw = [w for w in weekly if w['bot_attempted'] >= 20]
    if len(mw) >= 2:
        trend_val = mw[-1]['resolution_rate'] - mw[0]['resolution_rate']
        trend_ref = f"vs {mw[0]['week_label']}"
    else:
        trend_val = 0
        trend_ref = ''
    trend_cls = 'trend-up' if trend_val >= 0 else 'trend-down'
    trend_sym = '▲' if trend_val >= 0 else '▼'

    # Weekly table — each row is clickable and expands an intent deep dive panel
    # Build a map from week_seq to deep dive data
    wd_map = {wd['week_seq']: wd for wd in analysis.get('weekly_deepdives', []) if wd}

    INTENT_ORDER_WK = ['how-to', 'troubleshooting', 'access-request', 'notification']
    INTENT_LABELS_WK = {
        'how-to': 'How-To', 'troubleshooting': 'Troubleshooting',
        'access-request': 'Access Requests', 'notification': 'Notification/Enquiry',
    }

    weekly_rows_html = ''
    for w in weekly:
        rc = pct_color(w['resolution_rate'])
        wseq = w['week']
        wd = wd_map.get(wseq)

        # Build intent-level rows — each expands into the pre-rendered full analysis HTML
        intent_panel_html = ''
        if wd:
            intent_html_map = wd.get('intent_html_map', {})
            for intent in INTENT_ORDER_WK:
                if intent not in intent_html_map:
                    continue
                d = intent_html_map[intent]
                ilabel = INTENT_LABELS_WK.get(intent, intent)
                itotal = d['total']
                ibot = d['bot_resolved']
                iesc = d['escalated']
                ibot_pct = round(ibot / itotal * 100, 1) if itotal else 0
                iesc_pct = round(iesc / itotal * 100, 1) if itotal else 0
                icolor = pct_color(ibot_pct)
                panel_id = f'wd-intent-{wseq}-{intent.replace("/","-").replace(" ","-")}'

                intent_panel_html += f'''
                <div class="wk-intent-row">
                    <div class="wk-intent-summary" onclick="toggleWkIntent(this, \'{panel_id}\')">
                        <div class="wk-intent-left">
                            <span class="wk-intent-name">{esc(ilabel)}</span>
                            <span class="wk-intent-stats">
                                <span>{itotal} threads</span>
                                <span style="color:{icolor}">{ibot} bot resolved ({ibot_pct}%)</span>
                                <span style="color:var(--danger)">{iesc} escalated ({iesc_pct}%)</span>
                            </span>
                        </div>
                        <span class="wk-intent-arrow">▶</span>
                    </div>
                    <div class="wk-intent-detail" id="{panel_id}">
                        {d['html']}
                    </div>
                </div>'''

        # Week summary row + expandable intent panel
        weekly_rows_html += f'''
        <div class="wk-row{'  wk-row-alt' if w['week'] % 2 == 0 else ''}">
            <div class="wk-row-summary" onclick="toggleWkRow(this, 'wk-panel-{wseq}')">
                <div class="wk-row-cells">
                    <span class="wk-cell wk-label"><strong>{esc(w["week_label"])}</strong></span>
                    <span class="wk-cell">{w["total"]}</span>
                    <span class="wk-cell text-muted text-sm">{w["prefiltered"]} <span style="font-size:.66rem">({w["pf_escalated"]} direct-esc)</span></span>
                    <span class="wk-cell">{w["bot_attempted"]}</span>
                    <span class="wk-cell" style="color:var(--danger)">{w["escalated"]}</span>
                    <span class="wk-cell" style="color:var(--success)">{w["bot_resolved"]}</span>
                    <span class="wk-cell" style="color:{rc};font-weight:700">{w["resolution_rate"]}%</span>
                    <span class="wk-cell text-muted text-sm">{w["survey_rate"]}% ({w["survey_filled"]}/{w["escalated"]})</span>
                </div>
                <span class="wk-expand-arrow">▶</span>
            </div>
            <div class="wk-panel" id="wk-panel-{wseq}">
                <div class="wk-panel-inner">
                    <h5 class="wk-panel-title">Intent Breakdown — {esc(w["week_label"])} <span class="text-muted text-sm">(click an intent to see full deep dive)</span></h5>
                    {intent_panel_html if intent_panel_html else '<p class="text-muted text-sm">No data.</p>'}
                </div>
            </div>
        </div>'''

    # Intent summary table
    intent_table_rows = ''
    intent_chart_labels, intent_chart_esc, intent_chart_bot = [], [], []
    for d in analysis['intent_breakdown']:
        intent_table_rows += f'''
        <tr>
            <td><strong>{esc(d["intent"])}</strong></td>
            <td>{d["total"]}</td>
            <td style="color:var(--success)">{d["bot_resolved"]} <span class="text-muted">({d["pct_bot"]}%)</span></td>
            <td style="color:var(--danger)">{d["escalated"]} <span class="text-muted">({d["pct_esc"]}%)</span></td>
            <td>{d.get("faq_rendered", 0)}</td>
        </tr>'''
        intent_chart_labels.append(d['intent'])
        intent_chart_esc.append(d['escalated'])
        intent_chart_bot.append(d['bot_resolved'])

    # How-To section
    howto_bd = analysis.get('howto_breakdown', {})
    howto_total_count = sum(howto_bd.values())
    howto_stat_html, howto_detail_html = render_howto_section(
        howto_bd,
        analysis.get('howto_weekly', {}),
        analysis.get('howto_details', [])
    )
    howto_js_data = [
        howto_bd.get('no_context', 0),
        howto_bd.get('rag_insufficient', 0),
        howto_bd.get('rag_only', 0),
        howto_bd.get('faq_and_rag_then_no_context', 0),
        howto_bd.get('faq_and_rag', 0),
        howto_bd.get('faq_only', 0),
    ]

    # Clusters
    ts_html = render_cluster_section_full(analysis.get('ts_clusters', []), 'troubleshooting')
    ar_html = render_cluster_section_full(analysis.get('ar_clusters', []), 'access-request')
    notif_html = render_cluster_section_full(analysis.get('notif_clusters', []), 'notification/other')

    # Intent weekly drill-down (existing: per-intent tab showing all weeks)
    intent_weekly_html = render_intent_weekly_table(analysis.get('intent_weekly', {}))

    # Week deep dive (new: per-week tab showing all intents with clusters)
    week_deepdive_html = render_week_deepdive_tabs(analysis.get('weekly_deepdives', []))

    # Recommendations
    recs_html = ''
    for r in analysis['recommendations']:
        badge_cls = f'badge-{r["priority"].lower()}'
        recs_html += f'''
        <div class="recommendation-item">
            <div class="rec-meta">
                <span class="badge {badge_cls}">{esc(r["priority"])}</span>
                <span class="rec-category">{esc(r["category"])}</span>
            </div>
            <p class="rec-text">{esc(r["text"])}</p>
        </div>'''

    # Themed recommendations (Bot Platform vs Capability Team)
    themed_recs_html = render_themed_recommendations(analysis.get('themed_recommendations', {}))

    # Root cause summary (grouped + prioritized, with tagging logic)
    rc_counts_all = analysis.get('rc_counts_all', {})
    root_cause_summary_html = render_root_cause_summary(rc_counts_all, total_esc)

    # Thread explorer data
    thread_explorer_data = render_thread_explorer(df)

    # DQ notes
    dq_html = ''.join(f'<li>{esc(n)}</li>' for n in analysis.get('dq_notes', []))

    report_date = datetime.now().strftime('%B %d, %Y at %H:%M')

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Identity Support Bot — March 2026 Report v4</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#0f1117;--surface:#1a1d27;--surface2:#242836;--border:#2a2e3a;
  --text:#e8eaed;--text-sec:#9aa0b2;--text-muted:#6b7280;
  --accent:#6366f1;--accent-light:#818cf8;
  --success:#22c55e;--warning:#f59e0b;--danger:#ef4444;--info:#3b82f6;
  --radius:8px;--font:'IBM Plex Sans',-apple-system,sans-serif;--mono:'IBM Plex Mono',monospace;
}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:var(--font);background:var(--bg);color:var(--text);line-height:1.6;font-size:14px;}}

nav{{position:sticky;top:0;z-index:200;background:rgba(15,17,23,.97);backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border);padding:7px 20px;display:flex;align-items:center;gap:4px;overflow-x:auto;}}
.nav-brand{{font-weight:700;font-size:.85rem;color:var(--accent-light);white-space:nowrap;margin-right:14px;flex-shrink:0;}}
nav a{{color:var(--text-sec);text-decoration:none;font-size:.74rem;white-space:nowrap;padding:3px 8px;border-radius:4px;transition:all .15s;}}
nav a:hover{{color:var(--accent-light);background:rgba(99,102,241,.12);}}

.report-header{{background:linear-gradient(135deg,#1a1d27 0%,#13162a 100%);border-bottom:1px solid var(--border);padding:36px 28px;}}
.report-title{{font-size:1.7rem;font-weight:700;}}
.report-subtitle{{color:var(--text-sec);margin-top:3px;font-size:.9rem;}}
.report-meta{{display:flex;gap:16px;margin-top:10px;font-size:.74rem;color:var(--text-muted);}}

main{{max-width:1440px;margin:0 auto;padding:28px 20px;}}
section{{margin-bottom:44px;}}
.section-title{{font-size:1.25rem;font-weight:700;margin-bottom:16px;padding-bottom:8px;border-bottom:2px solid var(--accent);}}
.section-intro{{color:var(--text-sec);font-size:.84rem;margin-bottom:16px;line-height:1.7;}}

.card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px;margin-bottom:14px;}}
.card-title{{font-size:.74rem;font-weight:600;color:var(--text-sec);text-transform:uppercase;letter-spacing:.05em;margin-bottom:14px;}}
.collapsible-wrap{{padding:0;}}

.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px;margin-bottom:24px;}}
.metric-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px;text-align:center;}}
.metric-value{{font-size:1.9rem;font-weight:700;color:var(--accent);line-height:1.2;}}
.metric-label{{font-size:.68rem;color:var(--text-sec);text-transform:uppercase;letter-spacing:.05em;margin-top:3px;}}
.metric-trend{{font-size:.68rem;margin-top:3px;}}
.trend-up{{color:var(--success);}} .trend-down{{color:var(--danger);}}

.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
@media(max-width:768px){{.chart-grid{{grid-template-columns:1fr;}}}}
.chart-wrap{{position:relative;height:280px;}}
.chart-wrap-lg{{position:relative;height:340px;}}

.table-wrap{{overflow-x:auto;}}
table{{width:100%;border-collapse:collapse;font-size:.82rem;}}
th,td{{padding:7px 11px;text-align:left;border-bottom:1px solid var(--border);}}
th{{font-size:.68rem;color:var(--text-sec);text-transform:uppercase;letter-spacing:.04em;font-weight:600;background:rgba(99,102,241,.05);}}
tr:hover td{{background:var(--surface2);}}

.badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.66rem;font-weight:600;text-transform:uppercase;}}
.badge-high{{background:rgba(239,68,68,.15);color:var(--danger);}}
.badge-medium{{background:rgba(245,158,11,.15);color:var(--warning);}}
.badge-low{{background:rgba(34,197,94,.15);color:var(--success);}}
.badge-neutral{{background:rgba(99,102,241,.15);color:var(--accent-light);}}

/* Collapsible */
.collapsible-header{{cursor:pointer;display:flex;justify-content:space-between;align-items:center;
  padding:12px 18px;user-select:none;transition:background .15s;border-radius:var(--radius);}}
.collapsible-header:hover{{background:var(--surface2);}}
.arrow{{color:var(--text-muted);font-size:.7rem;flex-shrink:0;}}
.collapsible-content{{display:none;padding:14px 18px 18px;}}
.collapsible-content.active{{display:block;}}

/* How-To stats */
.howto-stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:18px;}}
.stat-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px;text-align:center;}}
.stat-num{{font-size:1.8rem;font-weight:700;line-height:1.1;}}
.stat-label{{font-size:.7rem;font-weight:600;margin-top:3px;text-transform:uppercase;letter-spacing:.04em;}}
.stat-sub{{font-size:.68rem;color:var(--text-muted);margin-top:2px;}}
.stat-desc{{font-size:.7rem;color:var(--text-muted);margin-top:6px;line-height:1.45;}}

/* Clusters */
.cluster-card{{border-left:3px solid var(--accent);padding:0;}}
.cluster-header-row{{display:flex;align-items:center;gap:9px;flex:1;}}
.cluster-label{{font-size:.9rem;font-weight:600;}}
.color-dot{{width:9px;height:9px;border-radius:50%;flex-shrink:0;}}

/* Two-col */
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:18px;}}
@media(max-width:768px){{.two-col{{grid-template-columns:1fr;}}}}

/* Bars */
.bar-chart{{margin:6px 0;}}
.bar-row{{display:flex;align-items:center;gap:7px;margin-bottom:4px;font-size:.74rem;}}
.bar-label{{width:220px;color:var(--text-sec);flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}}
.bar-track{{flex:1;background:var(--border);border-radius:3px;height:7px;}}
.bar-fill{{border-radius:3px;height:7px;min-width:2px;}}
.bar-val{{width:26px;text-align:right;color:var(--text-muted);}}

/* Thread cards */
.thread-card{{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);margin-bottom:6px;}}
.thread-card-header{{display:flex;align-items:flex-start;gap:8px;flex:1;min-width:0;}}
.thread-question{{font-size:.8rem;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;}}
.week-pill{{font-size:.66rem;background:rgba(99,102,241,.15);color:var(--accent-light);padding:2px 7px;border-radius:8px;white-space:nowrap;flex-shrink:0;}}
.faq-pill{{font-size:.66rem;background:rgba(59,130,246,.2);color:var(--info);padding:2px 7px;border-radius:8px;flex-shrink:0;}}
.rc-mini{{font-size:.9rem;flex-shrink:0;cursor:default;}}
.thread-full-question .sample-q{{white-space:pre-wrap;word-break:break-word;}}

/* Root cause badge inside thread card */
.rc-badge{{display:flex;align-items:flex-start;gap:10px;background:rgba(0,0,0,.2);border-radius:var(--radius);padding:8px 12px;margin:8px 0;}}
.rc-icon{{font-size:1.1rem;flex-shrink:0;margin-top:1px;}}
.rc-label{{font-size:.76rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;display:block;}}
.rc-detail{{font-size:.75rem;color:var(--text-sec);margin-top:3px;line-height:1.5;word-break:break-word;}}

/* Root cause tag in cluster summary */
.rc-tag{{display:inline-block;padding:3px 9px;border-radius:6px;border:1px solid;font-size:.72rem;margin:2px;white-space:nowrap;}}

/* Trace */
.trace-timeline{{border-left:2px solid var(--border);margin-left:8px;padding-left:12px;}}
.trace-step{{margin-bottom:10px;position:relative;}}
.trace-step::before{{content:'';position:absolute;left:-17px;top:7px;width:7px;height:7px;border-radius:50%;background:var(--border);}}
.trace-step-header{{display:flex;align-items:center;gap:8px;margin-bottom:3px;}}
.trace-icon{{font-size:.85rem;}}
.trace-event{{font-size:.72rem;font-weight:600;color:var(--text-sec);text-transform:uppercase;letter-spacing:.04em;}}
.trace-status{{font-size:.72rem;font-weight:700;text-transform:uppercase;}}
.trace-user-input{{font-size:.76rem;color:var(--accent-light);margin:2px 0;background:rgba(99,102,241,.07);padding:3px 8px;border-radius:4px;word-break:break-word;white-space:pre-wrap;}}
.trace-reasoning{{font-size:.75rem;color:var(--text-sec);margin:2px 0;word-break:break-word;white-space:pre-wrap;line-height:1.55;}}
.trace-field-label{{font-size:.66rem;font-weight:600;color:var(--text-muted);text-transform:uppercase;margin-right:4px;}}

/* Engineer */
.eng-block{{background:rgba(34,197,94,.05);border:1px solid rgba(34,197,94,.15);border-radius:var(--radius);padding:7px 11px;margin-bottom:5px;}}
.eng-name{{font-size:.68rem;color:var(--success);font-weight:600;text-transform:uppercase;display:block;margin-bottom:2px;}}

/* Misc */
.bot-quote{{border-left:3px solid var(--info);padding:5px 10px;margin:3px 0;font-size:.78rem;color:var(--text-sec);background:rgba(59,130,246,.05);border-radius:0 4px 4px 0;word-break:break-word;}}
.bot-answer-full{{font-size:.78rem;font-style:normal;white-space:pre-wrap;line-height:1.6;}}
.insight-quote{{border-left:3px solid var(--accent);padding:5px 10px;margin:3px 0;font-size:.78rem;color:var(--text-sec);font-style:italic;background:rgba(99,102,241,.05);border-radius:0 4px 4px 0;}}
.insight-summary{{color:var(--text-sec);font-style:italic;font-size:.8rem;}}
.sample-q{{font-size:.8rem;color:var(--text);font-family:var(--mono);background:rgba(0,0,0,.3);padding:7px 9px;border-radius:4px;word-break:break-word;margin:3px 0;}}
.subsection-title{{font-size:.72rem;font-weight:600;color:var(--text-sec);text-transform:uppercase;letter-spacing:.04em;margin-bottom:7px;}}
.field-label{{font-size:.68rem;color:var(--text-muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em;display:block;margin-bottom:3px;margin-top:10px;}}
.slack-link{{color:var(--accent-light);text-decoration:none;font-size:.72rem;white-space:nowrap;border:1px solid rgba(99,102,241,.3);padding:2px 7px;border-radius:4px;}}
.slack-link:hover{{background:rgba(99,102,241,.15);}}
.feedback-badge{{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);color:var(--warning);font-size:.74rem;padding:4px 10px;border-radius:4px;margin:8px 0;word-break:break-word;}}
.recommendation-block{{background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.2);border-radius:var(--radius);padding:11px 14px;margin-top:14px;}}
.rec-bullet-list{{list-style:none;padding:0;margin:8px 0 0;}}
.rec-bullet{{padding:6px 0;border-bottom:1px solid rgba(99,102,241,.15);font-size:.8rem;line-height:1.55;color:var(--text);}}
.rec-bullet:last-child{{border-bottom:none;}}
.gap-explain-list{{list-style:none;padding:0;margin-top:10px;}}
.gap-explain-list li{{padding:6px 0;border-bottom:1px solid var(--border);font-size:.78rem;line-height:1.5;}}
.gap-explain-list li:last-child{{border-bottom:none;}}

/* Tabs */
.tab-buttons{{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:12px;}}
.tab-btn{{background:var(--surface);border:1px solid var(--border);color:var(--text-sec);padding:5px 12px;border-radius:4px;cursor:pointer;font-size:.78rem;font-family:var(--font);transition:all .15s;}}
.tab-btn.active,.tab-btn:hover{{background:rgba(99,102,241,.15);color:var(--accent-light);border-color:rgba(99,102,241,.4);}}
.tab-panel{{display:none;}} .tab-panel.active{{display:block;}}

/* Recommendations */
.recommendation-item{{display:flex;gap:12px;padding:12px 14px;border:1px solid var(--border);border-radius:var(--radius);margin-bottom:7px;background:var(--surface);}}
.rec-meta{{display:flex;flex-direction:column;gap:3px;min-width:90px;flex-shrink:0;}}
.rec-category{{font-size:.64rem;color:var(--text-muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em;}}
.rec-text{{font-size:.82rem;color:var(--text);flex:1;}}

/* Themed Recommendations */
.themed-rec-group-title{{font-size:1rem;font-weight:700;margin-bottom:10px;}}
.themed-rec-item{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:12px 14px;margin-bottom:8px;}}
.themed-rec-body{{margin-top:7px;}}
.themed-rec-evidence{{font-size:.79rem;color:var(--text-sec);margin-bottom:5px;line-height:1.5;}}
.themed-rec-action{{font-size:.79rem;color:var(--text);line-height:1.5;}}

/* Thread Explorer */
.explorer-controls{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;align-items:center;}}
.search-box{{background:var(--surface);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:var(--radius);font-size:.78rem;font-family:var(--font);min-width:200px;}}
.filter-select{{background:var(--surface);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:var(--radius);font-size:.78rem;font-family:var(--font);cursor:pointer;}}
.search-box:focus,.filter-select:focus{{outline:none;border-color:var(--accent);}}
.page-info{{font-size:.74rem;color:var(--text-muted);margin-left:auto;}}
.pagination{{display:flex;gap:4px;margin-top:10px;flex-wrap:wrap;}}
.page-btn{{background:var(--surface);border:1px solid var(--border);color:var(--text-sec);padding:4px 9px;border-radius:4px;cursor:pointer;font-size:.74rem;}}
.page-btn.active,.page-btn:hover{{background:rgba(99,102,241,.2);color:var(--accent-light);border-color:var(--accent);}}
.explorer-table td{{max-width:380px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}}
.explorer-table .full-q{{max-width:320px;font-size:.78rem;}}

/* Weekly breakdown — clickable rows */
.wk-table-wrap{{overflow:hidden;}}
.wk-table-header{{display:flex;align-items:center;gap:0;padding:8px 14px;background:rgba(99,102,241,.05);border-bottom:1px solid var(--border);font-size:.68rem;color:var(--text-sec);text-transform:uppercase;letter-spacing:.04em;font-weight:600;}}
.wk-row{{border-bottom:1px solid var(--border);}}
.wk-row:last-child{{border-bottom:none;}}
.wk-row-summary{{display:flex;align-items:center;padding:10px 14px;cursor:pointer;transition:background .15s;gap:8px;}}
.wk-row-summary:hover{{background:var(--surface2);}}
.wk-row-cells{{display:flex;align-items:center;flex:1;gap:0;}}
.wk-cell{{flex:1;min-width:0;font-size:.8rem;padding-right:8px;}}
.wk-label{{min-width:110px;}}
.wk-expand-arrow{{color:var(--text-muted);font-size:.7rem;flex-shrink:0;width:24px;text-align:center;transition:transform .2s;}}
.wk-expand-arrow.open{{transform:rotate(90deg);}}
.wk-panel{{display:none;border-top:1px solid var(--border);background:rgba(99,102,241,.02);}}
.wk-panel.open{{display:block;}}
.wk-panel-inner{{padding:16px 18px;}}
.wk-panel-title{{font-size:.85rem;font-weight:700;margin-bottom:14px;color:var(--text);}}

/* Intent rows inside week panel */
.wk-intent-row{{border:1px solid var(--border);border-radius:var(--radius);margin-bottom:8px;overflow:hidden;}}
.wk-intent-summary{{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;cursor:pointer;transition:background .15s;}}
.wk-intent-summary:hover{{background:var(--surface2);}}
.wk-intent-left{{display:flex;align-items:center;gap:16px;flex:1;flex-wrap:wrap;}}
.wk-intent-name{{font-size:.85rem;font-weight:700;min-width:120px;}}
.wk-intent-stats{{display:flex;gap:16px;flex-wrap:wrap;font-size:.76rem;}}
.wk-intent-stats span{{white-space:nowrap;}}
.wk-intent-arrow{{color:var(--text-muted);font-size:.7rem;flex-shrink:0;width:20px;text-align:right;transition:transform .2s;}}
.wk-intent-arrow.open{{transform:rotate(90deg);}}
.wk-intent-detail{{display:none;padding:14px 16px;border-top:1px solid var(--border);background:var(--bg);}}
.wk-intent-detail.open{{display:block;}}

/* Week deep dive */
.week-deepdive-panel{{padding:4px 0;}}
.week-intent-block{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px;margin-bottom:14px;}}
.week-intent-header{{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:10px;}}
.week-intent-title{{font-size:.95rem;font-weight:700;min-width:140px;}}
.week-intent-metrics{{display:flex;gap:16px;flex-wrap:wrap;}}
.wim-stat{{display:flex;flex-direction:column;align-items:center;min-width:60px;}}
.wim-val{{font-size:1.2rem;font-weight:700;line-height:1.2;}}
.wim-lbl{{font-size:.64rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;text-align:center;}}
.howto-sub-inline{{margin-top:8px;}}
.week-cluster-card{{background:var(--bg);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:var(--radius);padding:12px;margin-bottom:8px;}}
.week-cluster-header{{display:flex;align-items:center;gap:8px;margin-bottom:10px;font-size:.85rem;}}
.week-cluster-body{{margin-top:6px;}}

/* Root Cause Summary */
.rc-summary{{}}
.rc-priority-bar{{display:flex;height:28px;border-radius:var(--radius);overflow:hidden;margin-bottom:8px;}}
.rc-bar-seg{{display:flex;align-items:center;justify-content:center;min-width:30px;transition:opacity .2s;}}
.rc-bar-seg:hover{{opacity:.85;cursor:default;}}
.rc-bar-label{{font-size:.65rem;font-weight:700;color:white;text-shadow:0 1px 2px rgba(0,0,0,.4);}}
.rc-groups{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px;}}
@media(max-width:900px){{.rc-groups{{grid-template-columns:1fr;}}}}
.rc-group-card{{border-radius:var(--radius);padding:14px;background:var(--surface);border:1px solid var(--border);}}
.rc-group-header{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;gap:8px;}}
.rc-group-label{{font-size:.92rem;font-weight:700;display:block;}}
.rc-priority-label{{font-size:.68rem;color:var(--text-muted);margin-top:2px;display:block;white-space:nowrap;}}
.rc-group-count{{text-align:right;flex-shrink:0;}}
.rc-group-num{{font-size:1.6rem;font-weight:700;display:block;line-height:1.1;}}
.rc-group-pct{{font-size:.68rem;display:block;}}
.rc-group-desc{{font-size:.76rem;color:var(--text-sec);margin-bottom:10px;line-height:1.5;border-bottom:1px solid var(--border);padding-bottom:8px;}}
.rc-tag-list{{}}
.rc-tag-row{{padding:6px 0;border-bottom:1px solid var(--border);}}
.rc-tag-row:last-child{{border-bottom:none;}}
.rc-tag-header{{display:flex;align-items:center;gap:7px;}}
.rc-tag-name{{font-size:.78rem;font-weight:600;flex:1;}}
.rc-tag-count{{font-size:.7rem;color:var(--text-muted);white-space:nowrap;}}
.rc-tag-logic{{font-size:.71rem;color:var(--text-muted);margin-top:3px;line-height:1.45;font-style:italic;padding-left:20px;}}

/* DQ */
.dq-list{{list-style:none;padding:0;}}
.dq-list li{{padding:6px 0;border-bottom:1px solid var(--border);font-size:.8rem;}}
.dq-list li::before{{content:"⚠ ";color:var(--warning);}}

.text-muted{{color:var(--text-muted);}} .text-secondary{{color:var(--text-sec);}} .text-accent{{color:var(--accent-light);}}
.text-sm{{font-size:.78rem;}} .mb-md{{margin-bottom:14px;}} .mt-md{{margin-top:14px;}}
code{{font-family:var(--mono);background:rgba(99,102,241,.1);padding:1px 4px;border-radius:3px;font-size:.82em;}}

@media print{{nav{{display:none;}} .collapsible-content{{display:block!important;}} body{{background:white;color:black;}} .card{{border:1px solid #ccc;break-inside:avoid;}}}}
</style>
</head>
<body>

<header class="report-header">
  <div class="report-title">Identity Support Bot — Performance Report</div>
  <div class="report-subtitle">identity-support channel · March 2026 · 6-week analysis · v4</div>
  <div class="report-meta">
    <span>Generated: {report_date}</span>
    <span>·</span><span>{total_requests} threads</span>
    <span>·</span><span>{total_esc} escalated ({round(total_esc/total_requests*100,1)}%)</span>
    <span>·</span><span>Bot Res Rate: {overall_res}%</span>
  </div>
</header>

<nav>
  <span class="nav-brand">📊 Bot Report</span>
  <a href="#summary">Summary</a>
  <a href="#weekly">Weekly</a>
  <a href="#intent">Intent</a>
  <a href="#intent-weekly">Intent by Week (6-Wk)</a>
  <a href="#week-deepdive">Weekly Deep Dive</a>
  <a href="#howto">How-To</a>
  <a href="#troubleshooting">Troubleshooting</a>
  <a href="#access">Access Requests</a>
  <a href="#notification">Notification</a>
  <a href="#thread-explorer">Thread Explorer</a>
  <a href="#recommendations">Recommendations</a>
  <a href="#themed-recommendations">Team Actions</a>
  <a href="#root-cause-analysis">Root Cause Analysis</a>
  <a href="#data-quality">Data Quality</a>
</nav>

<main>

<!-- ══ SUMMARY ════════════════════════════════════════════ -->
<section id="summary">
  <h2 class="section-title">Executive Summary</h2>
  <div class="metric-grid">
    <div class="metric-card">
      <div class="metric-value">{total_requests}</div>
      <div class="metric-label">Total Requests</div>
      <div class="metric-trend text-secondary">6-week total</div>
    </div>
    <div class="metric-card">
      <div class="metric-value" style="color:var(--danger)">{total_esc}</div>
      <div class="metric-label">Total Escalated</div>
      <div class="metric-trend text-muted">{round(total_esc/total_requests*100,1)}% of all threads</div>
    </div>
    <div class="metric-card">
      <div class="metric-value" style="color:{pct_color(overall_res)}">{overall_res}%</div>
      <div class="metric-label">Bot Resolution Rate</div>
      <div class="metric-trend {trend_cls}">{trend_sym} {abs(trend_val):.1f}% {trend_ref}</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{total_pf}</div>
      <div class="metric-label">Prefiltered</div>
      <div class="metric-trend text-muted">{total_pf_esc} direct-escalated</div>
    </div>
    <div class="metric-card">
      <div class="metric-value" style="color:var(--success)">{total_bot}</div>
      <div class="metric-label">Bot Resolved</div>
      <div class="metric-trend text-muted">of {total_att} attempted</div>
    </div>
    <div class="metric-card">
      <div class="metric-value" style="color:{'var(--danger)' if overall_surv < 10 else 'var(--warning)'}">{overall_surv}%</div>
      <div class="metric-label">Survey Fill Rate</div>
      <div class="metric-trend text-muted">{total_surv}/{total_esc} surveys</div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Bot Resolution Rate — Week over Week</div>
    <div class="chart-wrap-lg"><canvas id="resChart"></canvas></div>
  </div>
  <div class="chart-grid">
    <div class="card">
      <div class="card-title">Thread Volume by Week (Stacked)</div>
      <div class="chart-wrap"><canvas id="volChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Escalated vs Bot-Resolved</div>
      <div class="chart-wrap"><canvas id="escChart"></canvas></div>
    </div>
  </div>
</section>

<!-- ══ WEEKLY ══════════════════════════════════════════════ -->
<section id="weekly">
  <h2 class="section-title">Weekly Breakdown</h2>
  <p class="section-intro">Click any week to expand the intent breakdown. Click an intent to see the full deep dive for that week — engineer resolutions, patterns, and all escalated threads.</p>
  <div class="wk-table-wrap card" style="padding:0">
    <div class="wk-table-header">
      <span class="wk-cell wk-label">Week</span>
      <span class="wk-cell">Total</span>
      <span class="wk-cell">Prefiltered</span>
      <span class="wk-cell">Bot Attempted</span>
      <span class="wk-cell">Escalated</span>
      <span class="wk-cell">Bot Resolved</span>
      <span class="wk-cell">Resolution Rate</span>
      <span class="wk-cell">Survey Fill</span>
      <span style="width:24px;flex-shrink:0"></span>
    </div>
    {weekly_rows_html}
  </div>
  <p class="text-muted text-sm mt-md">Resolution Rate = Bot Resolved ÷ Bot Attempted. Prefilter-ESCALATED threads excluded from denominator.</p>
</section>

<!-- ══ INTENT OVERVIEW ════════════════════════════════════ -->
<section id="intent">
  <h2 class="section-title">Intent Category Breakdown</h2>
  <p class="section-intro">For each intent: total threads, how many the bot resolved, and how many escalated to engineers.</p>
  <div class="chart-grid">
    <div class="card">
      <div class="card-title">Bot Resolved vs Escalated by Intent</div>
      <div class="chart-wrap"><canvas id="intentChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Escalation Distribution</div>
      <div class="chart-wrap"><canvas id="intentDonut"></canvas></div>
    </div>
  </div>
  <div class="card">
    <div class="table-wrap">
      <table>
        <thead><tr><th>Intent</th><th>Total Requests</th><th>Bot Resolved</th><th>Escalated</th><th>FAQ Rendered</th></tr></thead>
        <tbody>{intent_table_rows}</tbody>
      </table>
    </div>
  </div>
</section>

<!-- ══ INTENT BY WEEK (6-week overview) ══════════════════ -->
<section id="intent-weekly">
  <h2 class="section-title">Intent Overview — Week by Week (6-Week)</h2>
  <p class="section-intro">Select an intent to see its full 6-week trend. For a deep dive into a specific week (with cluster analysis and engineer resolutions), see <a href="#week-deepdive" style="color:var(--accent-light)">Weekly Deep Dive</a> below.</p>
  {intent_weekly_html}
</section>

<!-- ══ WEEK DEEP DIVE ═════════════════════════════════════ -->
<section id="week-deepdive">
  <h2 class="section-title">Weekly Deep Dive — Intent &amp; Cluster Analysis per Week</h2>
  <p class="section-intro">Select a week to see all intents for that week — including escalation stats, how-to sub-breakdown, cluster analysis, engineer resolution patterns, and automation recommendations derived from conversation history.</p>
  {week_deepdive_html}
</section>

<!-- ══ HOW-TO ══════════════════════════════════════════════ -->
<section id="howto">
  <h2 class="section-title">How-To Escalation Deep Dive</h2>
  <p class="section-intro">
    Classifying all {howto_total_count} escalated how-to threads by context the bot had before escalating.
    Classification uses <code>faq_rendered</code> column and <code>llm_status</code> (READY/NO_CONTEXT) in orchestration trace.
    User feedback from <code>turn_user_input</code> (insufficient_answer / need_further_explanation) is also captured per thread.
    <br><br>
    <strong>Key finding:</strong> The 3 FAQ-rendered threads you identified (threadIds 1772739197, 1772827721, 1774943998) are
    correctly classified as <em>FAQ + RAG → No Context Follow-up</em> — the bot served an FAQ and found RAG docs for the initial
    answer, but when the user asked a follow-up question, the RAG system hit NO_CONTEXT. These were previously miscounted as "no context."
  </p>
  <div class="howto-stats">{howto_stat_html}</div>
  <div class="card">
    <div class="card-title">How-To Sub-Category Distribution</div>
    <div class="chart-wrap"><canvas id="howtoChart"></canvas></div>
  </div>
  {howto_detail_html}
</section>

<!-- ══ TROUBLESHOOTING ════════════════════════════════════ -->
<section id="troubleshooting">
  <h2 class="section-title">Troubleshooting Cluster Analysis</h2>
  <p class="section-intro">
    TF-IDF + KMeans clustering of {(df["intent_norm"]=="troubleshooting").sum()} troubleshooting threads (including {((df["intent_norm"]=="troubleshooting") & (df["escalated"]==True)).sum()} escalated).
    Each cluster shows: Bot Gap Analysis with explanations, what the bot answered, how engineers resolved it, and individual threads with full orchestration traces and Slack links.
  </p>
  {ts_html}
</section>

<!-- ══ ACCESS REQUESTS ════════════════════════════════════ -->
<section id="access">
  <h2 class="section-title">Access Request Cluster Analysis</h2>
  <p class="section-intro">
    Escalated access-request threads clustered by issue pattern. Each cluster includes engineer resolution extracted from Slack conversation history, full orchestration traces, and Slack thread links.
  </p>
  {ar_html}
</section>

<!-- ══ NOTIFICATION ═══════════════════════════════════════ -->
<section id="notification">
  <h2 class="section-title">Notification / Enquiry Cluster Analysis</h2>
  <p class="section-intro">
    Escalated notification and unclassified threads. Some are prefiltered; others are genuine support queries.
  </p>
  {notif_html}
</section>

<!-- ══ THREAD EXPLORER ════════════════════════════════════ -->
<section id="thread-explorer">
  <h2 class="section-title">Thread Explorer</h2>
  <p class="section-intro">Browse all {total_requests} threads. Filter by intent, sub-category (how-to breakdown), resolution status, or search by question text. Click Slack links to open the original thread.</p>
  <div class="card">
    <div class="explorer-controls">
      <input class="search-box" type="text" id="explorerSearch" placeholder="Search questions..." oninput="explorerApply()">
      <select class="filter-select" id="explorerIntent" onchange="explorerApply()">
        <option value="all">All Intents</option>
        <option value="troubleshooting">Troubleshooting</option>
        <option value="how-to">How-To</option>
        <option value="access-request">Access Request</option>
        <option value="notification">Notification</option>
        <option value="enquiry/other">Enquiry/Other</option>
      </select>
      <select class="filter-select" id="explorerSub" onchange="explorerApply()">
        <option value="all">All Sub-Categories</option>
        <option value="no_context">No Context</option>
        <option value="rag_insufficient">RAG Insufficient</option>
        <option value="rag_only">RAG Only</option>
        <option value="faq_and_rag_then_no_context">FAQ+RAG → No Context</option>
        <option value="faq_and_rag">FAQ + RAG</option>
        <option value="faq_only">FAQ Only</option>
      </select>
      <select class="filter-select" id="explorerStatus" onchange="explorerApply()">
        <option value="all">All Statuses</option>
        <option value="escalated">Escalated</option>
        <option value="bot">Bot Resolved</option>
      </select>
      <select class="filter-select" id="explorerRca" onchange="explorerApply()">
        <option value="all">All RCA Tags</option>
        <option value="INTENT_ERROR">Intent Error</option>
        <option value="INCOMPLETE_RESPONSE">Incomplete Response</option>
        <option value="AMBIGUOUS_USER">Ambiguous User</option>
        <option value="RETRIEVAL_FAILURE">Retrieval Failure</option>
        <option value="">No RCA Tag</option>
      </select>
      <span class="page-info" id="explorerInfo">Loading...</span>
    </div>
    <div class="table-wrap">
      <table class="explorer-table">
        <thead><tr><th>Week</th><th>Intent</th><th>Sub-Category</th><th>Status</th><th>RCA Tag</th><th>Question</th><th>Thread</th></tr></thead>
        <tbody id="explorerBody"></tbody>
      </table>
    </div>
    <div class="pagination" id="explorerPagination"></div>
  </div>
</section>

<!-- ══ RECOMMENDATIONS ═══════════════════════════════════ -->
<section id="recommendations">
  <h2 class="section-title">Prioritized Recommendations</h2>
  <p class="section-intro">Key findings and recommended actions based on 6-week analysis. For owner-specific action plans, see <a href="#themed-recommendations" style="color:var(--accent-light)">Team Actions</a> below.</p>
  {recs_html}
</section>

<!-- ══ THEMED RECOMMENDATIONS ════════════════════════════ -->
<section id="themed-recommendations">
  <h2 class="section-title">Team Action Plan</h2>
  <p class="section-intro">Recommendations split by responsible team, derived from cluster analysis, engineer resolution patterns, and KB coverage gaps identified across all 6 weeks.</p>
  {themed_recs_html}
</section>

<!-- ══ ROOT CAUSE ANALYSIS ═══════════════════════════════ -->
<section id="root-cause-analysis">
  <h2 class="section-title">Root Cause Analysis — Escalation Breakdown</h2>
  <p class="section-intro">
    Every escalated thread is tagged with a root cause using heuristic signals from the orchestration trace and conversation history.
    Tags are grouped into 5 priority categories. The priority bar shows the relative proportion of each group.
    Expand any group to see per-tag detection logic — exactly how the tag was assigned.
  </p>
  <div class="card">
    {root_cause_summary_html}
  </div>
</section>

<!-- ══ DATA QUALITY ══════════════════════════════════════ -->
<section id="data-quality">
  <h2 class="section-title">Data Quality &amp; Methodology Notes</h2>
  <div class="card">
    <ul class="dq-list">{dq_html}</ul>
  </div>
</section>

</main>

<script>
const DG='#2a2e3a',DT='#9aa0b2';
const COLORS=['#6366f1','#22c55e','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#f97316','#84cc16'];
const wLabels={json.dumps(week_labels_js)};
const wTotals={json.dumps(week_totals_js)};
const wEsc={json.dumps(week_esc_js)};
const wBot={json.dumps(week_bot_js)};
const wRes={json.dumps(week_res_js)};
const wPf={json.dumps(week_pf_js)};
const wAtt={json.dumps(week_att_js)};
const wNotRes=wAtt.map((v,i)=>v-wBot[i]);
const iLabels={json.dumps(intent_chart_labels)};
const iEsc={json.dumps(intent_chart_esc)};
const iBot={json.dumps(intent_chart_bot)};
const htLabels=['No Context','RAG Insufficient','RAG Only','FAQ+RAG→No Ctx','FAQ+RAG','FAQ Only'];
const htData={json.dumps(howto_js_data)};
const htColors=['#ef4444','#f59e0b','#22c55e','#8b5cf6','#6366f1','#3b82f6'];
const BS={{x:{{ticks:{{color:DT}},grid:{{color:DG}}}},y:{{ticks:{{color:DT}},grid:{{color:DG}},beginAtZero:true}}}};

// Charts
new Chart(document.getElementById('resChart'),{{type:'line',
  data:{{labels:wLabels,datasets:[{{label:'Resolution Rate (%)',data:wRes,borderColor:'#6366f1',backgroundColor:'rgba(99,102,241,.1)',fill:true,tension:.3,pointRadius:5}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:DT}}}}}},
    scales:{{...BS,y:{{...BS.y,min:0,ticks:{{callback:v=>v+'%',color:DT}}}}}}
  }}
}});
new Chart(document.getElementById('volChart'),{{type:'bar',
  data:{{labels:wLabels,datasets:[
    {{label:'Prefiltered',data:wPf,backgroundColor:'rgba(107,114,128,.6)',stack:'a'}},
    {{label:'Escalated',data:wEsc,backgroundColor:'rgba(239,68,68,.7)',stack:'a'}},
    {{label:'Bot Resolved',data:wBot,backgroundColor:'rgba(34,197,94,.7)',stack:'a'}},
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:DT}}}}}},
    scales:{{x:{{...BS.x,stacked:true}},y:{{...BS.y,stacked:true}}}}
  }}
}});
new Chart(document.getElementById('escChart'),{{type:'line',
  data:{{labels:wLabels,datasets:[
    {{label:'Escalated',data:wEsc,borderColor:'#ef4444',backgroundColor:'rgba(239,68,68,.08)',fill:true,tension:.3}},
    {{label:'Bot Resolved',data:wBot,borderColor:'#22c55e',backgroundColor:'rgba(34,197,94,.08)',fill:true,tension:.3}},
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:DT}}}}}},scales:BS}}
}});
new Chart(document.getElementById('intentChart'),{{type:'bar',
  data:{{labels:iLabels,datasets:[
    {{label:'Bot Resolved',data:iBot,backgroundColor:'rgba(34,197,94,.7)'}},
    {{label:'Escalated',data:iEsc,backgroundColor:'rgba(239,68,68,.7)'}},
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:DT}}}}}},scales:BS}}
}});
new Chart(document.getElementById('intentDonut'),{{type:'doughnut',
  data:{{labels:iLabels,datasets:[{{data:iEsc,backgroundColor:COLORS}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'bottom',labels:{{color:DT}}}}}}}}
}});
new Chart(document.getElementById('howtoChart'),{{type:'bar',
  data:{{labels:htLabels,datasets:[{{label:'Threads',data:htData,backgroundColor:htColors}}]}},
  options:{{responsive:true,maintainAspectRatio:false,indexAxis:'y',plugins:{{legend:{{display:false}}}},scales:BS}}
}});

// Collapsible
function toggleCollapsible(h){{
  const c=h.nextElementSibling,a=h.querySelector('.arrow');
  if(c&&c.classList.contains('collapsible-content')){{
    c.classList.toggle('active');
    if(a)a.textContent=c.classList.contains('active')?'▼':'▶';
  }}
}}

// Weekly breakdown — click week row to expand
function toggleWkRow(summaryEl, panelId){{
  const panel=document.getElementById(panelId);
  const arrow=summaryEl.querySelector('.wk-expand-arrow');
  if(!panel)return;
  const isOpen=panel.classList.contains('open');
  panel.classList.toggle('open');
  if(arrow)arrow.classList.toggle('open');
}}

// Weekly breakdown — click intent to expand deep dive
function toggleWkIntent(summaryEl, panelId){{
  const panel=document.getElementById(panelId);
  const arrow=summaryEl.querySelector('.wk-intent-arrow');
  if(!panel)return;
  panel.classList.toggle('open');
  if(arrow)arrow.classList.toggle('open');
}}

// Tab switching (intent weekly)
function switchTab(btn,panelId){{
  const container=btn.closest('.intent-weekly-tabs');
  container.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  container.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  const panel=document.getElementById(panelId);
  if(panel)panel.classList.add('active');
}}

// Smooth scroll
document.querySelectorAll('nav a').forEach(a=>{{
  a.addEventListener('click',function(e){{
    const href=this.getAttribute('href');
    if(href&&href.startsWith('#')){{
      e.preventDefault();
      const el=document.querySelector(href);
      if(el)el.scrollIntoView({{behavior:'smooth',block:'start'}});
    }}
  }});
}});

// ── Thread Explorer ────────────────────────────────────────
const ALL_THREADS = {thread_explorer_data};
let explorerFiltered = ALL_THREADS;
let explorerPage = 1;
const EXPLORER_PAGE_SIZE = 25;

const SUB_LABELS = {{
  'no_context': 'No Context',
  'rag_insufficient': 'RAG Insufficient',
  'rag_only': 'RAG Only',
  'faq_and_rag_then_no_context': 'FAQ+RAG→No Ctx',
  'faq_and_rag': 'FAQ+RAG',
  'faq_only': 'FAQ Only',
}};

function explorerApply(){{
  const search=document.getElementById('explorerSearch').value.toLowerCase();
  const intent=document.getElementById('explorerIntent').value;
  const sub=document.getElementById('explorerSub').value;
  const status=document.getElementById('explorerStatus').value;
  const rca=document.getElementById('explorerRca').value;

  explorerFiltered=ALL_THREADS.filter(t=>{{
    if(intent!=='all'&&t.intent!==intent)return false;
    if(sub!=='all'&&t.sub!==sub)return false;
    if(status==='escalated'&&!t.esc)return false;
    if(status==='bot'&&!t.bot)return false;
    if(rca!=='all'&&t.rca!==rca)return false;
    if(search&&!t.q.toLowerCase().includes(search))return false;
    return true;
  }});
  explorerPage=1;
  renderExplorer();
}}

function renderExplorer(){{
  const tbody=document.getElementById('explorerBody');
  const info=document.getElementById('explorerInfo');
  const total=explorerFiltered.length;
  const start=(explorerPage-1)*EXPLORER_PAGE_SIZE;
  const end=Math.min(start+EXPLORER_PAGE_SIZE,total);
  const page=explorerFiltered.slice(start,end);

  info.textContent=`Showing ${{start+1}}-${{end}} of ${{total}}`;

  tbody.innerHTML=page.map(t=>{{
    const statusBadge=t.esc?
      '<span class="badge badge-high">Escalated</span>':
      t.bot?'<span class="badge badge-low">Bot Resolved</span>':'<span class="badge badge-medium">Open</span>';
    const slackBtn=t.url&&t.url!=='nan'?
      `<a href="${{t.url}}" target="_blank" class="slack-link">↗ Slack</a>`:'';
    const subLabel=SUB_LABELS[t.sub]||t.sub||'—';
    const rcaB=t.rca?`<span class="badge badge-neutral">${{t.rca}}</span>`:'—';
    const q=(t.q||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    return `<tr>
      <td><span style="font-size:.72rem;color:var(--text-muted)">${{t.week}}</span></td>
      <td><span class="badge badge-neutral">${{t.intent}}</span></td>
      <td style="font-size:.72rem;color:var(--text-sec)">${{subLabel}}</td>
      <td>${{statusBadge}}</td>
      <td>${{rcaB}}</td>
      <td class="full-q" title="${{q}}">${{q.substring(0,120)}}${{q.length>120?'...':''}}</td>
      <td>${{slackBtn}}</td>
    </tr>`;
  }}).join('');

  // Pagination
  const pages=Math.ceil(total/EXPLORER_PAGE_SIZE);
  const pag=document.getElementById('explorerPagination');
  if(pages<=1){{pag.innerHTML='';return;}}

  let pagHtml='';
  const showPages=[1,pages];
  for(let p=Math.max(1,explorerPage-2);p<=Math.min(pages,explorerPage+2);p++)showPages.push(p);
  const uniquePages=[...new Set(showPages)].sort((a,b)=>a-b);
  let prev=-1;
  uniquePages.forEach(p=>{{
    if(prev>0&&p-prev>1)pagHtml+='<span style="color:var(--text-muted);padding:0 4px">…</span>';
    const ac=p===explorerPage?'active':'';
    pagHtml+=`<button class="page-btn ${{ac}}" onclick="explorerGo(${{p}})">${{p}}</button>`;
    prev=p;
  }});
  pag.innerHTML=pagHtml;
}}

function explorerGo(p){{explorerPage=p;renderExplorer();window.scrollTo(0,document.getElementById('thread-explorer').offsetTop-60);}}

// Init explorer
explorerApply();
</script>
</body>
</html>'''

# ── Main ──────────────────────────────────────────────────────────────────────
def run_analysis(csv_path, progress_callback=None):
    """
    Run full analysis pipeline on the given CSV path.
    Returns the HTML report as a string (does NOT write to disk).
    Optionally accepts a progress_callback(pct: float, msg: str) for UI updates.
    """
    def update(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    update(0.05, "Loading data...")
    df, wmap = load_and_prep(csv_path)

    update(0.10, "Computing weekly summary...")
    weekly = compute_weekly_summary(df)

    update(0.15, "Computing intent breakdown...")
    intent_data = []
    for intent, idf in df.groupby('intent_norm'):
        total = len(idf)
        esc = int((idf['escalated'] == True).sum())
        bot = int((idf['resolution_type'] == 'BOT').sum())
        faq = int(idf['faq_rendered'].sum())
        intent_data.append({
            'intent': intent, 'total': total,
            'bot_resolved': bot, 'escalated': esc,
            'faq_rendered': faq,
            'pct_esc': round(esc/total*100, 1) if total else 0,
            'pct_bot': round(bot/total*100, 1) if total else 0,
        })
    intent_data = sorted(intent_data, key=lambda x: -x['total'])

    update(0.20, "Computing intent weekly...")
    intent_weekly = compute_intent_weekly(df)

    update(0.25, "Analyzing how-to escalations...")
    howto_bd, howto_weekly, howto_details = analyze_howto(df)

    update(0.40, "Clustering troubleshooting...")
    ts_esc = df[(df['intent_norm'] == 'troubleshooting') & (df['escalated'] == True)]
    ts_clusters = build_cluster_data('troubleshooting', ts_esc)

    update(0.55, "Clustering access requests...")
    ar_esc = df[(df['intent_norm'] == 'access-request') & (df['escalated'] == True)]
    ar_clusters = build_cluster_data('access-request', ar_esc)

    update(0.65, "Clustering notifications...")
    notif_esc = df[df['intent_norm'].isin(['notification', 'enquiry/other']) & (df['escalated'] == True)]
    notif_clusters = build_cluster_data('notification/other', notif_esc)

    update(0.70, "Computing week deep dives (slowest step)...")
    all_weeks = sorted(df['week_seq'].unique())
    weekly_deepdives = []
    for i, wk in enumerate(all_weeks):
        wk_data = compute_week_deepdive(df, wk)
        weekly_deepdives.append(wk_data)
        update(0.70 + (i + 1) / len(all_weeks) * 0.15, f"Week {wk} deep dive...")

    update(0.88, "Generating recommendations...")
    total = len(df)
    total_esc = int(df['escalated'].sum())
    surv_cnt = int(df['rca_human_available'].sum()) if 'rca_human_available' in df.columns else 0
    rca_llm = int(df['rca_llm_available'].sum()) if 'rca_llm_available' in df.columns else 0
    faq_cnt = int(df['faq_rendered'].sum())

    dq_notes = [
        f"Dataset: {total} threads across {df['week_seq'].nunique()} weeks (Feb 23 – Mar 30, 2026), identity-support channel.",
        f"FAQ classification fix: The 3 user-identified FAQ threads (1772739197, 1772827721, 1774943998) are now correctly classified as 'FAQ+RAG→No Context Follow-up' — faq_rendered=True, READY in trace for initial answer, then NO_CONTEXT on follow-up turn.",
        f"faq_rendered=True for {faq_cnt} total threads ({round(faq_cnt/total*100,1)}% of all). Of 14 faq_rendered threads, 9 are escalated: 3 howTo, 4 troubleshooting, 1 accessRequest, 1 others.",
        f"Human survey completion: only {surv_cnt}/{total_esc} escalations ({round(surv_cnt/total_esc*100,1) if total_esc else 0}%) have human-filled RCA surveys. {rca_llm} rows have LLM auto-analysis.",
        f"rag_content, glean_content, rag_sources, glean_sources columns are empty for all rows — RAG usage inferred from ai_orchestrator_trace llm_status (READY=docs found, NO_CONTEXT=no docs).",
        f"Engineer identification: senderType='user' AND senderName ≠ original requester AND messageType ≠ IA_FEEDBACK_REQUEST.",
        f"User dissatisfaction captured from turn_user_input field in trace: insufficient_answer (~88 occurrences), need_further_explanation (~88), explicit escalate requests (~32).",
        f"Resolution rate formula: BOT-resolved ÷ (non-prefiltered + prefilter-SKIPPED threads). Prefilter-ESCALATED threads excluded from denominator.",
        f"Week 1 (Feb 23): 1 thread only. Week 6 (Mar 30): partial (68 threads). Trend comparisons use Wk2–Wk6.",
        f"Intent categories in raw data: howTo→how-to, accessRequest→access-request, troubleshooting, notification, NaN→enquiry/other.",
    ]

    overall_res = round(sum(w['bot_resolved'] for w in weekly) / sum(w['bot_attempted'] for w in weekly) * 100, 1)
    mw = [w for w in weekly if w['bot_attempted'] >= 20]
    recs = []
    if len(mw) >= 2:
        trend = mw[-1]['resolution_rate'] - mw[0]['resolution_rate']
        if trend < -5:
            recs.append({'priority': 'HIGH', 'category': 'Trend Alert',
                'text': f"Resolution rate declined {trend:+.1f}% from {mw[0]['week_label']} ({mw[0]['resolution_rate']}%, n={mw[0]['bot_attempted']}) to {mw[-1]['week_label']} ({mw[-1]['resolution_rate']}%, n={mw[-1]['bot_attempted']}). Week 6 is partial. Investigate bot quality or request complexity shift."})

    recs.append({'priority': 'HIGH', 'category': 'Survey Fill Rate — Critical',
        'text': f"Only {surv_cnt}/{total_esc} escalations have human RCA surveys (0.4%). Almost all insights rely on LLM auto-analysis. Implement mandatory post-escalation survey or auto-extract engineer resolution methods from conversation history."})

    no_ctx_total = howto_bd.get('no_context', 0) + howto_bd.get('rag_insufficient', 0) + howto_bd.get('faq_and_rag_then_no_context', 0)
    howto_esc_total = sum(howto_bd.values())
    if howto_esc_total > 0:
        recs.append({'priority': 'HIGH', 'category': 'Knowledge Base — How-To Coverage',
            'text': f"{no_ctx_total}/{howto_esc_total} ({round(no_ctx_total/howto_esc_total*100,0):.0f}%) escalated how-to threads hit NO_CONTEXT at some point. Bot has no KB coverage for most identity queries. Audit against top questions (account selector, offline tickets, GraphQL endpoints, SSO, onboarding) and add articles."})

    rag_insuff = howto_bd.get('rag_insufficient', 0)
    if rag_insuff > 0:
        recs.append({'priority': 'HIGH', 'category': 'RAG Multi-Turn — Insufficient Answers',
            'text': f"{rag_insuff} how-to threads follow this pattern: bot found docs and gave an initial answer → user said insufficient_answer or need_further_explanation → follow-up hit NO_CONTEXT. This is the dominant escalation theme. Add multi-turn RAG that retrieves fresh docs on each user follow-up turn, not just the initial query."})

    total_intent_err = sum(cl.get('root_causes', {}).get('Intent Error', 0) for g in [ts_clusters, ar_clusters, notif_clusters] for cl in g)
    if total_intent_err > 10:
        recs.append({'priority': 'HIGH', 'category': 'Intent Classification',
            'text': f"INTENT_ERROR root cause appears in {total_intent_err} threads across clusters. The howTo/troubleshooting boundary is blurry — both involve 'how do I fix/use X' phrasing. Retrain intent classifier with this dataset's actual question distribution."})

    total_incomp = sum(cl.get('root_causes', {}).get('Incomplete Response', 0) for g in [ts_clusters, ar_clusters, notif_clusters] for cl in g)
    if total_incomp > 10:
        recs.append({'priority': 'MEDIUM', 'category': 'Response Completeness',
            'text': f"INCOMPLETE_RESPONSE appears in {total_incomp} threads. Bot gives partial answers (API overview, doc link) but fails on follow-ups ('where exactly is the endpoint?', 'which environment?'). Add a 'did this answer your question?' confirmation step and multi-turn follow-up capability."})

    total_pf = sum(w['prefiltered'] for w in weekly)
    total_pf_esc = sum(w['pf_escalated'] for w in weekly)
    if total_pf > 0 and total_pf_esc / total_pf > 0.7:
        recs.append({'priority': 'MEDIUM', 'category': 'Prefilter Tuning',
            'text': f"{total_pf_esc}/{total_pf} prefiltered threads ({round(total_pf_esc/total_pf*100,0):.0f}%) were direct-escalated by the prefilter (not by bot failure). Review prefilter rules — some real support queries may be caught by announcement/notification rules."})

    recs.append({'priority': 'MEDIUM', 'category': 'User Dissatisfaction Signals',
        'text': "turn_user_input in orchestration traces captures 88 'insufficient_answer' and 88 'need_further_explanation' feedback events. Use these as training signals to identify which doc topics need richer coverage or clearer explanation depth."})

    recs = sorted(recs, key=lambda r: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[r['priority']])

    themed_recs = extract_themed_recommendations(howto_bd, ts_clusters, ar_clusters, notif_clusters, weekly)

    rc_counts_all = Counter()
    for cluster_list in [ts_clusters, ar_clusters, notif_clusters]:
        for cl in cluster_list:
            for thread in cl.get('threads', []):
                rc = thread.get('root_cause', 'OTHER') or 'OTHER'
                rc_counts_all[rc] += 1
    for cat in howto_details:
        for thread in cat.get('threads', []):
            rc = thread.get('root_cause', 'OTHER') or 'OTHER'
            rc_counts_all[rc] += 1

    analysis = {
        'weekly_summaries': weekly,
        'intent_breakdown': intent_data,
        'intent_weekly': intent_weekly,
        'weekly_deepdives': weekly_deepdives,
        'howto_breakdown': howto_bd,
        'howto_weekly': howto_weekly,
        'howto_details': howto_details,
        'ts_clusters': ts_clusters,
        'ar_clusters': ar_clusters,
        'notif_clusters': notif_clusters,
        'recommendations': recs,
        'themed_recommendations': themed_recs,
        'rc_counts_all': dict(rc_counts_all),
        'dq_notes': dq_notes,
    }

    update(0.95, "Rendering HTML report...")
    html_content = generate_html(analysis, df)
    update(1.0, "Done!")
    return html_content


def main():
    def _log(pct, msg):
        print(f"[{pct*100:3.0f}%] {msg}")

    html_content = run_analysis(CSV_PATH, progress_callback=_log)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html_content)

    size_kb = len(html_content) / 1024
    print(f"\n✓ Report saved: {OUTPUT_PATH}")
    print(f"  Size: {size_kb:.0f} KB")

if __name__ == '__main__':
    main()
