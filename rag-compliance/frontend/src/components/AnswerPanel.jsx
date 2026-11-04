/**
 * components/AnswerPanel.jsx — Streaming answer with token-by-token rendering,
 * refusal state, timing display, and feedback buttons.
 */
import React, { useEffect, useRef } from 'react';

function ParsedAnswer({ text }) {
  // Convert [1] citations to styled spans
  const parts = text.split(/(\[\d+\])/g);
  return (
    <div className="answer-text" role="region" aria-label="Answer">
      {parts.map((part, i) => {
        if (/^\[\d+\]$/.test(part)) {
          return (
            <sup key={i} className="citation-ref" title={`Source ${part.slice(1, -1)}`}>
              {part}
            </sup>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </div>
  );
}

function TimingBar({ timings }) {
  if (!timings) return null;
  return (
    <div className="timing-bar" role="status" aria-label="Performance timings">
      <span className="timing-item">
        <span className="timing-label">Retrieval</span>
        <span className="timing-value">{timings.retrieval_ms}ms</span>
      </span>
      {timings.ttft_ms != null && (
        <span className="timing-item">
          <span className="timing-label">First token</span>
          <span className="timing-value">{timings.ttft_ms}ms</span>
        </span>
      )}
      <span className="timing-item">
        <span className="timing-label">Total</span>
        <span className="timing-value">{timings.total_ms}ms</span>
      </span>
    </div>
  );
}

export function AnswerPanel({ state, onFeedback }) {
  const { status, answer, refused, timings, error } = state;
  const cursorRef = useRef(null);

  useEffect(() => {
    if (status === 'streaming' && cursorRef.current) {
      cursorRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [answer, status]);

  if (status === 'idle') {
    return (
      <div className="answer-panel answer-panel--empty">
        <div className="answer-empty-icon">⚖️</div>
        <h2 className="answer-empty-title">RBI &amp; SEBI Compliance Assistant</h2>
        <p className="answer-empty-subtitle">
          Ask any question about Indian regulatory circulars, master directions,
          or compliance requirements. Answers are grounded in official documents
          with numbered citations.
        </p>
        <div className="answer-chips">
          {[
            'KYC requirements',
            'NPA classification',
            'Basel III capital norms',
            'SEBI LODR disclosure',
          ].map(chip => (
            <span key={chip} className="answer-chip">{chip}</span>
          ))}
        </div>
      </div>
    );
  }

  if (status === 'loading') {
    return (
      <div className="answer-panel answer-panel--loading" role="status" aria-live="polite">
        <div className="pulse-loader">
          <span /><span /><span />
        </div>
        <p>Searching regulatory documents…</p>
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div className="answer-panel answer-panel--error" role="alert">
        <div className="error-icon">⚠️</div>
        <h3>Connection Error</h3>
        <p>{error}</p>
        <p className="error-hint">Make sure the backend is running at <code>/api</code>.</p>
      </div>
    );
  }

  return (
    <div className={`answer-panel ${refused ? 'answer-panel--refused' : ''}`}>
      {refused && (
        <div className="refusal-badge" role="alert">
          <span>🚫</span>
          <span>No sufficiently relevant circulars found for this query.</span>
        </div>
      )}

      <div className="answer-body">
        <ParsedAnswer text={answer} />
        {status === 'streaming' && (
          <span ref={cursorRef} className="streaming-cursor" aria-hidden="true">▋</span>
        )}
      </div>

      {status === 'done' && (
        <>
          <TimingBar timings={timings} />
          {!refused && onFeedback && (
            <div className="feedback-bar">
              <span className="feedback-label">Was this helpful?</span>
              <button
                id="feedback-up"
                className="feedback-btn feedback-btn--up"
                onClick={() => onFeedback(1)}
                aria-label="Thumbs up"
                title="Helpful"
              >
                👍
              </button>
              <button
                id="feedback-down"
                className="feedback-btn feedback-btn--down"
                onClick={() => onFeedback(-1)}
                aria-label="Thumbs down"
                title="Not helpful"
              >
                👎
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
