/**
 * components/SourceList.jsx — Citation sources with circular numbers, dates, and status badges.
 */
import React, { useState } from 'react';

const STATUS_CONFIG = {
  active: { label: 'Active', cls: 'status--active', icon: '🟢' },
  superseded: { label: 'Superseded', cls: 'status--superseded', icon: '🔴' },
  withdrawn: { label: 'Withdrawn', cls: 'status--withdrawn', icon: '⚪' },
};

function SourceCard({ source, index }) {
  const [expanded, setExpanded] = useState(false);
  const status = STATUS_CONFIG[source.status] || STATUS_CONFIG.active;

  return (
    <div
      id={`source-${index}`}
      className={`source-card ${source.status !== 'active' ? 'source-card--stale' : ''}`}
      role="article"
      aria-label={`Source ${index}: ${source.circular_no || source.document_id}`}
    >
      <div className="source-header">
        <span className="source-number">[{index}]</span>
        <div className="source-meta">
          <span className="source-circular">
            {source.circular_no || source.document_id}
          </span>
          {source.issue_date && (
            <span className="source-date">
              📅 {source.issue_date}
            </span>
          )}
        </div>
        <span className={`source-status ${status.cls}`}>
          {status.icon} {status.label}
        </span>
        <span className="source-sim">
          {(source.similarity * 100).toFixed(0)}% match
        </span>
      </div>

      {source.title && (
        <div className="source-title">{source.title}</div>
      )}

      {source.section && (
        <div className="source-section">
          <span className="source-section-icon">📑</span>
          {source.section}
        </div>
      )}

      <div className="source-snippet">
        {expanded ? source.text || source.snippet : source.snippet}
      </div>

      {source.text && source.text.length > 200 && (
        <button
          className="source-expand-btn"
          onClick={() => setExpanded(e => !e)}
          aria-expanded={expanded}
        >
          {expanded ? '▲ Show less' : '▼ Show full passage'}
        </button>
      )}
    </div>
  );
}

export function SourceList({ sources }) {
  if (!sources || sources.length === 0) return null;

  const staleSources = sources.filter(s => s.status !== 'active');

  return (
    <aside className="source-list" aria-label="Retrieved sources">
      <div className="source-list-header">
        <h3 className="source-list-title">
          <span>📚</span>
          <span>Sources ({sources.length})</span>
        </h3>
        {staleSources.length > 0 && (
          <div className="stale-warning" role="alert">
            ⚠️ {staleSources.length} superseded document{staleSources.length > 1 ? 's' : ''} in results
          </div>
        )}
      </div>
      <div className="source-list-body">
        {sources.map((source, i) => (
          <SourceCard key={source.chunk_id || i} source={source} index={i + 1} />
        ))}
      </div>
    </aside>
  );
}
