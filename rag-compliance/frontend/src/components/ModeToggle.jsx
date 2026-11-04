/**
 * components/ModeToggle.jsx — Toggle between Current Rules and Historical mode.
 */
import React from 'react';

export function ModeToggle({ historical, onChange }) {
  return (
    <div className="mode-toggle" role="group" aria-label="Query mode">
      <button
        id="mode-current"
        className={`mode-btn ${!historical ? 'mode-btn--active' : ''}`}
        onClick={() => onChange(false)}
        aria-pressed={!historical}
      >
        <span className="mode-icon">🟢</span>
        <span className="mode-label">Current Rules</span>
      </button>
      <button
        id="mode-historical"
        className={`mode-btn ${historical ? 'mode-btn--active' : ''}`}
        onClick={() => onChange(true)}
        aria-pressed={historical}
      >
        <span className="mode-icon">🕐</span>
        <span className="mode-label">Historical</span>
      </button>
    </div>
  );
}
