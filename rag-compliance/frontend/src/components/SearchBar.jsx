/**
 * components/SearchBar.jsx — Query input with submit/cancel controls.
 */
import React, { useRef, useEffect } from 'react';

const PLACEHOLDER_QUERIES = [
  'What is the KYC cash transaction limit under RBI?',
  'What are the NPA classification timelines?',
  'What are SEBI\'s insider trading restrictions?',
  'What is the minimum CRAR for scheduled banks?',
  'What are priority sector lending targets for banks?',
];

export function SearchBar({ onSubmit, onCancel, isLoading, historical }) {
  const inputRef = useRef(null);
  const placeholderIdx = useRef(Math.floor(Math.random() * PLACEHOLDER_QUERIES.length));

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    const value = inputRef.current?.value?.trim();
    if (!value || isLoading) return;
    onSubmit(value);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSubmit(e);
    }
    if (e.key === 'Escape' && isLoading) {
      onCancel();
    }
  };

  return (
    <form className="search-form" onSubmit={handleSubmit} role="search">
      <div className="search-wrapper">
        <div className="search-icon-wrap">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <path d="M21 21l-4.35-4.35" />
          </svg>
        </div>
        <textarea
          id="search-input"
          ref={inputRef}
          className="search-input"
          placeholder={PLACEHOLDER_QUERIES[placeholderIdx.current]}
          onKeyDown={handleKeyDown}
          rows={2}
          disabled={isLoading}
          aria-label="Ask a compliance question"
          aria-multiline="true"
        />
        <div className="search-actions">
          {isLoading ? (
            <button
              id="cancel-btn"
              type="button"
              className="btn btn--cancel"
              onClick={onCancel}
              aria-label="Cancel streaming"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
              Stop
            </button>
          ) : (
            <button
              id="submit-btn"
              type="submit"
              className="btn btn--primary"
              disabled={isLoading}
              aria-label="Submit question"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
              Ask
            </button>
          )}
        </div>
      </div>
      {historical && (
        <div className="search-badge search-badge--historical" role="status">
          <span>🕐</span> Historical mode — superseded circulars included
        </div>
      )}
    </form>
  );
}
