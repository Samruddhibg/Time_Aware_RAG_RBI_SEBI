/**
 * App.jsx — Root layout: header + search + answer + sources.
 */
import React, { useState, useCallback } from 'react';
import { useAsk } from './hooks/useAsk.js';
import { SearchBar } from './components/SearchBar.jsx';
import { AnswerPanel } from './components/AnswerPanel.jsx';
import { SourceList } from './components/SourceList.jsx';
import { ModeToggle } from './components/ModeToggle.jsx';
import './styles/components.css';

export default function App() {
  const [historical, setHistorical] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [feedbackGiven, setFeedbackGiven] = useState(false);
  const { ask, cancel, feedback, state } = useAsk();

  const handleSubmit = useCallback((question) => {
    setCurrentQuestion(question);
    setFeedbackGiven(false);
    ask(question, historical);
  }, [ask, historical]);

  const handleFeedback = useCallback((rating) => {
    if (feedbackGiven) return;
    feedback(rating);
    setFeedbackGiven(true);
  }, [feedback, feedbackGiven]);

  const isActive = state.status === 'loading' || state.status === 'streaming';

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header" role="banner">
        <div className="header-inner">
          <div className="header-brand">
            <div className="brand-logo" aria-hidden="true">
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <circle cx="16" cy="16" r="14" fill="url(#grad)" />
                <path d="M10 16l4 4 8-8" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                <defs>
                  <linearGradient id="grad" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stopColor="#6366f1" />
                    <stop offset="100%" stopColor="#06b6d4" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <div className="brand-text">
              <span className="brand-name">GuardianShield</span>
              <span className="brand-sub">RBI &amp; SEBI Compliance AI</span>
            </div>
          </div>
          <ModeToggle historical={historical} onChange={setHistorical} />
          <div className="header-badges">
            <span className="header-badge">pgvector</span>
            <span className="header-badge">Hybrid RAG</span>
            <span className="header-badge header-badge--accent">Time-Aware</span>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="app-main" id="main-content">
        {/* Search section */}
        <section className="search-section" aria-label="Search">
          <div className="search-section-inner">
            {currentQuestion && (
              <div className="current-question" role="status">
                <span className="current-question-label">Q:</span>
                <span className="current-question-text">{currentQuestion}</span>
              </div>
            )}
            <SearchBar
              onSubmit={handleSubmit}
              onCancel={cancel}
              isLoading={isActive}
              historical={historical}
            />
          </div>
        </section>

        {/* Results layout */}
        {(state.status !== 'idle' || state.sources?.length > 0) && (
          <div className="results-layout">
            {/* Answer */}
            <div className="results-answer">
              <AnswerPanel
                state={state}
                onFeedback={feedbackGiven ? null : handleFeedback}
              />
              {feedbackGiven && (
                <div className="feedback-thanks" role="status">
                  ✓ Thank you for your feedback!
                </div>
              )}
            </div>
            {/* Sources */}
            {state.sources?.length > 0 && (
              <div className="results-sources">
                <SourceList sources={state.sources} />
              </div>
            )}
          </div>
        )}

        {/* Idle state */}
        {state.status === 'idle' && (
          <div className="results-layout results-layout--idle">
            <div className="results-answer">
              <AnswerPanel state={state} />
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer" role="contentinfo">
        <div className="footer-inner">
          <span>Time-Aware RAG for RBI/SEBI Circulars</span>
          <span className="footer-sep">·</span>
          <span>Hybrid retrieval: dense + BM25 + RRF</span>
          <span className="footer-sep">·</span>
          <span>Answers grounded in official circulars only</span>
        </div>
      </footer>
    </div>
  );
}
