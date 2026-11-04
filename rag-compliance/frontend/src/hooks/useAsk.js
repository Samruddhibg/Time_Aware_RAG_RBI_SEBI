/**
 * hooks/useAsk.js — Streaming state management hook.
 *
 * Returns:
 *   { ask, cancel, state }
 *   state: { status, tokens, sources, queryId, refused, timings, error }
 */
import { useState, useRef, useCallback } from 'react';
import { askStream, submitFeedback } from '../api/askStream.js';

const INITIAL_STATE = {
  status: 'idle',        // 'idle' | 'loading' | 'streaming' | 'done' | 'error'
  tokens: [],
  answer: '',
  sources: [],
  queryId: null,
  refused: false,
  timings: null,
  error: null,
};

export function useAsk() {
  const [state, setState] = useState(INITIAL_STATE);
  const controllerRef = useRef(null);

  const cancel = useCallback(() => {
    if (controllerRef.current) {
      controllerRef.current.abort();
      controllerRef.current = null;
    }
    setState(prev => ({ ...prev, status: 'done' }));
  }, []);

  const ask = useCallback(async (question, historical = false) => {
    // Cancel any in-flight request
    if (controllerRef.current) controllerRef.current.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setState({
      ...INITIAL_STATE,
      status: 'loading',
    });

    try {
      const tokens = [];
      let sources = [];
      let queryId = null;
      let refused = false;
      let timings = null;

      for await (const event of askStream({ question, historical, controller })) {
        if (controller.signal.aborted) break;

        switch (event.type) {
          case 'sources':
            sources = event.data;
            setState(prev => ({
              ...prev,
              status: 'streaming',
              sources,
            }));
            break;

          case 'token':
            tokens.push(event.data);
            setState(prev => ({
              ...prev,
              status: 'streaming',
              tokens: [...tokens],
              answer: tokens.join(''),
            }));
            break;

          case 'done':
            queryId = event.data.query_id;
            refused = event.data.refused;
            timings = {
              retrieval_ms: event.data.retrieval_ms,
              ttft_ms: event.data.ttft_ms,
              total_ms: event.data.total_ms,
            };
            setState(prev => ({
              ...prev,
              status: 'done',
              queryId,
              refused,
              timings,
              answer: tokens.join(''),
            }));
            break;

          case 'error':
            throw new Error(event.data);
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      setState(prev => ({
        ...prev,
        status: 'error',
        error: err.message,
      }));
    } finally {
      controllerRef.current = null;
    }
  }, []);

  const feedback = useCallback(async (rating, comment = null) => {
    if (!state.queryId) return;
    try {
      await submitFeedback({ queryId: state.queryId, rating, comment });
    } catch (e) {
      console.error('Feedback error:', e);
    }
  }, [state.queryId]);

  return { ask, cancel, feedback, state };
}
