/**
 * api/askStream.js — SSE-over-fetch parser for /api/ask.
 *
 * Usage:
 *   const { stream, cancel } = askStream({ question, historical });
 *   for await (const event of stream) { ... }
 */

const API_BASE = import.meta.env.VITE_API_URL || '';

/**
 * @param {Object} opts
 * @param {string} opts.question
 * @param {boolean} opts.historical
 * @param {AbortController} opts.controller
 * @returns {AsyncGenerator<{type: string, data: any}>}
 */
export async function* askStream({ question, historical = false, controller }) {
  const signal = controller?.signal;

  const response = await fetch(`${API_BASE}/api/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, historical }),
    signal,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API error ${response.status}: ${text}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // incomplete line

      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const raw = line.slice(5).trim();
        if (raw === '[DONE]') return;
        try {
          const event = JSON.parse(raw);
          yield event;
        } catch {
          // malformed line — skip
        }
      }
    }
  } finally {
    reader.cancel();
  }
}

/**
 * Submit feedback for a completed query.
 */
export async function submitFeedback({ queryId, rating, comment = null }) {
  const response = await fetch(`${API_BASE}/api/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query_id: queryId, rating, comment }),
  });
  if (!response.ok) throw new Error(`Feedback error ${response.status}`);
  return response.json();
}
