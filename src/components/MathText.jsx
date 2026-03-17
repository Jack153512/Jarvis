/**
 * Renders text with LaTeX math expressions.
 * Supports \( \) for inline math and \[ \] for block (display) math.
 * Uses KaTeX for rendering — no raw backslashes or delimiters shown.
 */
import React from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

/**
 * Parse text and split into segments: plain text or math (inline/block).
 * Handles \( \) and \[ \] delimiters.
 */
function parseMathSegments(text) {
    if (!text || typeof text !== 'string') return [{ type: 'text', content: text || '' }];

    const segments = [];
    let i = 0;

    while (i < text.length) {
        const inlineStart = text.indexOf('\\(', i);
        const blockStart = text.indexOf('\\[', i);

        const next = Math.min(
            inlineStart === -1 ? Infinity : inlineStart,
            blockStart === -1 ? Infinity : blockStart
        );

        if (next === Infinity) {
            segments.push({ type: 'text', content: text.slice(i) });
            break;
        }

        if (next > i) {
            segments.push({ type: 'text', content: text.slice(i, next) });
        }

        const isBlock = blockStart !== -1 && blockStart <= inlineStart;
        const openDelim = isBlock ? '\\[' : '\\(';
        const closeDelim = isBlock ? '\\]' : '\\)';
        const closeIdx = text.indexOf(closeDelim, next + openDelim.length);

        if (closeIdx === -1) {
            segments.push({ type: 'text', content: text.slice(next) });
            break;
        }

        const mathContent = text.slice(next + openDelim.length, closeIdx);
        segments.push({ type: isBlock ? 'block' : 'inline', content: mathContent });
        i = closeIdx + closeDelim.length;
    }

    return segments;
}

/**
 * Render a single math expression with KaTeX.
 */
function renderMath(content, displayMode) {
    return katex.renderToString(content.trim(), {
        displayMode: !!displayMode,
        throwOnError: false,
        strict: false,
    });
}

/**
 * Renders text with LaTeX math. Use in place of plain {text} for chat messages.
 */
function MathText({ children, className = '' }) {
    const text = typeof children === 'string' ? children : String(children ?? '');
    const segments = parseMathSegments(text);

    if (segments.length === 1 && segments[0].type === 'text') {
        return <span className={className}>{segments[0].content}</span>;
    }

    return (
        <span className={className}>
            {segments.map((seg, idx) => {
                if (seg.type === 'text') {
                    return <React.Fragment key={idx}>{seg.content}</React.Fragment>;
                }
                const html = renderMath(seg.content, seg.type === 'block');
                return (
                    <span
                        key={idx}
                        className={seg.type === 'block' ? 'katex-block my-2 block' : 'katex-inline'}
                        dangerouslySetInnerHTML={{ __html: html }}
                    />
                );
            })}
        </span>
    );
}

export default MathText;
