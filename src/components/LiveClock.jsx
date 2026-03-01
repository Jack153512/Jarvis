import React, { useState, useEffect } from 'react';

/**
 * Self-contained clock component with its own 1-second timer.
 * Lives in its own module so React Fast Refresh can track it correctly
 * and so App never re-renders its entire tree just to tick the clock.
 */
const LiveClock = React.memo(function LiveClock() {
    const [time, setTime] = useState(() => new Date());

    useEffect(() => {
        const id = setInterval(() => setTime(new Date()), 1000);
        return () => clearInterval(id);
    }, []);

    return (
        <span className="tabular-nums">
            {time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
        </span>
    );
});

export default LiveClock;
