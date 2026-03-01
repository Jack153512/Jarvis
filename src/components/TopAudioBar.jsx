import React, { useEffect, useRef } from 'react';

const TopAudioBar = ({ audioData }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        const draw = () => {
            const width = canvas.width;
            const height = canvas.height;
            ctx.clearRect(0, 0, width, height);

            const barWidth = 4;
            const gap = 2;
            const totalBars = Math.floor(width / (barWidth + gap));

            // Simple visualization logic
            // Assuming audioData is an array of 0-255 values
            // We mirror it from center

            const center = width / 2;

            for (let i = 0; i < totalBars / 2; i++) {
                const value = audioData[i % audioData.length] || 0;
                const percent = value / 255;
                const barHeight = Math.max(2, percent * height);

                // Use the new palette: 0.2 opacity with a mix of cyan and royal blue
                const gradient = ctx.createLinearGradient(0, (height - barHeight) / 2, 0, (height + barHeight) / 2);
                gradient.addColorStop(0, 'rgba(92, 245, 255, 0.8)'); // Cyan
                gradient.addColorStop(1, 'rgba(21, 50, 120, 0.8)'); // Royal Blue

                ctx.fillStyle = gradient;

                // Right side
                ctx.fillRect(center + i * (barWidth + gap), (height - barHeight) / 2, barWidth, barHeight);

                // Left side
                ctx.fillRect(center - (i + 1) * (barWidth + gap), (height - barHeight) / 2, barWidth, barHeight);
            }
        };

        requestAnimationFrame(draw);
    }, [audioData]);

    return (
        <canvas
            ref={canvasRef}
            width={300}
            height={40}
            className="opacity-80"
        />
    );
};

export default React.memo(TopAudioBar);
