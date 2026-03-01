import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

const Visualizer = ({ audioData, isListening, intensity = 0, width = 600, height = 400 }) => {
    const canvasRef = useRef(null);
    const particlesRef = useRef([]);

    // Initialize particles once
    useEffect(() => {
        const particles = [];
        for (let i = 0; i < 20; i++) {
            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                size: Math.random() * 1.5 + 0.5,
                speedX: (Math.random() - 0.5) * 0.3,
                speedY: (Math.random() - 0.5) * 0.3,
                opacity: Math.random() * 0.3 + 0.1
            });
        }
        particlesRef.current = particles;
    }, [width, height]);

    // Use a ref for audioData to avoid re-creating the animation loop on every frame
    const audioDataRef = useRef(audioData);
    const intensityRef = useRef(intensity);
    const isListeningRef = useRef(isListening);

    useEffect(() => {
        audioDataRef.current = audioData;
        intensityRef.current = intensity;
        isListeningRef.current = isListening;
    }, [audioData, intensity, isListening]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        // Ensure canvas internal resolution matches display size for sharpness
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext('2d');
        let animationId;

        const draw = () => {
            const w = canvas.width;
            const h = canvas.height;
            const centerX = w / 2;
            const centerY = h / 2;

            // Use current audio data from ref if we were using it for visualization
            // Currently the effect only uses 'intensity', passed as prop. 
            // To ensure we aren't re-triggering this effect constantly, we use refs.

            const currentIntensity = intensityRef.current;
            const currentIsListening = isListeningRef.current;

            const baseRadius = Math.min(w, h) * 0.25;
            const radius = baseRadius + (currentIntensity * 40);

            ctx.clearRect(0, 0, w, h);

            // Draw Particles
            ctx.shadowBlur = 0;
            const particles = particlesRef.current;
            particles.forEach(p => {
                // Update particle position
                p.x += p.speedX * (1 + currentIntensity * 5);
                p.y += p.speedY * (1 + currentIntensity * 5);

                // Wrap around screen
                if (p.x < 0) p.x = w;
                if (p.x > w) p.x = 0;
                if (p.y < 0) p.y = h;
                if (p.y > h) p.y = 0;

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(92, 245, 255, ${p.opacity * (0.5 + currentIntensity)})`;
                ctx.fill();
            });

            // Base Circle (Glow)
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius - 10, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(92, 245, 255, 0.05)';
            ctx.lineWidth = 1;
            ctx.stroke();

            if (!currentIsListening) {
                // Idle State: Breathing Circle
                const time = Date.now() / 1000;
                const breath = Math.sin(time * 2) * 5;

                ctx.beginPath();
                ctx.arc(centerX, centerY, radius + breath, 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(92, 245, 255, 0.4)'; // Subtle Cyan
                ctx.lineWidth = 3;
                ctx.shadowBlur = 15;
                ctx.shadowColor = '#153278'; // Premium Royal Blue Glow
                ctx.stroke();
                ctx.shadowBlur = 0;
            } else {
                // Active State: Just the Circle causing the pulse
                ctx.beginPath();
                ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
                ctx.strokeStyle = '#5cf5ff'; // Vibrant Cyan
                ctx.lineWidth = 4;
                ctx.shadowBlur = 25;
                ctx.shadowColor = 'rgba(92, 245, 255, 0.6)';
                ctx.stroke();
                ctx.shadowBlur = 0;
            }

            animationId = requestAnimationFrame(draw);
        };

        draw();
        return () => cancelAnimationFrame(animationId);
    }, [width, height]);

    return (
        <div className="relative" style={{ width, height }}>
            {/* Central Logo/Text */}
            <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
                <motion.div
                    animate={{ scale: isListening ? [1, 1.1, 1] : 1 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                    className="text-cyan-100 font-bold tracking-widest drop-shadow-[0_0_15px_rgba(34,211,238,0.8)]"
                    style={{ fontSize: Math.min(width, height) * 0.1 }}
                >
                    Jarvis
                </motion.div>
            </div>

            <canvas
                ref={canvasRef}
                style={{ width: '100%', height: '100%' }}
            />
        </div>
    );
};

export default Visualizer;
