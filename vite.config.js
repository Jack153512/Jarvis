import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react({ fastRefresh: false })],
    base: './', // Important for Electron
    server: {
        port: 5173,
    }
    ,
    // MediaPipe Tasks can be sensitive to pre-bundling; excluding it avoids
    // Vite optimizing it into a form that may select Node-specific paths.
    optimizeDeps: {
        exclude: ['@mediapipe/tasks-vision'],
    },
})
