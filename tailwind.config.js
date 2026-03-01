/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                mono: ['"Share Tech Mono"', 'monospace'], // Sci-fi font
            },
            colors: {
                premium: {
                    dark: '#010409',
                    navy: '#050a15',
                    deep: '#0a0f1d',
                    blue: '#0d1117',
                    royal: '#161b22',
                },
                cyan: {
                    400: '#22d3ee',
                    500: '#06b6d4',
                    900: '#164e63',
                }
            },
            backgroundImage: {
                'premium-gradient': 'linear-gradient(135deg, #010409 0%, #050a15 50%, #0a0f1d 100%)',
            }
        },
    },
    plugins: [],
}
