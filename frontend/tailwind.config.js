/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        },
        recycling: {
          plastic: '#fbbf24',
          paper: '#3b82f6',
          glass: '#22c55e',
          metal: '#f97316',
          organic: '#92400e',
          ewaste: '#ef4444',
          textile: '#a855f7',
          cardboard: '#6366f1',
          trash: '#6b7280',
        }
      }
    },
  },
  plugins: [],
}
