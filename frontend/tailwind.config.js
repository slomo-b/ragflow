/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        // Progress Keyframes
        "progress-stripes": {
          "0%": { backgroundPosition: "1rem 0" },
          "100%": { backgroundPosition: "0 0" },
        },
        "progress-pulse": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.6" },
        },
        "progress-shimmer": {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
        "progress-indeterminate": {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        // Progress Animations
        "progress-stripes": "progress-stripes 1s linear infinite",
        "progress-pulse": "progress-pulse 2s ease-in-out infinite",
        "progress-shimmer": "progress-shimmer 2s infinite",
        "progress-indeterminate": "progress-indeterminate 2s linear infinite",
      },
      spacing: {
        "progress-xs": "2px",
        "progress-sm": "4px", 
        "progress-md": "8px",
        "progress-lg": "12px",
        "progress-xl": "16px",
      },
    },
  },
  plugins: [
    require("tailwindcss-animate"),
    // Minimaler Progress Plugin
    function({ addUtilities }) {
      addUtilities({
        '.progress-root': {
          position: 'relative',
          overflow: 'hidden',
          borderRadius: '9999px',
          backgroundColor: 'hsl(var(--progress-background))',
          width: '100%',
        },
        '.progress-indicator': {
          height: '100%',
          width: '100%',
          flex: '1',
          backgroundColor: 'hsl(var(--progress-foreground))',
          transition: 'transform 0.5s ease-out',
        },
        // Größen
        '.progress-xs': { height: '2px' },
        '.progress-sm': { height: '4px' },
        '.progress-md': { height: '8px' },
        '.progress-lg': { height: '12px' },
        '.progress-xl': { height: '16px' },
      })
    }
  ],
}