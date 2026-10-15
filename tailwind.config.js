/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: "#161A1D",
          soft: "#575F66",
          faint: "#8A9199",
        },
        surface: {
          DEFAULT: "#FFFFFF",
          sunk: "#F5F6F4",
          wash: "#EFF1EE",
        },
        line: {
          DEFAULT: "#E3E5E1",
          strong: "#CBCFC9",
        },
        signal: {
          DEFAULT: "#2A6F6B",
          dark: "#1F5450",
          soft: "#E3EFEE",
        },
        amber: {
          DEFAULT: "#B5762A",
          soft: "#F5EBDC",
        },
        good: {
          DEFAULT: "#2F7A4F",
          soft: "#E4F1E8",
        },
        bad: {
          DEFAULT: "#B4432F",
          soft: "#F6E7E3",
        },
      },
      fontFamily: {
        sans: ["'IBM Plex Sans'", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
      borderRadius: {
        sm: "4px",
        DEFAULT: "6px",
        lg: "10px",
      },
      boxShadow: {
        card: "0 1px 2px rgba(22,26,29,0.04)",
      },
    },
  },
  plugins: [],
};
