import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        neon: "#FFFF00",
        obsidian: "#0A0A0A",
        charcoal: "#111111",
      },
      boxShadow: {
        neon: "0 0 25px rgba(255, 255, 0, 0.22)",
        "neon-sm": "0 0 12px rgba(255, 255, 0, 0.16)",
      },
    },
  },
  plugins: [],
};

export default config;
