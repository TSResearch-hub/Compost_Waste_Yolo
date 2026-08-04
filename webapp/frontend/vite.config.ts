import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// Le build (dist/) est servi par FastAPI sur la même origine — voir
// app/main.py. En dev, l'API tourne sur :8000 et Vite proxifie /api :
// le cookie de session reste en même origine dans les deux modes.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
})
