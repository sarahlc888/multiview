import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'serve-outputs',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          // Serve triplets.json from outputs directory
          if (req.url?.startsWith('/triplets/')) {
            const tripletPath = req.url.replace('/triplets/', '');
            const outputsDirs = ['outputs/benchmark_fuzzy_debug', 'outputs/benchmark', 'outputs/benchmark_fuzzy_debug2'];

            for (const outputsDir of outputsDirs) {
              const fullPath = path.join('..', outputsDir, 'triplets', tripletPath);
              if (fs.existsSync(fullPath)) {
                res.setHeader('Content-Type', 'application/json');
                fs.createReadStream(fullPath).pipe(res);
                return;
              }
            }
          }
          next();
        });
      },
    },
  ],
  server: {
    port: 5173,
    open: true,
  },
  publicDir: '../outputs/viz',
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
        compare: path.resolve(__dirname, 'compare.html'),
      },
    },
  },
})
