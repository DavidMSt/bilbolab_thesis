import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  server: {
    host: true,     // listen on all interfaces (useful for testing on devices)
    port: 5173
  },
  preview: {
    host: true,
    port: 5174
  },
  // Make sure the Node 'events' polyfill is optimized for the browser bundle
  optimizeDeps: {
    include: ['events']
  },
  build: {
    target: 'esnext',
    sourcemap: true
  }
});