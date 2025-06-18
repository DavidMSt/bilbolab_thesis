import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue'; // ✅ Import Vue plugin

export default defineConfig({
    root: 'src', // Only if your index.html is in src/
    publicDir: 'lib/assets',
    plugins: [
        vue() // ✅ Add Vue plugin here
        // babylonPlugin()
    ],
    resolve: {
        alias: {
            // ✅ Add alias so Vue uses the build that supports templates
            'vue': 'vue/dist/vue.esm-bundler.js'
        }
    },
    build: {
        outDir: '../dist' // Build output goes to dist/ at the root
    },
    server: {
        host: true,  // 👈 This allows external access (e.g. via IP)
        port: 9200
    }
});
