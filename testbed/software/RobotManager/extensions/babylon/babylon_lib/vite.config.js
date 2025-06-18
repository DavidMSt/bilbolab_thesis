import {defineConfig} from 'vite';
// If you installed the Babylon plugin, uncomment:
// import babylonPlugin from '@babylonjs/dev-vite-plugin';

export default defineConfig({
    root: 'src',      // only if you kept index.html in src/
    publicDir: 'lib/assets',
    plugins: [
        // babylonPlugin()
    ],
    build: {
        outDir: '../dist'  // so build goes into dist/ next to public/
    }
});
