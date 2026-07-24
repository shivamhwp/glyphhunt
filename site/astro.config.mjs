import { defineConfig } from 'astro/config';

export default defineConfig({
  // Static output: the results are baked at build time and redeployed when
  // the grid advances, so there is nothing to render per-request.
  output: 'static',
  build: { format: 'file' },
});
