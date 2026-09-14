import { cpSync } from 'node:fs'
import { sep } from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: './',
  plugins: [
    react(),
    {
      name: 'copy-data',
      apply: 'build',
      writeBundle() {
        cpSync('data', 'dist/data', {
          recursive: true,
          filter: (src) => !src.includes(`${sep}raw${sep}`) && !src.endsWith(`${sep}raw`),
        })
      },
    },
  ],
  worker: {
    format: 'es',
  },
})