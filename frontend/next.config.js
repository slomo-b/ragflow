/** @type {import('next').NextConfig} */
const nextConfig = {
  // Existing config...
  experimental: {
    turbo: {
      rules: {
        '*.css': {
          loaders: ['css-loader'],
          as: '*.css',
        },
      },
    },
  },
}

module.exports = nextConfig