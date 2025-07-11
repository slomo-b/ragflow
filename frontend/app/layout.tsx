// frontend/app/layout.tsx
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from '@/components/providers/Providers'
import { Toaster } from 'react-hot-toast'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
})

export const metadata: Metadata = {
  title: {
    default: 'RagFlow - AI-Powered Document Analysis',
    template: '%s | RagFlow'
  },
  description: 'Powerful AI-driven document analysis and chat interface. Upload documents, ask questions, and get intelligent insights powered by Google Gemini.',
  keywords: [
    'AI',
    'document analysis',
    'machine learning',
    'chat interface',
    'Google Gemini',
    'PDF analysis',
    'document search',
    'knowledge management'
  ],
  authors: [{ name: 'RagFlow Team' }],
  creator: 'RagFlow',
  metadataBase: new URL('http://localhost:3000'),
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'http://localhost:3000',
    title: 'RagFlow - AI-Powered Document Analysis',
    description: 'Powerful AI-driven document analysis and chat interface',
    siteName: 'RagFlow',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'RagFlow - AI Document Analysis'
      }
    ]
  },
  twitter: {
    card: 'summary_large_image',
    title: 'RagFlow - AI-Powered Document Analysis',
    description: 'Powerful AI-driven document analysis and chat interface',
    images: ['/og-image.png']
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: 'your-google-site-verification',
  }
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html 
      lang="en" 
      className={`${inter.variable} antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-gray-50 antialiased">
        <Providers>
          {/* Main Application Container - REMOVED conflicting styles */}
          {children}
          
          {/* Toast Notifications */}
          <Toaster
            position="top-right"
            reverseOrder={false}
            gutter={8}
            toastOptions={{
              // Simplified toast options
              duration: 4000,
              style: {
                background: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '12px',
                color: '#1f2937',
                fontSize: '14px',
                fontWeight: '500',
                padding: '12px 16px',
                boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
              },
              success: {
                duration: 3000,
                style: {
                  background: '#f0fdf4',
                  border: '1px solid #bbf7d0',
                  color: '#15803d',
                },
              },
              error: {
                duration: 5000,
                style: {
                  background: '#fef2f2',
                  border: '1px solid #fecaca',
                  color: '#dc2626',
                },
              },
              loading: {
                style: {
                  background: '#eff6ff',
                  border: '1px solid #bfdbfe',
                  color: '#2563eb',
                },
              },
            }}
          />
        </Providers>
      </body>
    </html>
  )
}