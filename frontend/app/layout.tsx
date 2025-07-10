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
      <body className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800 transition-colors duration-300">
        <Providers>
          {/* Main Application */}
          <div className="flex h-screen overflow-hidden">
            {children}
          </div>
          
          {/* Toast Notifications */}
          <Toaster
            position="top-right"
            reverseOrder={false}
            gutter={8}
            containerClassName=""
            containerStyle={{}}
            toastOptions={{
              // Default options for all toasts
              className: '',
              duration: 4000,
              style: {
                background: 'rgba(255, 255, 255, 0.95)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '12px',
                color: '#0f172a',
                fontSize: '14px',
                fontWeight: '500',
                padding: '12px 16px',
                boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
              },
              // Success toast styling
              success: {
                duration: 3000,
                style: {
                  background: 'rgba(34, 197, 94, 0.1)',
                  border: '1px solid rgba(34, 197, 94, 0.3)',
                  color: '#059669',
                },
                iconTheme: {
                  primary: '#059669',
                  secondary: 'rgba(34, 197, 94, 0.1)',
                },
              },
              // Error toast styling
              error: {
                duration: 5000,
                style: {
                  background: 'rgba(239, 68, 68, 0.1)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  color: '#dc2626',
                },
                iconTheme: {
                  primary: '#dc2626',
                  secondary: 'rgba(239, 68, 68, 0.1)',
                },
              },
              // Loading toast styling
              loading: {
                style: {
                  background: 'rgba(59, 130, 246, 0.1)',
                  border: '1px solid rgba(59, 130, 246, 0.3)',
                  color: '#2563eb',
                },
                iconTheme: {
                  primary: '#2563eb',
                  secondary: 'rgba(59, 130, 246, 0.1)',
                },
              },
            }}
          />
          
          {/* Background Elements */}
          <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
            {/* Gradient orbs */}
            <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/20 to-purple-600/20 rounded-full blur-3xl animate-pulse-slow" />
            <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-gradient-to-br from-cyan-400/20 to-blue-600/20 rounded-full blur-3xl animate-pulse-slow animation-delay-2000" />
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-72 h-72 bg-gradient-to-br from-purple-400/10 to-pink-600/10 rounded-full blur-3xl animate-float" />
          </div>
        </Providers>
      </body>
    </html>
  )
}