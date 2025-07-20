'use client'

import React from 'react'
import { motion } from 'framer-motion'

interface MainContentProps {
  children: React.ReactNode
  className?: string
}

export function MainContent({ children, className }: MainContentProps) {
  return (
    <motion.main
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ 
        duration: 0.3, 
        ease: [0.4, 0, 0.2, 1] 
      }}
      className={`
        flex-1 
        flex 
        flex-col 
        min-h-screen 
        bg-gray-50 
        dark:bg-gray-900 
        overflow-hidden
        relative
        ${className || ''}
      `}
    >
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50/30 via-transparent to-purple-50/30 pointer-events-none" />
      
      {/* Content Container */}
      <div className="relative z-10 flex-1 flex flex-col">
        {children}
      </div>
    </motion.main>
  )
}

// Enhanced MainContent with Header Support
interface MainContentWithHeaderProps extends MainContentProps {
  header?: React.ReactNode
  footer?: React.ReactNode
  maxWidth?: 'full' | '7xl' | '6xl' | '5xl' | '4xl'
  padding?: 'none' | 'sm' | 'md' | 'lg' | 'xl'
}

export function MainContentWithHeader({ 
  children, 
  header, 
  footer, 
  maxWidth = 'full',
  padding = 'lg',
  className 
}: MainContentWithHeaderProps) {
  
  const maxWidthClasses = {
    'full': 'max-w-full',
    '7xl': 'max-w-7xl',
    '6xl': 'max-w-6xl', 
    '5xl': 'max-w-5xl',
    '4xl': 'max-w-4xl'
  }

  const paddingClasses = {
    'none': '',
    'sm': 'p-4',
    'md': 'p-6',
    'lg': 'p-8',
    'xl': 'p-12'
  }

  return (
    <motion.main
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ 
        duration: 0.4, 
        ease: [0.4, 0, 0.2, 1] 
      }}
      className={`
        flex-1 
        flex 
        flex-col 
        min-h-screen 
        bg-gray-50 
        dark:bg-gray-900 
        relative
        ${className || ''}
      `}
    >
      {/* Background Elements */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50/30 via-transparent to-purple-50/30 pointer-events-none" />
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-100 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob" />
      <div className="absolute top-0 right-1/4 w-96 h-96 bg-purple-100 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000" />
      <div className="absolute bottom-0 left-1/3 w-96 h-96 bg-pink-100 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000" />

      {/* Header */}
      {header && (
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ 
            duration: 0.3, 
            delay: 0.1,
            ease: [0.4, 0, 0.2, 1] 
          }}
          className="relative z-20 bg-white/80 backdrop-blur-md border-b border-gray-200/60 shadow-sm"
        >
          <div className={`mx-auto ${maxWidthClasses[maxWidth]} ${paddingClasses[padding]}`}>
            {header}
          </div>
        </motion.header>
      )}

      {/* Main Content Area */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ 
          duration: 0.4, 
          delay: 0.2,
          ease: [0.4, 0, 0.2, 1] 
        }}
        className="relative z-10 flex-1 flex flex-col overflow-hidden"
      >
        <div className={`flex-1 mx-auto w-full ${maxWidthClasses[maxWidth]} ${paddingClasses[padding]}`}>
          {children}
        </div>
      </motion.div>

      {/* Footer */}
      {footer && (
        <motion.footer
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ 
            duration: 0.3, 
            delay: 0.3,
            ease: [0.4, 0, 0.2, 1] 
          }}
          className="relative z-20 bg-white/80 backdrop-blur-md border-t border-gray-200/60"
        >
          <div className={`mx-auto ${maxWidthClasses[maxWidth]} ${paddingClasses[padding]}`}>
            {footer}
          </div>
        </motion.footer>
      )}
    </motion.main>
  )
}

// Specialized Content Containers
export function ChatMainContent({ children }: { children: React.ReactNode }) {
  return (
    <MainContent className="bg-gradient-to-br from-blue-50/50 to-indigo-50/50">
      <div className="flex-1 flex flex-col h-full">
        {children}
      </div>
    </MainContent>
  )
}

export function ProjectMainContent({ children }: { children: React.ReactNode }) {
  return (
    <MainContentWithHeader
      maxWidth="7xl"
      padding="lg"
      className="bg-gradient-to-br from-green-50/50 to-emerald-50/50"
    >
      {children}
    </MainContentWithHeader>
  )
}

export function DocumentMainContent({ children }: { children: React.ReactNode }) {
  return (
    <MainContentWithHeader
      maxWidth="6xl" 
      padding="md"
      className="bg-gradient-to-br from-purple-50/50 to-pink-50/50"
    >
      {children}
    </MainContentWithHeader>
  )
}

export function SettingsMainContent({ children }: { children: React.ReactNode }) {
  return (
    <MainContentWithHeader
      maxWidth="4xl"
      padding="lg"
      className="bg-gradient-to-br from-gray-50/50 to-slate-50/50"
    >
      {children}
    </MainContentWithHeader>
  )
}

// Loading State Component
export function MainContentLoading() {
  return (
    <MainContent>
      <div className="flex-1 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ 
            duration: 0.3,
            ease: [0.4, 0, 0.2, 1] 
          }}
          className="text-center"
        >
          <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center mx-auto mb-4">
            <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Loading...</h3>
          <p className="text-gray-600 text-sm">Please wait while we prepare your content</p>
        </motion.div>
      </div>
    </MainContent>
  )
}

// Error State Component
export function MainContentError({ 
  title = "Something went wrong",
  message = "We encountered an error while loading this content.",
  onRetry
}: {
  title?: string
  message?: string
  onRetry?: () => void
}) {
  return (
    <MainContent>
      <div className="flex-1 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ 
            duration: 0.3,
            ease: [0.4, 0, 0.2, 1] 
          }}
          className="text-center max-w-md"
        >
          <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center mx-auto mb-4">
            <div className="w-6 h-6 text-red-600">⚠️</div>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
          <p className="text-gray-600 text-sm mb-6">{message}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Try Again
            </button>
          )}
        </motion.div>
      </div>
    </MainContent>
  )
}