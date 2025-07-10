'use client'

import React from 'react'
import { motion } from 'framer-motion'

interface MainContentProps {
  children: React.ReactNode
}

export function MainContent({ children }: MainContentProps) {
  return (
    <motion.main
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
      className="flex-1 flex flex-col bg-gray-50 dark:bg-gray-900 overflow-hidden"
    >
      {children}
    </motion.main>
  )
}