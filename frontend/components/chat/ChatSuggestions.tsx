// === components/chat/ChatSuggestions.tsx ===
'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { Button } from "@/components/ui/button"

interface ChatSuggestionsProps {
  onSuggestionClick: (suggestion: string) => void
  hasDocuments?: boolean
}

export function ChatSuggestions({ onSuggestionClick, hasDocuments }: ChatSuggestionsProps) {
  const suggestions = hasDocuments
    ? [
        "Summarize the main points from my documents",
        "What are the key insights?",
        "Find information about...",
        "Compare different sections"
      ]
    : [
        "How can you help me?",
        "What can I upload?",
        "Explain how this works",
        "Get started guide"
      ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="grid grid-cols-2 gap-3 max-w-lg"
    >
      {suggestions.map((suggestion, index) => (
        <motion.div
          key={suggestion}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.1 * index }}
        >
          <Button
            variant="outline"
            onClick={() => onSuggestionClick(suggestion)}
            className="w-full text-left justify-start h-auto p-3 text-sm font-normal bg-white/50 dark:bg-gray-800/50 hover:bg-white dark:hover:bg-gray-800"
          >
            {suggestion}
          </Button>
        </motion.div>
      ))}
    </motion.div>
  )
}
