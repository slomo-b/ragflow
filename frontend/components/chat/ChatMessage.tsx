// === components/chat/ChatMessage.tsx ===
'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { CheckIcon, ClipboardDocumentIcon } from '@heroicons/react/24/outline'
import ReactMarkdown from 'react-markdown'
import { Button } from "@/components/ui/button"
import { cn } from '@/lib/utils'

interface ChatMessageProps {
  message: {
    id?: string
    content: string
    role: 'user' | 'assistant' | 'error'
    timestamp: Date
    sources?: Array<{
      name: string
      relevance_score: number
      excerpt: string
    }>
  }
  isLatest?: boolean
}

export function ChatMessage({ message, isLatest }: ChatMessageProps) {
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'
  const isError = message.role === 'error'

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const formatTimestamp = (timestamp: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    }).format(timestamp)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn('group relative flex flex-col', isUser ? 'items-end' : 'items-start')}
    >
      {/* Header */}
      <div className={cn(
        'flex items-center space-x-2 mb-2 px-1',
        isUser ? 'justify-end' : 'justify-start'
      )}>
        <span className="text-sm font-medium text-gray-900 dark:text-white">
          {isUser ? 'You' : isError ? 'Error' : 'AI Assistant'}
        </span>
        <span className="text-xs text-gray-500 dark:text-gray-400">
          {formatTimestamp(message.timestamp)}
        </span>
      </div>

      {/* Message Bubble */}
      <div className={cn(
        'relative rounded-2xl px-4 py-3 shadow-sm max-w-[80%]',
        isUser
          ? 'bg-blue-600 text-white'
          : isError
            ? 'bg-red-50 text-red-900 border border-red-200 dark:bg-red-900/20 dark:text-red-400 dark:border-red-800'
            : 'bg-white text-gray-900 border border-gray-200 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-700'
      )}>
        {/* Message Content */}
        <div className="prose prose-sm max-w-none">
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <ReactMarkdown
              components={{
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                ul: ({ children }) => <ul className="list-disc pl-4 mb-2">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal pl-4 mb-2">{children}</ol>,
                li: ({ children }) => <li className="mb-1">{children}</li>,
                code: ({ children, className }) => {
                  const isInline = !className
                  return isInline ? (
                    <code className="bg-gray-100 dark:bg-gray-700 px-1 py-0.5 rounded text-sm">
                      {children}
                    </code>
                  ) : (
                    <pre className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg overflow-x-auto">
                      <code>{children}</code>
                    </pre>
                  )
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {/* Copy Button */}
        <Button
          variant="ghost"
          size="sm"
          onClick={copyToClipboard}
          className={cn(
            'absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 h-6 w-6',
            isUser 
              ? 'text-blue-200 hover:text-white hover:bg-blue-700' 
              : 'text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300'
          )}
          title="Copy message"
        >
          {copied ? (
            <CheckIcon className="h-3 w-3" />
          ) : (
            <ClipboardDocumentIcon className="h-3 w-3" />
          )}
        </Button>
      </div>

      {/* Sources */}
      {message.sources && message.sources.length > 0 && !isUser && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          transition={{ duration: 0.2, delay: 0.1 }}
          className="mt-3 max-w-[80%]"
        >
          <div className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-2">
            Sources:
          </div>
          <div className="space-y-2">
            {message.sources.map((source, index) => (
              <div
                key={index}
                className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-gray-900 dark:text-white truncate">
                    {source.name}
                  </span>
                  <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                    {Math.round(source.relevance_score * 100)}% match
                  </span>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2">
                  {source.excerpt}
                </p>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}