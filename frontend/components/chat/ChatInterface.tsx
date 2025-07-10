'use client'

import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { PaperAirplaneIcon, StopIcon, SparklesIcon } from '@heroicons/react/24/solid'
import { Button } from "@/components/ui/Button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/Badge"
import { ChatMessage } from './ChatMessage'
import { TypingIndicator } from './TypingIndicator'
import { ChatSuggestions } from './ChatSuggestions'
import { useChat } from '@/hooks/useChat'
import { useStore } from '@/stores/useStore'
import { cn } from '@/lib/utils'

export function ChatInterface() {
  const [message, setMessage] = useState('')
  const [isComposing, setIsComposing] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textAreaRef = useRef<HTMLTextAreaElement>(null)
  
  const { currentProject } = useStore()
  const {
    messages,
    isLoading,
    sendMessage,
    stopGeneration,
    clearChat,
  } = useChat(currentProject?.id)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async () => {
    if (!message.trim() || isLoading) return

    const userMessage = message.trim()
    setMessage('')
    
    // Auto-resize textarea
    if (textAreaRef.current) {
      textAreaRef.current.style.height = 'auto'
    }

    await sendMessage(userMessage)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value)
    
    // Auto-resize textarea
    const textarea = e.target
    textarea.style.height = 'auto'
    textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`
  }

  const handleSuggestionClick = (suggestion: string) => {
    setMessage(suggestion)
    textAreaRef.current?.focus()
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex flex-col h-full">
      {/* Clean Header */}
      <div className="flex-shrink-0 px-8 py-6 border-b border-slate-200/50 dark:border-slate-700/50 bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gradient-to-br from-violet-500 to-blue-500 rounded-2xl flex items-center justify-center shadow-lg">
              <SparklesIcon className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-white">AI Assistant</h1>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                {currentProject ? `Working on: ${currentProject.name}` : 'Ready to help with your documents'}
              </p>
            </div>
          </div>
          
          {messages.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={clearChat}
              className="text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white"
            >
              Clear Chat
            </Button>
          )}
        </div>
      </div>

      {/* Messages Area - Clean and Centered */}
      <div className="flex-1 overflow-hidden bg-slate-50/50 dark:bg-slate-900/30">
        <div className="h-full overflow-y-auto">
          <div className="max-w-4xl mx-auto px-8 py-8">
            <AnimatePresence mode="wait">
              {isEmpty ? (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.4 }}
                  className="flex flex-col items-center justify-center min-h-[500px] text-center"
                >
                  {/* Welcome Section */}
                  <div className="mb-12">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 300, delay: 0.2 }}
                      className="w-20 h-20 bg-gradient-to-br from-violet-500 to-blue-500 rounded-3xl flex items-center justify-center mb-6 mx-auto shadow-2xl"
                    >
                      <SparklesIcon className="h-10 w-10 text-white" />
                    </motion.div>
                    <motion.h2 
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className="text-3xl font-bold text-slate-900 dark:text-white mb-4"
                    >
                      Welcome to RagFlow AI
                    </motion.h2>
                    <motion.p 
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.4 }}
                      className="text-lg text-slate-600 dark:text-slate-400 max-w-2xl mx-auto leading-relaxed"
                    >
                      I can help you analyze documents, answer questions, and provide insights. 
                      {currentProject ? ` Currently working with project "${currentProject.name}".` : ' Upload some documents to get started!'}
                    </motion.p>
                  </div>

                  {/* Modern Chat Suggestions */}
                  <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="w-full max-w-2xl"
                  >
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {(currentProject?.document_count > 0 ? [
                        "📄 Summarize my documents",
                        "🔍 What are the key insights?", 
                        "💡 Find specific information",
                        "📊 Compare different sections"
                      ] : [
                        "❓ How can you help me?",
                        "📤 What can I upload?",
                        "🔧 Explain how this works", 
                        "🚀 Get started guide"
                      ]).map((suggestion, index) => (
                        <motion.button
                          key={suggestion}
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.3, delay: 0.6 + index * 0.1 }}
                          whileHover={{ scale: 1.02, y: -2 }}
                          whileTap={{ scale: 0.98 }}
                          onClick={() => handleSuggestionClick(suggestion.slice(2))}
                          className="p-4 text-left bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 hover:border-violet-300 dark:hover:border-violet-600 shadow-sm hover:shadow-lg transition-all duration-200 group"
                        >
                          <div className="flex items-center space-x-3">
                            <span className="text-2xl group-hover:scale-110 transition-transform duration-200">
                              {suggestion.slice(0, 2)}
                            </span>
                            <span className="font-medium text-slate-900 dark:text-white group-hover:text-violet-600 dark:group-hover:text-violet-400 transition-colors">
                              {suggestion.slice(3)}
                            </span>
                          </div>
                        </motion.button>
                      ))}
                    </div>
                  </motion.div>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="space-y-8 pb-8"
                >
                  {messages.map((msg, index) => (
                    <ChatMessage
                      key={msg.id || index}
                      message={msg}
                      isLatest={index === messages.length - 1}
                    />
                  ))}
                  
                  {isLoading && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                    >
                      <TypingIndicator />
                    </motion.div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>
        </div>
      </div>

      {/* Clean Input Area */}
      <div className="flex-shrink-0 border-t border-slate-200/50 dark:border-slate-700/50 bg-white/70 dark:bg-slate-900/70 backdrop-blur-xl">
        <div className="max-w-4xl mx-auto px-8 py-6">
          <div className="relative">
            {/* Modern Input Container */}
            <div className="relative bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-lg hover:shadow-xl transition-all duration-200 focus-within:ring-2 focus-within:ring-violet-500/20 focus-within:border-violet-400">
              <Textarea
                ref={textAreaRef}
                value={message}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                onCompositionStart={() => setIsComposing(true)}
                onCompositionEnd={() => setIsComposing(false)}
                placeholder={
                  currentProject?.document_count > 0
                    ? "Ask a question about your documents..."
                    : "Ask me anything..."
                }
                className="min-h-[60px] max-h-[120px] py-4 pl-6 pr-16 text-base resize-none border-0 bg-transparent focus:ring-0 focus:outline-none placeholder:text-slate-500 dark:placeholder:text-slate-400"
                disabled={isLoading}
                rows={1}
              />
              
              {/* Send/Stop Button */}
              <div className="absolute right-3 bottom-3">
                {isLoading ? (
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={stopGeneration}
                    className="rounded-xl w-10 h-10 p-0 shadow-lg"
                    title="Stop generation"
                  >
                    <StopIcon className="h-5 w-5" />
                  </Button>
                ) : (
                  <Button
                    variant="default"
                    size="sm"
                    onClick={handleSendMessage}
                    disabled={!message.trim()}
                    className={cn(
                      "rounded-xl w-10 h-10 p-0 shadow-lg bg-gradient-to-r from-violet-500 to-blue-500 hover:from-violet-600 hover:to-blue-600 transition-all duration-200",
                      message.trim() ? "scale-100 opacity-100" : "scale-95 opacity-50"
                    )}
                    title="Send message (Enter)"
                  >
                    <PaperAirplaneIcon className="h-5 w-5" />
                  </Button>
                )}
              </div>
            </div>

            {/* Helper Text */}
            <div className="flex items-center justify-between mt-3 px-2">
              <span className="text-xs text-slate-500 dark:text-slate-400">
                Press Enter to send, Shift+Enter for new line
              </span>
              {currentProject?.document_count > 0 && (
                <Badge variant="secondary" className="text-xs bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-300">
                  {currentProject.document_count} document{currentProject.document_count !== 1 ? 's' : ''} available
                </Badge>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}