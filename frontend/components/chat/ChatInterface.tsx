'use client'

import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { PaperAirplaneIcon, StopIcon, SparklesIcon } from '@heroicons/react/24/solid'
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ChatMessage } from './ChatMessage'
import { TypingIndicator } from './TypingIndicator'
import { ChatSuggestions } from './ChatSuggestions'
import { useChat } from '@/hooks/useChat'
import { useStore } from '@/stores/useStore'
import { cn } from '@/utils/cn'

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
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`
  }

  const handleSuggestionClick = (suggestion: string) => {
    setMessage(suggestion)
    textAreaRef.current?.focus()
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <Card className="flex-shrink-0 border-b rounded-none">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 text-white">
                <SparklesIcon className="h-5 w-5" />
              </div>
              <div>
                <h1 className="text-lg font-semibold">AI Assistant</h1>
                <p className="text-sm text-muted-foreground">
                  {currentProject ? `Project: ${currentProject.name}` : 'Ask me anything about your documents'}
                </p>
              </div>
            </div>
            
            {messages.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={clearChat}
              >
                Clear Chat
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Messages Area */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full overflow-y-auto">
          <div className="max-w-4xl mx-auto px-6 py-6">
            <AnimatePresence mode="wait">
              {isEmpty ? (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className="flex flex-col items-center justify-center h-full min-h-[400px] text-center"
                >
                  {/* Welcome Message */}
                  <div className="mb-8">
                    <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 text-white mb-4 mx-auto">
                      <SparklesIcon className="h-8 w-8" />
                    </div>
                    <h2 className="text-2xl font-bold mb-2">Welcome to RagFlow AI</h2>
                    <p className="text-muted-foreground max-w-md">
                      I can help you analyze documents, answer questions, and provide insights. 
                      {currentProject ? ` Currently working with project "${currentProject.name}".` : ' Upload some documents to get started!'}
                    </p>
                  </div>

                  {/* Chat Suggestions */}
                  <ChatSuggestions 
                    onSuggestionClick={handleSuggestionClick}
                    hasDocuments={currentProject?.document_count > 0}
                  />
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="space-y-6"
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

      {/* Input Area */}
      <Card className="flex-shrink-0 border-t rounded-none">
        <CardContent className="p-6">
          <div className="relative">
            {/* Textarea */}
            <div className="relative">
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
                className="min-h-[60px] max-h-[200px] py-4 pl-4 pr-16 text-base resize-none rounded-2xl"
                disabled={isLoading}
                rows={1}
              />
              
              {/* Send/Stop Button */}
              <div className="absolute right-2 bottom-2">
                {isLoading ? (
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={stopGeneration}
                    className="rounded-xl p-2"
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
                      "rounded-xl p-2 transition-all duration-200",
                      message.trim() ? "scale-100" : "scale-95 opacity-50"
                    )}
                    title="Send message (Enter)"
                  >
                    <PaperAirplaneIcon className="h-5 w-5" />
                  </Button>
                )}
              </div>
            </div>

            {/* Helper Text */}
            <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
              <span>Press Enter to send, Shift+Enter for new line</span>
              {currentProject?.document_count > 0 && (
                <Badge variant="secondary" className="text-xs">
                  {currentProject.document_count} document{currentProject.document_count !== 1 ? 's' : ''} available
                </Badge>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}