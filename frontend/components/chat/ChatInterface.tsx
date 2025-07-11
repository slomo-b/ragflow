'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Send, Sparkles, Copy, Check, StopCircle } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/Badge'
import { useStore } from '@/stores/useStore'
import { cn } from '@/lib/utils'

interface Message {
  id: string
  content: string
  role: 'user' | 'assistant'
  timestamp: string
  sources?: Array<{
    name: string
    relevance_score: number
    excerpt: string
  }>
}

// Clean Message Component
const MessageBubble = ({ message, isLatest }: { message: Message; isLatest?: boolean }) => {
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  // Safe date formatting
  const formatTime = (dateString: string) => {
    try {
      const date = new Date(dateString)
      if (isNaN(date.getTime())) {
        return 'Now'
      }
      return new Intl.DateTimeFormat('en-US', {
        hour: '2-digit',
        minute: '2-digit',
      }).format(date)
    } catch {
      return 'Now'
    }
  }

  return (
    <div className={cn('flex w-full', isUser ? 'justify-end' : 'justify-start')}>
      <div className={cn(
        'max-w-[80%] group relative',
        isUser ? 'order-2' : 'order-1'
      )}>
        {/* Message Header */}
        <div className={cn(
          'flex items-center mb-2 text-xs text-gray-500',
          isUser ? 'justify-end' : 'justify-start'
        )}>
          <span className="font-medium">
            {isUser ? 'You' : 'AI Assistant'}
          </span>
          <span className="mx-2">•</span>
          <span>{formatTime(message.timestamp)}</span>
        </div>

        {/* Message Bubble */}
        <div className={cn(
          'relative px-4 py-3 rounded-2xl shadow-sm border',
          isUser
            ? 'bg-blue-600 text-white border-blue-600'
            : 'bg-white text-gray-900 border-gray-200'
        )}>
          <div className="whitespace-pre-wrap break-words">
            {message.content}
          </div>

          {/* Copy Button */}
          <button
            onClick={copyToClipboard}
            className={cn(
              'absolute top-2 right-2 p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity',
              isUser 
                ? 'text-blue-200 hover:text-white hover:bg-blue-700' 
                : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
            )}
            title="Copy message"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </button>
        </div>

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 space-y-2">
            <div className="text-xs font-medium text-gray-600">Sources:</div>
            {message.sources.map((source, index) => (
              <div
                key={index}
                className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-sm"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-gray-900 truncate">
                    {source.name}
                  </span>
                  <span className="text-xs text-gray-500">
                    {Math.round(source.relevance_score * 100)}% match
                  </span>
                </div>
                <p className="text-gray-600 text-xs line-clamp-2">
                  {source.excerpt}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// Typing Indicator Component
const TypingIndicator = () => (
  <div className="flex justify-start">
    <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3 shadow-sm">
      <div className="flex space-x-1">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
            style={{ animationDelay: `${i * 0.2}s` }}
          />
        ))}
      </div>
    </div>
  </div>
)

// Welcome Screen Component
const WelcomeScreen = ({ onSuggestionClick }: { onSuggestionClick: (suggestion: string) => void }) => {
  const { currentProject } = useStore()

  const suggestions = currentProject?.document_count && currentProject.document_count > 0 ? [
    "📄 Summarize my documents",
    "🔍 What are the key insights?",
    "💡 Find specific information", 
    "📊 Compare different sections"
  ] : [
    "❓ How can you help me?",
    "📤 What can I upload?",
    "🔧 Explain how this works",
    "🚀 Get started guide"
  ]

  return (
    <div className="flex flex-col items-center justify-center min-h-0 text-center px-8 py-8">
      {/* Welcome Icon */}
      <div className="w-16 h-16 bg-gradient-to-br from-blue-600 to-purple-600 rounded-3xl flex items-center justify-center mb-6 shadow-lg">
        <Sparkles className="h-8 w-8 text-white" />
      </div>

      {/* Welcome Text */}
      <h2 className="text-2xl font-bold text-gray-900 mb-3">
        Welcome to RagFlow AI
      </h2>
      <p className="text-base text-gray-600 max-w-xl mb-8 leading-relaxed">
        I can help you analyze documents, answer questions, and provide insights.{' '}
        {currentProject 
          ? `Currently working with project "${currentProject.name}".`
          : 'Upload some documents to get started!'
        }
      </p>

      {/* Suggestion Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            onClick={() => onSuggestionClick(suggestion.slice(2))} // Remove emoji
            className="p-3 text-left bg-white hover:bg-gray-50 rounded-xl border border-gray-200 hover:border-blue-300 hover:shadow-sm transition-all duration-200 group"
          >
            <div className="flex items-center space-x-3">
              <span className="text-lg group-hover:scale-105 transition-transform duration-200">
                {suggestion.slice(0, 2)}
              </span>
              <span className="font-medium text-sm text-gray-900 group-hover:text-blue-600 transition-colors">
                {suggestion.slice(3)}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

// Main Chat Interface Component
export function ChatInterface() {
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textAreaRef = useRef<HTMLTextAreaElement>(null)
  
  const { currentProject, addChat, updateChat } = useStore()

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Send message handler
  const handleSendMessage = async () => {
    if (!message.trim() || isLoading) return

    const userMessage: Message = {
      id: Math.random().toString(36),
      content: message.trim(),
      role: 'user',
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setMessage('')
    setIsLoading(true)

    // Auto-resize textarea
    if (textAreaRef.current) {
      textAreaRef.current.style.height = 'auto'
    }

    try {
      // Create or update chat in store
      let chatId = currentChatId
      if (!chatId && currentProject) {
        // Create new chat
        const newChatData = {
          title: message.trim().slice(0, 50) + (message.trim().length > 50 ? '...' : ''),
          projectId: currentProject.id,
          messages: [userMessage]
        }
        
        // This would be handled by the store, but we simulate it
        chatId = Math.random().toString(36)
        setCurrentChatId(chatId)
        
        addChat(newChatData)
      }

      // Simulate API call to backend
      const response = await fetch('http://localhost:8000/api/v1/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [...messages, userMessage].map(msg => ({
            role: msg.role,
            content: msg.content
          })),
          project_id: currentProject?.id || null
        })
      })

      if (response.ok) {
        const data = await response.json()
        
        const aiMessage: Message = {
          id: Math.random().toString(36),
          content: data.response || 'Sorry, I could not generate a response.',
          role: 'assistant',
          timestamp: new Date().toISOString(),
          sources: data.sources || []
        }

        setMessages(prev => [...prev, aiMessage])

        // Update chat in store
        if (chatId && currentProject) {
          updateChat(chatId, {
            messages: [...messages, userMessage, aiMessage],
            updatedAt: new Date().toISOString()
          })
        }
      } else {
        throw new Error('Failed to get response from server')
      }
      
    } catch (error) {
      console.error('Error sending message:', error)
      
      const errorMessage: Message = {
        id: Math.random().toString(36),
        content: 'Sorry, there was an error processing your request. Please make sure the backend is running.',
        role: 'assistant',
        timestamp: new Date().toISOString()
      }
      
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  // Stop generation
  const stopGeneration = () => {
    setIsLoading(false)
  }

  // Handle keyboard shortcuts
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  // Handle input change with auto-resize
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value)
    
    // Auto-resize textarea
    const textarea = e.target
    textarea.style.height = 'auto'
    textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`
  }

  // Handle suggestion click
  const handleSuggestionClick = (suggestion: string) => {
    setMessage(suggestion)
    textAreaRef.current?.focus()
  }

  // Clear chat
  const clearChat = () => {
    setMessages([])
    setCurrentChatId(null)
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 bg-white border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">AI Assistant</h1>
              <p className="text-sm text-gray-600">
                {currentProject 
                  ? `Working on: ${currentProject.name}` 
                  : 'Ready to help with your documents'
                }
              </p>
            </div>
          </div>
          
          {!isEmpty && (
            <Button
              variant="ghost"
              size="sm"
              onClick={clearChat}
              className="text-gray-600 hover:text-gray-900"
            >
              Clear Chat
            </Button>
          )}
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-hidden flex flex-col min-h-0">
        {isEmpty ? (
          <div className="flex-1 flex items-center justify-center min-h-0">
            <WelcomeScreen onSuggestionClick={handleSuggestionClick} />
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto">
            <div className="max-w-4xl mx-auto px-6 py-6 space-y-6">
              {messages.map((msg, index) => (
                <MessageBubble
                  key={msg.id}
                  message={msg}
                  isLatest={index === messages.length - 1}
                />
              ))}
              
              {isLoading && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 border-t border-gray-200 bg-white">
        <div className="max-w-4xl mx-auto px-6 py-6">
          <div className="relative">
            {/* Input Container */}
            <div className="relative bg-gray-50 border border-gray-200 rounded-2xl focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-blue-500 transition-all">
              <Textarea
                ref={textAreaRef}
                value={message}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder={
                  currentProject?.document_count && currentProject.document_count > 0
                    ? "Ask a question about your documents..."
                    : "Ask me anything..."
                }
                className="min-h-[56px] max-h-[120px] py-4 pl-6 pr-16 text-base resize-none border-0 bg-transparent focus:ring-0 focus:outline-none placeholder:text-gray-500"
                disabled={isLoading}
                rows={1}
              />
              
              {/* Send/Stop Button */}
              <div className="absolute right-3 bottom-3">
                {isLoading ? (
                  <Button
                    size="sm"
                    onClick={stopGeneration}
                    className="w-10 h-10 p-0 rounded-xl bg-red-600 hover:bg-red-700"
                  >
                    <StopCircle size={16} />
                  </Button>
                ) : (
                  <Button
                    size="sm"
                    onClick={handleSendMessage}
                    disabled={!message.trim()}
                    className={cn(
                      "w-10 h-10 p-0 rounded-xl bg-blue-600 hover:bg-blue-700 transition-all",
                      message.trim() ? "scale-100 opacity-100" : "scale-95 opacity-50"
                    )}
                  >
                    <Send size={16} />
                  </Button>
                )}
              </div>
            </div>

            {/* Helper Text & Project Badge */}
            <div className="flex items-center justify-between mt-3 px-2">
              <span className="text-xs text-gray-500">
                Press Enter to send, Shift+Enter for new line
              </span>
              {currentProject?.document_count && currentProject.document_count > 0 && (
                <Badge variant="secondary" className="text-xs">
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