import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Send, 
  StopCircle, 
  Sparkles, 
  Copy, 
  Check, 
  MessageSquare,
  FileText,
  User,
  Bot
} from 'lucide-react'
import { Button } from "@/components/ui/Button"
import { Badge } from "@/components/ui/Badge"
import { cn } from '@/lib/utils'
import { useStore } from '@/stores/useStore'
import toast from 'react-hot-toast'

interface Message {
  id: string
  content: string
  role: 'user' | 'assistant'
  timestamp: string
  sources?: Array<{
    id: string
    name: string
    excerpt: string
    relevance_score: number
  }>
}

// Chat API Service
class ChatAPI {
  static async sendMessage(content: string, projectId?: string) {
    console.log('Sending chat message:', { content, projectId })
    
    const payload = {
      message: content,
      project_id: projectId || null,
      use_documents: !!projectId
    }
    
    console.log('Chat API payload:', payload)
    
    try {
      const response = await fetch('http://localhost:8000/api/v1/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      })
      
      console.log('Chat response status:', response.status)
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error('Chat API error:', errorText)
        throw new Error(`Chat failed: ${response.status} ${response.statusText}`)
      }
      
      const result = await response.json()
      console.log('Chat API result:', result)
      return result
    } catch (error) {
      console.error('Chat API error:', error)
      throw error
    }
  }
  
  static async getChatHistory(projectId?: string) {
    const params = new URLSearchParams()
    if (projectId) {
      params.append('project_id', projectId)
    }
    
    const response = await fetch(`http://localhost:8000/api/v1/chats/?${params}`)
    
    if (!response.ok) {
      throw new Error(`Failed to fetch chat history: ${response.statusText}`)
    }
    
    return response.json()
  }
}

// Message Component
const MessageComponent = ({ message }: { message: Message }) => {
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "flex w-full",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div className={cn(
        "max-w-[70%] rounded-2xl px-4 py-3 shadow-sm",
        isUser 
          ? "bg-blue-600 text-white" 
          : "bg-white border border-gray-200"
      )}>
        {/* Message Header */}
        <div className="flex items-center gap-2 mb-2">
          <div className={cn(
            "w-6 h-6 rounded-full flex items-center justify-center",
            isUser ? "bg-blue-500" : "bg-gray-100"
          )}>
            {isUser ? (
              <User size={12} className="text-white" />
            ) : (
              <Bot size={12} className="text-gray-600" />
            )}
          </div>
          <span className={cn(
            "text-xs font-medium",
            isUser ? "text-blue-100" : "text-gray-600"
          )}>
            {isUser ? "You" : "AI Assistant"}
          </span>
          <span className={cn(
            "text-xs",
            isUser ? "text-blue-200" : "text-gray-400"
          )}>
            {new Date(message.timestamp).toLocaleTimeString('en-US', {
              hour: '2-digit',
              minute: '2-digit'
            })}
          </span>
        </div>

        {/* Message Content */}
        <div className={cn(
          "text-sm leading-relaxed whitespace-pre-wrap",
          isUser ? "text-white" : "text-gray-900"
        )}>
          {message.content}
        </div>

        {/* Copy Button */}
        <div className="flex justify-end mt-2">
          <button
            onClick={copyToClipboard}
            className={cn(
              "p-1 rounded transition-colors",
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
                  <span className="font-medium text-gray-900 truncate flex items-center gap-1">
                    <FileText size={12} />
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
    </motion.div>
  )
}

// Typing Indicator Component
const TypingIndicator = () => (
  <div className="flex justify-start">
    <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3 shadow-sm">
      <div className="flex items-center gap-2">
        <Bot size={16} className="text-gray-600" />
        <div className="flex space-x-1">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
              style={{ animationDelay: `${i * 0.2}s` }}
            />
          ))}
        </div>
        <span className="text-xs text-gray-500">AI is thinking...</span>
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

  // Handle suggestion clicks
  const handleSuggestionClick = (suggestion: string) => {
    setMessage(suggestion)
    if (textAreaRef.current) {
      textAreaRef.current.focus()
    }
  }

  // Auto-resize textarea
  const adjustTextareaHeight = () => {
    if (textAreaRef.current) {
      textAreaRef.current.style.height = 'auto'
      textAreaRef.current.style.height = `${Math.min(textAreaRef.current.scrollHeight, 120)}px`
    }
  }

  useEffect(() => {
    adjustTextareaHeight()
  }, [message])

  // Handle key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  // Send message handler
  const handleSendMessage = async () => {
    if (!message.trim() || isLoading) return

    console.log('Sending message with project:', currentProject)

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
      // Send to backend with correct project_id
      const response = await ChatAPI.sendMessage(
        userMessage.content, 
        currentProject?.id // This is the key fix!
      )

      const aiMessage: Message = {
        id: Math.random().toString(36),
        content: response.response || 'Sorry, I could not generate a response.',
        role: 'assistant',
        timestamp: new Date().toISOString(),
        sources: response.sources || []
      }

      setMessages(prev => [...prev, aiMessage])

      // Show success if sources were found
      if (response.sources && response.sources.length > 0) {
        toast.success(`Found ${response.sources.length} relevant document(s)`)
      }

      // Create or update chat in store
      if (currentProject) {
        if (!currentChatId) {
          const newChatData = {
            title: userMessage.content.slice(0, 50) + (userMessage.content.length > 50 ? '...' : ''),
            projectId: currentProject.id,
            messages: [userMessage, aiMessage]
          }
          
          const chatId = Math.random().toString(36)
          setCurrentChatId(chatId)
          addChat(newChatData)
        } else {
          updateChat(currentChatId, {
            messages: [...messages, userMessage, aiMessage],
            updatedAt: new Date().toISOString()
          })
        }
      }
      
    } catch (error) {
      console.error('Error sending message:', error)
      
      const errorMessage: Message = {
        id: Math.random().toString(36),
        content: `Sorry, there was an error: ${error.message}. Please make sure the backend is running and documents are properly uploaded.`,
        role: 'assistant',
        timestamp: new Date().toISOString()
      }
      
      setMessages(prev => [...prev, errorMessage])
      toast.error(`Chat error: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  // Stop generation (placeholder)
  const stopGeneration = () => {
    setIsLoading(false)
  }

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-gray-50 to-white">
      {/* Header */}
      <div className="flex-shrink-0 p-4 bg-white border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl">
              <MessageSquare className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900">
                AI Chat Assistant
              </h1>
              <p className="text-sm text-gray-600">
                {currentProject 
                  ? `Project: ${currentProject.name} • ${currentProject.document_count || 0} documents`
                  : 'No project selected'
                }
              </p>
            </div>
          </div>
          
          {currentProject && (
            <Badge variant="secondary" className="text-xs">
              {currentProject.document_count || 0} docs available
            </Badge>
          )}
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full overflow-y-auto">
          {messages.length === 0 ? (
            <WelcomeScreen onSuggestionClick={handleSuggestionClick} />
          ) : (
            <div className="p-4 space-y-4">
              <AnimatePresence>
                {messages.map((msg) => (
                  <MessageComponent key={msg.id} message={msg} />
                ))}
              </AnimatePresence>
              
              {isLoading && <TypingIndicator />}
              
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 p-4 bg-white border-t border-gray-200">
        <div className="max-w-4xl mx-auto">
          <div className="relative">
            {/* Input Container */}
            <div className="relative bg-gray-50 rounded-2xl border border-gray-200 focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500 transition-all">
              <textarea
                ref={textAreaRef}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder={
                  currentProject?.document_count && currentProject.document_count > 0
                    ? "Ask a question about your documents..."
                    : "Ask me anything..."
                }
                className="min-h-[56px] max-h-[120px] py-4 pl-6 pr-16 text-base resize-none border-0 bg-transparent focus:ring-0 focus:outline-none placeholder:text-gray-500 w-full"
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

            {/* Helper Text */}
            <div className="flex items-center justify-between mt-3 px-2">
              <span className="text-xs text-gray-500">
                Press Enter to send, Shift+Enter for new line
              </span>
              {currentProject?.document_count && currentProject.document_count > 0 && (
                <Badge variant="outline" className="text-xs">
                  RAG enabled
                </Badge>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}