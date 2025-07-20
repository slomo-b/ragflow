'use client'

import React from 'react'
import { 
  MessageCircle, 
  FolderOpen, 
  FileText, 
  Settings, 
  Plus, 
  Menu,
  Circle,
  ChevronLeft,
  ChevronRight
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { useStore } from '@/stores/useStore'
import { cn } from '@/lib/utils'

interface SidebarProps {
  currentView: string
  onViewChange: (view: string) => void
  isCollapsed: boolean
  onToggleCollapse: () => void
}

const navItems = [
  { id: 'chat', icon: MessageCircle, label: 'AI Chat', color: 'text-blue-600', bgColor: 'bg-blue-50', borderColor: 'border-blue-200' },
  { id: 'projects', icon: FolderOpen, label: 'Projects', color: 'text-green-600', bgColor: 'bg-green-50', borderColor: 'border-green-200' },
  { id: 'documents', icon: FileText, label: 'Documents', color: 'text-purple-600', bgColor: 'bg-purple-50', borderColor: 'border-purple-200' },
  { id: 'settings', icon: Settings, label: 'Settings', color: 'text-gray-600', bgColor: 'bg-gray-50', borderColor: 'border-gray-200' }
]

export function Sidebar({ 
  currentView, 
  onViewChange, 
  isCollapsed, 
  onToggleCollapse 
}: SidebarProps) {
  const { projects = [], currentProject, chats = [], setCurrentProject } = useStore()

  const formatDate = (dateString: string | Date) => {
    try {
      const date = typeof dateString === 'string' ? new Date(dateString) : dateString
      if (isNaN(date.getTime())) return 'Recently'
      return new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric'
      }).format(date)
    } catch {
      return 'Recently'
    }
  }

  const formatTime = (dateString: string | Date) => {
    try {
      const date = typeof dateString === 'string' ? new Date(dateString) : dateString
      if (isNaN(date.getTime())) return 'Recently'
      return new Intl.DateTimeFormat('en-US', {
        hour: '2-digit',
        minute: '2-digit'
      }).format(date)
    } catch {
      return 'Recently'
    }
  }

  return (
    <aside className={cn(
      "bg-white border-r border-gray-200 transition-all duration-300 flex flex-col flex-shrink-0 relative",
      isCollapsed ? "w-16" : "w-80"
    )}>
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          {!isCollapsed && (
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">R</span>
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900">RagFlow</h1>
                <p className="text-xs text-gray-500">AI Document Assistant</p>
              </div>
            </div>
          )}
          
          {isCollapsed && (
            <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center mx-auto">
              <span className="text-white font-bold text-sm">R</span>
            </div>
          )}
        </div>
      </div>

      {/* Toggle Button */}
      <button
        onClick={onToggleCollapse}
        className="absolute -right-3 top-20 w-6 h-6 bg-white border border-gray-200 rounded-full flex items-center justify-center shadow-md hover:shadow-lg transition-all duration-200 z-10"
        aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {isCollapsed ? (
          <ChevronRight className="w-3 h-3 text-gray-600" />
        ) : (
          <ChevronLeft className="w-3 h-3 text-gray-600" />
        )}
      </button>

      {/* Current Project Indicator */}
      {!isCollapsed && currentProject && (
        <div className="flex-shrink-0 mx-4 mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 min-w-0 flex-1">
              <div className="w-4 h-4 bg-blue-600 rounded flex items-center justify-center flex-shrink-0">
                <FolderOpen className="h-2.5 w-2.5 text-white" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="font-semibold text-sm text-blue-900 truncate">
                  {currentProject.name}
                </div>
                <div className="text-xs text-blue-600 flex items-center space-x-2">
                  <span>{currentProject.document_count || 0} docs</span>
                  <span>•</span>
                  <span>{currentProject.chat_count || 0} chats</span>
                </div>
              </div>
            </div>
            <div className="w-2 h-2 bg-green-500 rounded-full flex-shrink-0 animate-pulse" />
          </div>
        </div>
      )}

      {/* Navigation */}
      <nav className="flex-shrink-0 p-4 space-y-2">
        {navItems.map(item => {
          const Icon = item.icon
          const isActive = currentView === item.id
          
          return (
            <Button
              key={item.id}
              variant="ghost"
              onClick={() => onViewChange(item.id)}
              className={cn(
                "w-full justify-start px-3 py-3 h-auto rounded-lg transition-all duration-200",
                isActive 
                  ? `${item.bgColor} ${item.color} border ${item.borderColor} shadow-sm` 
                  : "text-gray-700 hover:bg-gray-50 hover:text-gray-900",
                isCollapsed && "justify-center px-2"
              )}
              title={isCollapsed ? item.label : undefined}
            >
              <Icon size={20} className="flex-shrink-0" />
              {!isCollapsed && (
                <span className="ml-3 font-medium truncate">{item.label}</span>
              )}
            </Button>
          )
        })}
      </nav>

      {/* Quick Action */}
      {!isCollapsed && (
        <div className="flex-shrink-0 px-4 pb-4">
          <Button 
            className="w-full justify-start bg-blue-50 text-blue-600 hover:bg-blue-100 h-10 rounded-lg"
            variant="ghost"
          >
            <Plus size={18} />
            <span className="ml-3 font-medium">New Chat</span>
          </Button>
        </div>
      )}

      {/* Dynamic Content Area */}
      {!isCollapsed && (
        <div className="flex-1 overflow-y-auto overflow-x-hidden px-4 pb-4">
          <div className="space-y-4">
            {/* Recent Chats */}
            {currentView === 'chat' && (
              <div>
                <h3 className="text-sm font-semibold text-gray-900 mb-3 sticky top-0 bg-white py-1 border-b border-gray-100">
                  Recent Chats
                </h3>
                <div className="space-y-2">
                  {(chats || []).slice(0, 8).map(chat => (
                    <div 
                      key={chat.id} 
                      className="p-3 rounded-lg hover:bg-gray-50 cursor-pointer group transition-all duration-200 border border-transparent hover:border-gray-200"
                    >
                      <div className="font-medium text-sm text-gray-900 truncate mb-1 group-hover:text-blue-600">
                        {chat.title}
                      </div>
                      <div className="text-xs text-gray-500">
                        {formatTime(chat.updatedAt)}
                      </div>
                    </div>
                  ))}
                  {(chats || []).length === 0 && (
                    <div className="text-center py-8 text-gray-500">
                      <MessageCircle className="h-8 w-8 mx-auto mb-3 opacity-40" />
                      <p className="text-sm font-medium">No chats yet</p>
                      <p className="text-xs mt-1">Start a conversation to see your chat history</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* All Projects */}
            {currentView === 'projects' && (
              <div>
                <div className="flex items-center justify-between mb-3 sticky top-0 bg-white py-1 border-b border-gray-100">
                  <h3 className="text-sm font-semibold text-gray-900">All Projects</h3>
                  <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded-full font-medium">
                    {(projects || []).length}
                  </span>
                </div>
                <div className="space-y-2">
                  {(projects || []).map(project => (
                    <div 
                      key={project.id} 
                      onClick={() => setCurrentProject(project)}
                      className={cn(
                        "p-3 rounded-lg cursor-pointer transition-all duration-200 border",
                        currentProject?.id === project.id
                          ? "bg-blue-600 text-white shadow-md border-blue-600"
                          : "hover:bg-gray-50 border-transparent hover:border-gray-200 hover:shadow-sm"
                      )}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="font-medium text-sm truncate flex-1">
                          {project.name}
                        </div>
                        {currentProject?.id === project.id ? (
                          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse ml-2" />
                        ) : (
                          <Circle size={8} className="text-gray-300 ml-2" />
                        )}
                      </div>
                      <div className={cn(
                        "text-xs flex items-center justify-between",
                        currentProject?.id === project.id ? "text-blue-100" : "text-gray-500"
                      )}>
                        <div className="flex items-center space-x-2">
                          <span>{project.document_count || 0} docs</span>
                          <span>•</span>
                          <span>{project.chat_count || 0} chats</span>
                        </div>
                        <span className="ml-2">{formatDate(project.createdAt)}</span>
                      </div>
                    </div>
                  ))}
                  {(projects || []).length === 0 && (
                    <div className="text-center py-8 text-gray-500">
                      <FolderOpen className="h-8 w-8 mx-auto mb-3 opacity-40" />
                      <p className="text-sm font-medium">No projects yet</p>
                      <p className="text-xs mt-1">Create your first project to get started</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Documents Preview */}
            {currentView === 'documents' && currentProject && (
              <div>
                <h3 className="text-sm font-semibold text-gray-900 mb-3 sticky top-0 bg-white py-1 border-b border-gray-100">
                  Recent Documents
                </h3>
                <div className="space-y-2">
                  {/* Placeholder for recent documents */}
                  <div className="text-center py-8 text-gray-500">
                    <FileText className="h-8 w-8 mx-auto mb-3 opacity-40" />
                    <p className="text-sm font-medium">No documents yet</p>
                    <p className="text-xs mt-1">Upload documents to see them here</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Collapsed Navigation Indicators */}
      {isCollapsed && (
        <div className="flex-1 flex flex-col items-center py-4 space-y-4">
          {navItems.map(item => {
            const isActive = currentView === item.id
            return (
              <div
                key={item.id}
                className={cn(
                  "w-1 h-8 rounded-full transition-all duration-200",
                  isActive ? "bg-blue-600" : "bg-gray-200"
                )}
              />
            )
          })}
        </div>
      )}
    </aside>
  )
}