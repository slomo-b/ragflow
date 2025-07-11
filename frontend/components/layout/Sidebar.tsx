'use client'

import React from 'react'
import { 
  MessageCircle, 
  FolderOpen, 
  FileText, 
  Settings, 
  Plus, 
  Menu,
  Circle
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
  { id: 'chat', icon: MessageCircle, label: 'Chat' },
  { id: 'projects', icon: FolderOpen, label: 'Projects' },
  { id: 'documents', icon: FileText, label: 'Documents' },
  { id: 'settings', icon: Settings, label: 'Settings' }
]

export function Sidebar({ 
  currentView, 
  onViewChange, 
  isCollapsed, 
  onToggleCollapse 
}: SidebarProps) {
  const { projects, currentProject, chats, setCurrentProject } = useStore()

  // Safe date formatting function
  const formatDate = (dateString: string | Date) => {
    try {
      const date = typeof dateString === 'string' ? new Date(dateString) : dateString
      if (isNaN(date.getTime())) {
        return 'Recently'
      }
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
      if (isNaN(date.getTime())) {
        return 'Recently'
      }
      return new Intl.DateTimeFormat('en-US', {
        hour: '2-digit',
        minute: '2-digit'
      }).format(date)
    } catch {
      return 'Recently'
    }
  }

  return (
    <div className={cn(
      "fixed left-0 top-0 h-full bg-white border-r border-gray-200 z-50 transition-all duration-300 flex flex-col",
      isCollapsed ? "w-20" : "w-80"
    )}>
      {/* Header - Fixed */}
      <div className="flex-shrink-0 flex items-center justify-between p-4 border-b border-gray-200">
        {!isCollapsed && (
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">R</span>
            </div>
            <span className="text-xl font-bold text-gray-900">RagFlow</span>
          </div>
        )}
        <Button
          variant="ghost"
          size="sm"
          onClick={onToggleCollapse}
          className="p-2 text-gray-500 hover:text-gray-700"
        >
          <Menu size={20} />
        </Button>
      </div>

      {/* Current Project - Compact Display */}
      {!isCollapsed && currentProject && (
        <div className="flex-shrink-0 mx-4 mt-4 p-3 bg-blue-50 border border-blue-200 rounded-xl">
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

      {/* Navigation - Fixed */}
      <div className="flex-shrink-0 p-4 space-y-2">
        {navItems.map(item => {
          const Icon = item.icon
          const isActive = currentView === item.id
          
          return (
            <Button
              key={item.id}
              variant="ghost"
              onClick={() => onViewChange(item.id)}
              className={cn(
                "w-full justify-start px-3 py-3 h-auto rounded-xl transition-all",
                isActive 
                  ? "bg-blue-50 text-blue-600 border border-blue-200" 
                  : "text-gray-700 hover:bg-gray-50"
              )}
            >
              <Icon size={20} />
              {!isCollapsed && (
                <span className="ml-3 font-medium">{item.label}</span>
              )}
            </Button>
          )
        })}
      </div>

      {/* Quick Action - Fixed */}
      {!isCollapsed && (
        <div className="flex-shrink-0 px-4 pb-4">
          <Button 
            className="w-full justify-start bg-blue-50 text-blue-600 hover:bg-blue-100 h-10"
            variant="ghost"
          >
            <Plus size={18} />
            <span className="ml-3 font-medium">New Chat</span>
          </Button>
        </div>
      )}

      {/* Scrollable Content Area */}
      {!isCollapsed && (
        <div className="flex-1 overflow-y-auto overflow-x-hidden px-4 pb-4">
          <div className="space-y-4">
            {/* Recent Chats */}
            {currentView === 'chat' && (
              <div>
                <h3 className="text-sm font-semibold text-gray-900 mb-3 sticky top-0 bg-white py-1">
                  Recent Chats
                </h3>
                <div className="space-y-2">
                  {chats.slice(0, 10).map(chat => (
                    <div 
                      key={chat.id} 
                      className="p-2.5 rounded-lg hover:bg-gray-50 cursor-pointer group transition-colors"
                    >
                      <div className="font-medium text-sm text-gray-900 truncate mb-1">
                        {chat.title}
                      </div>
                      <div className="text-xs text-gray-500">
                        {formatTime(chat.updatedAt)}
                      </div>
                    </div>
                  ))}
                  {chats.length === 0 && (
                    <div className="text-center py-6 text-gray-500">
                      <MessageCircle className="h-6 w-6 mx-auto mb-2 opacity-50" />
                      <p className="text-sm">No recent chats</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* All Projects */}
            {currentView === 'projects' && (
              <div>
                <div className="flex items-center justify-between mb-3 sticky top-0 bg-white py-1">
                  <h3 className="text-sm font-semibold text-gray-900">All Projects</h3>
                  <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded-full">
                    {projects.length}
                  </span>
                </div>
                <div className="space-y-2">
                  {projects.map(project => (
                    <div 
                      key={project.id} 
                      onClick={() => setCurrentProject(project)}
                      className={cn(
                        "p-3 rounded-xl cursor-pointer transition-all",
                        currentProject?.id === project.id
                          ? "bg-blue-600 text-white shadow-md"
                          : "hover:bg-gray-50 border border-transparent hover:border-gray-200"
                      )}
                    >
                      <div className="flex items-center justify-between mb-1">
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
                  {projects.length === 0 && (
                    <div className="text-center py-6 text-gray-500">
                      <FolderOpen className="h-6 w-6 mx-auto mb-2 opacity-50" />
                      <p className="text-sm">No projects yet</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            
          </div>
        </div>
      )}
    </div>
  )
}