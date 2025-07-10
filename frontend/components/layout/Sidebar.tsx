'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ChatBubbleLeftIcon,
  FolderIcon,
  DocumentTextIcon,
  Cog6ToothIcon,
  PlusIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  HeartIcon,
} from '@heroicons/react/24/outline'
import {
  ChatBubbleLeftIcon as ChatBubbleLeftSolidIcon,
  FolderIcon as FolderSolidIcon,
  DocumentTextIcon as DocumentTextSolidIcon,
  Cog6ToothIcon as Cog6ToothSolidIcon,
} from '@heroicons/react/24/solid'
import { Button } from '@/components/ui/button'
import { ProjectList } from '@/components/projects/ProjectList'
import { RecentChats } from '@/components/chat/RecentChats'
import { useStore } from '@/stores/useStore'
import { cn } from '@/lib/utils'

interface SidebarProps {
  currentView: string
  onViewChange: (view: string) => void
  isDarkMode: boolean
  onThemeToggle: () => void
}

const navigationItems = [
  {
    id: 'chat',
    label: 'AI Chat',
    icon: ChatBubbleLeftIcon,
    iconSolid: ChatBubbleLeftSolidIcon,
    description: 'Smart Conversations',
    shortcut: '⌘ 1',
  },
  {
    id: 'projects',
    label: 'Projects',
    icon: FolderIcon,
    iconSolid: FolderSolidIcon,
    description: 'Organize Workspace',
    shortcut: '⌘ 2',
  },
  {
    id: 'documents',
    label: 'Documents',
    icon: DocumentTextIcon,
    iconSolid: DocumentTextSolidIcon,
    description: 'File Management',
    shortcut: '⌘ 3',
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: Cog6ToothIcon,
    iconSolid: Cog6ToothSolidIcon,
    description: 'Preferences',
    shortcut: '⌘ ,',
  },
]

export function Sidebar({ currentView, onViewChange, isDarkMode, onThemeToggle }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [showNewProjectModal, setShowNewProjectModal] = useState(false)
  const { projects, currentProject } = useStore()

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed)
  }

  return (
    <motion.div
      initial={false}
      animate={{ width: isCollapsed ? 80 : 320 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className={`relative z-10 ${
        isDarkMode 
          ? 'bg-slate-900/50 border-slate-700/50' 
          : 'bg-white/70 border-slate-200/50'
      } backdrop-blur-xl border-r shadow-xl`}
    >
      {/* Modern Header */}
      <div className={`p-6 border-b ${isDarkMode ? 'border-slate-700/50' : 'border-slate-200/50'}`}>
        <div className="flex items-center justify-between">
          <AnimatePresence mode="wait">
            {!isCollapsed && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
                className="flex items-center space-x-4"
              >
                <div className="relative">
                  <div className="w-10 h-10 bg-gradient-to-br from-violet-500 via-purple-500 to-blue-500 rounded-xl flex items-center justify-center shadow-xl">
                    <span className="text-white font-bold text-lg">R</span>
                  </div>
                  <div className="absolute -top-1 -right-1 w-4 h-4 bg-emerald-400 rounded-full border-2 border-white shadow-lg animate-pulse" />
                </div>
                <div>
                  <h1 className={`text-xl font-bold bg-gradient-to-r from-violet-600 to-blue-600 bg-clip-text text-transparent ${
                    isDarkMode ? 'from-violet-400 to-blue-400' : ''
                  }`}>
                    RagFlow
                  </h1>
                  <p className={`text-xs ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                    AI Document Analysis
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <Button
            variant="ghost"
            size="sm"
            onClick={toggleCollapse}
            className={`flex-shrink-0 p-3 rounded-xl transition-all duration-200 ${
              isDarkMode 
                ? 'hover:bg-slate-800/50 text-slate-400 hover:text-white' 
                : 'hover:bg-slate-100/50 text-slate-500 hover:text-slate-700'
            }`}
            aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <motion.div
              animate={{ rotate: isCollapsed ? 180 : 0 }}
              transition={{ duration: 0.2 }}
            >
              {isCollapsed ? (
                <ChevronRightIcon className="h-4 w-4" />
              ) : (
                <ChevronLeftIcon className="h-4 w-4" />
              )}
            </motion.div>
          </Button>
        </div>
      </div>

      {/* Current Project Display */}
      {!isCollapsed && currentProject && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className={`p-6 border-b ${isDarkMode ? 'border-slate-700/50' : 'border-slate-200/50'}`}
        >
          <div className={`p-4 rounded-2xl ${
            isDarkMode 
              ? 'bg-gradient-to-r from-violet-500/10 to-blue-500/10 border border-violet-500/20' 
              : 'bg-gradient-to-r from-violet-50 to-blue-50 border border-violet-200/50'
          }`}>
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-violet-500 to-blue-500 rounded-lg flex items-center justify-center">
                <FolderIcon className="h-4 w-4 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <p className={`text-xs ${isDarkMode ? 'text-slate-400' : 'text-slate-500'} mb-1`}>
                  Active Project
                </p>
                <p className={`font-semibold ${isDarkMode ? 'text-white' : 'text-slate-900'} truncate`}>
                  {currentProject.name}
                </p>
                <p className={`text-xs ${isDarkMode ? 'text-violet-400' : 'text-violet-600'}`}>
                  {currentProject.document_count} documents
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Modern Navigation with inline content */}
      <nav className="flex-1 p-6 overflow-hidden">
        <div className="space-y-2 h-full flex flex-col">
          {navigationItems.map((item, index) => {
            const Icon = currentView === item.id ? item.iconSolid : item.icon
            const isActive = currentView === item.id

            return (
              <div key={item.id}>
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.1 * index }}
                >
                  <Button
                    variant="ghost"
                    onClick={() => onViewChange(item.id)}
                    className={cn(
                      'w-full group relative overflow-hidden rounded-2xl transition-all duration-300 p-4 h-auto',
                      isActive
                        ? isDarkMode
                          ? 'bg-gradient-to-r from-violet-500/20 to-blue-500/20 text-white border border-violet-500/30 shadow-lg shadow-violet-500/10'
                          : 'bg-gradient-to-r from-violet-100 to-blue-100 text-violet-900 border border-violet-200 shadow-lg'
                        : isDarkMode
                          ? 'hover:bg-slate-800/50 text-slate-400 hover:text-white border border-transparent hover:border-slate-700/50'
                          : 'hover:bg-slate-50/50 text-slate-600 hover:text-slate-900 border border-transparent hover:border-slate-200'
                    )}
                  >
                    <div className="flex items-center space-x-4 w-full">
                      <Icon className={`h-5 w-5 flex-shrink-0 transition-transform duration-200 ${
                        isActive ? 'scale-110' : 'group-hover:scale-105'
                      }`} />
                      <AnimatePresence mode="wait">
                        {!isCollapsed && (
                          <motion.div
                            initial={{ opacity: 0, width: 0 }}
                            animate={{ opacity: 1, width: 'auto' }}
                            exit={{ opacity: 0, width: 0 }}
                            transition={{ duration: 0.2 }}
                            className="flex-1 text-left"
                          >
                            <div className="font-semibold text-sm">{item.label}</div>
                            <div className={`text-xs opacity-70 ${
                              isActive ? '' : 'group-hover:opacity-100'
                            }`}>
                              {item.description}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                      {isActive && !isCollapsed && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className="w-2 h-2 bg-violet-400 rounded-full"
                        />
                      )}
                      <AnimatePresence mode="wait">
                        {!isCollapsed && (
                          <motion.span
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.2, delay: 0.1 }}
                            className={`ml-auto text-xs ${
                              isDarkMode ? 'text-slate-500' : 'text-slate-400'
                            }`}
                          >
                            {item.shortcut}
                          </motion.span>
                        )}
                      </AnimatePresence>
                    </div>
                    
                    {/* Active indicator */}
                    {isActive && (
                      <motion.div
                        layoutId="activeTab"
                        className="absolute inset-0 bg-gradient-to-r from-violet-500/10 to-blue-500/10 rounded-2xl"
                        transition={{ type: "spring", stiffness: 300, damping: 30 }}
                      />
                    )}
                  </Button>
                </motion.div>

                {/* Inline Content for Chat Tab */}
                <AnimatePresence>
                  {!isCollapsed && currentView === 'chat' && item.id === 'chat' && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3 }}
                      className="mt-3 ml-4 pr-2"
                    >
                      <div className={`p-4 rounded-xl ${
                        isDarkMode ? 'bg-slate-800/30' : 'bg-slate-50/50'
                      } border-l-2 border-violet-400/30`}>
                        <div className="flex items-center justify-between mb-3">
                          <h4 className={`text-sm font-medium ${
                            isDarkMode ? 'text-slate-300' : 'text-slate-700'
                          }`}>
                            Recent Chats
                          </h4>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0"
                          >
                            <PlusIcon className="h-3 w-3" />
                          </Button>
                        </div>
                        
                        {/* Scrollable Chat List */}
                        <div className="max-h-48 overflow-y-auto pr-2 space-y-2">
                          <RecentChats />
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Inline Content for Projects Tab */}
                <AnimatePresence>
                  {!isCollapsed && currentView === 'projects' && item.id === 'projects' && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3 }}
                      className="mt-3 ml-4 pr-2"
                    >
                      <div className={`p-4 rounded-xl ${
                        isDarkMode ? 'bg-slate-800/30' : 'bg-slate-50/50'
                      } border-l-2 border-violet-400/30`}>
                        <div className="flex items-center justify-between mb-3">
                          <h4 className={`text-sm font-medium ${
                            isDarkMode ? 'text-slate-300' : 'text-slate-700'
                          }`}>
                            Projects
                          </h4>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setShowNewProjectModal(true)}
                            className="h-6 w-6 p-0"
                          >
                            <PlusIcon className="h-3 w-3" />
                          </Button>
                        </div>
                        
                        {/* Scrollable Project List */}
                        <div className="max-h-48 overflow-y-auto pr-2">
                          <ProjectList />
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Inline Content for Documents Tab */}
                <AnimatePresence>
                  {!isCollapsed && currentView === 'documents' && item.id === 'documents' && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3 }}
                      className="mt-3 ml-4 pr-2"
                    >
                      <div className={`p-4 rounded-xl ${
                        isDarkMode ? 'bg-slate-800/30' : 'bg-slate-50/50'
                      } border-l-2 border-violet-400/30`}>
                        <h4 className={`text-sm font-medium mb-3 ${
                          isDarkMode ? 'text-slate-300' : 'text-slate-700'
                        }`}>
                          Quick Access
                        </h4>
                        <div className="space-y-1">
                          <Button variant="ghost" className="w-full justify-start text-xs h-8 px-2">
                            📄 Recent Uploads
                          </Button>
                          <Button variant="ghost" className="w-full justify-start text-xs h-8 px-2">
                            ⭐ Favorites
                          </Button>
                          <Button variant="ghost" className="w-full justify-start text-xs h-8 px-2">
                            ⏳ Processing
                          </Button>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )
          })}
        </div>
      </nav>

      {/* Modern Footer */}
      <div className={`border-t ${isDarkMode ? 'border-slate-700/50' : 'border-slate-200/50'} p-6 space-y-4`}>
        {/* Theme Toggle */}
        <Button
          variant="ghost"
          size="sm"
          onClick={onThemeToggle}
          className={cn(
            'w-full transition-all duration-300 p-4 rounded-2xl',
            isCollapsed ? 'justify-center px-2' : 'justify-start px-4',
            isDarkMode 
              ? 'bg-slate-800/50 hover:bg-slate-700/50 text-slate-300' 
              : 'bg-slate-100/50 hover:bg-slate-200/50 text-slate-600'
          )}
        >
          <motion.div 
            animate={{ rotate: isDarkMode ? 180 : 0 }}
            transition={{ duration: 0.5 }}
            className="text-xl mr-4 flex-shrink-0"
          >
            {isDarkMode ? '🌙' : '☀️'}
          </motion.div>
          <AnimatePresence mode="wait">
            {!isCollapsed && (
              <motion.div
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: 'auto' }}
                exit={{ opacity: 0, width: 0 }}
                transition={{ duration: 0.2 }}
                className="flex-1 text-left"
              >
                <div className="font-medium text-sm">
                  {isDarkMode ? 'Dark Mode' : 'Light Mode'}
                </div>
                <div className="text-xs opacity-70">
                  {isDarkMode ? 'Switch to light theme' : 'Switch to dark theme'}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </Button>

        {/* Made with Love */}
        <AnimatePresence mode="wait">
          {!isCollapsed && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              transition={{ duration: 0.2 }}
              className={`flex items-center justify-center gap-2 text-xs pt-2 ${
                isDarkMode ? 'text-slate-500' : 'text-slate-400'
              }`}
            >
              <span>Made with</span>
              <HeartIcon className="h-3 w-3 text-red-500 animate-pulse" />
              <span>by RagFlow</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}