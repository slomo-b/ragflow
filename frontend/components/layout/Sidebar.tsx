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
  SunIcon,
  MoonIcon,
  ComputerDesktopIcon,
  HeartIcon,
} from '@heroicons/react/24/outline'
import {
  ChatBubbleLeftIcon as ChatBubbleLeftSolidIcon,
  FolderIcon as FolderSolidIcon,
  DocumentTextIcon as DocumentTextSolidIcon,
  Cog6ToothIcon as Cog6ToothSolidIcon,
} from '@heroicons/react/24/solid'
import { useTheme } from 'next-themes'
import { Button } from '@/components/ui/button'  // Kleingeschrieben!
import { ProjectList } from '@/components/projects/ProjectList'
import { RecentChats } from '@/components/chat/RecentChats'
import { useStore } from '@/stores/useStore'
import { cn } from '@/utils/cn'

interface SidebarProps {
  currentView: string
  onViewChange: (view: string) => void
}

const navigationItems = [
  {
    id: 'chat',
    label: 'Chat',
    icon: ChatBubbleLeftIcon,
    iconSolid: ChatBubbleLeftSolidIcon,
    shortcut: '⌘ 1',
  },
  {
    id: 'projects',
    label: 'Projects',
    icon: FolderIcon,
    iconSolid: FolderSolidIcon,
    shortcut: '⌘ 2',
  },
  {
    id: 'documents',
    label: 'Documents',
    icon: DocumentTextIcon,
    iconSolid: DocumentTextSolidIcon,
    shortcut: '⌘ 3',
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: Cog6ToothIcon,
    iconSolid: Cog6ToothSolidIcon,
    shortcut: '⌘ ,',
  },
]

export function Sidebar({ currentView, onViewChange }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [showNewProjectModal, setShowNewProjectModal] = useState(false)
  const { theme, setTheme } = useTheme()
  const { projects, currentProject } = useStore()

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed)
  }

  const cycleTheme = () => {
    if (theme === 'light') {
      setTheme('dark')
    } else if (theme === 'dark') {
      setTheme('system')
    } else {
      setTheme('light')
    }
  }

  const getThemeIcon = () => {
    switch (theme) {
      case 'light':
        return SunIcon
      case 'dark':
        return MoonIcon
      default:
        return ComputerDesktopIcon
    }
  }

  const ThemeIcon = getThemeIcon()

  return (
    <motion.div
      initial={false}
      animate={{
        width: isCollapsed ? 80 : 320,
      }}
      transition={{
        duration: 0.3,
        ease: [0.4, 0, 0.2, 1],
      }}
      className="relative flex flex-col bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 shadow-lg"
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-800">
        <AnimatePresence mode="wait">
          {!isCollapsed && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              className="flex items-center space-x-3"
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 text-white font-bold text-sm">
                R
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">
                  RagFlow
                </h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">
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
          className="flex-shrink-0 p-2"
          aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {isCollapsed ? (
            <ChevronRightIcon className="h-4 w-4" />
          ) : (
            <ChevronLeftIcon className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navigationItems.map((item) => {
          const Icon = currentView === item.id ? item.iconSolid : item.icon
          const isActive = currentView === item.id

          return (
            <Button
              key={item.id}
              variant={isActive ? 'default' : 'ghost'}
              className={cn(
                'w-full justify-start gap-3 h-10',
                isCollapsed && 'justify-center px-2',
                !isCollapsed && 'px-3'
              )}
              onClick={() => onViewChange(item.id)}
            >
              <Icon className="h-5 w-5 flex-shrink-0" />
              <AnimatePresence mode="wait">
                {!isCollapsed && (
                  <motion.span
                    initial={{ opacity: 0, width: 0 }}
                    animate={{ opacity: 1, width: 'auto' }}
                    exit={{ opacity: 0, width: 0 }}
                    transition={{ duration: 0.2 }}
                    className="truncate text-sm font-medium"
                  >
                    {item.label}
                  </motion.span>
                )}
              </AnimatePresence>
              <AnimatePresence mode="wait">
                {!isCollapsed && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.2, delay: 0.1 }}
                    className="ml-auto text-xs text-gray-400 dark:text-gray-500"
                  >
                    {item.shortcut}
                  </motion.span>
                )}
              </AnimatePresence>
            </Button>
          )
        })}
      </nav>

      {/* Dynamic Content Area */}
      <div className="flex-1 overflow-hidden">
        <AnimatePresence mode="wait">
          {!isCollapsed && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              transition={{ duration: 0.2 }}
              className="p-4 h-full"
            >
              {currentView === 'projects' && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                      Projects
                    </h3>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowNewProjectModal(true)}
                      className="p-1"
                    >
                      <PlusIcon className="h-4 w-4" />
                    </Button>
                  </div>
                  <ProjectList onSelectProject={(id) => {}} />
                </div>
              )}

              {currentView === 'chat' && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                      Recent Chats
                    </h3>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="p-1"
                    >
                      <PlusIcon className="h-4 w-4" />
                    </Button>
                  </div>
                  <RecentChats />
                </div>
              )}

              {currentView === 'documents' && (
                <div className="space-y-4">
                  <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                    Quick Access
                  </h3>
                  <div className="space-y-2">
                    <Button variant="ghost" className="w-full justify-start text-sm">
                      Recent Uploads
                    </Button>
                    <Button variant="ghost" className="w-full justify-start text-sm">
                      Favorites
                    </Button>
                    <Button variant="ghost" className="w-full justify-start text-sm">
                      Unprocessed
                    </Button>
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Footer */}
      <div className="border-t border-gray-200 dark:border-gray-800 p-4 space-y-2">
        {/* Theme Toggle */}
        <Button
          variant="ghost"
          size="sm"
          onClick={cycleTheme}
          className={cn(
            'w-full gap-3',
            isCollapsed ? 'justify-center px-2' : 'justify-start px-3'
          )}
        >
          <ThemeIcon className="h-4 w-4 flex-shrink-0" />
          <AnimatePresence mode="wait">
            {!isCollapsed && (
              <motion.span
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: 'auto' }}
                exit={{ opacity: 0, width: 0 }}
                transition={{ duration: 0.2 }}
                className="truncate text-sm"
              >
                {theme === 'light' ? 'Light' : theme === 'dark' ? 'Dark' : 'System'}
              </motion.span>
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
              className="flex items-center justify-center gap-1 text-xs text-gray-400 dark:text-gray-500 pt-2"
            >
              <span>Made with</span>
              <HeartIcon className="h-3 w-3 text-red-500" />
              <span>by RagFlow</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}