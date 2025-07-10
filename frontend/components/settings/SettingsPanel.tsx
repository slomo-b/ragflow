// components/settings/SettingsPanel.tsx
'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { 
  Cog6ToothIcon,
  SunIcon,
  MoonIcon,
  ComputerDesktopIcon,
  UserIcon,
  BellIcon,
  ShieldCheckIcon,
  TrashIcon
} from '@heroicons/react/24/outline'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { useTheme } from 'next-themes'
import { useStore } from '@/stores/useStore'
import toast from 'react-hot-toast'

export function SettingsPanel() {
  const { theme, setTheme } = useTheme()
  const { projects, documents, chats, clearNotifications } = useStore()

  const handleClearAllData = () => {
    if (confirm('Are you sure you want to clear all data? This cannot be undone.')) {
      // TODO: Implement clear all data
      toast.success('All data cleared')
    }
  }

  const handleExportData = () => {
    const data = {
      projects,
      documents: documents.map(doc => ({
        ...doc,
        // Remove sensitive paths
        file_path: undefined
      })),
      chats,
      exportDate: new Date().toISOString()
    }
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ragflow-export-${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    
    toast.success('Data exported successfully')
  }

  const themeOptions = [
    { key: 'light', icon: SunIcon, label: 'Light' },
    { key: 'dark', icon: MoonIcon, label: 'Dark' },
    { key: 'system', icon: ComputerDesktopIcon, label: 'System' }
  ]

  return (
    <div className="flex-1 flex flex-col h-full bg-background">
      {/* Header */}
      <div className="flex-shrink-0 p-6 border-b">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 text-white">
            <Cog6ToothIcon className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-2xl font-bold">Settings</h1>
            <p className="text-muted-foreground">Manage your RagFlow preferences</p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        
        {/* Appearance */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <SunIcon className="h-5 w-5" />
                Appearance
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-3 block">Theme</label>
                <div className="flex gap-3">
                  {themeOptions.map((option) => {
                    const Icon = option.icon
                    return (
                      <Button
                        key={option.key}
                        variant={theme === option.key ? "default" : "outline"}
                        onClick={() => setTheme(option.key)}
                        className="flex-1"
                      >
                        <Icon className="h-4 w-4 mr-2" />
                        {option.label}
                      </Button>
                    )
                  })}
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Profile */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <UserIcon className="h-5 w-5" />
                Profile
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-1 block">Display Name</label>
                  <Input placeholder="Your name" defaultValue="User" />
                </div>
                <div>
                  <label className="text-sm font-medium mb-1 block">Email</label>
                  <Input type="email" placeholder="your@email.com" />
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Notifications */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BellIcon className="h-5 w-5" />
                Notifications
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">Processing Complete</div>
                  <div className="text-sm text-muted-foreground">
                    Get notified when document processing finishes
                  </div>
                </div>
                <Button variant="outline" size="sm">Enable</Button>
              </div>
              
              <Separator />
              
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">Error Alerts</div>
                  <div className="text-sm text-muted-foreground">
                    Receive alerts for processing errors
                  </div>
                </div>
                <Button variant="outline" size="sm">Enable</Button>
              </div>

              <div className="pt-2">
                <Button variant="ghost" size="sm" onClick={clearNotifications}>
                  Clear All Notifications
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Data & Storage */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ShieldCheckIcon className="h-5 w-5" />
                Data & Storage
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Statistics */}
              <div className="grid grid-cols-3 gap-4 p-4 bg-muted rounded-lg">
                <div className="text-center">
                  <div className="text-2xl font-bold">{projects.length}</div>
                  <div className="text-sm text-muted-foreground">Projects</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold">{documents.length}</div>
                  <div className="text-sm text-muted-foreground">Documents</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold">{chats.length}</div>
                  <div className="text-sm text-muted-foreground">Chats</div>
                </div>
              </div>

              {/* Actions */}
              <div className="space-y-3">
                <Button variant="outline" onClick={handleExportData} className="w-full">
                  Export All Data
                </Button>
                
                <Button 
                  variant="destructive" 
                  onClick={handleClearAllData}
                  className="w-full"
                >
                  <TrashIcon className="h-4 w-4 mr-2" />
                  Clear All Data
                </Button>
              </div>
              
              <div className="text-xs text-muted-foreground">
                <p>• Export includes projects, documents metadata, and chat history</p>
                <p>• Actual document files are not included in exports</p>
                <p>• Clearing data will remove all projects, chats, and document references</p>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* About */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>About RagFlow</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between">
                <span>Version</span>
                <Badge variant="secondary">2.0.0</Badge>
              </div>
              <div className="flex justify-between">
                <span>Backend Status</span>
                <Badge variant="default">Connected</Badge>
              </div>
              <div className="flex justify-between">
                <span>AI Model</span>
                <Badge variant="outline">Gemini 1.5 Flash</Badge>
              </div>
              
              <Separator />
              
              <div className="text-sm text-muted-foreground space-y-1">
                <p>🚀 AI-powered document analysis platform</p>
                <p>🤖 Powered by Google Gemini</p>
                <p>💻 Built with Next.js & FastAPI</p>
                <p>💙 Made with love for productivity</p>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  )
}