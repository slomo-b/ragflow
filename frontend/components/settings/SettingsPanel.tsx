// frontend/components/settings/SettingsPanel.tsx - Fehler behoben
'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  SettingsIcon,
  ServerIcon,
  DatabaseIcon,
  BrainIcon,
  KeyIcon,
  PaletteIcon,
  BellIcon,
  ShieldIcon,
  DownloadIcon,
  TrashIcon,
  RefreshCwIcon,
  CheckCircleIcon,
  AlertCircleIcon,
  InfoIcon,
  SaveIcon,
  RotateCcwIcon, // Ersetzt ResetIcon
  ExternalLinkIcon,
  ClipboardIcon,
  EyeIcon,
  EyeOffIcon
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { Input } from "@/components/ui/Input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/Badge"
import { Switch } from "@/components/ui/switch"
import { Separator } from "@/components/ui/separator"
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { 
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { 
  Tabs, 
  TabsContent, 
  TabsList, 
  TabsTrigger 
} from "@/components/ui/tabs"
import { cn } from '@/lib/utils'
import { copyToClipboard, downloadFile } from '@/lib/utils' // Aus utils importieren
import { ApiService } from '@/services/api' // Korrekte API import
import { useStore } from '@/stores/useStore'
import toast from 'react-hot-toast'

interface SystemInfo {
  app: {
    name: string
    version: string
    python_version: string
  }
  features: Record<string, boolean>
  stats: {
    projects: number
    documents: number
    chats: number
    rag_documents: number
  }
  settings: {
    chunk_size: number
    chunk_overlap: number
    top_k: number
  }
}

interface Settings {
  // API Settings
  googleApiKey: string
  openaiApiKey: string
  
  // RAG Settings
  chunkSize: number
  chunkOverlap: number
  topK: number
  minSimilarity: number
  
  // UI Settings
  theme: 'light' | 'dark' | 'system'
  sidebarCollapsed: boolean
  enableAnimations: boolean
  enableNotifications: boolean
  
  // Advanced Settings
  maxFileSize: number
  allowedFileTypes: string[]
  autoSave: boolean
  debugMode: boolean
}

const defaultSettings: Settings = {
  googleApiKey: '',
  openaiApiKey: '',
  chunkSize: 500,
  chunkOverlap: 50,
  topK: 5,
  minSimilarity: 0.1,
  theme: 'light',
  sidebarCollapsed: false,
  enableAnimations: true,
  enableNotifications: true,
  maxFileSize: 100,
  allowedFileTypes: ['.pdf', '.docx', '.txt', '.md'],
  autoSave: true,
  debugMode: false
}

export const SettingsPanel: React.FC = () => {
  // State Management
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null)
  const [settings, setSettings] = useState<Settings>(defaultSettings)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [activeTab, setActiveTab] = useState('general')
  const [showApiKeys, setShowApiKeys] = useState(false)
  const [showResetDialog, setShowResetDialog] = useState(false)
  const [showExportDialog, setShowExportDialog] = useState(false)
  
  // Store integration
  const store = useStore()

  // Load system info and settings
  useEffect(() => {
    loadSystemInfo()
    loadSettings()
  }, [])

  const loadSystemInfo = async () => {
    try {
      const result = await ApiService.getSystemInfo()
      setSystemInfo(result)
    } catch (error) {
      console.error('Failed to load system info:', error)
      toast.error('Failed to load system information')
    }
    setLoading(false)
  }

  const loadSettings = () => {
    try {
      const saved = localStorage.getItem('ragflow-settings')
      if (saved) {
        const parsed = JSON.parse(saved)
        setSettings({ ...defaultSettings, ...parsed })
      }
    } catch (error) {
      console.error('Failed to load settings:', error)
    }
  }

  const saveSettings = useCallback(async () => {
    setSaving(true)
    try {
      // Save to localStorage
      localStorage.setItem('ragflow-settings', JSON.stringify(settings))
      
      // Update store
      store.setTheme?.(settings.theme)
      store.setSidebarCollapsed?.(settings.sidebarCollapsed)
      
      toast.success('Settings saved successfully')
    } catch (error) {
      console.error('Failed to save settings:', error)
      toast.error('Failed to save settings')
    }
    setSaving(false)
  }, [settings, store])

  const resetSettings = () => {
    setSettings(defaultSettings)
    localStorage.removeItem('ragflow-settings')
    store.reset?.()
    toast.success('Settings reset to defaults')
    setShowResetDialog(false)
  }

  const exportSettings = () => {
    const exportData = {
      settings,
      systemInfo,
      timestamp: new Date().toISOString(),
      version: systemInfo?.app.version || 'unknown'
    }
    
    const content = JSON.stringify(exportData, null, 2)
    downloadFile(content, `ragflow-settings-${new Date().toISOString().split('T')[0]}.json`, 'application/json')
    toast.success('Settings exported successfully')
    setShowExportDialog(false)
  }

  const copySystemInfo = async () => {
    if (!systemInfo) return
    
    const info = `RagFlow System Information
Version: ${systemInfo.app.version}
Python: ${systemInfo.app.python_version.split(' ')[0]}
Features: ${Object.entries(systemInfo.features).map(([k, v]) => `${k}: ${v ? '✓' : '✗'}`).join(', ')}
Stats: ${systemInfo.stats.projects} projects, ${systemInfo.stats.documents} documents`

    const success = await copyToClipboard(info)
    if (success) {
      toast.success('System info copied to clipboard')
    } else {
      toast.error('Failed to copy system info')
    }
  }

  // System Information Component
  const SystemInfoSection: React.FC = () => {
    if (!systemInfo) return null

    const featureList = Object.entries(systemInfo.features).map(([key, value]) => ({
      name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      enabled: value,
      key
    }))

    return (
      <div className="space-y-6">
        {/* System Overview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ServerIcon className="w-5 h-5" />
              System Overview
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <Label className="text-gray-600">Application Version</Label>
                <div className="font-medium">{systemInfo.app.version}</div>
              </div>
              <div>
                <Label className="text-gray-600">Python Version</Label>
                <div className="font-medium">{systemInfo.app.python_version.split(' ')[0]}</div>
              </div>
            </div>
            
            <div className="flex items-center gap-2 pt-2">
              <Button variant="outline" size="sm" onClick={copySystemInfo}>
                <ClipboardIcon className="w-4 h-4 mr-2" />
                Copy Info
              </Button>
              <Button variant="outline" size="sm" onClick={() => setShowExportDialog(true)}>
                <DownloadIcon className="w-4 h-4 mr-2" />
                Export
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Features Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BrainIcon className="w-5 h-5" />
              Available Features
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3">
              {featureList.map((feature) => (
                <div key={feature.key} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-sm font-medium">{feature.name}</span>
                  <Badge variant={feature.enabled ? "default" : "secondary"}>
                    {feature.enabled ? (
                      <CheckCircleIcon className="w-3 h-3 mr-1" />
                    ) : (
                      <AlertCircleIcon className="w-3 h-3 mr-1" />
                    )}
                    {feature.enabled ? 'Enabled' : 'Disabled'}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Statistics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <DatabaseIcon className="w-5 h-5" />
              System Statistics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{systemInfo.stats.projects}</div>
                <div className="text-gray-600">Projects</div>
              </div>
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{systemInfo.stats.documents}</div>
                <div className="text-gray-600">Documents</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">{systemInfo.stats.chats}</div>
                <div className="text-gray-600">Chats</div>
              </div>
              <div className="text-center p-4 bg-orange-50 rounded-lg">
                <div className="text-2xl font-bold text-orange-600">{systemInfo.stats.rag_documents}</div>
                <div className="text-gray-600">Indexed</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  // Main Settings Tabs
  const renderSettingsTab = (tabId: string) => {
    switch (tabId) {
      case 'general':
        return (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <KeyIcon className="w-5 h-5" />
                  API Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="google-api-key">Google AI API Key</Label>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowApiKeys(!showApiKeys)}
                    >
                      {showApiKeys ? <EyeOffIcon className="w-4 h-4" /> : <EyeIcon className="w-4 h-4" />}
                    </Button>
                  </div>
                  <Input
                    id="google-api-key"
                    type={showApiKeys ? "text" : "password"}
                    value={settings.googleApiKey}
                    onChange={(e) => setSettings(prev => ({ ...prev, googleApiKey: e.target.value }))}
                    placeholder="Enter your Google AI API key"
                  />
                  <p className="text-xs text-gray-600">
                    Required for AI chat functionality. Get your key from{' '}
                    <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                      Google AI Studio
                      <ExternalLinkIcon className="w-3 h-3 inline ml-1" />
                    </a>
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BrainIcon className="w-5 h-5" />
                  RAG Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="chunk-size">Chunk Size</Label>
                    <Input
                      id="chunk-size"
                      type="number"
                      value={settings.chunkSize}
                      onChange={(e) => setSettings(prev => ({ ...prev, chunkSize: parseInt(e.target.value) || 500 }))}
                      min="100"
                      max="2000"
                    />
                    <p className="text-xs text-gray-600">Characters per document chunk (100-2000)</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="chunk-overlap">Chunk Overlap</Label>
                    <Input
                      id="chunk-overlap"
                      type="number"
                      value={settings.chunkOverlap}
                      onChange={(e) => setSettings(prev => ({ ...prev, chunkOverlap: parseInt(e.target.value) || 50 }))}
                      min="0"
                      max="500"
                    />
                    <p className="text-xs text-gray-600">Characters overlap between chunks (0-500)</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="top-k">Top K Results</Label>
                    <Input
                      id="top-k"
                      type="number"
                      value={settings.topK}
                      onChange={(e) => setSettings(prev => ({ ...prev, topK: parseInt(e.target.value) || 5 }))}
                      min="1"
                      max="20"
                    />
                    <p className="text-xs text-gray-600">Number of relevant chunks to retrieve (1-20)</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="min-similarity">Min Similarity</Label>
                    <Input
                      id="min-similarity"
                      type="number"
                      step="0.01"
                      value={settings.minSimilarity}
                      onChange={(e) => setSettings(prev => ({ ...prev, minSimilarity: parseFloat(e.target.value) || 0.1 }))}
                      min="0"
                      max="1"
                    />
                    <p className="text-xs text-gray-600">Minimum similarity threshold (0.0-1.0)</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )

      case 'ui':
        return (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <PaletteIcon className="w-5 h-5" />
                  Appearance
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="theme">Theme</Label>
                  <Select value={settings.theme} onValueChange={(value: 'light' | 'dark' | 'system') => setSettings(prev => ({ ...prev, theme: value }))}>
                    <SelectTrigger id="theme">
                      <SelectValue placeholder="Select theme" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">Light</SelectItem>
                      <SelectItem value="dark">Dark</SelectItem>
                      <SelectItem value="system">System</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Collapsed Sidebar</Label>
                    <p className="text-xs text-gray-600">Start with sidebar collapsed</p>
                  </div>
                  <Switch
                    checked={settings.sidebarCollapsed}
                    onCheckedChange={(checked) => setSettings(prev => ({ ...prev, sidebarCollapsed: checked }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Enable Animations</Label>
                    <p className="text-xs text-gray-600">Show smooth transitions and animations</p>
                  </div>
                  <Switch
                    checked={settings.enableAnimations}
                    onCheckedChange={(checked) => setSettings(prev => ({ ...prev, enableAnimations: checked }))}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BellIcon className="w-5 h-5" />
                  Notifications
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Enable Notifications</Label>
                    <p className="text-xs text-gray-600">Show toast notifications and alerts</p>
                  </div>
                  <Switch
                    checked={settings.enableNotifications}
                    onCheckedChange={(checked) => setSettings(prev => ({ ...prev, enableNotifications: checked }))}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        )

      case 'advanced':
        return (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ShieldIcon className="w-5 h-5" />
                  File Upload Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="max-file-size">Max File Size (MB)</Label>
                  <Input
                    id="max-file-size"
                    type="number"
                    value={settings.maxFileSize}
                    onChange={(e) => setSettings(prev => ({ ...prev, maxFileSize: parseInt(e.target.value) || 100 }))}
                    min="1"
                    max="1000"
                  />
                  <p className="text-xs text-gray-600">Maximum file size for uploads (1-1000 MB)</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="allowed-types">Allowed File Types</Label>
                  <Textarea
                    id="allowed-types"
                    value={settings.allowedFileTypes.join(', ')}
                    onChange={(e) => setSettings(prev => ({ 
                      ...prev, 
                      allowedFileTypes: e.target.value.split(',').map(t => t.trim()).filter(Boolean)
                    }))}
                    placeholder=".pdf, .docx, .txt, .md"
                    rows={2}
                  />
                  <p className="text-xs text-gray-600 mt-1">
                    Comma-separated list of allowed file extensions
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <SettingsIcon className="w-5 h-5" />
                  Advanced Options
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Auto-save Settings</Label>
                    <p className="text-xs text-gray-600">Automatically save changes</p>
                  </div>
                  <Switch
                    checked={settings.autoSave}
                    onCheckedChange={(checked) => setSettings(prev => ({ ...prev, autoSave: checked }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Debug Mode</Label>
                    <p className="text-xs text-gray-600">Show additional debugging information</p>
                  </div>
                  <Switch
                    checked={settings.debugMode}
                    onCheckedChange={(checked) => setSettings(prev => ({ ...prev, debugMode: checked }))}
                  />
                </div>

                <Separator />

                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-amber-600">
                    <AlertCircleIcon className="w-4 h-4" />
                    <Label>Danger Zone</Label>
                  </div>
                  
                  <div className="flex flex-col sm:flex-row gap-2">
                    <Button 
                      variant="outline" 
                      onClick={() => setShowResetDialog(true)}
                      className="border-red-200 text-red-600 hover:bg-red-50"
                    >
                      <RotateCcwIcon className="w-4 h-4 mr-2" />
                      Reset Settings
                    </Button>
                    
                    <Button 
                      variant="outline"
                      onClick={() => {
                        localStorage.clear()
                        window.location.reload()
                      }}
                      className="border-red-200 text-red-600 hover:bg-red-50"
                    >
                      <TrashIcon className="w-4 h-4 mr-2" />
                      Clear All Data
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )

      case 'system':
        return <SystemInfoSection />

      default:
        return null
    }
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"
          />
          <p className="text-gray-600">Loading settings...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-gray-600 to-gray-800 rounded-lg flex items-center justify-center">
              <SettingsIcon className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">Settings</h1>
              <p className="text-sm text-gray-600">Configure your RagFlow experience</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="outline" onClick={loadSystemInfo}>
              <RefreshCwIcon className="w-4 h-4 mr-2" />
              Refresh
            </Button>
            <Button onClick={saveSettings} disabled={saving}>
              {saving ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-4 h-4 border-2 border-current border-t-transparent rounded-full mr-2"
                />
              ) : (
                <SaveIcon className="w-4 h-4 mr-2" />
              )}
              {saving ? 'Saving...' : 'Save Settings'}
            </Button>
          </div>
        </div>
      </div>

      {/* Settings Content */}
      <div className="flex-1 overflow-hidden">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
          <div className="bg-white border-b border-gray-200 px-6">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="general" className="flex items-center gap-2">
                <KeyIcon className="w-4 h-4" />
                General
              </TabsTrigger>
              <TabsTrigger value="ui" className="flex items-center gap-2">
                <PaletteIcon className="w-4 h-4" />
                Appearance
              </TabsTrigger>
              <TabsTrigger value="advanced" className="flex items-center gap-2">
                <ShieldIcon className="w-4 h-4" />
                Advanced
              </TabsTrigger>
              <TabsTrigger value="system" className="flex items-center gap-2">
                <ServerIcon className="w-4 h-4" />
                System
              </TabsTrigger>
            </TabsList>
          </div>

          <div className="flex-1 overflow-y-auto p-6">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.2 }}
              >
                <TabsContent value={activeTab} className="mt-0">
                  {renderSettingsTab(activeTab)}
                </TabsContent>
              </motion.div>
            </AnimatePresence>
          </div>
        </Tabs>
      </div>

      {/* Reset Dialog */}
      <Dialog open={showResetDialog} onOpenChange={setShowResetDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertCircleIcon className="w-5 h-5 text-amber-600" />
              Reset Settings
            </DialogTitle>
            <DialogDescription>
              Are you sure you want to reset all settings to their default values?
              This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowResetDialog(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={resetSettings}>
              <RotateCcwIcon className="w-4 h-4 mr-2" />
              Reset Settings
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Export Dialog */}
      <Dialog open={showExportDialog} onOpenChange={setShowExportDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <DownloadIcon className="w-5 h-5" />
              Export Settings
            </DialogTitle>
            <DialogDescription>
              Export your current settings and system information for backup or sharing.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowExportDialog(false)}>
              Cancel
            </Button>
            <Button onClick={exportSettings}>
              <DownloadIcon className="w-4 h-4 mr-2" />
              Export
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default SettingsPanel