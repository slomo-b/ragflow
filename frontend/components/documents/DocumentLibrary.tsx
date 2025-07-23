// frontend/components/documents/DocumentLibrary.tsx - VOLLSTÄNDIG KORRIGIERT
'use client'

import React, { useState, useRef, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  DocumentTextIcon,
  CloudArrowUpIcon,
  EyeIcon,
  TrashIcon,
  ArrowDownTrayIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  PlusIcon,
  SparklesIcon,
  FolderIcon,
  TagIcon,
  CalendarIcon,
  ArrowPathIcon,
  XMarkIcon
} from '@heroicons/react/24/outline'
import {
  DocumentTextIcon as DocumentTextSolidIcon,
  CheckCircleIcon as CheckCircleSolidIcon,
  ClockIcon as ClockSolidIcon,
  ExclamationTriangleIcon as ExclamationTriangleSolidIcon,
} from '@heroicons/react/24/solid'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { Input } from "@/components/ui/Input"
import { Badge } from "@/components/ui/Badge"
import { Progress } from "@/components/ui/progress"
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel
} from "@/components/ui/dropdown-menu"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { cn, formatFileSize, formatDate } from '@/lib/utils'
import ApiService, { Document, Project } from '@/services/api'
import toast from 'react-hot-toast'

// ✅ Types korrigiert nach exakter Backend Schema
interface ExtendedDocument extends Document {
  uploading?: boolean
  uploadProgress?: number
}

interface UploadingFile {
  id: string
  file: File
  progress: number
  status: 'uploading' | 'processing' | 'completed' | 'error'
  error?: string
}

interface DocumentFilters {
  status: string
  type: string
  project: string
  sortBy: 'filename' | 'created_at' | 'file_size' | 'processing_status'
  sortOrder: 'asc' | 'desc'
}

interface DocumentStats {
  total: number
  byStatus: Record<string, number>
  byType: Record<string, number>
  totalSize: number
}

export const DocumentLibrary: React.FC = () => {
  // State Management
  const [documents, setDocuments] = useState<ExtendedDocument[]>([])
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null)
  const [uploadingFiles, setUploadingFiles] = useState<UploadingFile[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking')
  
  // Filters & Search
  const [searchQuery, setSearchQuery] = useState('')
  const [filters, setFilters] = useState<DocumentFilters>({
    status: 'all',
    type: 'all', 
    project: 'all',
    sortBy: 'created_at',
    sortOrder: 'desc'
  })
  
  // UI State
  const [selectedDocuments, setSelectedDocuments] = useState<Set<string>>(new Set())
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [documentToDelete, setDocumentToDelete] = useState<ExtendedDocument | null>(null)
  const [stats, setStats] = useState<DocumentStats>({
    total: 0,
    byStatus: {},
    byType: {},
    totalSize: 0
  })

  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null)

  // ===== LIFECYCLE =====
  useEffect(() => {
    initializeLibrary()
  }, [])

  useEffect(() => {
    loadDocuments()
  }, [selectedProjectId])

  useEffect(() => {
    calculateStats()
  }, [documents])

  // ===== INITIALIZATION =====
  const initializeLibrary = async () => {
    console.log('🚀 Initializing document library...')
    setConnectionStatus('checking')
    
    try {
      const healthCheck = await ApiService.healthCheck()
      
      if (healthCheck.status === 'healthy') {
        setConnectionStatus('connected')
        console.log('✅ Backend connection established')
        
        await loadProjects()
        toast.success('Document library ready!', { duration: 2000 })
      } else {
        throw new Error('Backend unhealthy')
      }
    } catch (error) {
      setConnectionStatus('disconnected')
      console.error('💥 Library initialization failed:', error)
      toast.error('Failed to connect to backend. Please check the server.')
    }
  }

  const loadProjects = async () => {
    try {
      const response = await ApiService.getProjects()
      const projectsData = response.projects || []
      
      setProjects(projectsData)
      console.log(`📁 Loaded ${projectsData.length} projects`)
      
      // Auto-select first project if available
      if (projectsData.length > 0 && !selectedProjectId) {
        setSelectedProjectId(projectsData[0].id)
        console.log(`🎯 Auto-selected project: ${projectsData[0].name}`)
      }
    } catch (error) {
      console.error('Failed to load projects:', error)
      toast.error('Failed to load projects')
    }
  }

  const loadDocuments = async () => {
    setIsLoading(true)
    try {
      const response = await ApiService.getDocuments({
        project_id: selectedProjectId || undefined
      })
      
      const documentsData = response.documents || []
      setDocuments(documentsData)
      console.log(`📄 Loaded ${documentsData.length} documents`)
    } catch (error) {
      console.error('Failed to load documents:', error)
      toast.error('Failed to load documents')
    } finally {
      setIsLoading(false)
    }
  }

  // ===== FILE UPLOAD - KORRIGIERT =====
  const handleFileUpload = useCallback(async (files: FileList | File[]) => {
    if (!selectedProjectId) {
      toast.error('Please select a project first')
      return
    }

    // ✅ Finde den project_name aus der project_id für Legacy Endpoint
    const selectedProject = projects.find(p => p.id === selectedProjectId)
    if (!selectedProject) {
      toast.error('Selected project not found')
      return
    }

    const fileArray = Array.from(files)
    
    // Validate files first
    try {
      const validation = await ApiService.validateFiles(fileArray)
      
      if (validation.invalid_files > 0) {
        const errors = validation.results
          .filter(r => !r.valid)
          .map(r => `${r.filename}: ${r.errors.join(', ')}`)
        
        toast.error(`Invalid files:\n${errors.join('\n')}`)
        return
      }
    } catch (error) {
      console.warn('File validation failed, proceeding with upload:', error)
    }

    const newUploadingFiles: UploadingFile[] = fileArray.map(file => ({
      id: `upload-${Date.now()}-${Math.random()}`,
      file,
      progress: 0,
      status: 'uploading'
    }))

    setUploadingFiles(prev => [...prev, ...newUploadingFiles])

    // ✅ Upload einzeln mit LEGACY /api/upload Endpoint (das ist was das Backend tatsächlich hat)
    try {
      for (const file of fileArray) {
        const formData = new FormData()
        formData.append('file', file)  // Backend erwartet 'file' (singular)
        formData.append('project_name', selectedProject.name)  // ✅ project_name für Legacy Endpoint!
        
        console.log(`📤 Uploading file: ${file.name} to project: ${selectedProject.name}`)
        
        const response = await fetch('http://localhost:8000/api/upload', {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}))
          throw new Error(errorData.detail || `Upload failed: ${response.status}`)
        }

        const result = await response.json()
        console.log('✅ Upload successful:', result)
      }

      // Update upload status to completed
      setUploadingFiles(prev => prev.map(f => 
        newUploadingFiles.some(nf => nf.id === f.id)
          ? { ...f, status: 'completed', progress: 100 }
          : f
      ))

      toast.success(`Successfully uploaded ${fileArray.length} file(s)`)

      // Remove completed uploads after delay
      setTimeout(() => {
        setUploadingFiles(prev => prev.filter(f => 
          !newUploadingFiles.some(nf => nf.id === f.id)
        ))
      }, 2000)

      // Reload documents
      setTimeout(() => {
        loadDocuments()
      }, 1000)

    } catch (error) {
      console.error('Upload error:', error)
      
      // Mark uploads as failed
      setUploadingFiles(prev => prev.map(f => 
        newUploadingFiles.some(nf => nf.id === f.id)
          ? { ...f, status: 'error', error: error instanceof Error ? error.message : 'Upload failed' }
          : f
      ))

      toast.error('Upload failed. Please try again.')
    }
  }, [selectedProjectId, projects])

  // ===== DRAG & DROP =====
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    
    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileUpload(files)
    }
  }, [handleFileUpload])

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileUpload(files)
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }, [handleFileUpload])

  // ===== DOCUMENT MANAGEMENT =====
  const deleteDocument = useCallback(async (documentId: string) => {
    try {
      await ApiService.deleteDocument(documentId)
      setDocuments(prev => prev.filter(doc => doc.id !== documentId))
      toast.success('Document deleted successfully')
    } catch (error) {
      console.error('Delete error:', error)
      toast.error('Failed to delete document')
    }
  }, [])

  const handleDeleteClick = useCallback((document: ExtendedDocument) => {
    setDocumentToDelete(document)
    setShowDeleteDialog(true)
  }, [])

  const confirmDelete = useCallback(async () => {
    if (documentToDelete) {
      await deleteDocument(documentToDelete.id)
      setShowDeleteDialog(false)
      setDocumentToDelete(null)
    }
  }, [documentToDelete, deleteDocument])

  // Bulk delete
  const deleteManyDocuments = useCallback(async () => {
    const idsToDelete = Array.from(selectedDocuments)
    
    try {
      await Promise.all(idsToDelete.map(id => ApiService.deleteDocument(id)))
      
      setDocuments(prev => prev.filter(doc => !selectedDocuments.has(doc.id)))
      setSelectedDocuments(new Set())
      
      toast.success(`Deleted ${idsToDelete.length} document(s)`)
    } catch (error) {
      console.error('Bulk delete error:', error)
      toast.error('Failed to delete some documents')
    }
  }, [selectedDocuments])

  // ===== FILTERING & SEARCH =====
  const filteredDocuments = documents
    .filter(doc => {
      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase()
        if (!doc.filename.toLowerCase().includes(query)) {
          return false
        }
      }
      
      // Status filter
      if (filters.status !== 'all' && doc.processing_status !== filters.status) {
        return false
      }
      
      // Type filter
      if (filters.type !== 'all' && doc.file_type !== filters.type) {
        return false
      }
      
      // Project filter
      if (filters.project !== 'all' && !doc.project_ids.includes(filters.project)) {
        return false
      }
      
      return true
    })
    .sort((a, b) => {
      const { sortBy, sortOrder } = filters
      let aValue: any, bValue: any
      
      switch (sortBy) {
        case 'filename':
          aValue = a.filename.toLowerCase()
          bValue = b.filename.toLowerCase()
          break
        case 'created_at':
          aValue = new Date(a.created_at).getTime()
          bValue = new Date(b.created_at).getTime()
          break
        case 'file_size':
          aValue = a.file_size
          bValue = b.file_size
          break
        case 'processing_status':
          aValue = a.processing_status
          bValue = b.processing_status
          break
        default:
          return 0
      }
      
      if (aValue < bValue) return sortOrder === 'asc' ? -1 : 1
      if (aValue > bValue) return sortOrder === 'asc' ? 1 : -1
      return 0
    })

  // ===== STATS CALCULATION =====
  const calculateStats = useCallback(() => {
    const byStatus: Record<string, number> = {}
    const byType: Record<string, number> = {}
    let totalSize = 0

    documents.forEach(doc => {
      byStatus[doc.processing_status] = (byStatus[doc.processing_status] || 0) + 1
      byType[doc.file_type] = (byType[doc.file_type] || 0) + 1
      totalSize += doc.file_size
    })

    setStats({
      total: documents.length,
      byStatus,
      byType,
      totalSize
    })
  }, [documents])

  // ===== STATUS HELPERS =====
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleSolidIcon className="w-4 h-4 text-green-500" />
      case 'processing':
        return <ClockSolidIcon className="w-4 h-4 text-yellow-500" />
      case 'failed':
        return <ExclamationTriangleSolidIcon className="w-4 h-4 text-red-500" />
      default:
        return <ClockIcon className="w-4 h-4 text-gray-400" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800'
      case 'processing': return 'bg-yellow-100 text-yellow-800'
      case 'failed': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  // ===== RENDER =====
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              📚 Document Library
            </h1>
            <p className="text-gray-600">
              Manage and organize your documents for RAG processing
            </p>
          </div>
          
          {/* Connection Status */}
          <div className="flex items-center gap-4">
            <div className={cn(
              "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium",
              connectionStatus === 'connected' 
                ? "bg-green-100 text-green-700"
                : connectionStatus === 'disconnected'
                ? "bg-red-100 text-red-700" 
                : "bg-yellow-100 text-yellow-700"
            )}>
              <div className={cn(
                "w-2 h-2 rounded-full",
                connectionStatus === 'connected' ? "bg-green-500" :
                connectionStatus === 'disconnected' ? "bg-red-500" : "bg-yellow-500"
              )} />
              {connectionStatus === 'connected' ? 'Connected' :
               connectionStatus === 'disconnected' ? 'Disconnected' : 'Checking...'}
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Total Documents</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
                </div>
                <DocumentTextSolidIcon className="w-8 h-8 text-blue-500" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Completed</p>
                  <p className="text-2xl font-bold text-green-600">{stats.byStatus.completed || 0}</p>
                </div>
                <CheckCircleSolidIcon className="w-8 h-8 text-green-500" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Processing</p>
                  <p className="text-2xl font-bold text-yellow-600">{stats.byStatus.processing || 0}</p>
                </div>
                <ClockSolidIcon className="w-8 h-8 text-yellow-500" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Total Size</p>
                  <p className="text-2xl font-bold text-gray-900">{formatFileSize(stats.totalSize)}</p>
                </div>
                <FolderIcon className="w-8 h-8 text-purple-500" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Controls Bar */}
        <Card className="mb-8">
          <CardContent className="p-6">
            <div className="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
              <div className="flex flex-col sm:flex-row gap-4 flex-1">
                {/* Project Selector */}
                <div className="min-w-0 flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Project
                  </label>
                  <select
                    value={selectedProjectId || ''}
                    onChange={(e) => setSelectedProjectId(e.target.value || null)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">All Projects</option>
                    {projects.map((project) => (
                      <option key={project.id} value={project.id}>
                        {project.name} ({project.document_count} docs)
                      </option>
                    ))}
                  </select>
                </div>

                {/* Search */}
                <div className="min-w-0 flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Search
                  </label>
                  <div className="relative">
                    <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <Input
                      type="text"
                      placeholder="Search documents..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>

                {/* Filters */}
                <div className="flex gap-2">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="sm">
                        <FunnelIcon className="w-4 h-4 mr-2" />
                        Filters
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-56">
                      <DropdownMenuLabel>Filter by Status</DropdownMenuLabel>
                      <DropdownMenuItem onClick={() => setFilters(prev => ({ ...prev, status: 'all' }))}>
                        All Status
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setFilters(prev => ({ ...prev, status: 'completed' }))}>
                        Completed
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setFilters(prev => ({ ...prev, status: 'processing' }))}>
                        Processing
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setFilters(prev => ({ ...prev, status: 'failed' }))}>
                        Failed
                      </DropdownMenuItem>
                      
                      <DropdownMenuSeparator />
                      
                      <DropdownMenuLabel>Sort by</DropdownMenuLabel>
                      <DropdownMenuItem onClick={() => setFilters(prev => ({ ...prev, sortBy: 'created_at', sortOrder: 'desc' }))}>
                        Newest First
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setFilters(prev => ({ ...prev, sortBy: 'created_at', sortOrder: 'asc' }))}>
                        Oldest First
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setFilters(prev => ({ ...prev, sortBy: 'filename', sortOrder: 'asc' }))}>
                        Name A-Z
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setFilters(prev => ({ ...prev, sortBy: 'file_size', sortOrder: 'desc' }))}>
                        Largest First
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-2">
                {selectedDocuments.size > 0 && (
                  <Button 
                    variant="destructive" 
                    size="sm"
                    onClick={deleteManyDocuments}
                  >
                    <TrashIcon className="w-4 h-4 mr-2" />
                    Delete ({selectedDocuments.size})
                  </Button>
                )}
                
                <Button 
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-blue-600 hover:bg-blue-700"
                  disabled={!selectedProjectId || connectionStatus !== 'connected'}
                >
                  <CloudArrowUpIcon className="w-4 h-4 mr-2" />
                  Upload Documents
                </Button>
                
                <Button 
                  variant="outline" 
                  onClick={loadDocuments}
                  disabled={isLoading}
                >
                  <ArrowPathIcon className={cn("w-4 h-4 mr-2", isLoading && "animate-spin")} />
                  Refresh
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Upload Area */}
        <Card 
          className={cn(
            "mb-8 border-2 border-dashed transition-colors cursor-pointer",
            selectedProjectId && connectionStatus === 'connected'
              ? "border-gray-300 hover:border-blue-400"
              : "border-gray-200 cursor-not-allowed"
          )}
          onDragOver={selectedProjectId && connectionStatus === 'connected' ? handleDragOver : undefined}
          onDrop={selectedProjectId && connectionStatus === 'connected' ? handleDrop : undefined}
          onClick={selectedProjectId && connectionStatus === 'connected' ? () => fileInputRef.current?.click() : undefined}
        >
          <CardContent className="p-8">
            <div className="text-center">
              <CloudArrowUpIcon className={cn(
                "mx-auto w-12 h-12 mb-4",
                selectedProjectId && connectionStatus === 'connected' ? "text-gray-400" : "text-gray-300"
              )} />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                {selectedProjectId && connectionStatus === 'connected' 
                  ? 'Upload Documents'
                  : !selectedProjectId 
                    ? 'Select a Project First'
                    : 'Backend Disconnected'
                }
              </h3>
              <p className="text-gray-600 mb-4">
                {selectedProjectId && connectionStatus === 'connected'
                  ? 'Drag and drop files here or click to browse'
                  : !selectedProjectId
                    ? 'Choose a project to upload documents to'
                    : 'Please check your backend connection'
                }
              </p>
              <p className="text-sm text-gray-500">
                Supports PDF, DOCX, DOC, TXT, MD files up to 50MB
              </p>
            </div>
            
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.docx,.doc,.txt,.md"
              onChange={handleFileInputChange}
              className="hidden"
              disabled={!selectedProjectId || connectionStatus !== 'connected'}
            />
          </CardContent>
        </Card>

        {/* Uploading Files */}
        {uploadingFiles.length > 0 && (
          <Card className="mb-8">
            <CardContent className="p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Uploading Files
              </h3>
              <div className="space-y-3">
                {uploadingFiles.map((upload) => (
                  <div key={upload.id} className="flex items-center gap-4">
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium text-gray-700">
                          {upload.file.name}
                        </span>
                        <span className="text-sm text-gray-500">
                          {upload.status === 'completed' ? '✅ Complete' :
                           upload.status === 'error' ? '❌ Failed' :
                           upload.status === 'processing' ? '⚙️ Processing' :
                           `📤 Uploading...`}
                        </span>
                      </div>
                      {upload.status !== 'completed' && upload.status !== 'error' && (
                        <Progress value={upload.progress} className="h-2" />
                      )}
                      {upload.error && (
                        <p className="text-sm text-red-600 mt-1">{upload.error}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Documents Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          <AnimatePresence mode="popLayout">
            {filteredDocuments.map((document) => (
              <motion.div
                key={document.id}
                layout
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.2 }}
              >
                <Card className={cn(
                  "relative group hover:shadow-lg transition-all duration-200 cursor-pointer border-2",
                  selectedDocuments.has(document.id) 
                    ? "border-blue-500 bg-blue-50" 
                    : "border-transparent hover:border-gray-200"
                )}>
                  <CardContent className="p-6">
                    {/* Selection Checkbox */}
                    <div className="absolute top-4 left-4">
                      <input
                        type="checkbox"
                        checked={selectedDocuments.has(document.id)}
                        onChange={(e) => {
                          const newSelected = new Set(selectedDocuments)
                          if (e.target.checked) {
                            newSelected.add(document.id)
                          } else {
                            newSelected.delete(document.id)
                          }
                          setSelectedDocuments(newSelected)
                        }}
                        className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                      />
                    </div>

                    {/* Actions Menu */}
                    <div className="absolute top-4 right-4">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="sm" className="opacity-0 group-hover:opacity-100 transition-opacity">
                            <TrashIcon className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem>
                            <EyeIcon className="w-4 h-4 mr-2" />
                            View Details
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <ArrowDownTrayIcon className="w-4 h-4 mr-2" />
                            Download
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem 
                            className="text-red-600"
                            onClick={() => handleDeleteClick(document)}
                          >
                            <TrashIcon className="w-4 h-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>

                    {/* Document Icon */}
                    <div className="flex justify-center mb-4 mt-4">
                      <DocumentTextSolidIcon className="w-12 h-12 text-blue-500" />
                    </div>

                    {/* Document Info */}
                    <div className="text-center">
                      <h3 className="font-medium text-gray-900 mb-2 truncate" title={document.filename}>
                        {document.filename}
                      </h3>
                      
                      <div className="flex items-center justify-center gap-2 mb-2">
                        {getStatusIcon(document.processing_status)}
                        <Badge 
                          variant="secondary" 
                          className={cn("text-xs", getStatusColor(document.processing_status))}
                        >
                          {document.processing_status}
                        </Badge>
                      </div>

                      <div className="text-sm text-gray-600 space-y-1">
                        <div className="flex items-center justify-center gap-1">
                          <CalendarIcon className="w-3 h-3" />
                          {formatDate(document.created_at)}
                        </div>
                        <div>
                          {formatFileSize(document.file_size)} • {document.file_type}
                        </div>
                        {document.project_ids.length > 0 && (
                          <div className="flex items-center justify-center gap-1">
                            <FolderIcon className="w-3 h-3" />
                            {document.project_ids.length} project(s)
                          </div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

        {/* Empty State */}
        {filteredDocuments.length === 0 && !isLoading && (
          <Card>
            <CardContent className="p-12">
              <div className="text-center">
                <DocumentTextIcon className="mx-auto w-16 h-16 text-gray-400 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  No documents found
                </h3>
                <p className="text-gray-600 mb-6">
                  {searchQuery || filters.status !== 'all' || filters.type !== 'all' 
                    ? "Try adjusting your search or filters"
                    : "Upload your first document to get started"
                  }
                </p>
                {!searchQuery && filters.status === 'all' && filters.type === 'all' && selectedProjectId && connectionStatus === 'connected' && (
                  <Button 
                    onClick={() => fileInputRef.current?.click()}
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    <CloudArrowUpIcon className="w-4 h-4 mr-2" />
                    Upload Documents
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="flex justify-center py-12">
            <div className="flex items-center gap-2">
              <ArrowPathIcon className="w-5 h-5 animate-spin text-blue-600" />
              <span className="text-gray-600">Loading documents...</span>
            </div>
          </div>
        )}
      </div>

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Document</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete "{documentToDelete?.filename}"? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowDeleteDialog(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmDelete}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default DocumentLibrary