'use client'

import React, { useState, useRef, useCallback } from 'react'
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
import { useStore } from '@/stores/useStore'
import { cn } from '@/lib/utils'
import toast from 'react-hot-toast'

export function DocumentLibrary() {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedStatus, setSelectedStatus] = useState('all')
  const [sortBy, setSortBy] = useState('uploadedAt')
  const [isUploading, setIsUploading] = useState(false)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [documentToDelete, setDocumentToDelete] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  const { documents, addDocument, deleteDocument, currentProject } = useStore()

  // Filtere Dokumente nach aktuellem Projekt
  const projectDocuments = React.useMemo(() => {
    if (!currentProject) return []
    return documents.filter(doc => doc.projectId === currentProject.id)
  }, [documents, currentProject])

  // Status-spezifische Konfiguration
  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'completed':
        return {
          icon: CheckCircleSolidIcon,
          color: 'text-emerald-500',
          bgColor: 'bg-emerald-50 dark:bg-emerald-900/20',
          borderColor: 'border-emerald-200 dark:border-emerald-800',
          label: 'Completed',
          description: 'Ready for analysis'
        }
      case 'processing':
        return {
          icon: ClockSolidIcon,
          color: 'text-blue-500',
          bgColor: 'bg-blue-50 dark:bg-blue-900/20',
          borderColor: 'border-blue-200 dark:border-blue-800',
          label: 'Processing',
          description: 'AI is analyzing...'
        }
      case 'failed':
        return {
          icon: ExclamationTriangleSolidIcon,
          color: 'text-red-500',
          bgColor: 'bg-red-50 dark:bg-red-900/20',
          borderColor: 'border-red-200 dark:border-red-800',
          label: 'Failed',
          description: 'Processing error'
        }
      default:
        return {
          icon: ClockIcon,
          color: 'text-gray-500',
          bgColor: 'bg-gray-50 dark:bg-gray-900/20',
          borderColor: 'border-gray-200 dark:border-gray-800',
          label: 'Uploaded',
          description: 'Waiting to process'
        }
    }
  }

  // Dateityp-Icons
  const getFileTypeIcon = (type: string, name: string) => {
    if (type.includes('pdf')) return '📄'
    if (type.includes('word') || type.includes('docx') || type.includes('document')) return '📝'
    if (type.includes('text') || name.endsWith('.txt') || name.endsWith('.md')) return '📄'
    if (type.includes('image')) return '🖼️'
    return '📄'
  }

  // Formatiere Dateigröße
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
  }

  // Datei-Upload Handler
  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return

    if (!currentProject) {
      toast.error('Please select a project first!')
      return
    }

    setIsUploading(true)
    
    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        
        // Füge Dokument zum Store hinzu
        addDocument({
          name: file.name,
          type: file.type,
          size: file.size,
          projectId: currentProject.id
        })
      }
      
      toast.success(`${files.length} document(s) uploaded successfully!`)
    } catch (error) {
      console.error('Upload failed:', error)
      toast.error('Upload failed. Please try again.')
    } finally {
      setIsUploading(false)
    }
  }

  // Drag & Drop Handler
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const files = e.dataTransfer.files
    handleFileUpload(files)
  }, [currentProject])

  // Dokument löschen
  const handleDeleteDocument = (documentId: string) => {
    setDocumentToDelete(documentId)
    setDeleteDialogOpen(true)
  }

  const confirmDelete = async () => {
    if (!documentToDelete) return

    try {
      deleteDocument(documentToDelete)
      toast.success('Document deleted successfully')
    } catch (error) {
      console.error('Delete failed:', error)
      toast.error('Failed to delete document')
    } finally {
      setDeleteDialogOpen(false)
      setDocumentToDelete(null)
    }
  }

  // Filter und Such-Logik für Projekt-Dokumente
  const filteredDocuments = projectDocuments
    .filter(doc => {
      const matchesSearch = doc.name.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesStatus = selectedStatus === 'all' || doc.status === selectedStatus
      return matchesSearch && matchesStatus
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name)
        case 'size':
          return b.size - a.size
        case 'uploadedAt':
        default:
          return new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime()
      }
    })

  // Statistiken für aktuelles Projekt
  const stats = {
    total: projectDocuments.length,
    completed: projectDocuments.filter(doc => doc.status === 'completed').length,
    processing: projectDocuments.filter(doc => doc.status === 'processing').length,
    failed: projectDocuments.filter(doc => doc.status === 'failed').length,
    totalSize: projectDocuments.reduce((sum, doc) => sum + doc.size, 0)
  }

  const documentToDeleteInfo = projectDocuments.find(doc => doc.id === documentToDelete)

  return (
    <div 
      className={cn(
        "h-full transition-colors duration-300",
        isDragOver 
          ? "bg-violet-50/50 dark:bg-violet-900/20" 
          : "bg-slate-50/50 dark:bg-slate-900/30"
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="h-full overflow-y-auto">
        <div className="max-w-7xl mx-auto px-8 py-8">
          
          {/* Modern Header */}
          <div className="mb-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="flex items-center justify-between"
            >
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-br from-violet-500 to-blue-500 rounded-2xl flex items-center justify-center shadow-lg">
                  <DocumentTextIcon className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
                    {currentProject ? `${currentProject.name} Documents` : 'Document Library'}
                  </h1>
                  <p className="text-slate-600 dark:text-slate-400 mt-1">
                    {currentProject 
                      ? `Manage documents in ${currentProject.name}`
                      : 'Please select a project to view documents'
                    }
                  </p>
                </div>
              </div>

              {/* Upload Button - nur wenn Projekt ausgewählt */}
              {currentProject && (
                <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                  <Button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                    className="bg-gradient-to-r from-violet-500 to-blue-500 hover:from-violet-600 hover:to-blue-600 text-white px-6 py-3 rounded-2xl shadow-lg"
                  >
                    {isUploading ? (
                      <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
                    ) : (
                      <CloudArrowUpIcon className="h-5 w-5 mr-2" />
                    )}
                    {isUploading ? 'Uploading...' : 'Upload Documents'}
                  </Button>
                </motion.div>
              )}
            </motion.div>

            {/* Projekt-spezifische Statistiken */}
            {currentProject && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
                className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-8"
              >
                {[
                  { 
                    label: 'Total Documents', 
                    value: stats.total, 
                    icon: DocumentTextIcon, 
                    color: 'from-blue-500 to-cyan-500',
                    description: `in ${currentProject.name}`
                  },
                  { 
                    label: 'Processed', 
                    value: stats.completed, 
                    icon: CheckCircleIcon, 
                    color: 'from-emerald-500 to-teal-500',
                    description: 'ready for AI analysis'
                  },
                  { 
                    label: 'Processing', 
                    value: stats.processing, 
                    icon: ClockIcon, 
                    color: 'from-amber-500 to-orange-500',
                    description: 'being analyzed'
                  },
                  { 
                    label: 'Storage Used', 
                    value: formatFileSize(stats.totalSize), 
                    icon: SparklesIcon, 
                    color: 'from-purple-500 to-pink-500',
                    description: 'in this project'
                  }
                ].map((stat, index) => (
                  <Card key={stat.label} className="bg-white/70 dark:bg-slate-800/70 backdrop-blur-xl border-slate-200/50 dark:border-slate-700/50 hover:shadow-lg transition-all duration-200">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">{stat.label}</p>
                          <p className="text-2xl font-bold text-slate-900 dark:text-white">{stat.value}</p>
                          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">{stat.description}</p>
                        </div>
                        <div className={`w-12 h-12 bg-gradient-to-br ${stat.color} rounded-2xl flex items-center justify-center shadow-lg`}>
                          <stat.icon className="h-6 w-6 text-white" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </motion.div>
            )}
          </div>

          {/* Kein Projekt ausgewählt */}
          {!currentProject && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="flex flex-col items-center justify-center min-h-[500px] text-center"
            >
              <div className="w-20 h-20 bg-gradient-to-br from-slate-400 to-slate-600 rounded-3xl flex items-center justify-center mb-6 mx-auto shadow-2xl">
                <FolderIcon className="h-10 w-10 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">
                No Project Selected
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-6 max-w-lg mx-auto">
                Please select a project from the sidebar to view and manage documents.
              </p>
            </motion.div>
          )}

          {/* Projekt ausgewählt - zeige Dokumente */}
          {currentProject && (
            <>
              {/* Search and Filters */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="mb-8"
              >
                <Card className="bg-white/70 dark:bg-slate-800/70 backdrop-blur-xl border-slate-200/50 dark:border-slate-700/50">
                  <CardContent className="p-6">
                    <div className="flex flex-col lg:flex-row gap-4 items-center">
                      {/* Search */}
                      <div className="relative flex-1">
                        <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-slate-400" />
                        <Input
                          placeholder="Search documents in this project..."
                          value={searchTerm}
                          onChange={(e) => setSearchTerm(e.target.value)}
                          className="pl-10 bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 rounded-xl"
                        />
                      </div>

                      {/* Filters */}
                      <div className="flex gap-3">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="outline" className="rounded-xl border-slate-200 dark:border-slate-700">
                              <FunnelIcon className="h-4 w-4 mr-2" />
                              Status: {selectedStatus === 'all' ? 'All' : selectedStatus}
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent>
                            <DropdownMenuLabel>Filter by Status</DropdownMenuLabel>
                            <DropdownMenuItem onClick={() => setSelectedStatus('all')}>
                              All Documents
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => setSelectedStatus('completed')}>
                              Completed
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => setSelectedStatus('processing')}>
                              Processing
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => setSelectedStatus('failed')}>
                              Failed
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>

                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="outline" className="rounded-xl border-slate-200 dark:border-slate-700">
                              Sort: {sortBy === 'uploadedAt' ? 'Date' : sortBy === 'name' ? 'Name' : 'Size'}
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent>
                            <DropdownMenuItem onClick={() => setSortBy('uploadedAt')}>
                              Upload Date
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => setSortBy('name')}>
                              Name
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => setSortBy('size')}>
                              File Size
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Document Grid/List */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
              >
                {filteredDocuments.length === 0 ? (
                  // Empty State
                  <div className={cn(
                    "border-2 border-dashed rounded-3xl p-16 text-center bg-white/50 dark:bg-slate-800/50 backdrop-blur-xl transition-all duration-300",
                    isDragOver 
                      ? "border-violet-400 bg-violet-50/50 dark:bg-violet-900/20" 
                      : "border-slate-300 dark:border-slate-600"
                  )}>
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 300, delay: 0.5 }}
                      className="mb-6"
                    >
                      <div className="w-20 h-20 bg-gradient-to-br from-violet-500 to-blue-500 rounded-3xl flex items-center justify-center mx-auto shadow-2xl">
                        <CloudArrowUpIcon className="h-10 w-10 text-white" />
                      </div>
                    </motion.div>
                    
                    <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
                      {searchTerm ? 'No documents found' : `No documents in ${currentProject.name}`}
                    </h3>
                    <p className="text-slate-600 dark:text-slate-400 mb-6 max-w-lg mx-auto">
                      {searchTerm 
                        ? `No documents match your search "${searchTerm}". Try adjusting your filters.`
                        : `Drag and drop files here or click the upload button to add documents to ${currentProject.name}.`
                      }
                    </p>
                    
                    {!searchTerm && (
                      <Button
                        onClick={() => fileInputRef.current?.click()}
                        className="bg-gradient-to-r from-violet-500 to-blue-500 hover:from-violet-600 hover:to-blue-600 text-white px-8 py-3 rounded-2xl shadow-lg"
                      >
                        <PlusIcon className="h-5 w-5 mr-2" />
                        Upload Your First Document
                      </Button>
                    )}
                  </div>
                ) : (
                  // Document Grid
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredDocuments.map((doc, index) => {
                      const statusConfig = getStatusConfig(doc.status || 'uploaded')
                      
                      return (
                        <motion.div
                          key={doc.id}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.3, delay: index * 0.1 }}
                          whileHover={{ y: -4, scale: 1.02 }}
                          className="group"
                        >
                          <Card className="h-full bg-white/70 dark:bg-slate-800/70 backdrop-blur-xl border-slate-200/50 dark:border-slate-700/50 hover:shadow-xl transition-all duration-300 overflow-hidden">
                            <CardHeader className="pb-4">
                              <div className="flex items-start justify-between">
                                {/* File Icon & Status */}
                                <div className="flex items-center space-x-3 flex-1 min-w-0">
                                  <div className="text-3xl flex-shrink-0">{getFileTypeIcon(doc.type, doc.name)}</div>
                                  <div className="flex-1 min-w-0">
                                    <CardTitle className="text-lg font-semibold text-slate-900 dark:text-white truncate">
                                      {doc.name}
                                    </CardTitle>
                                    <div className="flex items-center space-x-2 mt-1">
                                      <statusConfig.icon className={`h-4 w-4 ${statusConfig.color}`} />
                                      <span className={`text-sm font-medium ${statusConfig.color}`}>
                                        {statusConfig.label}
                                      </span>
                                    </div>
                                  </div>
                                </div>

                                {/* Action Menu */}
                                <DropdownMenu>
                                  <DropdownMenuTrigger asChild>
                                    <Button 
                                      variant="ghost" 
                                      size="sm" 
                                      className="opacity-0 group-hover:opacity-100 transition-opacity p-2 flex-shrink-0"
                                    >
                                      <span className="sr-only">Actions</span>
                                      <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                                        <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z" />
                                      </svg>
                                    </Button>
                                  </DropdownMenuTrigger>
                                  <DropdownMenuContent>
                                    <DropdownMenuItem>
                                      <EyeIcon className="h-4 w-4 mr-2" />
                                      View
                                    </DropdownMenuItem>
                                    <DropdownMenuItem>
                                      <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
                                      Download
                                    </DropdownMenuItem>
                                    <DropdownMenuSeparator />
                                    <DropdownMenuItem 
                                      className="text-red-600"
                                      onClick={() => handleDeleteDocument(doc.id)}
                                    >
                                      <TrashIcon className="h-4 w-4 mr-2" />
                                      Delete
                                    </DropdownMenuItem>
                                  </DropdownMenuContent>
                                </DropdownMenu>
                              </div>
                            </CardHeader>

                            <CardContent className="pt-0">
                              {/* Document Stats */}
                              <div className="grid grid-cols-2 gap-4 mb-4">
                                <div className="text-center p-3 bg-slate-50 dark:bg-slate-700/50 rounded-xl">
                                  <div className="text-lg font-semibold text-slate-900 dark:text-white">
                                    {formatFileSize(doc.size)}
                                  </div>
                                  <div className="text-xs text-slate-500 dark:text-slate-400">Size</div>
                                </div>
                                <div className="text-center p-3 bg-slate-50 dark:bg-slate-700/50 rounded-xl">
                                  <div className="text-lg font-semibold text-slate-900 dark:text-white">
                                    {doc.type.split('/')[1]?.toUpperCase() || 'DOC'}
                                  </div>
                                  <div className="text-xs text-slate-500 dark:text-slate-400">Type</div>
                                </div>
                              </div>

                              {/* Upload Date */}
                              <div className="flex items-center text-xs text-slate-500 dark:text-slate-400">
                                <CalendarIcon className="h-3 w-3 mr-1" />
                                {new Date(doc.uploadedAt).toLocaleDateString()}
                              </div>
                            </CardContent>
                          </Card>
                        </motion.div>
                      )
                    })}
                  </div>
                )}
              </motion.div>
            </>
          )}
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-red-600">
              <TrashIcon className="h-5 w-5" />
              Delete Document
            </DialogTitle>
            <DialogDescription>
              Are you sure you want to delete "{documentToDeleteInfo?.name}"? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setDeleteDialogOpen(false)}
              className="rounded-xl"
            >
              Cancel
            </Button>
            <Button 
              variant="destructive" 
              onClick={confirmDelete}
              className="rounded-xl"
            >
              <TrashIcon className="h-4 w-4 mr-2" />
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.doc,.docx,.txt,.md"
        onChange={(e) => handleFileUpload(e.target.files)}
        className="hidden"
      />
    </div>
  )
}