// frontend/components/projects/ProjectWorkspace.tsx - Vollständig funktionsfähig für ChromaDB
'use client'

import React, { useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  PlusIcon,
  FolderIcon,
  DocumentTextIcon,
  ChatBubbleLeftRightIcon,
  EllipsisVerticalIcon,
  PencilIcon,
  TrashIcon,
  CalendarIcon,
  SparklesIcon,
  ArrowRightIcon,
  ExclamationTriangleIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
  CloudArrowUpIcon,
  EyeIcon,
  CpuChipIcon,
  BoltIcon
} from '@heroicons/react/24/outline'
import {
  FolderIcon as FolderSolidIcon,
  CheckCircleIcon as CheckCircleSolidIcon,
  CpuChipIcon as CpuChipSolidIcon
} from '@heroicons/react/24/solid'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { Input } from "@/components/ui/Input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/Badge"
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger,
  DropdownMenuSeparator
} from "@/components/ui/dropdown-menu"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { cn, formatDate } from '@/lib/utils'
import ApiService, { type Project as ApiProject, type SystemInfo } from '@/services/api'
import toast from 'react-hot-toast'

// Create a simple Database icon
const DatabaseIcon = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2 4.5 4 8 4s8-2 8-4V7M4 7c0 2 4.5 4 8 4s8-2 8-4M4 7c0-2 4.5-4 8-4s8 2 8 4" />
  </svg>
)

// Extended Project interface matching backend exactly
interface Project extends ApiProject {
  last_activity?: string
  status?: 'active' | 'archived' | 'draft'
  processing_info?: {
    total_chunks?: number
    ai_ready?: boolean
    embedding_status?: string
  }
}

interface CreateProjectData {
  name: string
  description?: string
}

interface UpdateProjectData {
  name?: string
  description?: string
}

const ProjectWorkspace: React.FC = () => {
  // State Management
  const [projects, setProjects] = useState<Project[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedProject, setSelectedProject] = useState<Project | null>(null)
  const [systemStats, setSystemStats] = useState<SystemInfo | null>(null)
  
  // Dialog states
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false)
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false)
  const [isProjectDetailOpen, setIsProjectDetailOpen] = useState(false)
  const [projectToDelete, setProjectToDelete] = useState<Project | null>(null)
  
  // Form states
  const [createForm, setCreateForm] = useState<CreateProjectData>({ name: '', description: '' })
  const [editForm, setEditForm] = useState<UpdateProjectData>({ name: '', description: '' })
  const [isSubmitting, setIsSubmitting] = useState(false)

  // Load projects and system stats
  const loadProjects = useCallback(async () => {
    try {
      setIsLoading(true)
      console.log('🔄 Loading projects from ChromaDB backend...')
      
      // Load projects - backend returns array directly
      const projectsResponse = await ApiService.getProjects()
      console.log('✅ Projects response:', projectsResponse)
      
      // Enhance projects with status and processing info
      const enhancedProjects: Project[] = projectsResponse.map(project => ({
        ...project,
        last_activity: project.created_at,
        status: project.document_count > 0 
          ? (project.chat_count > 0 ? 'active' : 'draft') 
          : 'draft',
        processing_info: {
          ai_ready: project.document_count > 0,
          embedding_status: project.document_count > 0 ? 'ready' : 'pending'
        }
      }))
      
      setProjects(enhancedProjects)
      
      // Load system stats
      try {
        const statsResponse = await ApiService.getSystemInfo()
        setSystemStats(statsResponse)
        console.log('✅ System stats loaded:', statsResponse)
      } catch (error) {
        console.warn('⚠️ Could not load system stats:', error)
      }
      
      toast.success(`Loaded ${enhancedProjects.length} projects from ChromaDB`)
    } catch (error) {
      console.error('❌ Failed to load projects:', error)
      toast.error('Failed to load projects. Please check backend connection.')
      setProjects([])
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Initialize component
  useEffect(() => {
    loadProjects()
  }, [loadProjects])

  // Create new project with exact backend format
  const handleCreateProject = useCallback(async () => {
    if (!createForm.name.trim()) {
      toast.error('Project name is required')
      return
    }

    if (createForm.name.length > 100) {
      toast.error('Project name must be 100 characters or less')
      return
    }

    if (createForm.description && createForm.description.length > 500) {
      toast.error('Project description must be 500 characters or less')
      return
    }

    try {
      setIsSubmitting(true)
      console.log('🚀 Creating project in ChromaDB:', createForm)
      
      // Prepare data exactly as backend expects
      const projectData = {
        name: createForm.name.trim(),
        description: createForm.description?.trim() || ''
      }
      
      console.log('📤 Sending project data:', projectData)
      
      const newProject = await ApiService.createProject(projectData)
      
      console.log('✅ Project created successfully:', newProject)
      
      // Add to projects list with enhanced stats
      const enhancedProject: Project = {
        ...newProject,
        last_activity: newProject.created_at,
        status: 'draft',
        processing_info: {
          ai_ready: false,
          embedding_status: 'pending',
          total_chunks: 0
        }
      }
      
      setProjects(prev => [enhancedProject, ...prev])
      
      // Reset form and close dialog
      setCreateForm({ name: '', description: '' })
      setIsCreateDialogOpen(false)
      
      toast.success(`Project "${newProject.name}" created successfully in ChromaDB!`)
      
    } catch (error: any) {
      console.error('❌ Failed to create project:', error)
      
      // Show specific error message
      let errorMessage = 'Failed to create project. Please try again.'
      
      if (error.response?.data?.detail) {
        errorMessage = `Error: ${error.response.data.detail}`
      } else if (error.message) {
        errorMessage = `Error: ${error.message}`
      }
      
      toast.error(errorMessage)
    } finally {
      setIsSubmitting(false)
    }
  }, [createForm])

  // Edit project with backend validation
  const handleEditProject = useCallback(async () => {
    if (!selectedProject) return

    if (!editForm.name?.trim()) {
      toast.error('Project name is required')
      return
    }

    if (editForm.name && editForm.name.length > 100) {
      toast.error('Project name must be 100 characters or less')
      return
    }

    if (editForm.description && editForm.description.length > 500) {
      toast.error('Project description must be 500 characters or less')
      return
    }

    try {
      setIsSubmitting(true)
      console.log('🔄 Updating project in ChromaDB:', selectedProject.id, editForm)
      
      const updateData = {
        name: editForm.name?.trim() || selectedProject.name,
        description: editForm.description?.trim() || selectedProject.description
      }
      
      console.log('📤 Sending update data:', updateData)
      
      const updatedProject = await ApiService.updateProject(selectedProject.id, updateData)
      console.log('✅ Project updated via ChromaDB:', updatedProject)
      
      // Update local state
      setProjects(prev => prev.map(p => 
        p.id === selectedProject.id ? {
          ...p,
          ...updatedProject,
          last_activity: new Date().toISOString(),
          processing_info: p.processing_info
        } : p
      ))
      
      // Reset states
      setEditForm({ name: '', description: '' })
      setSelectedProject(null)
      setIsEditDialogOpen(false)
      
      toast.success(`Project "${updateData.name}" updated successfully!`)
      
    } catch (error: any) {
      console.error('❌ Failed to update project:', error)
      
      let errorMessage = 'Failed to update project. Please try again.'
      
      if (error.response?.data?.detail) {
        errorMessage = `Error: ${error.response.data.detail}`
      } else if (error.message) {
        errorMessage = `Error: ${error.message}`
      }
      
      toast.error(errorMessage)
      
      // Reload projects to ensure consistency
      loadProjects()
    } finally {
      setIsSubmitting(false)
    }
  }, [selectedProject, editForm, loadProjects])

  // Delete project with backend confirmation
  const handleDeleteProject = useCallback(async () => {
    if (!projectToDelete) return

    try {
      setIsSubmitting(true)
      console.log('🗑️ Deleting project from ChromaDB:', projectToDelete.id)
      
      const deleteResponse = await ApiService.deleteProject(projectToDelete.id)
      console.log('✅ Project deleted from ChromaDB:', deleteResponse)
      
      // Remove from local state
      setProjects(prev => prev.filter(p => p.id !== projectToDelete.id))
      
      // Reset states
      setProjectToDelete(null)
      setIsDeleteDialogOpen(false)
      
      toast.success(`Project "${projectToDelete.name}" permanently deleted from ChromaDB!`)
      
      // Show deletion details if provided
      if (deleteResponse.details) {
        const details = deleteResponse.details
        setTimeout(() => {
          toast.success(`Cleanup: ${details.documents_affected} documents, ${details.chats_deleted} chats removed`)
        }, 1000)
      }
      
    } catch (error: any) {
      console.error('❌ Failed to delete project:', error)
      
      let errorMessage = 'Failed to delete project. Please try again.'
      
      if (error.response?.data?.detail) {
        errorMessage = `Error: ${error.response.data.detail}`
      } else if (error.message) {
        errorMessage = `Error: ${error.message}`
      }
      
      toast.error(errorMessage)
      
      // Reload projects to ensure consistency
      loadProjects()
    } finally {
      setIsSubmitting(false)
    }
  }, [projectToDelete, loadProjects])

  // View project details with backend data
  const handleViewProject = useCallback(async (project: Project) => {
    try {
      console.log('👁️ Loading detailed project info:', project.id)
      
      const detailedProject = await ApiService.getProject(project.id)
      console.log('✅ Project details loaded:', detailedProject)
      
      setSelectedProject({
        ...project,
        processing_info: {
          ...project.processing_info,
          total_chunks: detailedProject.statistics?.document_chunks || 0,
          ai_ready: (detailedProject.statistics?.document_count || 0) > 0
        }
      })
      setIsProjectDetailOpen(true)
      
    } catch (error) {
      console.error('❌ Failed to load project details:', error)
      toast.error('Failed to load project details')
    }
  }, [])

  // Open edit dialog
  const openEditDialog = useCallback((project: Project) => {
    setSelectedProject(project)
    setEditForm({
      name: project.name,
      description: project.description
    })
    setIsEditDialogOpen(true)
  }, [])

  // Open delete dialog
  const openDeleteDialog = useCallback((project: Project) => {
    setProjectToDelete(project)
    setIsDeleteDialogOpen(true)
  }, [])

  // Filter projects based on search
  const filteredProjects = projects.filter(project =>
    project.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    project.description.toLowerCase().includes(searchQuery.toLowerCase())
  )

  // Get status color for ChromaDB status
  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'active':
        return 'bg-emerald-100 text-emerald-800 border-emerald-200'
      case 'archived':
        return 'bg-gray-100 text-gray-800 border-gray-200'
      case 'draft':
        return 'bg-blue-100 text-blue-800 border-blue-200'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  // Get status icon
  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'active':
        return <CheckCircleSolidIcon className="w-3 h-3" />
      case 'draft':
        return <PencilIcon className="w-3 h-3" />
      default:
        return <FolderSolidIcon className="w-3 h-3" />
    }
  }

  // Get AI readiness indicator
  const getAIReadinessIndicator = (project: Project) => {
    const isReady = project.processing_info?.ai_ready
    const hasDocuments = project.document_count > 0
    
    if (isReady && hasDocuments) {
      return (
        <div className="flex items-center gap-1 text-xs text-emerald-600">
          <BoltIcon className="w-3 h-3" />
          <span>AI Ready</span>
        </div>
      )
    } else if (hasDocuments) {
      return (
        <div className="flex items-center gap-1 text-xs text-amber-600">
          <CpuChipIcon className="w-3 h-3" />
          <span>Processing</span>
        </div>
      )
    } else {
      return (
        <div className="flex items-center gap-1 text-xs text-gray-500">
          <CloudArrowUpIcon className="w-3 h-3" />
          <span>Upload docs</span>
        </div>
      )
    }
  }

  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-gray-50 to-white">
      {/* Enhanced Header with ChromaDB Info */}
      <div className="flex-none p-6 border-b border-gray-200 bg-white/70 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-purple-500 to-blue-600 rounded-lg text-white">
              <DatabaseIcon className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Projects</h1>
              <div className="flex items-center gap-2 mt-1">
                <p className="text-sm text-gray-600">
                  Powered by ChromaDB vector database
                </p>
                {systemStats?.database.stats && (
                  <Badge variant="outline" className="text-xs bg-emerald-50 text-emerald-700 border-emerald-200">
                    <DatabaseIcon className="w-3 h-3 mr-1" />
                    Connected
                  </Badge>
                )}
              </div>
            </div>
          </div>
          
          <Button
            onClick={() => setIsCreateDialogOpen(true)}
            className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white shadow-lg hover:shadow-xl transition-all duration-200"
          >
            <PlusIcon className="w-4 h-4 mr-2" />
            New Project
          </Button>
        </div>

        {/* Enhanced Search and Stats */}
        <div className="flex items-center justify-between mt-6">
          <div className="relative flex-1 max-w-md">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <Input
              placeholder="Search projects..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 pr-10 bg-white border-gray-200 focus:border-purple-300 focus:ring-purple-100"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                <XMarkIcon className="w-4 h-4" />
              </button>
            )}
          </div>
          
          <div className="flex items-center gap-6 text-sm text-gray-600">
            <div className="flex items-center gap-2">
              <FolderIcon className="w-4 h-4" />
              <span>{projects.length} Projects</span>
            </div>
            <div className="flex items-center gap-2">
              <DocumentTextIcon className="w-4 h-4" />
              <span>{systemStats?.database.stats.documents?.total || 0} Documents</span>
            </div>
            <div className="flex items-center gap-2">
              <CpuChipSolidIcon className="w-4 h-4" />
              <span>{systemStats?.database.stats.documents?.chunks_total || 0} Vectors</span>
            </div>
            <div className="flex items-center gap-2">
              <SparklesIcon className="w-4 h-4" />
              <span>{projects.filter(p => p.status === 'active').length} Active</span>
            </div>
          </div>
        </div>

        {/* ChromaDB Status Indicator */}
        {systemStats && (
          <div className="mt-4 flex items-center justify-between bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-3">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-sm text-blue-700">
                <DatabaseIcon className="w-4 h-4" />
                <span className="font-medium">ChromaDB Status:</span>
                <Badge variant="outline" className="bg-emerald-50 text-emerald-700 border-emerald-200">
                  Healthy
                </Badge>
              </div>
            </div>
            <div className="flex items-center gap-4 text-xs text-gray-600">
              <span>AI Models: {systemStats.capabilities.ai_providers.length || 0}</span>
              <span>Document Formats: {systemStats.capabilities.document_formats.length || 0}</span>
              <span>OCR Engines: {systemStats.capabilities.ocr_engines.length || 0}</span>
            </div>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="flex items-center gap-3 text-gray-600">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
              <span>Loading projects from ChromaDB...</span>
            </div>
          </div>
        ) : filteredProjects.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-center">
            {searchQuery ? (
              <>
                <MagnifyingGlassIcon className="w-16 h-16 text-gray-300 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No projects found</h3>
                <p className="text-gray-600 mb-4">
                  No projects match "{searchQuery}". Try a different search term.
                </p>
                <Button
                  variant="outline"
                  onClick={() => setSearchQuery('')}
                  className="border-gray-300 text-gray-600 hover:bg-gray-50"
                >
                  Clear search
                </Button>
              </>
            ) : (
              <>
                <DatabaseIcon className="w-16 h-16 text-gray-300 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No projects yet</h3>
                <p className="text-gray-600 mb-4">
                  Create your first project to start using ChromaDB for document analysis and AI conversations.
                </p>
                <Button
                  onClick={() => setIsCreateDialogOpen(true)}
                  className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white"
                >
                  <PlusIcon className="w-4 h-4 mr-2" />
                  Create Project
                </Button>
              </>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <AnimatePresence mode="popLayout">
              {filteredProjects.map((project) => (
                <motion.div
                  key={project.id}
                  layout
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.2 }}
                >
                  <Card className="group hover:shadow-lg transition-all duration-200 border-gray-200 hover:border-purple-200 bg-white">
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3 flex-1 min-w-0">
                          <div className="p-2 bg-gradient-to-br from-purple-100 to-blue-100 rounded-lg group-hover:from-purple-200 group-hover:to-blue-200 transition-colors">
                            <FolderSolidIcon className="w-5 h-5 text-purple-600" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <CardTitle className="text-lg font-semibold text-gray-900 truncate group-hover:text-purple-700 transition-colors">
                              {project.name}
                            </CardTitle>
                            {project.description && (
                              <p className="text-sm text-gray-600 mt-1 line-clamp-2">
                                {project.description}
                              </p>
                            )}
                          </div>
                        </div>
                        
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="opacity-0 group-hover:opacity-100 transition-opacity h-8 w-8 p-0"
                            >
                              <EllipsisVerticalIcon className="w-4 h-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="w-48">
                            <DropdownMenuItem onClick={() => handleViewProject(project)}>
                              <EyeIcon className="w-4 h-4 mr-2" />
                              View Details
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => openEditDialog(project)}>
                              <PencilIcon className="w-4 h-4 mr-2" />
                              Edit Project
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem 
                              onClick={() => openDeleteDialog(project)}
                              className="text-red-600 focus:text-red-600"
                            >
                              <TrashIcon className="w-4 h-4 mr-2" />
                              Delete Project
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </CardHeader>

                    <CardContent className="pt-0">
                      {/* Enhanced Status and Stats */}
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                          <Badge 
                            variant="outline" 
                            className={cn(
                              "text-xs font-medium",
                              getStatusColor(project.status)
                            )}
                          >
                            {getStatusIcon(project.status)}
                            <span className="ml-1 capitalize">{project.status || 'draft'}</span>
                          </Badge>
                          {getAIReadinessIndicator(project)}
                        </div>
                      </div>

                      {/* ChromaDB Stats */}
                      <div className="grid grid-cols-3 gap-3 mb-4 text-xs">
                        <div className="text-center">
                          <div className="flex items-center justify-center gap-1 text-gray-500 mb-1">
                            <DocumentTextIcon className="w-3 h-3" />
                          </div>
                          <div className="font-semibold text-gray-900">{project.document_count}</div>
                          <div className="text-gray-500">Docs</div>
                        </div>
                        <div className="text-center">
                          <div className="flex items-center justify-center gap-1 text-gray-500 mb-1">
                            <CpuChipIcon className="w-3 h-3" />
                          </div>
                          <div className="font-semibold text-gray-900">
                            {project.processing_info?.total_chunks || '0'}
                          </div>
                          <div className="text-gray-500">Vectors</div>
                        </div>
                        <div className="text-center">
                          <div className="flex items-center justify-center gap-1 text-gray-500 mb-1">
                            <ChatBubbleLeftRightIcon className="w-3 h-3" />
                          </div>
                          <div className="font-semibold text-gray-900">{project.chat_count}</div>
                          <div className="text-gray-500">Chats</div>
                        </div>
                      </div>

                      {/* Timestamps */}
                      <div className="flex items-center justify-between text-xs text-gray-500 mb-4">
                        <div className="flex items-center gap-1">
                          <CalendarIcon className="w-3 h-3" />
                          <span>Created {formatDate(project.created_at)}</span>
                        </div>
                        {project.last_activity && (
                          <span>Updated {formatDate(project.last_activity)}</span>
                        )}
                      </div>

                      {/* Action Button */}
                      <Button
                        onClick={() => handleViewProject(project)}
                        variant="outline"
                        size="sm"
                        className="w-full group/btn border-gray-200 hover:border-purple-300 hover:bg-purple-50 text-gray-700 hover:text-purple-700"
                      >
                        <span>Open Project</span>
                        <ArrowRightIcon className="w-3 h-3 ml-2 group-hover/btn:translate-x-0.5 transition-transform" />
                      </Button>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Create Project Dialog - Fixed for backend */}
      <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <PlusIcon className="w-5 h-5 text-purple-600" />
              Create New Project
            </DialogTitle>
            <DialogDescription>
              Create a new project in ChromaDB to organize your documents and AI conversations with vector embeddings.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="create-name">Project Name *</Label>
              <Input
                id="create-name"
                placeholder="Enter project name..."
                value={createForm.name}
                onChange={(e) => setCreateForm(prev => ({ ...prev, name: e.target.value }))}
                className="focus:border-purple-300 focus:ring-purple-100"
                maxLength={100}
                disabled={isSubmitting}
              />
              <p className="text-xs text-gray-500">
                {createForm.name.length}/100 characters
              </p>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="create-description">Description</Label>
              <Textarea
                id="create-description"
                placeholder="Describe your project..."
                value={createForm.description}
                onChange={(e) => setCreateForm(prev => ({ ...prev, description: e.target.value }))}
                className="resize-none focus:border-purple-300 focus:ring-purple-100"
                rows={3}
                maxLength={500}
                disabled={isSubmitting}
              />
              <p className="text-xs text-gray-500">
                {createForm.description?.length || 0}/500 characters
              </p>
            </div>

            {/* ChromaDB Info */}
            <div className="bg-blue-50 rounded-lg p-3">
              <div className="flex items-center gap-2 text-sm text-blue-700 mb-1">
                <DatabaseIcon className="w-4 h-4" />
                <span className="font-medium">ChromaDB Features</span>
              </div>
              <p className="text-xs text-blue-600">
                Your project will support vector embeddings, semantic search, and AI-powered document analysis.
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsCreateDialogOpen(false)
                setCreateForm({ name: '', description: '' })
              }}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreateProject}
              disabled={!createForm.name.trim() || isSubmitting}
              className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
            >
              {isSubmitting ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin mr-2" />
                  Creating...
                </>
              ) : (
                <>
                  <PlusIcon className="w-4 h-4 mr-2" />
                  Create Project
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Project Dialog */}
      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <PencilIcon className="w-5 h-5 text-purple-600" />
              Edit Project
            </DialogTitle>
            <DialogDescription>
              Update your project information in ChromaDB.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="edit-name">Project Name *</Label>
              <Input
                id="edit-name"
                placeholder="Enter project name..."
                value={editForm.name || ''}
                onChange={(e) => setEditForm(prev => ({ ...prev, name: e.target.value }))}
                className="focus:border-purple-300 focus:ring-purple-100"
                maxLength={100}
                disabled={isSubmitting}
              />
              <p className="text-xs text-gray-500">
                {editForm.name?.length || 0}/100 characters
              </p>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="edit-description">Description</Label>
              <Textarea
                id="edit-description"
                placeholder="Describe your project..."
                value={editForm.description || ''}
                onChange={(e) => setEditForm(prev => ({ ...prev, description: e.target.value }))}
                className="resize-none focus:border-purple-300 focus:ring-purple-100"
                rows={3}
                maxLength={500}
                disabled={isSubmitting}
              />
              <p className="text-xs text-gray-500">
                {editForm.description?.length || 0}/500 characters
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsEditDialogOpen(false)
                setEditForm({ name: '', description: '' })
                setSelectedProject(null)
              }}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button
              onClick={handleEditProject}
              disabled={!editForm.name?.trim() || isSubmitting}
              className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
            >
              {isSubmitting ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin mr-2" />
                  Saving...
                </>
              ) : (
                <>
                  <CheckCircleSolidIcon className="w-4 h-4 mr-2" />
                  Save Changes
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <DialogContent className="sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-red-700">
              <ExclamationTriangleIcon className="w-5 h-5" />
              Delete Project
            </DialogTitle>
            <DialogDescription>
              Are you sure you want to delete "{projectToDelete?.name}"? 
              This will permanently remove the project and all associated data from ChromaDB.
            </DialogDescription>
          </DialogHeader>

          {/* Deletion Impact Warning */}
          {projectToDelete && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3">
              <div className="text-sm text-red-800">
                <p className="font-medium mb-2">This action will delete:</p>
                <ul className="space-y-1 text-xs">
                  <li>• {projectToDelete.document_count} documents and their vector embeddings</li>
                  <li>• {projectToDelete.chat_count} chat conversations</li>
                  <li>• {projectToDelete.processing_info?.total_chunks || 0} vector chunks from ChromaDB</li>
                  <li>• All project metadata and associations</li>
                </ul>
                <p className="font-medium mt-2 text-red-700">This action cannot be undone!</p>
              </div>
            </div>
          )}

          <DialogFooter className="mt-6">
            <Button
              variant="outline"
              onClick={() => {
                setIsDeleteDialogOpen(false)
                setProjectToDelete(null)
              }}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button
              onClick={handleDeleteProject}
              disabled={isSubmitting}
              className="bg-red-600 hover:bg-red-700 text-white"
            >
              {isSubmitting ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin mr-2" />
                  Deleting...
                </>
              ) : (
                <>
                  <TrashIcon className="w-4 h-4 mr-2" />
                  Delete Permanently
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Project Detail Dialog */}
      <Dialog open={isProjectDetailOpen} onOpenChange={setIsProjectDetailOpen}>
        <DialogContent className="sm:max-w-[600px] max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <EyeIcon className="w-5 h-5 text-purple-600" />
              Project Details
            </DialogTitle>
            <DialogDescription>
              Detailed information about your ChromaDB project
            </DialogDescription>
          </DialogHeader>
          
          {selectedProject && (
            <div className="space-y-6 py-4">
              {/* Basic Info */}
              <div className="space-y-3">
                <h4 className="font-medium text-gray-900">Basic Information</h4>
                <div className="bg-gray-50 rounded-lg p-4 space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Name:</span>
                    <span className="text-sm font-medium">{selectedProject.name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Description:</span>
                    <span className="text-sm">{selectedProject.description || 'No description'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Status:</span>
                    <Badge className={getStatusColor(selectedProject.status)}>
                      {selectedProject.status || 'draft'}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Created:</span>
                    <span className="text-sm">{formatDate(selectedProject.created_at)}</span>
                  </div>
                </div>
              </div>

              {/* ChromaDB Statistics */}
              <div className="space-y-3">
                <h4 className="font-medium text-gray-900 flex items-center gap-2">
                  <DatabaseIcon className="w-4 h-4" />
                  ChromaDB Statistics
                </h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-blue-600">{selectedProject.document_count}</div>
                    <div className="text-sm text-blue-700">Documents</div>
                    <div className="text-xs text-blue-500 mt-1">Indexed in ChromaDB</div>
                  </div>
                  <div className="bg-purple-50 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      {selectedProject.processing_info?.total_chunks || 0}
                    </div>
                    <div className="text-sm text-purple-700">Vector Chunks</div>
                    <div className="text-xs text-purple-500 mt-1">Embedding vectors</div>
                  </div>
                  <div className="bg-green-50 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-green-600">{selectedProject.chat_count}</div>
                    <div className="text-sm text-green-700">Conversations</div>
                    <div className="text-xs text-green-500 mt-1">AI chat sessions</div>
                  </div>
                  <div className="bg-amber-50 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-amber-600">
                      {selectedProject.processing_info?.ai_ready ? 'Ready' : 'Pending'}
                    </div>
                    <div className="text-sm text-amber-700">AI Status</div>
                    <div className="text-xs text-amber-500 mt-1">Vector search ready</div>
                  </div>
                </div>
              </div>

              {/* Processing Information */}
              <div className="space-y-3">
                <h4 className="font-medium text-gray-900 flex items-center gap-2">
                  <CpuChipIcon className="w-4 h-4" />
                  Processing Information
                </h4>
                <div className="bg-gray-50 rounded-lg p-4 space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Embedding Status:</span>
                    <span className="text-sm font-medium">
                      {selectedProject.processing_info?.embedding_status || 'unknown'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">AI Ready:</span>
                    <Badge variant={selectedProject.processing_info?.ai_ready ? "default" : "secondary"}>
                      {selectedProject.processing_info?.ai_ready ? 'Yes' : 'No'}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Vector Search:</span>
                    <Badge variant={selectedProject.document_count > 0 ? "default" : "secondary"}>
                      {selectedProject.document_count > 0 ? 'Available' : 'No documents'}
                    </Badge>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="space-y-3">
                <h4 className="font-medium text-gray-900">Quick Actions</h4>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    onClick={() => openEditDialog(selectedProject)}
                    className="flex-1"
                  >
                    <PencilIcon className="w-4 h-4 mr-2" />
                    Edit Project
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      setIsProjectDetailOpen(false)
                      // Navigate to project documents view
                      // This would typically route to a documents page
                    }}
                    className="flex-1"
                  >
                    <DocumentTextIcon className="w-4 h-4 mr-2" />
                    View Documents
                  </Button>
                </div>
              </div>
            </div>
          )}

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsProjectDetailOpen(false)
                setSelectedProject(null)
              }}
            >
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default ProjectWorkspace