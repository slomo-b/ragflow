'use client'

import React, { useState } from 'react'
import { FolderOpen, Plus, Calendar, FileText, MessageCircle, MoreVertical, Trash2, Edit3, Settings } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { useStore } from '@/stores/useStore'
import { cn } from '@/lib/utils'

// Create Project Modal Component
const CreateProjectModal = ({ isOpen, onClose, onSubmit }: {
  isOpen: boolean
  onClose: () => void
  onSubmit: (data: { name: string; description: string }) => void
}) => {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) return
    
    onSubmit({ name: name.trim(), description: description.trim() })
    setName('')
    setDescription('')
    onClose()
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-md">
        <div className="p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Create New Project</h2>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Project Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter project name..."
                className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                autoFocus
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description (Optional)
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Brief description of your project..."
                rows={3}
                className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
              />
            </div>
            
            <div className="flex items-center justify-end space-x-3 pt-2">
              <Button
                type="button"
                variant="ghost"
                onClick={onClose}
                className="px-4 py-2"
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={!name.trim()}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700"
              >
                Create Project
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

// Edit Project Modal Component
const EditProjectModal = ({ isOpen, onClose, onSubmit, project }: {
  isOpen: boolean
  onClose: () => void
  onSubmit: (data: { name: string; description: string }) => void
  project: any
}) => {
  const [name, setName] = useState(project?.name || '')
  const [description, setDescription] = useState(project?.description || '')

  React.useEffect(() => {
    if (project) {
      setName(project.name || '')
      setDescription(project.description || '')
    }
  }, [project])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) return
    
    onSubmit({ name: name.trim(), description: description.trim() })
    onClose()
  }

  if (!isOpen || !project) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-md">
        <div className="p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Edit Project</h2>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Project Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter project name..."
                className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                autoFocus
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description (Optional)
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Brief description of your project..."
                rows={3}
                className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
              />
            </div>
            
            <div className="flex items-center justify-end space-x-3 pt-2">
              <Button
                type="button"
                variant="ghost"
                onClick={onClose}
                className="px-4 py-2"
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={!name.trim()}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700"
              >
                Update Project
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

// Project Card Component
const ProjectCard = ({ project, isActive, onClick }: {
  project: any
  isActive: boolean
  onClick: () => void
}) => {
  const [showMenu, setShowMenu] = useState(false)
  const [showEditModal, setShowEditModal] = useState(false)
  const { deleteProject, updateProject } = useStore()

  // Safe date formatting
  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString)
      if (isNaN(date.getTime())) {
        return 'Recently created'
      }
      return new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric'
      }).format(date)
    } catch {
      return 'Recently created'
    }
  }

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (confirm(`Are you sure you want to delete "${project.name}"?\n\nThis will also delete all associated documents and chats.`)) {
      deleteProject(project.id)
    }
    setShowMenu(false)
  }

  const handleEdit = (e: React.MouseEvent) => {
    e.stopPropagation()
    setShowEditModal(true)
    setShowMenu(false)
  }

  const handleEditSubmit = (data: { name: string; description: string }) => {
    updateProject(project.id, data)
  }

  return (
    <>
      <div
        onClick={onClick}
        className={cn(
          "group relative p-6 bg-white border rounded-2xl transition-all duration-200 cursor-pointer",
          isActive
            ? "border-blue-500 shadow-lg shadow-blue-500/10 bg-blue-50"
            : "border-gray-200 hover:border-gray-300 hover:shadow-md"
        )}
      >
        {/* Project Icon & Status */}
        <div className="flex items-start justify-between mb-4">
          <div className={cn(
            "w-12 h-12 rounded-xl flex items-center justify-center",
            isActive ? "bg-blue-600" : "bg-gray-100 group-hover:bg-gray-200"
          )}>
            <FolderOpen 
              size={24} 
              className={cn(
                "transition-colors",
                isActive ? "text-white" : "text-gray-600"
              )} 
            />
          </div>
          
          {/* Menu Button */}
          <div className="relative">
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation()
                setShowMenu(!showMenu)
              }}
              className="opacity-0 group-hover:opacity-100 transition-opacity p-2 h-8 w-8"
            >
              <MoreVertical size={16} />
            </Button>
            
            {/* Dropdown Menu */}
            {showMenu && (
              <>
                {/* Backdrop to close menu */}
                <div 
                  className="fixed inset-0 z-10" 
                  onClick={() => setShowMenu(false)}
                />
                <div className="absolute right-0 top-full mt-2 w-48 bg-white border border-gray-200 rounded-xl shadow-lg z-20">
                  <div className="py-2">
                    <button
                      onClick={handleEdit}
                      className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 flex items-center space-x-2"
                    >
                      <Edit3 size={16} />
                      <span>Edit Project</span>
                    </button>
                    <div className="border-t border-gray-100 my-1"></div>
                    <button
                      onClick={handleDelete}
                      className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center space-x-2"
                    >
                      <Trash2 size={16} />
                      <span>Delete Project</span>
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Project Info */}
        <div className="mb-4">
          <h3 className={cn(
            "text-lg font-semibold mb-2 line-clamp-2",
            isActive ? "text-blue-900" : "text-gray-900"
          )}>
            {project.name}
          </h3>
          
          {project.description && (
            <p className="text-sm text-gray-600 line-clamp-2 mb-3">
              {project.description}
            </p>
          )}
        </div>

        {/* Project Stats */}
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1 text-gray-500">
              <FileText size={16} />
              <span>{project.document_count || 0} docs</span>
            </div>
            <div className="flex items-center space-x-1 text-gray-500">
              <MessageCircle size={16} />
              <span>{project.chat_count || 0} chats</span>
            </div>
          </div>
          
          <div className="text-xs text-gray-400">
            {formatDate(project.createdAt)}
          </div>
        </div>

        {/* Active Project Badge */}
        {isActive && (
          <div className="absolute top-4 right-12">
            <Badge className="bg-green-100 text-green-700 text-xs">
              Active
            </Badge>
          </div>
        )}
      </div>

      {/* Edit Modal */}
      <EditProjectModal
        isOpen={showEditModal}
        onClose={() => setShowEditModal(false)}
        onSubmit={handleEditSubmit}
        project={project}
      />
    </>
  )
}

// Empty State Component
const EmptyProjectsState = ({ onCreateProject }: { onCreateProject: () => void }) => (
  <div className="flex flex-col items-center justify-center py-20 text-center">
    <div className="w-20 h-20 bg-gray-100 rounded-3xl flex items-center justify-center mb-6">
      <FolderOpen className="h-10 w-10 text-gray-400" />
    </div>
    
    <h3 className="text-xl font-semibold text-gray-900 mb-2">
      No projects yet
    </h3>
    <p className="text-gray-600 mb-8 max-w-md">
      Create your first project to organize your documents and start analyzing them with AI.
    </p>
    
    <Button 
      onClick={onCreateProject}
      className="bg-blue-600 hover:bg-blue-700"
    >
      <Plus size={16} className="mr-2" />
      Create Your First Project
    </Button>
  </div>
)

// Main Project Workspace Component
export function ProjectWorkspace() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  
  const { projects, currentProject, setCurrentProject, addProject } = useStore()

  // Filter projects based on search
  const filteredProjects = projects.filter(project =>
    project.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (project.description && project.description.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  const handleCreateProject = (data: { name: string; description: string }) => {
    addProject(data)
  }

  const handleProjectSelect = (project: any) => {
    setCurrentProject(project)
  }

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 bg-white border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
              <FolderOpen className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Projects</h1>
              <p className="text-sm text-gray-600">
                Organize your document analysis projects
              </p>
            </div>
          </div>
          
          <Button 
            onClick={() => setShowCreateModal(true)}
            className="bg-blue-600 hover:bg-blue-700"
          >
            <Plus size={16} className="mr-2" />
            New Project
          </Button>
        </div>
      </div>

      {/* Search & Filters */}
      {projects.length > 0 && (
        <div className="flex-shrink-0 px-6 py-4 bg-white border-b border-gray-100">
          <div className="flex items-center space-x-4">
            <div className="flex-1 relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search projects..."
                className="w-full px-4 py-2 pl-4 pr-4 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <span>{filteredProjects.length} of {projects.length} projects</span>
            </div>
          </div>
        </div>
      )}

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto">
        {projects.length === 0 ? (
          <EmptyProjectsState onCreateProject={() => setShowCreateModal(true)} />
        ) : (
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {filteredProjects.map((project) => (
                <ProjectCard
                  key={project.id}
                  project={project}
                  isActive={currentProject?.id === project.id}
                  onClick={() => handleProjectSelect(project)}
                />
              ))}
            </div>
            
            {filteredProjects.length === 0 && searchQuery && (
              <div className="text-center py-12">
                <p className="text-gray-500">No projects found matching "{searchQuery}"</p>
                <Button
                  variant="ghost"
                  onClick={() => setSearchQuery('')}
                  className="mt-2"
                >
                  Clear search
                </Button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Create Project Modal */}
      <CreateProjectModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onSubmit={handleCreateProject}
      />
    </div>
  )
}