
'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { FolderIcon, PlusIcon } from '@heroicons/react/24/outline'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useStore } from '@/stores/useStore'

export function ProjectWorkspace() {
  const { projects, addProject, setCurrentProject } = useStore()

  const handleCreateProject = () => {
    const name = prompt('Project name:')
    if (name) {
      addProject({
        name,
        description: 'New project'
      })
    }
  }

  return (
    <div className="h-full bg-gray-50 dark:bg-gray-900">
      <div className="h-full overflow-y-auto">
        <div className="max-w-7xl mx-auto px-6 py-8">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Projects</h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">Manage your document analysis projects</p>
            </div>
            <Button onClick={handleCreateProject}>
              <PlusIcon className="h-4 w-4 mr-2" />
              New Project
            </Button>
          </div>

          {/* Projects Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projects.map((project, index) => (
              <motion.div
                key={project.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                <Card className="cursor-pointer hover:shadow-lg transition-shadow" onClick={() => setCurrentProject(project)}>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <FolderIcon className="h-5 w-5 mr-2" />
                      {project.name}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{project.description}</p>
                    <div className="mt-4 text-xs text-gray-500">
                      Created {new Intl.DateTimeFormat('en-US').format(project.createdAt)}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>

          {projects.length === 0 && (
            <div className="text-center py-12">
              <FolderIcon className="h-12 w-12 mx-auto mb-4 text-gray-400" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">No projects yet</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">Create your first project to get started</p>
              <Button onClick={handleCreateProject}>
                <PlusIcon className="h-4 w-4 mr-2" />
                Create Project
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
