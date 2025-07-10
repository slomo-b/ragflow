// === components/documents/DocumentLibrary.tsx ===
'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { DocumentTextIcon, CloudArrowUpIcon } from '@heroicons/react/24/outline'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useStore } from '@/stores/useStore'

export function DocumentLibrary() {
  const { documents, addDocument } = useStore()

  const handleFileUpload = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = true
    input.accept = '.pdf,.doc,.docx,.txt'
    input.onchange = (e) => {
      const files = (e.target as HTMLInputElement).files
      if (files) {
        Array.from(files).forEach(file => {
          addDocument({
            name: file.name,
            type: file.type,
            size: file.size,
            projectId: 'default'
          })
        })
      }
    }
    input.click()
  }

  return (
    <div className="h-full bg-gray-50 dark:bg-gray-900">
      <div className="h-full overflow-y-auto">
        <div className="max-w-7xl mx-auto px-6 py-8">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Documents</h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">Upload and manage your documents</p>
            </div>
            <Button onClick={handleFileUpload}>
              <CloudArrowUpIcon className="h-4 w-4 mr-2" />
              Upload Documents
            </Button>
          </div>

          {/* Documents Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {documents.map((doc, index) => (
              <motion.div
                key={doc.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <DocumentTextIcon className="h-5 w-5 mr-2" />
                      {doc.name}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      Size: {(doc.size / 1024).toFixed(1)} KB
                    </div>
                    <div className="mt-2 text-xs text-gray-500">
                      Uploaded {new Intl.DateTimeFormat('en-US').format(doc.uploadedAt)}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>

          {documents.length === 0 && (
            <div className="text-center py-12">
              <DocumentTextIcon className="h-12 w-12 mx-auto mb-4 text-gray-400" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">No documents yet</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">Upload your first document to get started</p>
              <Button onClick={handleFileUpload}>
                <CloudArrowUpIcon className="h-4 w-4 mr-2" />
                Upload Documents
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}