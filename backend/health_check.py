#!/usr/bin/env python3
"""
RagFlow Backend Health Check Script
Automatisierte Überprüfung aller Backend-Komponenten und AI-Kommunikation
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import io

# HTTP Client
try:
    import httpx
except ImportError:
    print("❌ httpx nicht installiert. Installieren mit: pip install httpx")
    sys.exit(1)

# Rich für schöne Ausgaben
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.live import Live
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None
    def rprint(*args, **kwargs):
        print(*args, **kwargs)

class BackendHealthChecker:
    """Umfassender Backend Health Checker"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        self.results = {
            'connection': {},
            'configuration': {},
            'ai_communication': {},
            'features': {},
            'performance': {},
            'errors': []
        }
        self.start_time = datetime.now()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def log_result(self, category: str, test_name: str, status: bool, details: str = "", response_time: float = 0):
        """Log test result"""
        self.results[category][test_name] = {
            'status': status,
            'details': details,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if RICH_AVAILABLE:
            icon = "✅" if status else "❌"
            color = "green" if status else "red"
            console.print(f"{icon} [{color}]{test_name}[/{color}]: {details}")
        else:
            icon = "✓" if status else "✗"
            print(f"{icon} {test_name}: {details}")
    
    def log_error(self, error: str):
        """Log error"""
        self.results['errors'].append({
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        
        if RICH_AVAILABLE:
            console.print(f"[red]🚨 ERROR: {error}[/red]")
        else:
            print(f"ERROR: {error}")
    
    async def test_basic_connection(self):
        """Test grundlegende Backend-Erreichbarkeit"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]🔗 Testing Basic Connection[/bold blue]")
        else:
            print("\n🔗 Testing Basic Connection")
        
        # Root endpoint
        try:
            start_time = time.time()
            response = await self.client.get(f"{self.base_url}/")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                self.log_result('connection', 'Root Endpoint', True, 
                              f"Status: {response.status_code}", response_time)
            else:
                self.log_result('connection', 'Root Endpoint', False, 
                              f"Status: {response.status_code}")
        except Exception as e:
            self.log_result('connection', 'Root Endpoint', False, f"Error: {str(e)}")
            self.log_error(f"Cannot reach backend at {self.base_url}: {str(e)}")
            return False
        
        # Health endpoint
        try:
            start_time = time.time()
            response = await self.client.get(f"{self.base_url}/api/health")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                self.log_result('connection', 'Health Endpoint', True, 
                              f"Status: {health_data.get('status', 'unknown')}", response_time)
            else:
                self.log_result('connection', 'Health Endpoint', False, 
                              f"Status: {response.status_code}")
        except Exception as e:
            self.log_result('connection', 'Health Endpoint', False, f"Error: {str(e)}")
        
        # API Docs
        try:
            start_time = time.time()
            response = await self.client.get(f"{self.base_url}/docs")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                self.log_result('connection', 'API Documentation', True, 
                              "Available", response_time)
            else:
                self.log_result('connection', 'API Documentation', False, 
                              f"Status: {response.status_code}")
        except Exception as e:
            self.log_result('connection', 'API Documentation', False, f"Error: {str(e)}")
        
        return True
    
    async def test_configuration(self):
        """Test Backend-Konfiguration"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]⚙️  Testing Configuration[/bold blue]")
        else:
            print("\n⚙️ Testing Configuration")
        
        try:
            response = await self.client.get(f"{self.base_url}/api/config")
            
            if response.status_code == 200:
                config = response.json()
                
                # Google API Key Check
                has_google_key = config.get('google_api_configured', False)
                self.log_result('configuration', 'Google AI API Key', has_google_key,
                              "Configured" if has_google_key else "Not configured")
                
                # Upload Directory
                upload_dir = config.get('upload_dir', 'unknown')
                self.log_result('configuration', 'Upload Directory', True, upload_dir)
                
                # Data Directory
                data_dir = config.get('data_dir', 'unknown')
                self.log_result('configuration', 'Data Directory', True, data_dir)
                
                # RAG Settings
                chunk_size = config.get('chunk_size', 'unknown')
                self.log_result('configuration', 'RAG Chunk Size', True, f"{chunk_size} chars")
                
                # File Size Limit
                max_size = config.get('max_file_size', 0)
                size_mb = max_size / (1024 * 1024) if max_size else 0
                self.log_result('configuration', 'Max File Size', True, f"{size_mb:.1f} MB")
                
            else:
                self.log_result('configuration', 'Config Endpoint', False, 
                              f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_result('configuration', 'Config Access', False, f"Error: {str(e)}")
    
    async def test_ai_communication(self):
        """Test AI-Kommunikation"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]🤖 Testing AI Communication[/bold blue]")
        else:
            print("\n🤖 Testing AI Communication")
        
        # Test Chat ohne Kontext
        try:
            start_time = time.time()
            chat_payload = {
                "message": "Hello, this is a health check test message. Please respond briefly.",
                "project_id": None
            }
            
            response = await self.client.post(f"{self.base_url}/api/chat", json=chat_payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                chat_data = response.json()
                ai_response = chat_data.get('response', '')
                
                if ai_response and len(ai_response) > 10:
                    self.log_result('ai_communication', 'Basic Chat', True, 
                                  f"Response received ({len(ai_response)} chars)", response_time)
                else:
                    self.log_result('ai_communication', 'Basic Chat', False, 
                                  "Empty or too short response")
            else:
                self.log_result('ai_communication', 'Basic Chat', False, 
                              f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_result('ai_communication', 'Basic Chat', False, f"Error: {str(e)}")
        
        # Test AI Model Info
        try:
            response = await self.client.get(f"{self.base_url}/api/ai/info")
            
            if response.status_code == 200:
                ai_info = response.json()
                model_name = ai_info.get('model', 'unknown')
                self.log_result('ai_communication', 'AI Model Info', True, f"Model: {model_name}")
            else:
                self.log_result('ai_communication', 'AI Model Info', False, 
                              f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_result('ai_communication', 'AI Model Info', False, f"Error: {str(e)}")
    
    async def test_file_features(self):
        """Test File Upload und Processing Features"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]📁 Testing File Features[/bold blue]")
        else:
            print("\n📁 Testing File Features")
        
        # Test File Upload
        try:
            # Erstelle temporäre Test-Datei
            test_content = "This is a test document for RagFlow health check.\n\nIt contains some sample text to test document processing and RAG functionality."
            
            # Simulate file upload
            files = {
                'file': ('test_health_check.txt', io.BytesIO(test_content.encode()), 'text/plain')
            }
            data = {
                'project_name': 'health_check_project'
            }
            
            start_time = time.time()
            response = await self.client.post(f"{self.base_url}/api/upload", files=files, data=data)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                upload_result = response.json()
                project_id = upload_result.get('project_id')
                
                self.log_result('features', 'File Upload', True, 
                              f"Project ID: {project_id}", response_time)
                
                # Test RAG Search wenn Upload erfolgreich
                if project_id:
                    await self.test_rag_search(project_id)
                    
            else:
                self.log_result('features', 'File Upload', False, 
                              f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_result('features', 'File Upload', False, f"Error: {str(e)}")
        
        # Test Projects List
        try:
            response = await self.client.get(f"{self.base_url}/api/projects")
            
            if response.status_code == 200:
                projects = response.json()
                project_count = len(projects)
                self.log_result('features', 'Projects List', True, f"{project_count} projects found")
            else:
                self.log_result('features', 'Projects List', False, 
                              f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_result('features', 'Projects List', False, f"Error: {str(e)}")
    
    async def test_rag_search(self, project_id: str):
        """Test RAG Search Funktionalität"""
        try:
            search_payload = {
                "query": "test document",
                "project_id": project_id,
                "top_k": 3
            }
            
            start_time = time.time()
            response = await self.client.post(f"{self.base_url}/api/search", json=search_payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                search_results = response.json()
                results_count = len(search_results.get('results', []))
                
                self.log_result('features', 'RAG Search', True, 
                              f"{results_count} results found", response_time)
            else:
                self.log_result('features', 'RAG Search', False, 
                              f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_result('features', 'RAG Search', False, f"Error: {str(e)}")
    
    async def test_performance(self):
        """Test Performance Metrics"""
        if RICH_AVAILABLE:
            console.print("\n[bold blue]⚡ Testing Performance[/bold blue]")
        else:
            print("\n⚡ Testing Performance")
        
        # Multiple requests für Performance-Test
        response_times = []
        success_count = 0
        
        for i in range(5):
            try:
                start_time = time.time()
                response = await self.client.get(f"{self.base_url}/api/health")
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if response.status_code == 200:
                    success_count += 1
                    
            except Exception:
                pass
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            # Performance bewerten
            performance_good = avg_response_time < 1.0 and max_response_time < 2.0
            
            self.log_result('performance', 'Response Time', performance_good, 
                          f"Avg: {avg_response_time:.3f}s, Max: {max_response_time:.3f}s")
            
            self.log_result('performance', 'Reliability', success_count == 5, 
                          f"{success_count}/5 requests successful")
        else:
            self.log_result('performance', 'Performance Test', False, "No successful requests")
    
    def generate_report(self):
        """Generate comprehensive health report"""
        if RICH_AVAILABLE:
            console.print("\n[bold green]📊 Health Check Report[/bold green]")
            
            # Summary Table
            table = Table(title="Backend Health Check Summary")
            table.add_column("Category", style="cyan", no_wrap=True)
            table.add_column("Status", style="bold")
            table.add_column("Details", style="dim")
            
            for category, tests in self.results.items():
                if category == 'errors':
                    continue
                    
                if tests:
                    passed = sum(1 for test in tests.values() if test['status'])
                    total = len(tests)
                    
                    if passed == total:
                        status = "[green]✅ All Pass[/green]"
                    elif passed > 0:
                        status = f"[yellow]⚠️ {passed}/{total} Pass[/yellow]"
                    else:
                        status = "[red]❌ All Fail[/red]"
                    
                    table.add_row(category.title(), status, f"{passed}/{total} tests")
            
            console.print(table)
            
            # Errors
            if self.results['errors']:
                console.print("\n[bold red]🚨 Errors Found:[/bold red]")
                for error in self.results['errors']:
                    console.print(f"  • {error['error']}")
            
            # Recommendations
            console.print("\n[bold blue]💡 Recommendations:[/bold blue]")
            self._generate_recommendations()
            
        else:
            print("\n📊 Health Check Report")
            print("=" * 50)
            
            for category, tests in self.results.items():
                if category == 'errors':
                    continue
                    
                if tests:
                    passed = sum(1 for test in tests.values() if test['status'])
                    total = len(tests)
                    print(f"{category.title()}: {passed}/{total} tests passed")
            
            if self.results['errors']:
                print("\nErrors:")
                for error in self.results['errors']:
                    print(f"  • {error['error']}")
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check Google AI
        if not self.results['configuration'].get('Google AI API Key', {}).get('status'):
            recommendations.append("Set GOOGLE_API_KEY environment variable for AI functionality")
        
        # Check performance
        perf_tests = self.results.get('performance', {})
        if perf_tests.get('Response Time', {}).get('status') is False:
            recommendations.append("Backend response times are slow - check server resources")
        
        # Check basic connectivity
        conn_tests = self.results.get('connection', {})
        failed_conn = [name for name, result in conn_tests.items() if not result.get('status')]
        if failed_conn:
            recommendations.append(f"Fix connectivity issues with: {', '.join(failed_conn)}")
        
        # Check AI communication
        ai_tests = self.results.get('ai_communication', {})
        if not any(test.get('status') for test in ai_tests.values()):
            recommendations.append("AI communication failed - check Google AI API key and network")
        
        if recommendations:
            for rec in recommendations:
                if RICH_AVAILABLE:
                    console.print(f"  • [yellow]{rec}[/yellow]")
                else:
                    print(f"  • {rec}")
        else:
            if RICH_AVAILABLE:
                console.print("  • [green]All systems operational! 🎉[/green]")
            else:
                print("  • All systems operational! 🎉")
    
    async def run_full_check(self):
        """Run complete health check"""
        total_start = time.time()
        
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold]RagFlow Backend Health Check[/bold]\n"
                f"Target: {self.base_url}\n"
                f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                title="🏥 Health Check",
                border_style="blue"
            ))
        else:
            print("🏥 RagFlow Backend Health Check")
            print(f"Target: {self.base_url}")
            print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all tests
        try:
            await self.test_basic_connection()
            await self.test_configuration()
            await self.test_ai_communication()
            await self.test_file_features()
            await self.test_performance()
        except KeyboardInterrupt:
            if RICH_AVAILABLE:
                console.print("\n[yellow]Health check interrupted by user[/yellow]")
            else:
                print("\nHealth check interrupted by user")
            return
        
        total_time = time.time() - total_start
        
        if RICH_AVAILABLE:
            console.print(f"\n[dim]Total check time: {total_time:.2f} seconds[/dim]")
        else:
            print(f"\nTotal check time: {total_time:.2f} seconds")
        
        # Generate report
        self.generate_report()
        
        # Return overall status
        all_categories = ['connection', 'configuration', 'ai_communication', 'features']
        overall_success = True
        
        for category in all_categories:
            tests = self.results.get(category, {})
            if tests and not all(test.get('status') for test in tests.values()):
                overall_success = False
                break
        
        return overall_success

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RagFlow Backend Health Check")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Backend URL (default: http://localhost:8000)")
    parser.add_argument("--save-report", help="Save detailed report to file")
    
    args = parser.parse_args()
    
    async with BackendHealthChecker(args.url) as checker:
        success = await checker.run_full_check()
        
        # Save report if requested
        if args.save_report:
            with open(args.save_report, 'w') as f:
                json.dump(checker.results, f, indent=2)
            print(f"\nDetailed report saved to: {args.save_report}")
        
        # Exit code
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())