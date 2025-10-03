"""
Output collector and organizer - Enhanced with project-based timestamped storage and AI Council support
"""
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import hashlib
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class CollectedFile:
    """Represents a collected/processed file"""
    original_path: Path
    output_path: Path
    language: str
    size: int
    processing_time: float
    model_used: str
    success: bool
    error: Optional[str] = None
    content_hash: Optional[str] = None
    is_council_response: bool = False
    council_member: Optional[str] = None

class Collector:
    """Manages output collection and organization with project-based storage and AI Council support"""
    
    def __init__(self, output_dir: Path, project_name: str, organize_by: str = 'language'):
        # Create timestamped project directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_project_name = self._sanitize_project_name(project_name)
        
        self.output_dir = Path(output_dir) / f"{safe_project_name}_{timestamp}"
        self.project_name = project_name
        self.organize_by = organize_by
        self.timestamp = timestamp
        
        self.manifest = {
            'project': {
                'name': project_name,
                'safe_name': safe_project_name,
                'timestamp': timestamp,
                'created_at': datetime.now().isoformat()
            },
            'harvest_config': {
                'organize_by': organize_by,
                'output_dir': str(self.output_dir)
            },
            'files': [],
            'council_sessions': [],  # Track council processing sessions
            'statistics': {
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'council_files': 0,
                'synthesized_files': 0,
                'by_language': {},
                'by_model': {},
                'total_size_bytes': 0,
                'total_processing_time': 0
            }
        }
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self._setup_directories()
        
        # Save initial manifest
        self.save_manifest()
    
    def _sanitize_project_name(self, name: str) -> str:
        """Sanitize project name for filesystem"""
        # Remove path separators and invalid characters
        safe_name = name.replace('/', '_').replace('\\', '_')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '-_')
        return safe_name[:50]  # Limit length
    
    def _setup_directories(self):
        """Setup output directory structure"""
        # Create base directories
        (self.output_dir / 'processed').mkdir(exist_ok=True)
        (self.output_dir / 'errors').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # Create council-specific directories
        (self.output_dir / 'consensus').mkdir(exist_ok=True)
        (self.output_dir / 'council_responses').mkdir(exist_ok=True)
        (self.output_dir / 'synthesized').mkdir(exist_ok=True)
        
        if self.organize_by == 'language':
            # Create language-specific directories
            common_languages = [
                'python', 'javascript', 'typescript', 'java', 
                'go', 'rust', 'cpp', 'csharp', 'ruby', 'php',
                'swift', 'kotlin', 'scala', 'r', 'julia'
            ]
            for lang in common_languages:
                (self.output_dir / 'processed' / lang).mkdir(exist_ok=True)
            
            (self.output_dir / 'processed' / 'misc').mkdir(exist_ok=True)
            
        elif self.organize_by == 'preserve':
            # Will preserve original directory structure
            pass
    
    def collect(
        self,
        original_path: Path,
        processed_content: str,
        metadata: Dict[str, Any],
        is_council_run: bool = False  # NEW PARAMETER
    ) -> CollectedFile:
        """
        Collect and save processed file with enhanced metadata
        """
        # Check if this is a council run
        if is_council_run or metadata.get('processing_mode') == 'ai_council':
            # For council runs, we handle things differently
            return self._collect_council_response(original_path, processed_content, metadata)
        
        # Regular collection logic
        try:
            # Calculate content hash
            content_hash = hashlib.sha256(processed_content.encode()).hexdigest()[:16]
            
            # Determine output path
            output_path = self._determine_output_path(original_path, metadata)
            
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write processed content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            # Save original metadata
            metadata_path = self.output_dir / 'metadata' / f"{output_path.stem}_meta.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metadata_path, 'w') as f:
                json.dump({
                    'original_path': str(original_path),
                    'output_path': str(output_path),
                    'content_hash': content_hash,
                    'processing_metadata': metadata,
                    'collected_at': datetime.now().isoformat(),
                    'council_mode': False
                }, f, indent=2)
            
            # Create collected file record
            collected = CollectedFile(
                original_path=original_path,
                output_path=output_path,
                language=metadata.get('language', 'unknown'),
                size=len(processed_content),
                processing_time=metadata.get('processing_time', 0),
                model_used=metadata.get('model', 'unknown'),
                success=True,
                content_hash=content_hash,
                is_council_response=False
            )
            
            # Update manifest
            self._update_manifest(collected)
            
            logger.info(f"Collected: {original_path} -> {output_path}")
            return collected
            
        except Exception as e:
            logger.error(f"Error collecting {original_path}: {e}")
            return self._handle_collection_error(original_path, metadata, e)

    def _collect_council_response(
        self,
        original_path: Path,
        processed_content: str,
        metadata: Dict[str, Any]
    ) -> CollectedFile:
        """
        Special handling for AI Council responses
        """
        try:
            # Create consensus subdirectory for this file
            consensus_dir = self.output_dir / 'consensus' / original_path.stem
            consensus_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine output path based on metadata
            if 'synthesizer_model' in metadata:
                # This is the final synthesized result
                output_path = self.output_dir / 'synthesized' / self._determine_output_path(original_path, metadata).name
                is_synthesis = True
            else:
                # This is an individual council member response
                model_name = metadata.get('model', metadata.get('council_member', 'unknown'))
                safe_model_name = self._sanitize_filename(model_name)
                output_path = consensus_dir / f"{safe_model_name}_response.md"
                is_synthesis = False
            
            # Calculate content hash
            content_hash = hashlib.sha256(processed_content.encode()).hexdigest()[:16]
            
            # Write content
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            # Create metadata record
            metadata_path = self.output_dir / 'metadata' / f"{output_path.stem}_meta.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metadata_path, 'w') as f:
                json.dump({
                    'original_path': str(original_path),
                    'output_path': str(output_path),
                    'content_hash': content_hash,
                    'processing_metadata': metadata,
                    'collected_at': datetime.now().isoformat(),
                    'council_mode': True,
                    'is_synthesis': is_synthesis,
                    'council_member': metadata.get('council_member'),
                    'synthesizer_model': metadata.get('synthesizer_model')
                }, f, indent=2)
            
            # Create collected file record
            collected = CollectedFile(
                original_path=original_path,
                output_path=output_path,
                language=metadata.get('language', 'unknown'),
                size=len(processed_content),
                processing_time=metadata.get('processing_time', 0),
                model_used=metadata.get('synthesizer_model', metadata.get('model', 'council')),
                success=True,
                content_hash=content_hash,
                is_council_response=True,
                council_member=metadata.get('council_member')
            )
            
            # Update manifest with council-specific tracking
            self._update_manifest(collected, is_council=True, is_synthesis=is_synthesis)
            
            # Track council session if this is a new one
            self._track_council_session(original_path, metadata)
            
            logger.info(f"Collected council response: {original_path.name} -> {output_path.name}")
            return collected
            
        except Exception as e:
            logger.error(f"Error collecting council response for {original_path}: {e}")
            return self._handle_collection_error(original_path, metadata, e, is_council=True)

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem"""
        # Remove invalid characters and replace with underscores
        safe_name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in filename)
        return safe_name[:50]  # Limit length

    def _track_council_session(self, original_path: Path, metadata: Dict[str, Any]):
        """Track council processing sessions"""
        session_id = metadata.get('original_job_id', str(original_path))
        
        # Check if this session already exists
        existing_session = None
        for session in self.manifest['council_sessions']:
            if session['session_id'] == session_id:
                existing_session = session
                break
        
        if existing_session is None:
            # Create new session
            new_session = {
                'session_id': session_id,
                'original_file': str(original_path),
                'started_at': datetime.now().isoformat(),
                'council_members': [],
                'synthesizer_model': None,
                'status': 'in_progress'
            }
            self.manifest['council_sessions'].append(new_session)
            existing_session = new_session
        
        # Update session with member info
        if 'council_member' in metadata:
            member_info = {
                'model': metadata['council_member'],
                'completed_at': datetime.now().isoformat(),
                'attempts': metadata.get('attempts', 1)
            }
            if member_info not in existing_session['council_members']:
                existing_session['council_members'].append(member_info)
        
        # Update synthesizer info
        if 'synthesizer_model' in metadata:
            existing_session['synthesizer_model'] = metadata['synthesizer_model']
            existing_session['status'] = 'completed'
            existing_session['completed_at'] = datetime.now().isoformat()

    def _handle_collection_error(
        self, 
        original_path: Path, 
        metadata: Dict[str, Any], 
        error: Exception,
        is_council: bool = False
    ) -> CollectedFile:
        """Handle collection errors"""
        # Save error information
        error_dir = self.output_dir / 'errors'
        if is_council:
            error_dir = error_dir / 'council'
        error_dir.mkdir(parents=True, exist_ok=True)
        
        error_path = error_dir / f"{original_path.stem}_error.json"
        
        with open(error_path, 'w') as f:
            json.dump({
                'original_path': str(original_path),
                'error': str(error),
                'metadata': metadata,
                'timestamp': datetime.now().isoformat(),
                'is_council': is_council
            }, f, indent=2)
        
        # Create error record
        collected = CollectedFile(
            original_path=original_path,
            output_path=error_path,
            language=metadata.get('language', 'unknown'),
            size=0,
            processing_time=metadata.get('processing_time', 0),
            model_used=metadata.get('model', 'unknown'),
            success=False,
            error=str(error),
            is_council_response=is_council,
            council_member=metadata.get('council_member')
        )
        
        self._update_manifest(collected, is_council=is_council)
        return collected
    
    def _determine_output_path(self, original_path: Path, metadata: Dict) -> Path:
        """Determine where to save the processed file"""
        if self.organize_by == 'language':
            language = metadata.get('language', 'misc')
            lang_dir = self.output_dir / 'processed' / language
            
            if not lang_dir.exists():
                lang_dir = self.output_dir / 'processed' / 'misc'
            
            # Include part of original path in filename for uniqueness
            original_parts = original_path.parts[-2:]  # Last 2 parts of path
            safe_name = '_'.join(original_parts).replace('/', '_')
            
            return lang_dir / safe_name
            
        elif self.organize_by == 'flat':
            return self.output_dir / 'processed' / original_path.name
            
        elif self.organize_by == 'preserve':
            # Preserve relative path structure
            relative_path = metadata.get('relative_path', original_path.name)
            return self.output_dir / 'processed' / relative_path
        
        else:
            return self.output_dir / 'processed' / original_path.name
    
    def _update_manifest(self, collected: CollectedFile, is_council: bool = False, is_synthesis: bool = False):
        """Update manifest with collected file info"""
        # Add to files list
        file_record = {
            'original_path': str(collected.original_path),
            'output_path': str(collected.output_path),
            'language': collected.language,
            'size': collected.size,
            'processing_time': collected.processing_time,
            'model_used': collected.model_used,
            'success': collected.success,
            'error': collected.error,
            'content_hash': collected.content_hash,
            'collected_at': datetime.now().isoformat(),
            'is_council_response': collected.is_council_response,
            'council_member': collected.council_member
        }
        
        self.manifest['files'].append(file_record)
        
        # Update statistics
        stats = self.manifest['statistics']
        stats['total_files'] += 1
        stats['total_size_bytes'] += collected.size
        stats['total_processing_time'] += collected.processing_time
        
        if collected.success:
            stats['successful'] += 1
        else:
            stats['failed'] += 1
        
        # Council-specific stats
        if is_council:
            stats['council_files'] += 1
        if is_synthesis:
            stats['synthesized_files'] += 1
        
        # Update language statistics
        if collected.language not in stats['by_language']:
            stats['by_language'][collected.language] = 0
        stats['by_language'][collected.language] += 1
        
        # Update model statistics
        if collected.model_used not in stats['by_model']:
            stats['by_model'][collected.model_used] = {
                'count': 0,
                'successful': 0,
                'failed': 0,
                'council_responses': 0,
                'syntheses': 0
            }
        
        model_stats = stats['by_model'][collected.model_used]
        model_stats['count'] += 1
        if collected.success:
            model_stats['successful'] += 1
        else:
            model_stats['failed'] += 1
        
        if is_council:
            model_stats['council_responses'] += 1
        if is_synthesis:
            model_stats['syntheses'] += 1
        
        # Save manifest after each update
        self.save_manifest()
    
    def save_manifest(self):
        """Save collection manifest"""
        manifest_path = self.output_dir / 'manifest.json'
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2, default=str)
        
        logger.debug(f"Manifest saved to {manifest_path}")
    
    def create_index(self):
        """Create an index file for easy browsing with council support"""
        index_path = self.output_dir / 'index.html'
        
        # Separate council and regular files
        regular_files = [f for f in self.manifest['files'] if not f.get('is_council_response', False)]
        council_files = [f for f in self.manifest['files'] if f.get('is_council_response', False)]
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Harvest Results - {self.project_name} - {self.timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        .stats {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .file-list {{ margin-top: 20px; }}
        .file {{ margin: 5px 0; padding: 5px; background: #fff; border: 1px solid #ddd; }}
        .success {{ border-left: 3px solid #4CAF50; }}
        .failed {{ border-left: 3px solid #f44336; }}
        .council {{ border-left: 3px solid #2196F3; }}
        .synthesis {{ border-left: 3px solid #FF9800; }}
        .council-session {{ background: #e3f2fd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .tabs {{ margin: 20px 0; }}
        .tab {{ display: inline-block; padding: 10px 20px; background: #ddd; margin-right: 5px; cursor: pointer; }}
        .tab.active {{ background: #4CAF50; color: white; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
    </style>
    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            var contents = document.getElementsByClassName('tab-content');
            for (var i = 0; i < contents.length; i++) {{
                contents[i].classList.remove('active');
            }}
            
            // Remove active class from all tabs
            var tabs = document.getElementsByClassName('tab');
            for (var i = 0; i < tabs.length; i++) {{
                tabs[i].classList.remove('active');
            }}
            
            // Show selected tab content and mark tab as active
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}
    </script>
</head>
<body>
    <h1>Harvest Results: {self.project_name}</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="stats">
        <h2>Overall Statistics</h2>
        <ul>
            <li>Total Files: {self.manifest['statistics']['total_files']}</li>
            <li>Successful: {self.manifest['statistics']['successful']}</li>
            <li>Failed: {self.manifest['statistics']['failed']}</li>
            <li>Council Files: {self.manifest['statistics'].get('council_files', 0)}</li>
            <li>Synthesized Files: {self.manifest['statistics'].get('synthesized_files', 0)}</li>
            <li>Total Size: {self.manifest['statistics']['total_size_bytes'] / 1024 / 1024:.2f} MB</li>
        </ul>
    </div>
    
    <div class="tabs">
        <div class="tab active" onclick="showTab('regular-files')">Regular Files</div>
        <div class="tab" onclick="showTab('council-files')">Council Responses</div>
        <div class="tab" onclick="showTab('council-sessions')">Council Sessions</div>
    </div>
    
    <div id="regular-files" class="tab-content active">
        <h2>Regular Processed Files</h2>
        <div class="file-list">
            {"".join(f'<div class="file {("success" if f["success"] else "failed")}">{f["original_path"]} → {f["output_path"]}</div>' for f in regular_files)}
        </div>
    </div>
    
    <div id="council-files" class="tab-content">
        <h2>Council Responses</h2>
        <div class="file-list">
            {"".join(f'<div class="file council {("success" if f["success"] else "failed")}">[{f.get("council_member", "Unknown")}] {f["original_path"]} → {f["output_path"]}</div>' for f in council_files)}
        </div>
    </div>
    
    <div id="council-sessions" class="tab-content">
        <h2>Council Processing Sessions</h2>
        {"".join(f'''
        <div class="council-session">
            <h3>Session: {session["session_id"]}</h3>
            <p><strong>Original File:</strong> {session["original_file"]}</p>
            <p><strong>Status:</strong> {session["status"]}</p>
            <p><strong>Council Members:</strong> {", ".join([m["model"] for m in session["council_members"]])}</p>
            <p><strong>Synthesizer:</strong> {session.get("synthesizer_model", "None")}</p>
        </div>
        ''' for session in self.manifest.get('council_sessions', []))}
    </div>
</body>
</html>
        """
        
        with open(index_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Enhanced index with council support created at {index_path}")

    def get_council_summary(self) -> Dict[str, Any]:
        """Get summary of council processing activities"""
        council_files = [f for f in self.manifest['files'] if f.get('is_council_response', False)]
        
        summary = {
            'total_council_responses': len(council_files),
            'total_sessions': len(self.manifest.get('council_sessions', [])),
            'completed_sessions': len([s for s in self.manifest.get('council_sessions', []) if s['status'] == 'completed']),
            'models_used': list(set(f.get('council_member') for f in council_files if f.get('council_member'))),
            'synthesized_files': self.manifest['statistics'].get('synthesized_files', 0)
        }
        
        return summary