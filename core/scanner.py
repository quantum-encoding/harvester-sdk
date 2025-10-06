"""
Recursive file scanner with configurable profiles
"""
import os
import yaml
import fnmatch
from pathlib import Path
from typing import List, Dict, Generator, Set, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ScanResult:
    path: Path
    size: int
    language: str
    relative_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ScanResult to dictionary for JSON serialization
        
        Returns:
            Dictionary representation of the scan result
        """
        return {
            "path": str(self.path),
            "relative_path": self.relative_path,
            "language": self.language,
            "size": self.size
        }

class Scanner:
    """Recursively scans directories for code files based on profiles"""
    
    LANGUAGE_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.jsx': 'javascript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.c': 'c',
        '.cs': 'csharp',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.m': 'matlab',
        '.jl': 'julia',
        '.lua': 'lua',
        '.dart': 'dart',
        '.ex': 'elixir',
        '.clj': 'clojure',
        '.hs': 'haskell',
        '.ml': 'ocaml',
        '.nim': 'nim',
        '.v': 'vlang',
        '.zig': 'zig',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.less': 'less',
        '.vue': 'vue',
        '.svelte': 'svelte',
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',
        '.sql': 'sql',
        '.dockerfile': 'dockerfile',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'config',
        '.conf': 'config',
    }
    
    def __init__(self, profile: Dict[str, Any]):
        """
        Initialize Scanner with a profile configuration dictionary.
        
        Args:
            profile: Profile configuration dict containing extensions, ignore_patterns, etc.
        """
        # The Scanner now directly receives its configuration profile.
        # It doesn't need to know how to load it from a file.
        self.profile = profile
        self.stats = {
            'total_files_scanned': 0,
            'total_size_bytes': 0,
            'by_language': {},
            'skipped_files': 0
        }
        logger.info(f"Scanner initialized with profile containing: {list(profile.keys())}")
    
    def scan(self, root_path: Path) -> Generator[ScanResult, None, None]:
        """
        Recursively scan directory for matching files
        
        Args:
            root_path: Root directory to scan
            
        Yields:
            ScanResult objects for each matching file
        """
        root_path = Path(root_path).resolve()
        profile_name = self.profile.get('name', 'custom')
        logger.info(f"Starting scan of {root_path} with profile '{profile_name}'")
        logger.info(f"Extensions: {self.profile.get('extensions', [])}")
        logger.info(f"Ignore patterns: {self.profile.get('ignore_patterns', [])}")
        
        for file_path in self._walk_directory(root_path):
            if self._should_process_file(file_path, root_path):
                try:
                    size = file_path.stat().st_size
                    language = self._detect_language(file_path)
                    relative = file_path.relative_to(root_path)
                    
                    result = ScanResult(
                        path=file_path,
                        size=size,
                        language=language,
                        relative_path=str(relative)
                    )
                    
                    self._update_stats(result)
                    yield result
                    
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    self.stats['skipped_files'] += 1
    
    def _walk_directory(self, root: Path) -> Generator[Path, None, None]:
        """Walk directory tree, respecting ignore patterns"""
        for dirpath, dirnames, filenames in os.walk(root):
            current_dir = Path(dirpath)
            
            # Filter out ignored directories
            dirnames[:] = [
                d for d in dirnames 
                if not self._should_ignore_path(current_dir / d, root)
            ]
            
            # Yield files
            for filename in filenames:
                file_path = current_dir / filename
                if not self._should_ignore_path(file_path, root):
                    yield file_path
    
    def _should_process_file(self, file_path: Path, root: Path) -> bool:
        """Check if file should be processed based on profile"""
        # Check extension
        extensions = self.profile.get('extensions', [])
        if extensions and file_path.suffix not in extensions:
            return False
        
        # Check file size limits
        try:
            size = file_path.stat().st_size
            max_size = self.profile.get('max_file_size', 10 * 1024 * 1024)  # 10MB default
            min_size = self.profile.get('min_file_size', 1)  # 1 byte default
            
            if size > max_size or size < min_size:
                logger.debug(f"Skipping {file_path}: size {size} outside limits [{min_size}, {max_size}]")
                return False
                
        except Exception as e:
            logger.debug(f"Error checking file size for {file_path}: {e}")
            return False
        
        return True
    
    def _should_ignore_path(self, path: Path, root: Path) -> bool:
        """Check if path matches any ignore pattern"""
        try:
            relative = str(path.relative_to(root))
            
            for pattern in self.profile.get('ignore_patterns', []):
                # Support both glob patterns and simple directory names
                if fnmatch.fnmatch(relative, pattern) or pattern in relative.split(os.sep):
                    logger.debug(f"Ignoring {relative} (matches pattern: {pattern})")
                    return True
        except ValueError:
            # path is not relative to root
            return True
        
        return False
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        extension = file_path.suffix.lower()
        
        # Special cases for files without extensions
        if not extension:
            filename = file_path.name.lower()
            if filename in ['dockerfile', 'makefile', 'jenkinsfile']:
                return filename
            elif filename.startswith('.'):
                return 'config'
            else:
                return 'unknown'
        
        return self.LANGUAGE_MAP.get(extension, 'unknown')
    
    def _update_stats(self, result: ScanResult):
        """Update scanning statistics"""
        self.stats['total_files_scanned'] += 1
        self.stats['total_size_bytes'] += result.size
        
        if result.language not in self.stats['by_language']:
            self.stats['by_language'][result.language] = {
                'count': 0,
                'total_size': 0
            }
        
        self.stats['by_language'][result.language]['count'] += 1
        self.stats['by_language'][result.language]['total_size'] += result.size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scanning statistics"""
        total_size_mb = self.stats['total_size_bytes'] / (1024 * 1024) if self.stats['total_size_bytes'] > 0 else 0
        return {
            'total_files': self.stats['total_files_scanned'],
            'total_size': self.stats['total_size_bytes'],
            'total_size_mb': round(total_size_mb, 2),
            'by_language': self.stats['by_language'],
            'skipped_files': self.stats['skipped_files']
        }