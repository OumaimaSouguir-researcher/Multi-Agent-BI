"""
File Tool Module

Provides file operations including read, write, copy, move, delete,
and various file management utilities.
"""

import os
import shutil
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import mimetypes
import base64

from base_tool import BaseTool, ToolResult, ToolStatus, ToolCategory, ToolConfig


class FileTool(BaseTool):
    """
    Tool for file operations.
    """
    
    def __init__(self, config: Optional[ToolConfig] = None):
        """Initialize the File Tool"""
        super().__init__(
            name="FileTool",
            description="Perform file operations: read, write, copy, move, delete, and manage files",
            category=ToolCategory.FILE_OPERATIONS,
            config=config
        )
        self.allowed_extensions = None  # None means all extensions allowed
        self.max_file_size = 100 * 1024 * 1024  # 100 MB default
    
    def execute(self, operation: str, **kwargs) -> ToolResult:
        """
        Execute a file operation.
        
        Args:
            operation: Operation to perform (read, write, copy, move, delete, list, info, search)
            **kwargs: Operation-specific parameters
            
        Returns:
            ToolResult with operation results
        """
        operations = {
            "read": self._read_file,
            "write": self._write_file,
            "append": self._append_file,
            "copy": self._copy_file,
            "move": self._move_file,
            "delete": self._delete_file,
            "list": self._list_directory,
            "info": self._get_file_info,
            "search": self._search_files,
            "exists": self._check_exists,
            "create_dir": self._create_directory,
            "hash": self._calculate_hash,
            "size": self._get_size,
            "compress": self._compress_file,
            "decompress": self._decompress_file
        }
        
        if operation not in operations:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Unknown operation: {operation}",
                execution_time=0.0,
                errors=[f"Valid operations: {', '.join(operations.keys())}"]
            )
        
        return operations[operation](**kwargs)
    
    def validate_params(self, operation: str, **kwargs) -> bool:
        """Validate parameters for file operations"""
        required_params = {
            "read": ["path"],
            "write": ["path", "content"],
            "append": ["path", "content"],
            "copy": ["source", "destination"],
            "move": ["source", "destination"],
            "delete": ["path"],
            "list": ["path"],
            "info": ["path"],
            "search": ["directory", "pattern"],
            "exists": ["path"],
            "create_dir": ["path"],
            "hash": ["path"],
            "size": ["path"],
            "compress": ["path"],
            "decompress": ["path"]
        }
        
        if operation not in required_params:
            return False
        
        for param in required_params[operation]:
            if param not in kwargs:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        
        return True
    
    def _read_file(
        self,
        path: str,
        encoding: str = "utf-8",
        binary: bool = False
    ) -> ToolResult:
        """Read file contents"""
        try:
            file_path = Path(path)
            
            if not file_path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"File not found: {path}",
                    execution_time=0.0,
                    errors=["File does not exist"]
                )
            
            if file_path.stat().st_size > self.max_file_size:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"File too large: {file_path.stat().st_size} bytes",
                    execution_time=0.0,
                    errors=["File exceeds maximum size limit"]
                )
            
            if binary:
                with open(file_path, 'rb') as f:
                    content = base64.b64encode(f.read()).decode('utf-8')
            else:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "content": content,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "binary": binary
                },
                message=f"Successfully read file: {path}",
                execution_time=0.0,
                metadata={"encoding": encoding}
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to read file: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _write_file(
        self,
        path: str,
        content: Union[str, bytes],
        encoding: str = "utf-8",
        create_dirs: bool = True
    ) -> ToolResult:
        """Write content to file"""
        try:
            file_path = Path(path)
            
            # Create parent directories if needed
            if create_dirs and not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            if isinstance(content, bytes):
                with open(file_path, 'wb') as f:
                    f.write(content)
            else:
                with open(file_path, 'w', encoding=encoding) as f:
                    f.write(content)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "created": not file_path.exists()
                },
                message=f"Successfully wrote to file: {path}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to write file: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _append_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8"
    ) -> ToolResult:
        """Append content to file"""
        try:
            file_path = Path(path)
            
            with open(file_path, 'a', encoding=encoding) as f:
                f.write(content)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "path": str(file_path),
                    "size": file_path.stat().st_size
                },
                message=f"Successfully appended to file: {path}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to append to file: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _copy_file(
        self,
        source: str,
        destination: str,
        overwrite: bool = False
    ) -> ToolResult:
        """Copy file from source to destination"""
        try:
            src_path = Path(source)
            dst_path = Path(destination)
            
            if not src_path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Source file not found: {source}",
                    execution_time=0.0,
                    errors=["Source does not exist"]
                )
            
            if dst_path.exists() and not overwrite:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Destination already exists: {destination}",
                    execution_time=0.0,
                    errors=["Use overwrite=True to replace existing file"]
                )
            
            # Create destination directory if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(src_path, dst_path)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "source": str(src_path),
                    "destination": str(dst_path),
                    "size": dst_path.stat().st_size
                },
                message=f"Successfully copied file from {source} to {destination}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to copy file: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _move_file(
        self,
        source: str,
        destination: str,
        overwrite: bool = False
    ) -> ToolResult:
        """Move file from source to destination"""
        try:
            src_path = Path(source)
            dst_path = Path(destination)
            
            if not src_path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Source file not found: {source}",
                    execution_time=0.0,
                    errors=["Source does not exist"]
                )
            
            if dst_path.exists() and not overwrite:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Destination already exists: {destination}",
                    execution_time=0.0,
                    errors=["Use overwrite=True to replace existing file"]
                )
            
            # Create destination directory if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(src_path), str(dst_path))
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "source": str(src_path),
                    "destination": str(dst_path)
                },
                message=f"Successfully moved file from {source} to {destination}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to move file: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _delete_file(self, path: str, recursive: bool = False) -> ToolResult:
        """Delete file or directory"""
        try:
            file_path = Path(path)
            
            if not file_path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Path not found: {path}",
                    execution_time=0.0,
                    errors=["Path does not exist"]
                )
            
            if file_path.is_dir():
                if recursive:
                    shutil.rmtree(file_path)
                else:
                    file_path.rmdir()  # Only removes empty directories
            else:
                file_path.unlink()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"path": str(file_path), "was_directory": file_path.is_dir()},
                message=f"Successfully deleted: {path}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to delete: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _list_directory(
        self,
        path: str,
        recursive: bool = False,
        pattern: Optional[str] = None
    ) -> ToolResult:
        """List directory contents"""
        try:
            dir_path = Path(path)
            
            if not dir_path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Directory not found: {path}",
                    execution_time=0.0,
                    errors=["Directory does not exist"]
                )
            
            if not dir_path.is_dir():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Not a directory: {path}",
                    execution_time=0.0,
                    errors=["Path is not a directory"]
                )
            
            files = []
            if recursive:
                items = dir_path.rglob(pattern or "*")
            else:
                items = dir_path.glob(pattern or "*")
            
            for item in items:
                files.append({
                    "name": item.name,
                    "path": str(item),
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0,
                    "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                })
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "path": str(dir_path),
                    "files": files,
                    "count": len(files)
                },
                message=f"Listed {len(files)} items in {path}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to list directory: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _get_file_info(self, path: str) -> ToolResult:
        """Get detailed file information"""
        try:
            file_path = Path(path)
            
            if not file_path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Path not found: {path}",
                    execution_time=0.0,
                    errors=["Path does not exist"]
                )
            
            stat = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            info = {
                "name": file_path.name,
                "path": str(file_path),
                "absolute_path": str(file_path.absolute()),
                "is_file": file_path.is_file(),
                "is_dir": file_path.is_dir(),
                "size": stat.st_size,
                "size_human": self._format_size(stat.st_size),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "mime_type": mime_type,
                "extension": file_path.suffix
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=info,
                message=f"Retrieved info for: {path}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to get file info: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _search_files(
        self,
        directory: str,
        pattern: str,
        content_search: Optional[str] = None,
        recursive: bool = True
    ) -> ToolResult:
        """Search for files matching pattern"""
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Directory not found: {directory}",
                    execution_time=0.0,
                    errors=["Directory does not exist"]
                )
            
            results = []
            if recursive:
                items = dir_path.rglob(pattern)
            else:
                items = dir_path.glob(pattern)
            
            for item in items:
                match_info = {
                    "path": str(item),
                    "name": item.name,
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0
                }
                
                # Content search if specified
                if content_search and item.is_file():
                    try:
                        with open(item, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content_search.lower() in content.lower():
                                match_info["content_match"] = True
                                results.append(match_info)
                    except:
                        pass  # Skip files that can't be read
                else:
                    results.append(match_info)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "directory": str(dir_path),
                    "pattern": pattern,
                    "results": results,
                    "count": len(results)
                },
                message=f"Found {len(results)} matching items",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to search files: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _check_exists(self, path: str) -> ToolResult:
        """Check if path exists"""
        try:
            file_path = Path(path)
            exists = file_path.exists()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "path": str(file_path),
                    "exists": exists,
                    "is_file": file_path.is_file() if exists else None,
                    "is_dir": file_path.is_dir() if exists else None
                },
                message=f"Path {'exists' if exists else 'does not exist'}: {path}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to check existence: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _create_directory(self, path: str, parents: bool = True) -> ToolResult:
        """Create directory"""
        try:
            dir_path = Path(path)
            dir_path.mkdir(parents=parents, exist_ok=True)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"path": str(dir_path)},
                message=f"Created directory: {path}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create directory: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _calculate_hash(
        self,
        path: str,
        algorithm: str = "md5"
    ) -> ToolResult:
        """Calculate file hash"""
        try:
            file_path = Path(path)
            
            if not file_path.exists() or not file_path.is_file():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"File not found: {path}",
                    execution_time=0.0,
                    errors=["File does not exist"]
                )
            
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "path": str(file_path),
                    "algorithm": algorithm,
                    "hash": hash_obj.hexdigest()
                },
                message=f"Calculated {algorithm} hash for: {path}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to calculate hash: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _get_size(self, path: str, recursive: bool = True) -> ToolResult:
        """Get size of file or directory"""
        try:
            file_path = Path(path)
            
            if not file_path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Path not found: {path}",
                    execution_time=0.0,
                    errors=["Path does not exist"]
                )
            
            if file_path.is_file():
                size = file_path.stat().st_size
            else:
                size = sum(
                    f.stat().st_size
                    for f in file_path.rglob('*') if f.is_file()
                ) if recursive else 0
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "path": str(file_path),
                    "size": size,
                    "size_human": self._format_size(size)
                },
                message=f"Size: {self._format_size(size)}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to get size: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _compress_file(self, path: str, output: Optional[str] = None) -> ToolResult:
        """Compress file to zip"""
        try:
            import zipfile
            
            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Path not found: {path}",
                    execution_time=0.0,
                    errors=["Path does not exist"]
                )
            
            output_path = Path(output) if output else Path(f"{path}.zip")
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if file_path.is_file():
                    zipf.write(file_path, file_path.name)
                else:
                    for file in file_path.rglob('*'):
                        if file.is_file():
                            zipf.write(file, file.relative_to(file_path.parent))
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "source": str(file_path),
                    "output": str(output_path),
                    "size": output_path.stat().st_size
                },
                message=f"Compressed to: {output_path}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to compress: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _decompress_file(self, path: str, output: Optional[str] = None) -> ToolResult:
        """Decompress zip file"""
        try:
            import zipfile
            
            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"File not found: {path}",
                    execution_time=0.0,
                    errors=["File does not exist"]
                )
            
            output_dir = Path(output) if output else file_path.parent / file_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(file_path, 'r') as zipf:
                zipf.extractall(output_dir)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "source": str(file_path),
                    "output": str(output_dir)
                },
                message=f"Decompressed to: {output_dir}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to decompress: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    @staticmethod
    def _format_size(size: int) -> str:
        """Format size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"