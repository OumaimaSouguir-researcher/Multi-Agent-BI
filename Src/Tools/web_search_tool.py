"""
Web Search Tool Module

Provides web search and scraping capabilities including search queries,
content extraction, and web data collection.
"""

import requests
from bs4 import BeautifulSoup
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse, quote_plus
import re
import time
from datetime import datetime

from base_tool import BaseTool, ToolResult, ToolStatus, ToolCategory, ToolConfig


class WebSearchTool(BaseTool):
    """
    Tool for web search and content extraction.
    """
    
    def __init__(self, config: Optional[ToolConfig] = None):
        """Initialize the Web Search Tool"""
        super().__init__(
            name="WebSearchTool",
            description="Perform web searches and extract content from websites",
            category=ToolCategory.WEB,
            config=config
        )
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.rate_limit_delay = 1.0  # seconds between requests
        self._last_request_time = 0
    
    def execute(self, operation: str, **kwargs) -> ToolResult:
        """
        Execute a web operation.
        
        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters
            
        Returns:
            ToolResult with operation results
        """
        operations = {
            "search": self._search,
            "fetch": self._fetch_page,
            "extract_text": self._extract_text,
            "extract_links": self._extract_links,
            "extract_images": self._extract_images,
            "extract_metadata": self._extract_metadata,
            "scrape": self._scrape_structured,
            "download": self._download_file,
            "check_status": self._check_status,
            "get_headers": self._get_headers,
            "extract_tables": self._extract_tables,
            "crawl": self._crawl_site
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
        """Validate parameters for web operations"""
        required_params = {
            "search": ["query"],
            "fetch": ["url"],
            "extract_text": ["url"],
            "extract_links": ["url"],
            "extract_images": ["url"],
            "extract_metadata": ["url"],
            "scrape": ["url", "selectors"],
            "download": ["url", "output_path"],
            "check_status": ["url"],
            "get_headers": ["url"],
            "extract_tables": ["url"],
            "crawl": ["start_url"]
        }
        
        if operation not in required_params:
            return False
        
        for param in required_params[operation]:
            if param not in kwargs:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        
        return True
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self._last_request_time = time.time()
    
    def _search(
        self,
        query: str,
        num_results: int = 10,
        engine: str = "duckduckgo"
    ) -> ToolResult:
        """
        Perform web search (simulated - in production would use actual search APIs).
        
        Note: This is a simplified implementation. In production, you would use
        actual search engine APIs like Google Custom Search, Bing API, etc.
        """
        try:
            self._rate_limit()
            
            # Simulate search results (in production, use real search API)
            results = {
                "query": query,
                "engine": engine,
                "results": [
                    {
                        "title": f"Result {i+1} for '{query}'",
                        "url": f"https://example.com/result{i+1}",
                        "snippet": f"This is a simulated search result snippet for query: {query}"
                    }
                    for i in range(min(num_results, 5))
                ],
                "total_results": num_results
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=results,
                message=f"Search completed: found {len(results['results'])} results",
                execution_time=0.0,
                warnings=["Using simulated search results. Integrate with actual search API for production use."]
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to perform search: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _fetch_page(
        self,
        url: str,
        timeout: int = 10,
        allow_redirects: bool = True
    ) -> ToolResult:
        """Fetch web page content"""
        try:
            self._rate_limit()
            
            response = self.session.get(
                url,
                timeout=timeout,
                allow_redirects=allow_redirects
            )
            response.raise_for_status()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "url": url,
                    "final_url": response.url,
                    "status_code": response.status_code,
                    "content": response.text,
                    "content_type": response.headers.get('Content-Type'),
                    "encoding": response.encoding,
                    "size": len(response.content)
                },
                message=f"Fetched page successfully: {url}",
                execution_time=0.0
            )
            
        except requests.exceptions.RequestException as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to fetch page: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _extract_text(
        self,
        url: str,
        clean: bool = True
    ) -> ToolResult:
        """Extract text content from a web page"""
        try:
            # Fetch page
            fetch_result = self._fetch_page(url)
            if not fetch_result.is_success():
                return fetch_result
            
            html_content = fetch_result.data['content']
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            if clean:
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "url": url,
                    "text": text,
                    "length": len(text),
                    "word_count": len(text.split())
                },
                message=f"Extracted {len(text)} characters of text",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to extract text: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _extract_links(
        self,
        url: str,
        internal_only: bool = False,
        external_only: bool = False
    ) -> ToolResult:
        """Extract all links from a web page"""
        try:
            fetch_result = self._fetch_page(url)
            if not fetch_result.is_success():
                return fetch_result
            
            html_content = fetch_result.data['content']
            soup = BeautifulSoup(html_content, 'html.parser')
            base_domain = urlparse(url).netloc
            
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(url, href)
                link_domain = urlparse(absolute_url).netloc
                
                is_internal = link_domain == base_domain
                
                if internal_only and not is_internal:
                    continue
                if external_only and is_internal:
                    continue
                
                links.append({
                    "url": absolute_url,
                    "text": link.get_text(strip=True),
                    "title": link.get('title', ''),
                    "is_internal": is_internal
                })
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "source_url": url,
                    "links": links,
                    "count": len(links),
                    "internal_count": sum(1 for l in links if l['is_internal']),
                    "external_count": sum(1 for l in links if not l['is_internal'])
                },
                message=f"Extracted {len(links)} links",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to extract links: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _extract_images(self, url: str) -> ToolResult:
        """Extract all images from a web page"""
        try:
            fetch_result = self._fetch_page(url)
            if not fetch_result.is_success():
                return fetch_result
            
            html_content = fetch_result.data['content']
            soup = BeautifulSoup(html_content, 'html.parser')
            
            images = []
            for img in soup.find_all('img'):
                src = img.get('src')
                if src:
                    absolute_url = urljoin(url, src)
                    images.append({
                        "url": absolute_url,
                        "alt": img.get('alt', ''),
                        "title": img.get('title', ''),
                        "width": img.get('width'),
                        "height": img.get('height')
                    })
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "source_url": url,
                    "images": images,
                    "count": len(images)
                },
                message=f"Extracted {len(images)} images",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to extract images: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _extract_metadata(self, url: str) -> ToolResult:
        """Extract metadata from a web page"""
        try:
            fetch_result = self._fetch_page(url)
            if not fetch_result.is_success():
                return fetch_result
            
            html_content = fetch_result.data['content']
            soup = BeautifulSoup(html_content, 'html.parser')
            
            metadata = {
                "title": soup.title.string if soup.title else None,
                "description": None,
                "keywords": None,
                "author": None,
                "og_tags": {},
                "twitter_tags": {},
                "meta_tags": []
            }
            
            # Extract meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name', '').lower()
                property_val = meta.get('property', '').lower()
                content = meta.get('content', '')
                
                if name == 'description':
                    metadata['description'] = content
                elif name == 'keywords':
                    metadata['keywords'] = content
                elif name == 'author':
                    metadata['author'] = content
                elif property_val.startswith('og:'):
                    metadata['og_tags'][property_val] = content
                elif name.startswith('twitter:'):
                    metadata['twitter_tags'][name] = content
                
                if name or property_val:
                    metadata['meta_tags'].append({
                        'name': name or property_val,
                        'content': content
                    })
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "url": url,
                    "metadata": metadata
                },
                message="Extracted metadata successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to extract metadata: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _scrape_structured(
        self,
        url: str,
        selectors: Dict[str, str]
    ) -> ToolResult:
        """Scrape structured data using CSS selectors"""
        try:
            fetch_result = self._fetch_page(url)
            if not fetch_result.is_success():
                return fetch_result
            
            html_content = fetch_result.data['content']
            soup = BeautifulSoup(html_content, 'html.parser')
            
            results = {}
            for key, selector in selectors.items():
                elements = soup.select(selector)
                if len(elements) == 1:
                    results[key] = elements[0].get_text(strip=True)
                elif len(elements) > 1:
                    results[key] = [el.get_text(strip=True) for el in elements]
                else:
                    results[key] = None
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "url": url,
                    "results": results,
                    "selectors": selectors
                },
                message="Scraped structured data successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to scrape structured data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _download_file(
        self,
        url: str,
        output_path: str,
        chunk_size: int = 8192
    ) -> ToolResult:
        """Download file from URL"""
        try:
            self._rate_limit()
            
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "url": url,
                    "output_path": output_path,
                    "size": downloaded,
                    "content_type": response.headers.get('Content-Type')
                },
                message=f"Downloaded file: {downloaded} bytes",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to download file: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _check_status(self, url: str) -> ToolResult:
        """Check HTTP status of URL"""
        try:
            self._rate_limit()
            
            response = self.session.head(url, allow_redirects=True, timeout=10)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "url": url,
                    "status_code": response.status_code,
                    "status_text": response.reason,
                    "final_url": response.url,
                    "redirected": url != response.url,
                    "content_type": response.headers.get('Content-Type'),
                    "content_length": response.headers.get('Content-Length')
                },
                message=f"Status: {response.status_code} {response.reason}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to check status: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _get_headers(self, url: str) -> ToolResult:
        """Get HTTP headers from URL"""
        try:
            self._rate_limit()
            
            response = self.session.head(url, allow_redirects=True, timeout=10)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "url": url,
                    "headers": dict(response.headers)
                },
                message="Retrieved headers successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to get headers: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _extract_tables(self, url: str) -> ToolResult:
        """Extract HTML tables from page"""
        try:
            fetch_result = self._fetch_page(url)
            if not fetch_result.is_success():
                return fetch_result
            
            html_content = fetch_result.data['content']
            soup = BeautifulSoup(html_content, 'html.parser')
            
            tables = []
            for idx, table in enumerate(soup.find_all('table')):
                rows = []
                for tr in table.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells:
                        rows.append(cells)
                
                if rows:
                    tables.append({
                        "index": idx,
                        "rows": rows,
                        "row_count": len(rows),
                        "column_count": len(rows[0]) if rows else 0
                    })
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "url": url,
                    "tables": tables,
                    "count": len(tables)
                },
                message=f"Extracted {len(tables)} table(s)",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to extract tables: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _crawl_site(
        self,
        start_url: str,
        max_pages: int = 10,
        max_depth: int = 2,
        same_domain_only: bool = True
    ) -> ToolResult:
        """Crawl website starting from URL"""
        try:
            visited = set()
            to_visit = [(start_url, 0)]  # (url, depth)
            results = []
            base_domain = urlparse(start_url).netloc
            
            while to_visit and len(visited) < max_pages:
                current_url, depth = to_visit.pop(0)
                
                if current_url in visited or depth > max_depth:
                    continue
                
                visited.add(current_url)
                
                # Fetch page
                fetch_result = self._fetch_page(current_url)
                if not fetch_result.is_success():
                    continue
                
                results.append({
                    "url": current_url,
                    "depth": depth,
                    "status": "success"
                })
                
                # Extract links if not at max depth
                if depth < max_depth:
                    html_content = fetch_result.data['content']
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(current_url, link['href'])
                        link_domain = urlparse(next_url).netloc
                        
                        # Filter by domain if required
                        if same_domain_only and link_domain != base_domain:
                            continue
                        
                        if next_url not in visited:
                            to_visit.append((next_url, depth + 1))
                
                # Rate limiting
                self._rate_limit()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "start_url": start_url,
                    "pages_crawled": len(results),
                    "pages": results
                },
                message=f"Crawled {len(results)} page(s)",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to crawl site: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def __del__(self):
        """Cleanup session on deletion"""
        if hasattr(self, 'session'):
            self.session.close()