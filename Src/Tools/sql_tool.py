"""
SQL Tool Module

Provides database operations including connections, queries, 
schema management, and data manipulation.
"""

import sqlite3
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import json

from base_tool import BaseTool, ToolResult, ToolStatus, ToolCategory, ToolConfig


class SQLTool(BaseTool):
    """
    Tool for SQL database operations.
    """
    
    def __init__(self, config: Optional[ToolConfig] = None):
        """Initialize the SQL Tool"""
        super().__init__(
            name="SQLTool",
            description="Perform SQL database operations: queries, schema management, data manipulation",
            category=ToolCategory.DATABASE,
            config=config
        )
        self._connections: Dict[str, sqlite3.Connection] = {}
    
    def execute(self, operation: str, **kwargs) -> ToolResult:
        """
        Execute a SQL operation.
        
        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters
            
        Returns:
            ToolResult with operation results
        """
        operations = {
            "connect": self._connect,
            "disconnect": self._disconnect,
            "query": self._execute_query,
            "insert": self._insert_data,
            "update": self._update_data,
            "delete": self._delete_data,
            "create_table": self._create_table,
            "drop_table": self._drop_table,
            "list_tables": self._list_tables,
            "describe_table": self._describe_table,
            "bulk_insert": self._bulk_insert,
            "execute_script": self._execute_script,
            "export_to_csv": self._export_to_csv,
            "import_from_csv": self._import_from_csv,
            "backup": self._backup_database,
            "optimize": self._optimize_database
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
        """Validate parameters for SQL operations"""
        required_params = {
            "connect": ["database"],
            "disconnect": ["conn_id"],
            "query": ["conn_id", "sql"],
            "insert": ["conn_id", "table", "data"],
            "update": ["conn_id", "table", "data", "where"],
            "delete": ["conn_id", "table", "where"],
            "create_table": ["conn_id", "table", "schema"],
            "drop_table": ["conn_id", "table"],
            "list_tables": ["conn_id"],
            "describe_table": ["conn_id", "table"],
            "bulk_insert": ["conn_id", "table", "data"],
            "execute_script": ["conn_id", "script"],
            "export_to_csv": ["conn_id", "table", "output_path"],
            "import_from_csv": ["conn_id", "table", "csv_path"],
            "backup": ["conn_id", "backup_path"],
            "optimize": ["conn_id"]
        }
        
        if operation not in required_params:
            return False
        
        for param in required_params[operation]:
            if param not in kwargs:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        
        return True
    
    def _connect(
        self,
        database: str,
        conn_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Connect to a database"""
        try:
            conn_id = conn_id or f"conn_{len(self._connections)}"
            
            # Create database file if it doesn't exist
            db_path = Path(database)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            conn = sqlite3.connect(database, **kwargs)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            
            self._connections[conn_id] = conn
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "conn_id": conn_id,
                    "database": database,
                    "active_connections": list(self._connections.keys())
                },
                message=f"Connected to database: {database}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to connect to database: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _disconnect(self, conn_id: str) -> ToolResult:
        """Disconnect from a database"""
        try:
            if conn_id not in self._connections:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Connection not found: {conn_id}",
                    execution_time=0.0,
                    errors=[f"Active connections: {', '.join(self._connections.keys())}"]
                )
            
            self._connections[conn_id].close()
            del self._connections[conn_id]
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"conn_id": conn_id},
                message=f"Disconnected: {conn_id}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to disconnect: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _execute_query(
        self,
        conn_id: str,
        sql: str,
        params: Optional[Union[Tuple, Dict]] = None,
        fetch_all: bool = True
    ) -> ToolResult:
        """Execute a SQL query"""
        try:
            conn = self._get_connection(conn_id)
            if conn is None:
                return self._connection_not_found_error(conn_id)
            
            cursor = conn.cursor()
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            # Check if query returns results
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                
                if fetch_all:
                    rows = cursor.fetchall()
                    data = [dict(zip(columns, row)) for row in rows]
                else:
                    row = cursor.fetchone()
                    data = dict(zip(columns, row)) if row else None
                
                result_data = {
                    "columns": columns,
                    "rows": data,
                    "row_count": len(data) if fetch_all else (1 if data else 0)
                }
            else:
                # Query doesn't return results (INSERT, UPDATE, DELETE)
                conn.commit()
                result_data = {
                    "rows_affected": cursor.rowcount,
                    "last_row_id": cursor.lastrowid
                }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result_data,
                message="Query executed successfully",
                execution_time=0.0,
                metadata={"sql": sql}
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to execute query: {str(e)}",
                execution_time=0.0,
                errors=[str(e)],
                metadata={"sql": sql}
            )
    
    def _insert_data(
        self,
        conn_id: str,
        table: str,
        data: Dict[str, Any]
    ) -> ToolResult:
        """Insert data into table"""
        try:
            conn = self._get_connection(conn_id)
            if conn is None:
                return self._connection_not_found_error(conn_id)
            
            columns = list(data.keys())
            values = list(data.values())
            placeholders = ', '.join(['?' for _ in values])
            
            sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            
            cursor = conn.cursor()
            cursor.execute(sql, values)
            conn.commit()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "last_row_id": cursor.lastrowid,
                    "rows_affected": cursor.rowcount
                },
                message=f"Inserted 1 row into {table}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to insert data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _update_data(
        self,
        conn_id: str,
        table: str,
        data: Dict[str, Any],
        where: str,
        params: Optional[Tuple] = None
    ) -> ToolResult:
        """Update data in table"""
        try:
            conn = self._get_connection(conn_id)
            if conn is None:
                return self._connection_not_found_error(conn_id)
            
            set_clause = ', '.join([f"{col} = ?" for col in data.keys()])
            values = list(data.values())
            
            if params:
                values.extend(params)
            
            sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
            
            cursor = conn.cursor()
            cursor.execute(sql, values)
            conn.commit()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"rows_affected": cursor.rowcount},
                message=f"Updated {cursor.rowcount} row(s) in {table}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to update data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _delete_data(
        self,
        conn_id: str,
        table: str,
        where: str,
        params: Optional[Tuple] = None
    ) -> ToolResult:
        """Delete data from table"""
        try:
            conn = self._get_connection(conn_id)
            if conn is None:
                return self._connection_not_found_error(conn_id)
            
            sql = f"DELETE FROM {table} WHERE {where}"
            
            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            conn.commit()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"rows_affected": cursor.rowcount},
                message=f"Deleted {cursor.rowcount} row(s) from {table}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to delete data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _create_table(
        self,
        conn_id: str,
        table: str,
        schema: Dict[str, str],
        primary_key: Optional[str] = None,
        if_not_exists: bool = True
    ) -> ToolResult:
        """Create a table"""
        try:
            conn = self._get_connection(conn_id)
            if conn is None:
                return self._connection_not_found_error(conn_id)
            
            # Build column definitions
            columns = []
            for col_name, col_type in schema.items():
                col_def = f"{col_name} {col_type}"
                if primary_key and col_name == primary_key:
                    col_def += " PRIMARY KEY"
                columns.append(col_def)
            
            exists_clause = "IF NOT EXISTS " if if_not_exists else ""
            sql = f"CREATE TABLE {exists_clause}{table} ({', '.join(columns)})"
            
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"table": table, "schema": schema},
                message=f"Created table: {table}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create table: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _drop_table(
        self,
        conn_id: str,
        table: str,
        if_exists: bool = True
    ) -> ToolResult:
        """Drop a table"""
        try:
            conn = self._get_connection(conn_id)
            if conn is None:
                return self._connection_not_found_error(conn_id)
            
            exists_clause = "IF EXISTS " if if_exists else ""
            sql = f"DROP TABLE {exists_clause}{table}"
            
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"table": table},
                message=f"Dropped table: {table}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to drop table: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _list_tables(self, conn_id: str) -> ToolResult:
        """List all tables in database"""
        try:
            conn = self._get_connection(conn_id)
            if conn is None:
                return self._connection_not_found_error(conn_id)
            
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"tables": tables, "count": len(tables)},
                message=f"Found {len(tables)} table(s)",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to list tables: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _describe_table(self, conn_id: str, table: str) -> ToolResult:
        """Describe table schema"""
        try:
            conn = self._get_connection(conn_id)
            if conn is None:
                return self._connection_not_found_error(conn_id)
            
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table})")
            
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    "cid": row[0],
                    "name": row[1],
                    "type": row[2],
                    "not_null": bool(row[3]),
                    "default_value": row[4],
                    "primary_key": bool(row[5])
                })
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"table": table, "columns": columns},
                message=f"Table {table} has {len(columns)} column(s)",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to describe table: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _bulk_insert(
        self,
        conn_id: str,
        table: str,
        data: List[Dict[str, Any]]
    ) -> ToolResult:
        """Bulk insert data"""
        try:
            conn = self._get_connection(conn_id)
            if conn is None:
                return self._connection_not_found_error(conn_id)
            
            if not data:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message="No data provided",
                    execution_time=0.0,
                    errors=["Data list is empty"]
                )
            
            columns = list(data[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            
            values_list = [tuple(row[col] for col in columns) for row in data]
            
            cursor = conn.cursor()
            cursor.executemany(sql, values_list)
            conn.commit()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"rows_inserted": len(data)},
                message=f"Inserted {len(data)} row(s) into {table}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to bulk insert: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _execute_script(self, conn_id: str, script: str) -> ToolResult:
        """Execute SQL script"""
        try:
            conn = self._get_connection(conn_id)
            if conn is None:
                return self._connection_not_found_error(conn_id)
            
            cursor = conn.cursor()
            cursor.executescript(script)
            conn.commit()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"script_length": len(script)},
                message="Script executed successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to execute script: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _export_to_csv(
        self,
        conn_id: str,
        table: str,
        output_path: str,
        query: Optional[str] = None
    ) -> ToolResult:
        """Export table to CSV"""
        try:
            conn = self._get_connection(conn_id)
            if conn is None:
                return self._connection_not_found_error(conn_id)
            
            sql = query or f"SELECT * FROM {table}"
            df = pd.read_sql_query(sql, conn)
            
            df.to_csv(output_path, index=False)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "output_path": output_path,
                    "rows_exported": len(df),
                    "columns": df.columns.tolist()
                },
                message=f"Exported {len(df)} row(s) to {output_path}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to export to CSV: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _import_from_csv(
        self,
        conn_id: str,
        table: str,
        csv_path: str,
        if_exists: str = "append"
    ) -> ToolResult:
        """Import CSV to table"""
        try:
            conn = self._get_connection(conn_id)
            if conn is None:
                return self._connection_not_found_error(conn_id)
            
            df = pd.read_csv(csv_path)
            df.to_sql(table, conn, if_exists=if_exists, index=False)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "table": table,
                    "rows_imported": len(df),
                    "columns": df.columns.tolist()
                },
                message=f"Imported {len(df)} row(s) from {csv_path}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to import from CSV: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _backup_database(self, conn_id: str, backup_path: str) -> ToolResult:
        """Backup database"""
        try:
            conn = self._get_connection(conn_id)
            if conn is None:
                return self._connection_not_found_error(conn_id)
            
            backup_conn = sqlite3.connect(backup_path)
            conn.backup(backup_conn)
            backup_conn.close()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"backup_path": backup_path},
                message=f"Database backed up to {backup_path}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to backup database: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _optimize_database(self, conn_id: str) -> ToolResult:
        """Optimize database (VACUUM)"""
        try:
            conn = self._get_connection(conn_id)
            if conn is None:
                return self._connection_not_found_error(conn_id)
            
            cursor = conn.cursor()
            cursor.execute("VACUUM")
            conn.commit()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={},
                message="Database optimized successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to optimize database: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _get_connection(self, conn_id: str) -> Optional[sqlite3.Connection]:
        """Get connection by ID"""
        return self._connections.get(conn_id)
    
    def _connection_not_found_error(self, conn_id: str) -> ToolResult:
        """Return connection not found error"""
        return ToolResult(
            status=ToolStatus.ERROR,
            data=None,
            message=f"Connection not found: {conn_id}",
            execution_time=0.0,
            errors=[f"Active connections: {', '.join(self._connections.keys())}"]
        )
    
    def close_all_connections(self):
        """Close all active connections"""
        for conn_id, conn in self._connections.items():
            try:
                conn.close()
            except:
                pass
        self._connections.clear()
    
    def __del__(self):
        """Cleanup connections on deletion"""
        self.close_all_connections()