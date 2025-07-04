#!/usr/bin/env python3
"""
Database verification script
Check the current state of all databases and tables
"""

import sqlite3
import os
from datetime import datetime

def check_database_exists(db_path):
    """Check if database file exists"""
    if os.path.exists(db_path):
        print(f"✅ Database found: {db_path}")
        return True
    else:
        print(f"❌ Database not found: {db_path}")
        return False

def analyze_database(db_path, db_name):
    """Analyze database structure and content"""
    print(f"\n📊 Analyzing {db_name} ({db_path})")
    print("-" * 50)
    
    if not check_database_exists(db_path):
        return
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            print(f"Tables found: {len(tables)}")
            
            for table in tables:
                table_name = table[0]
                print(f"\n  📋 Table: {table_name}")
                
                # Get column info
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                print(f"    Columns: {[col[1] for col in columns]}")
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"    Records: {count}")
                
                # Show recent records for key tables
                if table_name in ['raw_data', 'processed', 'fact_check_results'] and count > 0:
                    cursor.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT 3")
                    recent = cursor.fetchall()
                    print(f"    Recent records:")
                    for i, record in enumerate(recent):
                        print(f"      {i+1}. {record[:3]}...")  # Show first 3 fields
                        
    except sqlite3.Error as e:
        print(f"❌ Error analyzing {db_name}: {e}")

def check_data_consistency():
    """Check data consistency across databases"""
    print(f"\n🔍 Checking data consistency")
    print("-" * 50)
    
    raw_db = "databases/raw_data.db"
    
    if not os.path.exists(raw_db):
        print("❌ Raw data database not found")
        return
    
    try:
        with sqlite3.connect(raw_db) as conn:
            cursor = conn.cursor()
            
            # Check if all required tables exist
            required_tables = ['raw_data', 'processed', 'fact_check_results']
            existing_tables = []
            
            for table in required_tables:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if cursor.fetchone():
                    existing_tables.append(table)
                    print(f"✅ {table} table exists")
                else:
                    print(f"❌ {table} table missing")
            
            # Check data relationships if all tables exist
            if len(existing_tables) == len(required_tables):
                print("\n🔗 Checking data relationships:")
                
                # Count orphaned records
                cursor.execute("""
                    SELECT COUNT(*) FROM processed p 
                    LEFT JOIN raw_data r ON p.message_id = r.ID 
                    WHERE r.ID IS NULL
                """)
                orphaned_processed = cursor.fetchone()[0]
                print(f"   Orphaned processed records: {orphaned_processed}")
                
                cursor.execute("""
                    SELECT COUNT(*) FROM fact_check_results f 
                    LEFT JOIN raw_data r ON f.message_id = r.ID 
                    WHERE r.ID IS NULL
                """)
                orphaned_fact_check = cursor.fetchone()[0]
                print(f"   Orphaned fact_check records: {orphaned_fact_check}")
                
                # Show successful pipeline completions
                cursor.execute("""
                    SELECT COUNT(*) FROM raw_data r
                    INNER JOIN processed p ON r.ID = p.message_id
                    INNER JOIN fact_check_results f ON r.ID = f.message_id
                """)
                complete_pipeline = cursor.fetchone()[0]
                print(f"   Complete pipeline records: {complete_pipeline}")
                
                if complete_pipeline > 0:
                    print("✅ Pipeline is working - messages are flowing through all stages")
                else:
                    print("⚠️ No complete pipeline records found")
                    
    except sqlite3.Error as e:
        print(f"❌ Error checking consistency: {e}")

def main():
    print("🔍 WhatsApp Bridge Database Verification")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Check all known database locations
    databases = [
        ("databases/raw_data.db", "Raw Data DB"),
        ("store/messages.db", "Messages DB"),
        ("store/whatsapp.db", "WhatsApp DB"),
        ("store/raw_data.db", "Store Raw Data DB")
    ]
    
    for db_path, db_name in databases:
        analyze_database(db_path, db_name)
    
    # Check consistency
    check_data_consistency()
    
    print(f"\n✅ Database verification complete")

if __name__ == "__main__":
    main()
