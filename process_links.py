import asyncio
import sqlite3
import json
from typing import List, Dict, Any
import sys
import os

# Add the parent directory to the path to import the link_crawler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Extractors.link_crawler import crawl_link_records

class DatabaseLinkProcessor:
    """
    Integration class to process links from your WhatsApp database
    """
    
    def __init__(self, db_path: str = "databases/raw_data.db"):
        self.db_path = db_path
    
    def get_link_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get records from database where content_type is 'link'
        
        Args:
            limit: Maximum number of records to fetch
            
        Returns:
            List of database records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            
            cursor = conn.cursor()
            
            # Query for link records that haven't been crawled yet
            query = """
            SELECT 
                ID, UUID, source_type, sender_phone, is_group_message, 
                group_name, channel_name, chat_jid, content_type, content_url,
                raw_text, submission_timestamp, processing_status, user_identifier,
                priority, metadata
            FROM raw_data 
            WHERE content_type = 'link' 
            AND (processing_status IS NULL OR processing_status != 'crawled')
            ORDER BY submission_timestamp DESC            LIMIT ?
            """
            
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
              # Convert to list of dictionaries with standardized field names
            records = []
            for row in rows:
                record = {
                    # Keep original database fields
                    "ID": row["ID"],  # Keep the original ID
                    "UUID": row["UUID"],
                    # Map database fields to expected crawler fields
                    "message_id": row["UUID"],  # Using UUID as message_id
                    "chat_jid": row["chat_jid"],
                    "sender_jid": row["sender_phone"],  # Map sender_phone to sender_jid
                    "user_identifier": row["user_identifier"],
                    "content": row["raw_text"],  # Map raw_text to content
                    "content_type": row["content_type"],
                    "content_url": row["content_url"],
                    "timestamp": row["submission_timestamp"],
                    "source_type": row["source_type"],
                    "is_group": bool(row["is_group_message"]),
                    "group_name": row["group_name"],
                    "channel_name": row["channel_name"],
                    "processing_status": row["processing_status"],
                    "priority": row["priority"],
                    "metadata": row["metadata"],
                    # Store original database ID for updates
                    "_db_id": row["ID"]
                }
                records.append(record)
            
            conn.close()
            
            print(f"📊 Found {len(records)} link records to process")
            return records
            
        except Exception as e:
            print(f"❌ Error fetching link records: {e}")
            return []
    
    def update_crawled_record(self, record: Dict[str, Any]) -> bool:
        """
        Update database record with crawled content
        
        Args:
            record: Processed record with crawled content
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
              # Update the record with crawled content
            update_query = """
            UPDATE raw_data 
            SET 
                processing_status = ?
            WHERE ID = ?
            """
            
            # Print crawled results to standard output
            print(f"\n{'='*80}")
            print(f"🔗 CRAWLED CONTENT FOR RECORD {record.get('message_id')}")
            print(f"{'='*80}")
            print(f"📍 Original URL: {record.get('content', 'N/A')}")
            print(f"📊 Crawl Status: {record.get('crawl_status', 'unknown')}")
            
            if record.get("crawl_status") == "success":
                print(f"📝 Title: {record.get('crawled_title', 'N/A')}")
                print(f"📄 Description: {record.get('crawled_description', 'N/A')}")
                print(f"🔢 Word Count: {record.get('crawled_word_count', 0)}")
                print(f"🌐 Crawled URL: {record.get('crawled_url', 'N/A')}")
                print(f"\n📖 EXTRACTED CONTENT:")
                print(f"{'-'*40}")
                crawled_content = record.get("crawled_content", "")
                if crawled_content:
                    # Print first 1000 characters for readability
                    if len(crawled_content) > 1000:
                        print(f"{crawled_content[:1000]}...")
                        print(f"\n[Content truncated - Total length: {len(crawled_content)} characters]")
                    else:
                        print(crawled_content)
                else:
                    print("No content extracted")
            else:
                print(f"❌ Crawl failed: {record.get('crawl_reason', 'Unknown error')}")
            
            print(f"{'='*80}\n")
            
            # Just update the processing status
            status = "crawled" if record.get("crawl_status") == "success" else "crawl_failed"
            
            cursor.execute(update_query, (
                status,
                record.get("_db_id")  # Use the original database ID
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating record {record.get('message_id')} (DB ID: {record.get('_db_id')}): {e}")
            return False
    
    async def process_all_links(self, batch_size: int = 10) -> Dict[str, int]:
        """
        Process all link records in the database
        
        Args:
            batch_size: Number of records to process at once
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
        
        print("🚀 Starting link processing...")
        
        while True:
            # Get batch of link records
            records = self.get_link_records(batch_size)
            
            if not records:
                print("✅ No more link records to process")
                break
            
            # Process the batch
            processed_records = await crawl_link_records(records)
            
            # Update database with results
            for record in processed_records:
                stats["total"] += 1
                
                crawl_status = record.get("crawl_status", "unknown")
                if crawl_status == "success":
                    stats["success"] += 1
                elif crawl_status == "failed":
                    stats["failed"] += 1
                else:
                    stats["skipped"] += 1
                
                # Update database
                self.update_crawled_record(record)
                
                print(f"📝 Updated record {record.get('message_id')} - Status: {crawl_status}")
        
        print(f"🎯 Processing complete! Stats: {stats}")
        return stats


async def main():
    """Example usage of the database link processor"""
    
    # Initialize the processor
    processor = DatabaseLinkProcessor()
    
    # Process all links in the database
    stats = await processor.process_all_links(batch_size=5)
    
    print(f"\n📈 Final Statistics:")
    print(f"   Total processed: {stats['total']}")
    print(f"   Successfully crawled: {stats['success']}")
    print(f"   Failed to crawl: {stats['failed']}")
    print(f"   Skipped: {stats['skipped']}")


if __name__ == "__main__":
    asyncio.run(main())
