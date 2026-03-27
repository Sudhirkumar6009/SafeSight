"""
MongoDB Connection Module
=========================
Handles MongoDB connection for user authentication
"""

import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings

logger = logging.getLogger(__name__)

# Global MongoDB client
mongodb_client: Optional[AsyncIOMotorClient] = None
mongodb_database: Optional[AsyncIOMotorDatabase] = None


async def init_mongodb() -> None:
    """Initialize MongoDB connection."""
    global mongodb_client, mongodb_database
    
    try:
        logger.info(f"Connecting to MongoDB at {settings.mongodb_url}...")
        
        mongodb_client = AsyncIOMotorClient(
            settings.mongodb_url,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
        )
        
        # Verify connection
        await mongodb_client.admin.command('ping')
        
        mongodb_database = mongodb_client[settings.mongodb_database]
        
        # Create indexes for users collection
        users_collection = mongodb_database.users
        await users_collection.create_index("email", unique=True)
        await users_collection.create_index("username", unique=True)
        
        logger.info(f"Connected to MongoDB database: {settings.mongodb_database}")
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_mongodb() -> None:
    """Close MongoDB connection."""
    global mongodb_client, mongodb_database
    
    if mongodb_client:
        mongodb_client.close()
        mongodb_client = None
        mongodb_database = None
        logger.info("MongoDB connection closed")


def get_mongodb() -> AsyncIOMotorDatabase:
    """Get MongoDB database instance."""
    if mongodb_database is None:
        raise RuntimeError("MongoDB not initialized. Call init_mongodb() first.")
    return mongodb_database
