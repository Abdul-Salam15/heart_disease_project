"""
Blockchain utility functions
This module provides blockchain functionality integrated with Django models
"""

import hashlib
import time
import json
from django.utils import timezone


class Block:
    """
    Represents a single block in the blockchain
    """
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()
    
    def calculate_hash(self):
        """
        Calculate SHA-256 hash of the block
        """
        block_string = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self):
        """
        Convert block to dictionary
        """
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'hash': self.hash
        }


class BlockchainManager:
    """
    Manages blockchain operations with Django integration
    """
    
    @staticmethod
    def create_genesis_block():
        """
        Create the first block in the chain
        """
        return Block(0, time.time(), "Genesis Block", "0")
    
    @staticmethod
    def create_block(index, data, previous_hash):
        """
        Create a new block
        
        Args:
            index: Block index number
            data: Dictionary of prediction data
            previous_hash: Hash of the previous block
            
        Returns:
            Block object
        """
        timestamp = time.time()
        return Block(index, timestamp, json.dumps(data), previous_hash)
    
    @staticmethod
    def verify_chain(blockchain_records):
        """
        Verify the integrity of the blockchain
        
        Args:
            blockchain_records: QuerySet of BlockchainRecord objects
            
        Returns:
            Boolean indicating if chain is valid
        """
        for i, record in enumerate(blockchain_records):
            # Verify hash
            calculated_hash = hashlib.sha256(
                f"{record.block_index}{record.timestamp}{record.data}{record.previous_hash}".encode()
            ).hexdigest()
            
            if calculated_hash != record.block_hash:
                return False
            
            # Verify chain linking (except for genesis block)
            if i > 0:
                previous_record = blockchain_records[i - 1]
                if record.previous_hash != previous_record.block_hash:
                    return False
        
        return True
    
    @staticmethod
    def get_chain_info(blockchain_records):
        """
        Get information about the blockchain
        
        Args:
            blockchain_records: QuerySet of BlockchainRecord objects
            
        Returns:
            Dictionary with chain statistics
        """
        return {
            'total_blocks': blockchain_records.count(),
            'is_valid': BlockchainManager.verify_chain(blockchain_records),
            'latest_block': blockchain_records.last() if blockchain_records.exists() else None,
            'genesis_block': blockchain_records.first() if blockchain_records.exists() else None
        }
