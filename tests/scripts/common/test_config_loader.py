#!/usr/bin/env python3
"""
Tests for config_loader module.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Import the module under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts', 'common'))

from config_loader import (
    load_config,
    filter_sensitive_fields,
    add_config_to_metadata,
    SENSITIVE_FIELDS,
)


def test_load_json_config():
    """Test loading JSON configuration file."""
    config_data = {
        "defaults": {
            "prefilter_model_type": "CTRANSPATH",
            "batch_size": 64,
        },
        "tasks": [
            {
                "task_id": "task_1",
                "slide_path": "/path/to/slide1.svs",
                "output_h5_path": "/path/to/output1.h5",
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_file = f.name
    
    try:
        config = load_config(config_file)
        assert config == config_data
        assert "defaults" in config
        assert "tasks" in config
        assert len(config["tasks"]) == 1
    finally:
        os.unlink(config_file)


def test_load_yaml_config():
    """Test loading YAML configuration file."""
    yaml_content = """
defaults:
  prefilter_model_type: CTRANSPATH
  batch_size: 64

tasks:
  - task_id: task_1
    slide_path: /path/to/slide1.svs
    output_h5_path: /path/to/output1.h5
  - task_id: task_2
    slide_path: /path/to/slide2.svs
    output_h5_path: /path/to/output2.h5
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        config = load_config(config_file)
        assert "defaults" in config
        assert "tasks" in config
        assert len(config["tasks"]) == 2
        assert config["defaults"]["prefilter_model_type"] == "CTRANSPATH"
        assert config["defaults"]["batch_size"] == 64
        assert config["tasks"][0]["task_id"] == "task_1"
    finally:
        os.unlink(config_file)


def test_load_yaml_config_yml_extension():
    """Test loading YAML configuration file with .yml extension."""
    yaml_content = """
defaults:
  use_gpu: true
  
tasks:
  - task_id: test_task
    slide_path: /test/slide.svs
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write(yaml_content)
        config_file = f.name
    
    try:
        config = load_config(config_file)
        assert "defaults" in config
        assert config["defaults"]["use_gpu"] is True
    finally:
        os.unlink(config_file)


def test_load_config_file_not_found():
    """Test that FileNotFoundError is raised for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_config("/path/that/does/not/exist.json")


def test_load_config_unsupported_format():
    """Test that ValueError is raised for unsupported file format."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("some content")
        config_file = f.name
    
    try:
        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            load_config(config_file)
    finally:
        os.unlink(config_file)


def test_filter_sensitive_fields():
    """Test filtering sensitive fields from configuration."""
    config = {
        "prefilter_model_type": "CTRANSPATH",
        "batch_size": 64,
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "hf_token": "hf_token_value",
        "use_gpu": True,
    }
    
    filtered = filter_sensitive_fields(config)
    
    # Non-sensitive fields should be present
    assert filtered["prefilter_model_type"] == "CTRANSPATH"
    assert filtered["batch_size"] == 64
    assert filtered["use_gpu"] is True
    
    # Sensitive fields should be removed
    assert "aws_access_key_id" not in filtered
    assert "aws_secret_access_key" not in filtered
    assert "hf_token" not in filtered


def test_filter_sensitive_fields_nested():
    """Test filtering sensitive fields from nested configuration."""
    config = {
        "defaults": {
            "prefilter_model_type": "CTRANSPATH",
            "aws_access_key_id": "SENSITIVE",
        },
        "tasks": [
            {
                "task_id": "task_1",
                "hf_token": "SENSITIVE",
                "batch_size": 64,
            }
        ]
    }
    
    filtered = filter_sensitive_fields(config)
    
    # Check nested defaults
    assert "prefilter_model_type" in filtered["defaults"]
    assert "aws_access_key_id" not in filtered["defaults"]
    
    # Check nested tasks list
    assert len(filtered["tasks"]) == 1
    assert filtered["tasks"][0]["task_id"] == "task_1"
    assert filtered["tasks"][0]["batch_size"] == 64
    assert "hf_token" not in filtered["tasks"][0]


def test_add_config_to_metadata():
    """Test adding configuration to task metadata."""
    task_metadata = {}
    config = {
        "task_id": "task_1",
        "slide_path": "/path/to/slide.svs",
        "prefilter_model_type": "CTRANSPATH",
        "aws_secret_access_key": "SECRET",
        "batch_size": 64,
    }
    
    add_config_to_metadata(task_metadata, config, "task_1")
    
    # Check metadata was added
    assert "task_1" in task_metadata
    assert "config" in task_metadata["task_1"]
    
    # Check non-sensitive fields are present
    stored_config = task_metadata["task_1"]["config"]
    assert stored_config["task_id"] == "task_1"
    assert stored_config["slide_path"] == "/path/to/slide.svs"
    assert stored_config["prefilter_model_type"] == "CTRANSPATH"
    assert stored_config["batch_size"] == 64
    
    # Check sensitive field was filtered out
    assert "aws_secret_access_key" not in stored_config


def test_add_config_to_metadata_updates_existing():
    """Test that add_config_to_metadata updates existing metadata."""
    task_metadata = {
        "task_1": {
            "slide_path": "/old/path.svs",
            "output_h5_path": "/output.h5",
        }
    }
    
    config = {
        "task_id": "task_1",
        "prefilter_model_type": "CTRANSPATH",
        "batch_size": 64,
    }
    
    add_config_to_metadata(task_metadata, config, "task_1")
    
    # Check that existing fields are preserved
    assert task_metadata["task_1"]["slide_path"] == "/old/path.svs"
    assert task_metadata["task_1"]["output_h5_path"] == "/output.h5"
    
    # Check that config was added
    assert "config" in task_metadata["task_1"]
    assert task_metadata["task_1"]["config"]["prefilter_model_type"] == "CTRANSPATH"


def test_sensitive_fields_constant():
    """Test that SENSITIVE_FIELDS contains expected fields."""
    expected_fields = {
        'aws_access_key_id',
        'aws_secret_access_key',
        'hf_token',
        'batch_account_key',
        'storage_account_key',
        'azure_files_share_name',
    }
    
    assert SENSITIVE_FIELDS == expected_fields
