#!/usr/bin/env python3
"""
Script to generate comprehensive summary documents for specified folders in the Amazon Bedrock Workshop.
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path

# Configuration
REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = REPO_ROOT / "my-analysis"

# List of folders to analyze - you can modify this list as needed
FOLDERS_TO_ANALYZE = [
    "01_Text_generation",
    "02_Knowledge_Bases_and_RAG",
    "03_Model_customization",
    "04_Image_and_Multimodal",
    "05_Agents",
    "06_OpenSource_examples",
    "07_Cross_Region_Inference"
]

def read_file(file_path):
    """Read the contents of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def list_files(directory):
    """List all files in a directory recursively."""
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                result.append(os.path.join(root, file))
    except Exception as e:
        print(f"Error listing files in {directory}: {e}")
    return result

def extract_code_blocks(content):
    """Extract code blocks from markdown or notebook content."""
    code_blocks = []
    # Simple pattern for markdown code blocks
    md_pattern = r'```(?:python)?\s*(.*?)\s*```'
    md_matches = re.findall(md_pattern, content, re.DOTALL)
    code_blocks.extend(md_matches)
    
    # Try to parse as JSON for notebooks
    try:
        notebook = json.loads(content)
        if 'cells' in notebook:
            for cell in notebook['cells']:
                if cell.get('cell_type') == 'code' and 'source' in cell:
                    code = ''.join(cell['source'])
                    if code.strip():
                        code_blocks.append(code)
    except:
        pass
    
    return code_blocks

def analyze_folder(folder_name):
    """Analyze a folder and generate a comprehensive summary."""
    folder_path = REPO_ROOT / folder_name
    if not folder_path.exists():
        print(f"Folder {folder_name} does not exist.")
        return None
    
    print(f"Analyzing folder: {folder_name}")
    
    # Read README.md if it exists
    readme_path = folder_path / "README.md"
    readme_content = read_file(readme_path) if readme_path.exists() else ""
    
    # Get all files in the folder
    all_files = list_files(folder_path)
    
    # Categorize files
    notebooks = [f for f in all_files if f.endswith('.ipynb')]
    python_files = [f for f in all_files if f.endswith('.py')]
    markdown_files = [f for f in all_files if f.endswith('.md')]
    other_files = [f for f in all_files if not (f.endswith('.ipynb') or f.endswith('.py') or f.endswith('.md'))]
    
    # Extract key information from notebooks and Python files
    code_samples = []
    key_concepts = []
    
    for notebook in notebooks:
        content = read_file(notebook)
        if content:
            code_blocks = extract_code_blocks(content)
            if code_blocks:
                # Take a sample of code blocks (to avoid making the summary too long)
                sample = code_blocks[0] if code_blocks else ""
                if len(sample) > 500:
                    sample = sample[:500] + "..."
                code_samples.append((os.path.basename(notebook), sample))
    
    for py_file in python_files:
        content = read_file(py_file)
        if content:
            if len(content) > 500:
                content = content[:500] + "..."
            code_samples.append((os.path.basename(py_file), content))
    
    # Generate the summary
    summary = f"# Amazon Bedrock Workshop - {folder_name} Module Analysis\n\n"
    
    # Executive Summary
    summary += "## Executive Summary\n\n"
    if readme_content:
        # Extract first paragraph or first few lines from README
        first_para = readme_content.split('\n\n')[0] if '\n\n' in readme_content else readme_content.split('\n')[0]
        summary += f"{first_para}\n\n"
    else:
        summary += f"This module focuses on {folder_name.replace('_', ' ')} capabilities in Amazon Bedrock.\n\n"
    
    # Implementation Details
    summary += "## Implementation Details Breakdown\n\n"
    
    # List notebooks and their purpose
    if notebooks:
        summary += "### Notebooks\n\n"
        for notebook in notebooks:
            notebook_name = os.path.basename(notebook)
            summary += f"- **{notebook_name}**: "
            # Try to infer purpose from filename
            name_parts = notebook_name.replace('.ipynb', '').split('_')
            purpose = ' '.join(name_parts).capitalize()
            summary += f"{purpose}\n"
        summary += "\n"
    
    # List Python files and their purpose
    if python_files:
        summary += "### Python Files\n\n"
        for py_file in python_files:
            py_file_name = os.path.basename(py_file)
            summary += f"- **{py_file_name}**: "
            # Try to infer purpose from filename
            name_parts = py_file_name.replace('.py', '').split('_')
            purpose = ' '.join(name_parts).capitalize()
            summary += f"{purpose}\n"
        summary += "\n"
    
    # Code samples
    if code_samples:
        summary += "### Key Code Samples\n\n"
        for file_name, code in code_samples[:3]:  # Limit to 3 samples
            summary += f"#### From {file_name}\n\n"
            summary += f"```python\n{code}\n```\n\n"
    
    # Technical Architecture
    summary += "## Technical Architecture Overview\n\n"
    summary += "```mermaid\nflowchart TD\n"
    summary += f"    A[Client Application] --> B[Amazon Bedrock Service]\n"
    
    # Add module-specific components to the diagram
    if "Text_generation" in folder_name:
        summary += "    B --> C1[Invoke API]\n"
        summary += "    B --> C2[Converse API]\n"
        summary += "    C2 --> D1[Text Generation]\n"
        summary += "    C2 --> D2[Multi-turn Conversations]\n"
        summary += "    C2 --> D3[Function Calling]\n"
    elif "Knowledge_Bases" in folder_name:
        summary += "    B --> C1[Knowledge Base]\n"
        summary += "    C1 --> D1[Document Ingestion]\n"
        summary += "    C1 --> D2[Retrieval]\n"
        summary += "    B --> C2[RAG]\n"
        summary += "    C2 --> D3[Query Processing]\n"
        summary += "    D2 --> D3\n"
    elif "Model_customization" in folder_name:
        summary += "    B --> C1[Fine-tuning]\n"
        summary += "    B --> C2[Continued Pre-training]\n"
        summary += "    C1 --> D1[Custom Models]\n"
    elif "Image" in folder_name:
        summary += "    B --> C1[Image Generation]\n"
        summary += "    B --> C2[Image Editing]\n"
        summary += "    B --> C3[Multimodal Understanding]\n"
    elif "Agents" in folder_name:
        summary += "    B --> C1[Agent Creation]\n"
        summary += "    B --> C2[Knowledge Base Association]\n"
        summary += "    B --> C3[Agent Invocation]\n"
        summary += "    C1 --> D1[Action Groups]\n"
        summary += "    C2 --> D2[Document Retrieval]\n"
    elif "OpenSource" in folder_name:
        summary += "    B --> C1[LangChain Integration]\n"
        summary += "    B --> C2[LangGraph Integration]\n"
        summary += "    B --> C3[CrewAI Integration]\n"
    elif "Cross_Region" in folder_name:
        summary += "    B --> C1[Cross-Region Inference]\n"
        summary += "    C1 --> D1[Higher Throughput]\n"
        summary += "    C1 --> D2[Traffic Management]\n"
    
    summary += "```\n\n"
    
    # Key Takeaways
    summary += "## Key Takeaways and Lessons Learned\n\n"
    summary += "1. **Module Focus**: This module demonstrates " + folder_name.replace('_', ' ') + " capabilities in Amazon Bedrock.\n\n"
    summary += "2. **Integration Patterns**: The examples show how to integrate Amazon Bedrock services into applications.\n\n"
    summary += "3. **Best Practices**: The code demonstrates recommended patterns for working with Amazon Bedrock APIs.\n\n"
    
    # Recommendations
    summary += "## Recommendations and Next Steps\n\n"
    summary += "1. **Explore Further**: Experiment with different parameters and configurations to understand their impact.\n\n"
    summary += "2. **Combine Capabilities**: Consider how the capabilities demonstrated in this module can be combined with other Amazon Bedrock features.\n\n"
    summary += "3. **Production Considerations**: When moving to production, consider aspects like error handling, monitoring, and scaling.\n\n"
    
    return summary

def generate_summary_file(folder_name):
    """Generate a summary file for a folder."""
    summary = analyze_folder(folder_name)
    if summary:
        # Create output filename: SUMMARY-folder_name.md
        clean_folder_name = folder_name.split('_')[0] if '_' in folder_name else folder_name
        output_file = OUTPUT_DIR / f"SUMMARY-{clean_folder_name}.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"Summary generated: {output_file}")
        return True
    return False

def main():
    """Main function to generate summaries for specified folders."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process command line arguments if provided, otherwise use default list
    folders = sys.argv[1:] if len(sys.argv) > 1 else FOLDERS_TO_ANALYZE
    
    print(f"Generating summaries for {len(folders)} folders...")
    
    success_count = 0
    for folder in folders:
        if generate_summary_file(folder):
            success_count += 1
    
    print(f"Summary generation complete. {success_count}/{len(folders)} summaries generated.")

if __name__ == "__main__":
    main()
