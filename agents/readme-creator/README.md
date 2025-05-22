# README Generation Agent

## Overview of the Agent

This agent automates the process of generating professional README.md files for GitHub repositories by analyzing repository structure and content. It enhances project documentation by creating comprehensive, standardized READMEs tailored to each repository's specific characteristics.

## Agent Details

The README generation agent has the following capabilities:

* Clones GitHub repositories to local machine for analysis
* Scans repository structure to understand project organization
* Extracts and analyzes key files (READMEs, requirements, config files)
* Generates well-structured, professional README.md files
* Updates existing READMEs with standardized sections

| Feature | Description |
|---------|-------------|
| **Interaction Type** | Programmatic (API calls) |
| **Complexity** | Medium |
| **Agent Type** | Single Agent |
| **Components** | Git command-line tools for repository cloning<br>File system access for repository analysis<br>Generative AI model (Gemini 2.0 Flash) for content generation<br>Markdown formatting for README output |
| **Vertical** | Software Development/DevOps |

## Architecture

The agent follows this workflow:
1. Repository cloning via git commands
2. File system analysis to determine structure
3. Key file content extraction
4. Prompt engineering for the AI model
5. README content generation
6. File system write operation

```
[Git Repository URL] 
    → [Clone Repository] 
    → [Analyze Structure] 
    → [Extract Key Files] 
    → [Generate Prompt] 
    → [AI Model] 
    → [Generate README] 
    → [Write to File System]
```

## Key Features

* **Repository Analysis**: Deep inspection of repository structure and key files
* **Intelligent Content Generation**: Uses Gemini 2.0 Flash model to create context-aware documentation
* **Standardized Formatting**: Produces professional, well-structured markdown files
* **Flexible Integration**: Can be used as standalone script or integrated into larger workflows

## Setup and Installation

### Folder Structure
```
.
├── README.md
├── readme-creator.png
├── pyproject.toml
├── readme_creator/
├── tests/
├── eval/
    └── data/
```

### Prerequisites

1. Python 3.8 or higher
2. Git command-line tools installed and in system PATH
3. Google API key for Gemini access
4. Basic file system permissions for repository cloning

### Installation

1. Clone this repository:
   ```bash
   git clone _clone this repo_
   cd readme-creator
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install google-generativeai python-dotenv
   ```

4. Set up environment variables:
   Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Running the Agent

### Basic Usage

- You may run the agent on a web interface. This will start a web server on your machine. You may go to the URL printed on the screen and interact with the agent in a chatbot interface.
    ```bash
    cd readme-creator
    adk web
    ```
    Please select the `readme_creator` option from the dropdown list located at the top left of the screen. Now you can start talking to the agent!

## Example Output

For a Python project, the agent generate:

```markdown
# Project Name

## Overview
This repository contains a Python implementation of [brief description]. The project appears to be a [type of application] with [key features].

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/username/repository.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Contributing
Pull requests are welcome. For major changes, please open an issue first.

```

## Evaluation

The agent's performance can be evaluated on:

1. **Completeness**: Presence of all standard README sections
2. **Accuracy**: Correct interpretation of project purpose
3. **Usefulness**: Practical installation and usage instructions
4. **Formatting**: Proper markdown syntax and readability

Sample test cases are provided in `/tests` directory.

## Troubleshooting

### Common Issues

1. **Git Command Errors**:
   - Ensure git is installed and in PATH
   - Verify network connectivity to GitHub

2. **Permission Errors**:
   - Check write permissions in target directory
   - Run with appropriate user privileges

3. **API Key Problems**:
   - Verify `.env` file is in correct location
   - Confirm key has proper Gemini API permissions

## Security Considerations

1. The agent executes git clone commands - only use with trusted repositories
2. API keys should be properly secured and not committed to version control
3. File system operations are performed with the same permissions as the running user

## Future Enhancements

1. Multi-language documentation generation
2. Interactive mode for user input/confirmation
3. Template system for different project types

## License

This project is licensed under the [MIT License] - see the LICENSE file for details.

## Acknowledgements

- Google for the Gemini AI models
- GitHub for repository hosting
- The open source community for inspiration