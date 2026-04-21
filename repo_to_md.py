import os

def repo_to_markdown(repo_path='.', output_file='repo_dump.md'):
    """
    Scans a repository and compiles all code files into a single Markdown file.
    """
    # Folders to completely ignore to save space and time
    ignore_dirs = {
        '.git', 'node_modules', 'venv', 'env', '__pycache__', 
        'dist', 'build', '.idea', '.vscode', 'coverage'
    }
    
    # File extensions to ignore (images, compiled binaries, etc.)
    ignore_exts = {
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.pdf', 
        '.zip', '.tar', '.gz', '.mp4', '.mp3', '.wav', 
        '.pyc', '.exe', '.dll', '.so', '.o', '.a', '.class', '.jar', '.ttf'
    }

    # Files to specifically ignore (like the script itself or the output file)
    ignore_files = {'repo_to_md.py', 'repo_dump.md', 'package-lock.json', 'yarn.lock'}

    print(f"Scanning repository at: {os.path.abspath(repo_path)}...")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write("# Repository Code Dump\n\n")

        for root, dirs, files in os.walk(repo_path):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                
                # Skip ignored extensions and specific files
                if ext in ignore_exts or file in ignore_files:
                    continue

                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_path)

                try:
                    # Attempt to read the file as UTF-8
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Determine markdown code block language for syntax highlighting
                    language = ext.lstrip('.') if ext else 'text'
                    
                    # Map common extensions to markdown language tags
                    if language == 'js': language = 'javascript'
                    elif language == 'py': language = 'python'
                    elif language == 'ts': language = 'typescript'

                    # Write to the markdown file
                    out_f.write(f"## File: `{rel_path}`\n\n")
                    out_f.write(f"```{language}\n")
                    out_f.write(content)
                    
                    # Ensure the code block ends on a new line
                    if not content.endswith('\n'):
                        out_f.write('\n')
                    out_f.write(f"```\n\n")

                except UnicodeDecodeError:
                    # Silently skip files that aren't plain text (caught as decode errors)
                    print(f"Skipped unreadable/binary file: {rel_path}")
                except Exception as e:
                    print(f"Error reading {rel_path}: {e}")

    print(f"\nDone! Your code has been compiled into '{output_file}'.")

if __name__ == "__main__":
    # Run the function in the current directory
    repo_to_markdown()